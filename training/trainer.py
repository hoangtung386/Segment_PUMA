import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, List, Optional, NamedTuple
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from monai.networks.utils import one_hot
from monai.metrics import DiceMetric
from models.losses import SegmentationLoss
from configs.constants import IMAGENET_MEAN, IMAGENET_STD


class ValidationResult(NamedTuple):
    """Result of a validation epoch."""
    dice: float
    loss: float
    loss_details: Dict[str, float]
    per_class_dice: Optional[np.ndarray]
    vis_images: Optional[torch.Tensor] = None
    vis_masks: Optional[torch.Tensor] = None
    vis_preds: Optional[torch.Tensor] = None


class Trainer:
    def __init__(self, model: nn.Module, train_loader: 'DataLoader',
                 val_loader: 'DataLoader', config: 'TrainingConfig',
                 device: torch.device, multi_gpu: bool = False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.multi_gpu = multi_gpu
        self.use_wandb = config.USE_WANDB and WANDB_AVAILABLE

        # Class weights
        if hasattr(config, 'CUSTOM_CLASS_WEIGHTS') and config.CUSTOM_CLASS_WEIGHTS:
            class_weights = torch.tensor(config.CUSTOM_CLASS_WEIGHTS, dtype=torch.float32).to(device)
        else:
            from utils.data_utils import compute_class_weights
            class_weights = compute_class_weights(
                train_loader.dataset,
                num_classes=config.NUM_CLASSES,
                num_samples=2000,
            ).to(device)

        print(f"Class weights: {class_weights.cpu().tolist()}")

        self.criterion = SegmentationLoss(
            num_classes=config.NUM_CLASSES,
            class_weights=class_weights,
            dice_weight=config.DICE_WEIGHT,
            focal_weight=config.FOCAL_WEIGHT,
            ce_weight=config.CE_WEIGHT,
            cluster_weight=config.CLUSTER_WEIGHT,
            boundary_weight=config.BOUNDARY_WEIGHT,
            fp_penalty_weight=config.FP_PENALTY_WEIGHT,
        )

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Warmup + CosineAnnealingWarmRestarts
        self.warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 0)
        self.base_lr = config.LEARNING_RATE

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.SCHEDULER_T0,
            T_mult=config.SCHEDULER_T_MULT,
            eta_min=config.SCHEDULER_ETA_MIN,
        )

        # Per-class Dice (include_background=False → returns N-1 classes)
        self.dice_metric = DiceMetric(
            include_background=False, reduction='mean_batch',
        )
        self.best_dice = 0.0
        self.patience_counter = 0
        self.history = []
        self.global_step = 0
        self.start_epoch = 1

        # AMP (Automatic Mixed Precision)
        self.use_amp = getattr(config, 'USE_AMP', False) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.use_amp:
            print("AMP (Mixed Precision) enabled")

        # Resolve class names for logging
        self.class_names = self._get_class_names()

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a saved checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}. Starting from scratch.")
            return

        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load model state
        model = self.model.module if self.multi_gpu else self.model
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer & scheduler state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load AMP scaler state
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_dice = checkpoint.get('best_dice', 0.0)
        print(f"Resumed at epoch {self.start_epoch}, best_dice={self.best_dice:.4f}")

    def _get_class_names(self) -> List[str]:
        """Get class names for per-class logging."""
        return self.config.get_class_names()

    def train_epoch(self, epoch: int) -> tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0
        loss_accumulators = {}
        grad_norms = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                pred = outputs['pred']
                multiscale = outputs.get('multiscale_preds', None)
                loss, loss_dict = self.criterion(pred, masks, multiscale)

            # NaN guard: skip batch if loss is NaN/Inf
            if not torch.isfinite(loss):
                print(f"  Warning: NaN/Inf loss detected, skipping batch")
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()

            # Unscale before clipping so grad_norm is in original scale
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.GRAD_CLIP_NORM
            )
            grad_norms.append(grad_norm.item())

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_accumulators[k] = loss_accumulators.get(k, 0.0) + v

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Batch-level W&B logging
            if self.use_wandb:
                batch_log = {
                    'batch/train_loss': loss.item(),
                    'batch/grad_norm': grad_norm.item(),
                    'batch/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                }
                # Log each loss component per batch
                for k, v in loss_dict.items():
                    batch_log[f'batch/{k}'] = v
                wandb.log(batch_log)

        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in loss_accumulators.items()}
        avg_metrics['grad_norm_mean'] = np.mean(grad_norms)
        avg_metrics['grad_norm_max'] = np.max(grad_norms)

        return avg_loss, avg_metrics

    def validate(self, epoch: int) -> ValidationResult:
        self.model.eval()
        self.dice_metric.reset()
        total_val_loss = 0
        num_batches = 0
        val_loss_accumulators: Dict[str, float] = {}

        # For prediction visualization
        vis_images, vis_masks, vis_preds = None, None, None

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(images)
                    pred = outputs['pred']
                    multiscale = outputs.get('multiscale_preds', None)
                    loss, loss_dict = self.criterion(pred, masks, multiscale)
                total_val_loss += loss.item()
                num_batches += 1

                for k, v in loss_dict.items():
                    val_loss_accumulators[k] = val_loss_accumulators.get(k, 0.0) + v

                # Per-class Dice via MONAI
                masks_metric = masks.unsqueeze(1) if masks.ndim == 3 else masks
                y_pred_idx = torch.argmax(pred, dim=1, keepdim=True)
                y_pred_onehot = one_hot(y_pred_idx, num_classes=self.config.NUM_CLASSES)
                y_target_onehot = one_hot(masks_metric, num_classes=self.config.NUM_CLASSES)

                self.dice_metric(y_pred=y_pred_onehot, y=y_target_onehot)
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

                # Save first batch for visualization
                if vis_images is None:
                    vis_images = images[:4].cpu()
                    vis_masks = masks[:4].cpu()
                    vis_preds = y_pred_idx[:4, 0].cpu()

        if num_batches == 0:
            print("\nWarning: Validation set is empty, skipping metrics.")
            return ValidationResult(0.0, 0.0, {}, None)

        # Per-class Dice: shape (num_classes-1,)
        per_class_dice = self.dice_metric.aggregate()
        if per_class_dice.ndim > 0:
            per_class_dice = per_class_dice.cpu().numpy()
        else:
            per_class_dice = np.array([per_class_dice.item()])

        val_dice = float(np.nanmean(per_class_dice))
        val_loss = total_val_loss / num_batches
        val_avg_losses = {k: v / num_batches for k, v in val_loss_accumulators.items()}

        # Print per-class dice
        print(f"\nValidation: Loss={val_loss:.4f}, Mean Dice={val_dice:.4f}")
        for i, d in enumerate(per_class_dice):
            cls_name = self.class_names[i + 1] if (i + 1) < len(self.class_names) else f'Class{i+1}'
            print(f"  {cls_name}: Dice={d:.4f}")

        return ValidationResult(
            dice=val_dice, loss=val_loss, loss_details=val_avg_losses,
            per_class_dice=per_class_dice,
            vis_images=vis_images, vis_masks=vis_masks, vis_preds=vis_preds,
        )

    def _log_wandb_visualizations(self, epoch, vis_images, vis_masks, vis_preds):
        """Log sample predictions to W&B as images."""
        if vis_images is None or not self.use_wandb:
            return

        import matplotlib.pyplot as plt

        num_samples = min(vis_images.shape[0], 4)
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]

        mean = np.array(IMAGENET_MEAN)
        std = np.array(IMAGENET_STD)

        for i in range(num_samples):
            img = vis_images[i].numpy()
            img = (img * std[:, None, None] + mean[:, None, None])
            img = np.clip(img.transpose(1, 2, 0), 0, 1)

            mask = vis_masks[i].numpy()
            pred = vis_preds[i].numpy()

            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask, cmap='tab10', vmin=0, vmax=self.config.NUM_CLASSES - 1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='tab10', vmin=0, vmax=self.config.NUM_CLASSES - 1)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.tight_layout()
        wandb.log({
            'predictions': wandb.Image(fig, caption=f'Epoch {epoch}'),
            'epoch': epoch,
        })
        plt.close(fig)

    def train(self, num_epochs: int) -> None:
        print(f"\nStarting Training\nEpochs: {num_epochs}\nDevice: {self.device}\n{'=' * 60}")

        # Log model architecture summary
        if self.use_wandb:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.run.summary['model/total_params'] = total_params
            wandb.run.summary['model/trainable_params'] = trainable_params
            wandb.run.summary['model/total_params_M'] = total_params / 1e6
            wandb.run.summary['data/train_samples'] = len(self.train_loader.dataset)
            wandb.run.summary['data/val_samples'] = len(self.val_loader.dataset)

        for epoch in range(self.start_epoch, num_epochs + 1):
            epoch_start = time.time()

            # Warmup: linearly ramp LR from 1e-6 to base_lr
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                warmup_lr = 1e-6 + (self.base_lr - 1e-6) * (epoch / self.warmup_epochs)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr
                print(f"  Warmup LR: {warmup_lr:.6f}")

            train_loss, train_metrics = self.train_epoch(epoch)

            val = self.validate(epoch)

            epoch_time = time.time() - epoch_start

            # Only step cosine scheduler after warmup
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            if val.dice > self.best_dice:
                self.best_dice = val.dice
                self.patience_counter = 0
                self._save_checkpoint(epoch, val.dice, is_best=True)
            else:
                self.patience_counter += 1

            self._save_checkpoint(epoch, val.dice, is_best=False)
            self._log_history(epoch, train_loss, val.loss, val.dice,
                              train_metrics, val.loss_details, val.per_class_dice, epoch_time)

            # Log visualizations every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                self._log_wandb_visualizations(epoch, val.vis_images, val.vis_masks, val.vis_preds)

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best Dice: {self.best_dice:.4f}")
                break

        print(f"\nTraining Complete! Best Dice: {self.best_dice:.4f}")

        if self.use_wandb:
            wandb.run.summary['best_dice'] = self.best_dice

    def _save_checkpoint(self, epoch: int, val_dice: float, is_best: bool = False) -> None:
        model_state = self.model.module.state_dict() if self.multi_gpu else self.model.state_dict()
        prefix = 'best_' if is_best else ''
        path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'{prefix}{self.config.DATASET_NAME}.pth',
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_dice': self.best_dice,
            'val_dice': val_dice,
            'config': self.config.to_dict(),
        }, path)

        if is_best:
            print(f"Best model saved: {path} (Dice: {val_dice:.4f})")

    def _log_history(self, epoch: int, train_loss: float, val_loss: float,
                     val_dice: float, train_metrics: Dict[str, float],
                     val_avg_losses: Dict[str, float],
                     per_class_dice: Optional[np.ndarray], epoch_time: float) -> None:
        history_dict = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'epoch_time': epoch_time,
        }
        history_dict.update({f'train_{k}': v for k, v in train_metrics.items()})
        self.history.append(history_dict)

        # CSV log
        log_file = os.path.join(self.config.OUTPUT_DIR, f"training_{self.config.DATASET_NAME}_log.csv")
        mode = 'w' if (epoch == 1 and self.start_epoch == 1) else 'a'
        with open(log_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history_dict.keys())
            if epoch == 1:
                writer.writeheader()
            writer.writerow(history_dict)

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"Dice={val_dice:.4f}, Best={self.best_dice:.4f}, Time={epoch_time:.1f}s")

        # W&B epoch-level logging
        if self.use_wandb:
            log = {
                'epoch': epoch,
                'epoch_time_sec': epoch_time,

                # Train losses (averaged over epoch)
                'train/loss': train_loss,
                'train/dice_loss': train_metrics.get('dice', 0),
                'train/focal_loss': train_metrics.get('focal', 0),
                'train/ce_loss': train_metrics.get('ce', 0),
                'train/fp_penalty': train_metrics.get('fp_penalty', 0),
                'train/boundary_dice': train_metrics.get('boundary_dice', 0),
                'train/cluster_total': train_metrics.get('cluster_total', 0),

                # Gradient stats
                'train/grad_norm_mean': train_metrics.get('grad_norm_mean', 0),
                'train/grad_norm_max': train_metrics.get('grad_norm_max', 0),

                # Val losses
                'val/loss': val_loss,
                'val/dice_loss': val_avg_losses.get('dice', 0),
                'val/focal_loss': val_avg_losses.get('focal', 0),
                'val/ce_loss': val_avg_losses.get('ce', 0),

                # Val metrics
                'val/dice': val_dice,
                'val/best_dice': self.best_dice,
                'val/patience': self.patience_counter,

                # Learning rate
                'learning_rate': self.optimizer.param_groups[0]['lr'],
            }

            # Per-class Dice scores
            if per_class_dice is not None:
                for i, d in enumerate(per_class_dice):
                    cls_name = self.class_names[i + 1] if (i + 1) < len(self.class_names) else f'class{i+1}'
                    log[f'val/dice_{cls_name}'] = float(d) if not np.isnan(d) else 0.0

            # GPU memory
            if torch.cuda.is_available():
                log['system/gpu_memory_allocated_GB'] = torch.cuda.max_memory_allocated() / 1e9
                log['system/gpu_memory_reserved_GB'] = torch.cuda.max_memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()

            wandb.log(log)
