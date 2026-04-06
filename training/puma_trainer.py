"""
PUMATrainer - Multi-task trainer for PUMANet dual-decoder architecture.

Handles joint training of tissue segmentation + nuclei instance segmentation.
Supports three modes: 'tissue', 'nuclei', 'joint'.

Key differences from original Trainer:
  - Multi-task loss (PUMALoss) with separate tissue/nuclei components
  - Dict-based batch format (image + multiple masks)
  - Separate validation metrics for tissue Dice and nuclei Dice
  - Best model selection based on combined metric
"""
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
from models.losses_v2 import PUMALoss
from configs.constants import IMAGENET_MEAN, IMAGENET_STD


class ValidationResult(NamedTuple):
    score: float          # Combined score used for model selection
    tissue_dice: float
    nuclei_dice: float
    loss: float
    loss_details: Dict[str, float]
    per_class_tissue_dice: Optional[np.ndarray]
    per_class_nuclei_dice: Optional[np.ndarray]


class PUMATrainer:
    """Multi-task trainer for PUMANet.

    Args:
        model: PUMANet instance.
        train_loader: DataLoader returning dict batches.
        val_loader: DataLoader returning dict batches.
        config: TrainingConfig with V2 fields.
        device: torch.device.
    """

    def __init__(self, model: nn.Module, train_loader, val_loader,
                 config, device: torch.device, multi_gpu: bool = False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.multi_gpu = multi_gpu
        self.mode = config.MODE
        self.use_wandb = config.USE_WANDB and WANDB_AVAILABLE

        # Compute class weights
        tissue_weights = None
        nuclei_weights = None
        if self.mode in ('tissue', 'joint'):
            tissue_weights = self._compute_class_weights('tissue_mask', config.NUM_TISSUE_CLASSES)
        if self.mode in ('nuclei', 'joint'):
            nuclei_weights = self._compute_class_weights('nuclei_mask', config.NUM_NUCLEI_CLASSES)

        # Multi-task loss
        self.criterion = PUMALoss(
            num_tissue_classes=config.NUM_TISSUE_CLASSES,
            num_nuclei_classes=config.NUM_NUCLEI_CLASSES,
            tissue_class_weights=tissue_weights,
            nuclei_class_weights=nuclei_weights,
            mode=self.mode,
            w_tissue=config.W_TISSUE,
            w_np=config.W_NP,
            w_hv=config.W_HV,
            w_nc=config.W_NC,
            w_ms=config.W_MS,
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 0)
        self.base_lr = config.LEARNING_RATE
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.SCHEDULER_T0,
            T_mult=config.SCHEDULER_T_MULT,
            eta_min=config.SCHEDULER_ETA_MIN,
        )

        # Validation metrics
        if self.mode in ('tissue', 'joint'):
            self.tissue_dice_metric = DiceMetric(include_background=False, reduction='mean_batch')
        if self.mode in ('nuclei', 'joint'):
            self.nuclei_dice_metric = DiceMetric(include_background=False, reduction='mean_batch')

        self.best_score = 0.0
        self.patience_counter = 0
        self.history = []
        self.global_step = 0
        self.start_epoch = 1

        # AMP
        self.use_amp = getattr(config, 'USE_AMP', False) and torch.cuda.is_available()
        if self.use_amp and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            self.scaler = torch.amp.GradScaler('cuda', enabled=False)
            print("AMP enabled with bfloat16")
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            if self.use_amp:
                print("AMP enabled with float16")

    def _compute_class_weights(self, mask_key, num_classes):
        """Compute inverse frequency class weights from training data."""
        counts = torch.zeros(num_classes)
        num_samples = min(len(self.train_loader.dataset), 500)

        for i in range(num_samples):
            sample = self.train_loader.dataset[i]
            if mask_key in sample:
                mask = sample[mask_key]
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy()
                for c in range(num_classes):
                    counts[c] += (mask == c).sum()

        # Inverse frequency with smoothing
        total = counts.sum()
        weights = total / (num_classes * counts.clamp(min=1))
        weights = weights / weights.sum() * num_classes

        print(f"Class weights ({mask_key}): {weights.tolist()}")
        return weights.to(self.device)

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}. Starting from scratch.")
            return

        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model = self.model.module if self.multi_gpu else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_score = checkpoint.get('best_score', 0.0)
        print(f"Resumed at epoch {self.start_epoch}, best_score={self.best_score:.4f}")

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        loss_accumulators = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != 'image'}

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)

            if not torch.isfinite(loss):
                print(f"  Warning: NaN/Inf loss, skipping batch")
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.GRAD_CLIP_NORM,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1
            num_batches += 1

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_accumulators[k] = loss_accumulators.get(k, 0.0) + v

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if self.use_wandb:
                wandb.log({
                    'batch/train_loss': loss.item(),
                    'batch/grad_norm': grad_norm.item(),
                    'batch/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                    **{f'batch/{k}': v for k, v in loss_dict.items()},
                })

        avg_loss = total_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in loss_accumulators.items()}
        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> ValidationResult:
        self.model.eval()
        total_loss = 0
        num_batches = 0
        loss_accumulators = {}

        if self.mode in ('tissue', 'joint'):
            self.tissue_dice_metric.reset()
        if self.mode in ('nuclei', 'joint'):
            self.nuclei_dice_metric.reset()

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != 'image'}

            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1
            for k, v in loss_dict.items():
                loss_accumulators[k] = loss_accumulators.get(k, 0.0) + v

            # Tissue Dice
            if self.mode in ('tissue', 'joint') and 'tissue_pred' in outputs:
                t_pred = torch.argmax(outputs['tissue_pred'], dim=1, keepdim=True)
                t_pred_oh = one_hot(t_pred, num_classes=self.config.NUM_TISSUE_CLASSES)
                t_target = targets['tissue_mask'].unsqueeze(1)
                t_target_oh = one_hot(t_target, num_classes=self.config.NUM_TISSUE_CLASSES)
                self.tissue_dice_metric(y_pred=t_pred_oh, y=t_target_oh)

            # Nuclei Dice (on NC head)
            if self.mode in ('nuclei', 'joint') and 'nc_pred' in outputs:
                n_pred = torch.argmax(outputs['nc_pred'], dim=1, keepdim=True)
                n_pred_oh = one_hot(n_pred, num_classes=self.config.NUM_NUCLEI_CLASSES)
                n_target = targets['nuclei_mask'].unsqueeze(1)
                n_target_oh = one_hot(n_target, num_classes=self.config.NUM_NUCLEI_CLASSES)
                self.nuclei_dice_metric(y_pred=n_pred_oh, y=n_target_oh)

            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        if num_batches == 0:
            return ValidationResult(0.0, 0.0, 0.0, 0.0, {}, None, None)

        val_loss = total_loss / num_batches
        val_losses = {k: v / num_batches for k, v in loss_accumulators.items()}

        # Tissue dice
        tissue_dice = 0.0
        per_class_tissue = None
        if self.mode in ('tissue', 'joint'):
            per_class_tissue = self.tissue_dice_metric.aggregate().cpu().numpy()
            tissue_dice = float(np.nanmean(per_class_tissue))

        # Nuclei dice
        nuclei_dice = 0.0
        per_class_nuclei = None
        if self.mode in ('nuclei', 'joint'):
            per_class_nuclei = self.nuclei_dice_metric.aggregate().cpu().numpy()
            nuclei_dice = float(np.nanmean(per_class_nuclei))

        # Combined score for model selection
        if self.mode == 'tissue':
            score = tissue_dice
        elif self.mode == 'nuclei':
            score = nuclei_dice
        else:  # joint
            score = 0.4 * tissue_dice + 0.6 * nuclei_dice

        # Print results
        print(f"\nValidation: Loss={val_loss:.4f}, Score={score:.4f}")
        if per_class_tissue is not None:
            tissue_names = ['Tumor', 'Stroma', 'Epithelium', 'Blood Vessel', 'Necrosis']
            print(f"  Tissue Dice={tissue_dice:.4f}: " +
                  ", ".join(f"{n}={d:.3f}" for n, d in zip(tissue_names, per_class_tissue)))
        if per_class_nuclei is not None:
            from configs.constants import get_task_config
            nuclei_names = get_task_config('nuclei', self.config.NUCLEI_TRACK).class_names[1:]
            print(f"  Nuclei Dice={nuclei_dice:.4f}: " +
                  ", ".join(f"{n}={d:.3f}" for n, d in zip(nuclei_names, per_class_nuclei)))

        return ValidationResult(
            score=score, tissue_dice=tissue_dice, nuclei_dice=nuclei_dice,
            loss=val_loss, loss_details=val_losses,
            per_class_tissue_dice=per_class_tissue,
            per_class_nuclei_dice=per_class_nuclei,
        )

    def train(self, num_epochs: int) -> None:
        print(f"\nStarting PUMANet Training (mode={self.mode})")
        print(f"Epochs: {num_epochs}, Device: {self.device}")
        print("=" * 60)

        if self.use_wandb:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            wandb.run.summary['model/total_params'] = total_params
            wandb.run.summary['model/total_params_M'] = total_params / 1e6
            wandb.run.summary['model/trainable_params'] = trainable_params

        for epoch in range(self.start_epoch, num_epochs + 1):
            epoch_start = time.time()

            # Warmup
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                warmup_lr = 1e-6 + (self.base_lr - 1e-6) * (epoch / self.warmup_epochs)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

            train_loss, train_metrics = self.train_epoch(epoch)
            val = self.validate(epoch)
            epoch_time = time.time() - epoch_start

            if epoch > self.warmup_epochs:
                self.scheduler.step()

            # Model selection
            if val.score > self.best_score:
                self.best_score = val.score
                self.patience_counter = 0
                self._save_checkpoint(epoch, val.score, is_best=True)
            else:
                self.patience_counter += 1

            self._save_checkpoint(epoch, val.score, is_best=False)
            self._log_history(epoch, train_loss, val, epoch_time)

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best Score: {self.best_score:.4f}")
                break

        print(f"\nTraining Complete! Best Score: {self.best_score:.4f}")

        if self.use_wandb:
            wandb.run.summary['best_score'] = self.best_score

    def _save_checkpoint(self, epoch, score, is_best=False):
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
            'best_score': self.best_score,
            'score': score,
            'config': self.config.to_dict(),
            'mode': self.mode,
        }, path)

        if is_best:
            print(f"Best model saved: {path} (Score: {score:.4f})")

    def _log_history(self, epoch, train_loss, val, epoch_time):
        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val.loss:.4f}, "
              f"Score={val.score:.4f}, Best={self.best_score:.4f}, Time={epoch_time:.1f}s")

        if self.use_wandb:
            log = {
                'epoch': epoch,
                'epoch_time_sec': epoch_time,
                'train/loss': train_loss,
                'val/loss': val.loss,
                'val/score': val.score,
                'val/best_score': self.best_score,
                'val/patience': self.patience_counter,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
            }

            if val.tissue_dice > 0:
                log['val/tissue_dice'] = val.tissue_dice
            if val.nuclei_dice > 0:
                log['val/nuclei_dice'] = val.nuclei_dice

            for k, v in val.loss_details.items():
                log[f'val/{k}'] = v

            if val.per_class_tissue_dice is not None:
                tissue_names = ['Tumor', 'Stroma', 'Epithelium', 'BloodVessel', 'Necrosis']
                for i, d in enumerate(val.per_class_tissue_dice):
                    name = tissue_names[i] if i < len(tissue_names) else f'class{i+1}'
                    log[f'val/tissue_dice_{name}'] = float(d) if not np.isnan(d) else 0.0

            if val.per_class_nuclei_dice is not None:
                from configs.constants import get_task_config
                nuclei_names = get_task_config('nuclei', self.config.NUCLEI_TRACK).class_names[1:]
                for i, d in enumerate(val.per_class_nuclei_dice):
                    name = nuclei_names[i] if i < len(nuclei_names) else f'class{i+1}'
                    log[f'val/nuclei_dice_{name}'] = float(d) if not np.isnan(d) else 0.0

            if torch.cuda.is_available():
                log['system/gpu_memory_GB'] = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()

            wandb.log(log)
