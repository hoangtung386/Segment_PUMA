"""
Multi-task loss for PUMANet panoptic segmentation.

Combines losses for all output heads:
  1. Tissue loss:  Dice + Focal + CE (semantic segmentation)
  2. NP loss:      BCE (nuclei pixel detection)
  3. HV loss:      MSE + gradient MSE (horizontal-vertical maps)
  4. NC loss:      Dice + Focal + CE (nuclei classification)
  5. Deep supervision: multi-scale losses for both tissue and nuclei

Total loss = w_tissue * L_tissue + w_np * L_np + w_hv * L_hv + w_nc * L_nc
           + w_ms * (L_tissue_ms + L_nc_ms)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss


class TissueLoss(nn.Module):
    """Semantic segmentation loss for tissue (Dice + Focal + CE)."""

    def __init__(self, num_classes=6, class_weights=None,
                 dice_weight=0.5, focal_weight=0.3, ce_weight=0.3):
        super().__init__()
        self.dice = DiceLoss(
            include_background=False, to_onehot_y=True,
            softmax=True, reduction='mean',
        )
        self.focal = FocalLoss(
            include_background=False, to_onehot_y=True,
            gamma=2.0, reduction='mean',
        )
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] logits.
            target: [B, H, W] class indices.
        """
        target_dice = target.unsqueeze(1) if target.ndim == 3 else target
        target_ce = target.long() if target.ndim == 3 else target.squeeze(1).long()

        loss = (self.dice_weight * self.dice(pred, target_dice)
                + self.focal_weight * self.focal(pred, target_dice)
                + self.ce_weight * self.ce(pred, target_ce))

        return loss


class NucleiPixelLoss(nn.Module):
    """Binary cross-entropy loss for nuclei pixel detection (NP head)."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, np_pred, np_target):
        """
        Args:
            np_pred: [B, 2, H, W] logits (bg, fg).
            np_target: [B, H, W] binary (0=bg, 1=nuclei).
        """
        return self.ce(np_pred, np_target.long())


class HVLoss(nn.Module):
    """Loss for horizontal-vertical gradient maps.

    Combines:
      1. MSE loss: pixel-wise regression on HV values
      2. Gradient MSE: penalizes incorrect gradient directions at instance
         boundaries, encouraging sharp transitions between touching nuclei

    The gradient component is critical for separating touching instances.
    """

    def __init__(self, mse_weight=1.0, grad_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight

    def _compute_gradient(self, x):
        """Compute spatial gradients using Sobel-like finite differences.

        Args:
            x: [B, 2, H, W] HV maps.

        Returns:
            grad: [B, 2, H, W] gradient magnitude for each channel.
        """
        # Horizontal gradient (along W)
        grad_h = x[:, :, :, 2:] - x[:, :, :, :-2]
        # Vertical gradient (along H)
        grad_v = x[:, :, 2:, :] - x[:, :, :-2, :]

        # Pad to original size
        grad_h = F.pad(grad_h, (1, 1, 0, 0), mode='replicate')
        grad_v = F.pad(grad_v, (0, 0, 1, 1), mode='replicate')

        return grad_h, grad_v

    def forward(self, hv_pred, hv_target, mask):
        """
        Args:
            hv_pred: [B, 2, H, W] predicted HV maps.
            hv_target: [B, 2, H, W] ground truth HV maps.
            mask: [B, H, W] binary mask of nuclei pixels (only compute loss where nuclei exist).
        """
        # Expand mask for 2-channel HV maps
        mask_2ch = mask.unsqueeze(1).expand_as(hv_pred).float()

        # MSE loss (only on nuclei pixels)
        mse = F.mse_loss(hv_pred * mask_2ch, hv_target * mask_2ch, reduction='sum')
        num_pixels = mask_2ch.sum().clamp(min=1.0)
        mse_loss = mse / num_pixels

        # Gradient MSE loss (important for instance boundary sharpness)
        pred_gh, pred_gv = self._compute_gradient(hv_pred)
        target_gh, target_gv = self._compute_gradient(hv_target)

        grad_mse_h = F.mse_loss(pred_gh * mask_2ch, target_gh * mask_2ch, reduction='sum')
        grad_mse_v = F.mse_loss(pred_gv * mask_2ch, target_gv * mask_2ch, reduction='sum')
        grad_loss = (grad_mse_h + grad_mse_v) / num_pixels

        return self.mse_weight * mse_loss + self.grad_weight * grad_loss


class NucleiClassLoss(nn.Module):
    """Classification loss for nuclei (Dice + Focal + CE), same as TissueLoss."""

    def __init__(self, num_classes=4, class_weights=None,
                 dice_weight=0.5, focal_weight=0.3, ce_weight=0.3):
        super().__init__()
        self.dice = DiceLoss(
            include_background=False, to_onehot_y=True,
            softmax=True, reduction='mean',
        )
        self.focal = FocalLoss(
            include_background=False, to_onehot_y=True,
            gamma=2.0, reduction='mean',
        )
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        target_dice = target.unsqueeze(1) if target.ndim == 3 else target
        target_ce = target.long() if target.ndim == 3 else target.squeeze(1).long()

        loss = (self.dice_weight * self.dice(pred, target_dice)
                + self.focal_weight * self.focal(pred, target_dice)
                + self.ce_weight * self.ce(pred, target_ce))
        return loss


class MultiScaleLoss(nn.Module):
    """Deep supervision loss applied to multi-scale predictions."""

    def __init__(self, num_classes, scale_weights=(0.5, 0.3, 0.15, 0.05)):
        super().__init__()
        self.dice = DiceLoss(
            include_background=False, to_onehot_y=True,
            softmax=True, reduction='mean',
        )
        self.focal = FocalLoss(
            include_background=False, to_onehot_y=True,
            gamma=2.0, reduction='mean',
        )
        self.scale_weights = list(scale_weights)
        w_sum = sum(self.scale_weights)
        self.scale_weights = [w / w_sum for w in self.scale_weights]

    def forward(self, multiscale_preds, target):
        if not multiscale_preds:
            return torch.tensor(0.0, device=target.device)

        total = 0.0
        target_float = target.unsqueeze(1).float() if target.ndim == 3 else target.float()

        for pred, w in zip(multiscale_preds, self.scale_weights):
            t_scaled = F.interpolate(
                target_float, size=pred.shape[-2:], mode='nearest',
            ).long()
            total += w * (self.dice(pred, t_scaled) + self.focal(pred, t_scaled))

        return total


class PUMALoss(nn.Module):
    """Combined multi-task loss for PUMANet.

    Args:
        num_tissue_classes: Number of tissue classes.
        num_nuclei_classes: Number of nuclei classes.
        tissue_class_weights: Optional class weights for tissue CE loss.
        nuclei_class_weights: Optional class weights for nuclei CE loss.
        mode: 'tissue', 'nuclei', or 'joint'.
        w_tissue: Weight for tissue loss.
        w_np: Weight for NP loss.
        w_hv: Weight for HV loss.
        w_nc: Weight for NC loss.
        w_ms: Weight for multi-scale deep supervision.
    """

    def __init__(self, num_tissue_classes=6, num_nuclei_classes=4,
                 tissue_class_weights=None, nuclei_class_weights=None,
                 mode='joint',
                 w_tissue=1.0, w_np=1.0, w_hv=2.0, w_nc=1.0, w_ms=0.1):
        super().__init__()
        self.mode = mode

        self.w_tissue = w_tissue
        self.w_np = w_np
        self.w_hv = w_hv
        self.w_nc = w_nc
        self.w_ms = w_ms

        if mode in ('tissue', 'joint'):
            self.tissue_loss = TissueLoss(
                num_classes=num_tissue_classes,
                class_weights=tissue_class_weights,
            )
            self.tissue_ms_loss = MultiScaleLoss(num_tissue_classes)

        if mode in ('nuclei', 'joint'):
            self.np_loss = NucleiPixelLoss()
            self.hv_loss = HVLoss()
            self.nc_loss = NucleiClassLoss(
                num_classes=num_nuclei_classes,
                class_weights=nuclei_class_weights,
            )
            self.nuclei_ms_loss = MultiScaleLoss(num_nuclei_classes)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from PUMANet forward pass.
            targets: dict with keys:
                'tissue_mask': [B, H, W] tissue class labels
                'nuclei_mask': [B, H, W] nuclei class labels
                'hv_map':      [B, 2, H, W] HV gradient targets
                'np_map':      [B, H, W] binary nuclei mask

        Returns:
            total_loss: scalar tensor.
            loss_dict: dict of individual loss components for logging.
        """
        loss_dict = {}
        total = 0.0

        # --- Tissue losses ---
        if self.mode in ('tissue', 'joint') and 'tissue_pred' in outputs:
            tissue_target = targets['tissue_mask']

            l_tissue = self.tissue_loss(outputs['tissue_pred'], tissue_target)
            loss_dict['tissue'] = l_tissue.item()
            total += self.w_tissue * l_tissue

            l_tissue_ms = self.tissue_ms_loss(
                outputs.get('tissue_multiscale', []), tissue_target,
            )
            loss_dict['tissue_ms'] = l_tissue_ms.item()
            total += self.w_ms * l_tissue_ms

        # --- Nuclei losses ---
        if self.mode in ('nuclei', 'joint') and 'np_pred' in outputs:
            np_target = targets['np_map']
            hv_target = targets['hv_map']
            nc_target = targets['nuclei_mask']

            # NP loss
            l_np = self.np_loss(outputs['np_pred'], np_target)
            loss_dict['np'] = l_np.item()
            total += self.w_np * l_np

            # HV loss (only on nuclei pixels)
            l_hv = self.hv_loss(outputs['hv_pred'], hv_target, np_target)
            loss_dict['hv'] = l_hv.item()
            total += self.w_hv * l_hv

            # NC loss
            l_nc = self.nc_loss(outputs['nc_pred'], nc_target)
            loss_dict['nc'] = l_nc.item()
            total += self.w_nc * l_nc

            # Multi-scale NC loss
            l_nc_ms = self.nuclei_ms_loss(
                outputs.get('nuclei_multiscale', []), nc_target,
            )
            loss_dict['nc_ms'] = l_nc_ms.item()
            total += self.w_ms * l_nc_ms

        loss_dict['total'] = total.item()
        return total, loss_dict
