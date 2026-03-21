"""
Loss functions for cell segmentation.

Components:
1. Dice + Focal + CrossEntropy (main segmentation loss)
2. Multi-scale deep supervision (cluster loss)
3. Boundary refinement loss
4. False positive penalty
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes=2, class_weights=None,
                 dice_weight=0.7, focal_weight=0.3, ce_weight=0.3,
                 cluster_weight=0.1, boundary_weight=0.1,
                 fp_penalty_weight=0.3):
        super().__init__()

        self.num_classes = num_classes

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
        self.cluster_weight = cluster_weight
        self.boundary_weight = boundary_weight
        self.fp_penalty_weight = fp_penalty_weight

        self.scale_weights = [0.5, 0.3, 0.15, 0.05]

    def compute_main_loss(self, output, target):
        if target.ndim == 3:
            target_dice = target.unsqueeze(1)
            target_ce = target.long()
        else:
            target_dice = target
            target_ce = target.squeeze(1).long()

        dice_loss = self.dice(output, target_dice)
        focal_loss = self.focal(output, target_dice)
        ce_loss = self.ce(output, target_ce)

        main_loss = (
            self.dice_weight * dice_loss
            + self.focal_weight * focal_loss
            + self.ce_weight * ce_loss
        )

        # False positive penalty (must use probabilities, not raw logits)
        output_probs = F.softmax(output, dim=1)

        if target_dice.shape[1] == 1:
            y_true_bg = (target_dice == 0).float()
        else:
            y_true_bg = target_dice[:, 0:1, :, :]

        y_pred_nobg = 1.0 - output_probs[:, 0:1, :, :]
        fp_loss = (y_pred_nobg * y_true_bg).mean()
        main_loss += self.fp_penalty_weight * fp_loss

        return main_loss, {
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'ce': ce_loss.item(),
            'fp_penalty': fp_loss.item(),
        }

    def compute_cluster_loss(self, cluster_outputs, target):
        if not cluster_outputs:
            return torch.tensor(0.0, device=target.device), {}

        total_loss = 0.0
        num_stages = len(cluster_outputs)
        weights = self.scale_weights[:num_stages]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        for cluster_out, w in zip(cluster_outputs, weights):
            H_out, W_out = cluster_out.shape[-2:]
            target_scaled = F.interpolate(
                target.unsqueeze(1).float() if target.ndim == 3 else target.float(),
                size=(H_out, W_out), mode='nearest',
            )
            dice_loss = self.dice(cluster_out, target_scaled.long())
            focal_loss = self.focal(cluster_out, target_scaled.long())
            total_loss += (dice_loss + focal_loss) * w

        return total_loss, {'cluster_total': total_loss.item()}

    def compute_boundary_loss(self, output, target):
        # Use soft predictions (softmax) to keep gradient flow
        pred_probs = F.softmax(output, dim=1)
        # Sum of non-background class probabilities as a soft foreground map
        pred_fg = pred_probs[:, 1:, :, :].sum(dim=1, keepdim=True)

        target_float = target.unsqueeze(1).float() if target.ndim == 3 else target.float()
        target_fg = (target_float > 0).float()

        # Morphological gradient for target boundary
        target_dilated = F.max_pool2d(target_fg, 3, stride=1, padding=1)
        target_eroded = -F.max_pool2d(-target_fg, 3, stride=1, padding=1)
        target_boundary = (target_dilated - target_eroded).clamp(0, 1)

        # Soft morphological gradient for prediction (differentiable)
        pred_dilated = F.max_pool2d(pred_fg, 3, stride=1, padding=1)
        pred_eroded = -F.max_pool2d(-pred_fg, 3, stride=1, padding=1)
        pred_boundary = (pred_dilated - pred_eroded).clamp(0, 1)

        # Soft boundary Dice (differentiable)
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum()
        boundary_dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

        return boundary_dice, {'boundary_dice': boundary_dice.item()}

    def forward(self, output, target, cluster_outputs=None):
        loss_dict = {}

        main_loss, main_dict = self.compute_main_loss(output, target)
        loss_dict.update(main_dict)

        cluster_loss, cluster_dict = self.compute_cluster_loss(cluster_outputs, target)
        loss_dict.update(cluster_dict)

        boundary_loss, boundary_dict = self.compute_boundary_loss(output, target)
        loss_dict.update(boundary_dict)

        total_loss = (
            main_loss
            + self.cluster_weight * cluster_loss
            + self.boundary_weight * boundary_loss
        )

        loss_dict['total'] = total_loss.item()
        loss_dict['main'] = main_loss.item()

        return total_loss, loss_dict
