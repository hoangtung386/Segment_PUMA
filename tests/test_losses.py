"""Tests for loss functions."""
import pytest
import torch
from models.losses import SegmentationLoss


class TestSegmentationLoss:
    def setup_method(self):
        self.criterion = SegmentationLoss(num_classes=4, class_weights=None)

    def test_basic_forward(self):
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        loss, loss_dict = self.criterion(pred, target)

        assert loss.ndim == 0  # scalar
        assert loss.item() > 0
        assert 'total' in loss_dict
        assert 'dice' in loss_dict
        assert 'focal' in loss_dict
        assert 'ce' in loss_dict

    def test_with_multiscale(self):
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        multiscale = [
            torch.randn(2, 4, 4, 4),
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 4, 16, 16),
        ]
        loss, loss_dict = self.criterion(pred, target, multiscale)

        assert loss.item() > 0
        assert 'cluster_total' in loss_dict

    def test_without_multiscale(self):
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        loss, loss_dict = self.criterion(pred, target, cluster_outputs=None)

        assert loss.item() > 0

    def test_gradient_flows(self):
        pred = torch.randn(2, 4, 32, 32, requires_grad=True)
        target = torch.randint(0, 4, (2, 32, 32))
        loss, _ = self.criterion(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_target_3d_and_4d(self):
        pred = torch.randn(2, 4, 16, 16)
        # 3D target
        target_3d = torch.randint(0, 4, (2, 16, 16))
        loss_3d, _ = self.criterion(pred, target_3d)
        assert loss_3d.item() > 0

        # 4D target
        target_4d = target_3d.unsqueeze(1)
        loss_4d, _ = self.criterion(pred, target_4d)
        assert loss_4d.item() > 0

    def test_boundary_loss_component(self):
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        _, loss_dict = self.criterion(pred, target)
        assert 'boundary_dice' in loss_dict

    def test_fp_penalty_component(self):
        pred = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        _, loss_dict = self.criterion(pred, target)
        assert 'fp_penalty' in loss_dict

    def test_with_class_weights(self):
        weights = torch.tensor([0.5, 1.0, 1.5, 2.0])
        criterion = SegmentationLoss(num_classes=4, class_weights=weights)
        pred = torch.randn(2, 4, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16))
        loss, _ = criterion(pred, target)
        assert loss.item() > 0
