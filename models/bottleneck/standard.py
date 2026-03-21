import torch
import torch.nn as nn
from .base import BaseBottleneck


class StandardBottleneck(BaseBottleneck):
    """Standard convolutional bottleneck (fallback when Mamba is unavailable)."""

    def __init__(self, in_channels: int = 1024):
        super().__init__(in_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x
