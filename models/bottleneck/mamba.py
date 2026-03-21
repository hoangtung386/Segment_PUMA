import torch
import torch.nn as nn
from .base import BaseBottleneck


class MambaBottleneckWrapper(BaseBottleneck):
    """Mamba SSM bottleneck with fallback to standard conv."""

    def __init__(self, in_channels: int = 1024, mamba_depth: int = 4):
        super().__init__(in_channels)

        try:
            from models.layers.mamba import MambaBlock
            self.block = MambaBlock(in_channels=in_channels, depth=mamba_depth)
        except ImportError:
            print("Warning: Mamba not available. Using standard conv bottleneck.")
            from .standard import StandardBottleneck
            self.block = StandardBottleneck(in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
