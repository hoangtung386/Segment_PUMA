import torch
import torch.nn as nn
from .base import BaseBottleneck


class MambaBottleneckWrapper(BaseBottleneck):
    """Mamba SSM bottleneck with fallback to standard conv."""

    def __init__(self, in_channels: int = 1024, mamba_depth: int = 4):
        super().__init__(in_channels)

        try:
            from models.layers.mamba import MambaBlock, MAMBA_AVAILABLE
            if not MAMBA_AVAILABLE:
                raise ImportError("mamba-ssm not installed")
            block = MambaBlock(in_channels=in_channels, depth=mamba_depth)
            # Smoke-test CUDA kernels (causal-conv1d) to catch runtime failures early
            with torch.no_grad():
                _dummy = torch.randn(1, in_channels, 4, 4, device='cuda')
                block.cuda()(_dummy)
                del _dummy
            torch.cuda.empty_cache()
            self.block = block
            print("Mamba bottleneck: CUDA kernels verified OK")
        except Exception as e:
            print(f"Warning: Mamba not available ({e}). Using standard conv bottleneck.")
            from .standard import StandardBottleneck
            self.block = StandardBottleneck(in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
