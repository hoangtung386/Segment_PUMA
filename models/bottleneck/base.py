import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseBottleneck(nn.Module, ABC):
    """Abstract base class for bottlenecks."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        pass
