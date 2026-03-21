from .base import BaseBottleneck
from .mamba import MambaBottleneckWrapper
from .standard import StandardBottleneck


def get_bottleneck(bottleneck_type: str, in_channels: int, **kwargs) -> BaseBottleneck:
    if bottleneck_type.lower() == 'mamba':
        return MambaBottleneckWrapper(in_channels=in_channels, **kwargs)
    elif bottleneck_type.lower() == 'standard':
        return StandardBottleneck(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown bottleneck type: {bottleneck_type}. Available: mamba, standard")


__all__ = ['BaseBottleneck', 'MambaBottleneckWrapper', 'StandardBottleneck', 'get_bottleneck']
