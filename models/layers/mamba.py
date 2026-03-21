"""
Mamba-2 State Space Model layers for sequence modeling.
"""
import torch
import torch.nn as nn
from einops import rearrange


try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MambaBlock(nn.Module):
    """Mamba SSM block for 2D feature maps (flattened to sequences)."""

    def __init__(self, in_channels=1024, depth=4, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.in_channels = in_channels

        if MAMBA_AVAILABLE:
            self.layers = nn.ModuleList([
                Mamba2(d_model=in_channels, d_state=d_state,
                       d_conv=d_conv, expand=expand)
                for _ in range(depth)
            ])
        else:
            self.layers = nn.ModuleList([
                SimplifiedSSM(in_channels, d_state)
                for _ in range(depth)
            ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(in_channels) for _ in range(depth)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        seq = rearrange(x, 'b c h w -> b (h w) c')

        for layer, norm in zip(self.layers, self.norms):
            seq = seq + layer(norm(seq))

        return rearrange(seq, 'b (h w) c -> b c h w', h=H, w=W)


class SimplifiedSSM(nn.Module):
    """Simplified State Space Model (fallback when mamba-ssm not available)."""

    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))

        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        B, L, D = x.shape
        u = self.in_proj(x)

        dt = 0.001
        A_bar = torch.eye(self.d_state, device=x.device) + dt * self.A
        B_bar = dt * self.B

        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []

        for t in range(L):
            h = h @ A_bar.T + u[:, t, :] @ B_bar.T
            y = h @ self.C.T + u[:, t, :] * self.D
            outputs.append(y)

        return self.out_proj(torch.stack(outputs, dim=1))
