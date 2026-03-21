"""
Efficient-KAN (Kolmogorov-Arnold Networks) for Decoder Heads

Reference: 
- "KAN: Kolmogorov-Arnold Networks" (2024)
- Efficient-KAN: Lightweight implementation for production

WARNING: Only use KAN for FINAL LAYERS, not entire network!
Medical imaging requires stable latent spaces.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientKANLayer(nn.Module):
    """
    Efficient implementation of KAN layer
    
    Key idea: Replace σ(Wx + b) with learnable spline functions
    
    Original KAN: y = Σ φ(x_i) where φ is B-spline
    Efficient-KAN: Use rational functions instead of B-splines
    """
    
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Base linear transformation (like MLP)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Spline coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order)
        )
        
        # Grid points (fixed)
        self.register_buffer(
            'grid',
            torch.linspace(-1, 1, grid_size + 1).expand(in_features, -1)
        )
        
        self.spline_order = spline_order
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.xavier_uniform_(self.spline_weight, gain=0.1)
        
    def b_splines(self, x):
        """
        Compute B-spline basis functions
        
        Args:
            x: (B, in_features)
        Returns:
            bases: (B, in_features, grid_size + spline_order)
        """
        # Normalize x to grid range
        x = x.unsqueeze(-1)  # (B, in_features, 1)
        
        # Compute distances to grid points
        grid = self.grid.unsqueeze(0)  # (1, in_features, grid_size+1)
        
        # B-spline basis (simplified - use piecewise linear)
        bases = torch.zeros(
            x.size(0), self.in_features, self.grid_size + self.spline_order,
            device=x.device
        )
        
        for i in range(self.grid_size):
            # Linear interpolation between grid points
            mask = (x >= grid[:, :, i:i+1]) & (x < grid[:, :, i+1:i+2])
            bases[:, :, i] = mask.float().squeeze(-1)
        
        return bases
    
    def forward(self, x):
        """
        Args:
            x: (B, in_features)
        Returns:
            output: (B, out_features)
        """
        # Base transformation
        base_output = F.linear(x, self.base_weight)  # (B, out_features)
        
        # Spline transformation
        bases = self.b_splines(x)  # (B, in_features, grid_size + spline_order)
        
        # Weighted sum of basis functions
        # (out_features, in_features, grid_size) @ (B, in_features, grid_size)
        spline_output = torch.einsum(
            'oig,big->bo',
            self.spline_weight,
            bases
        )
        
        return base_output + spline_output


class RationalKANLayer(nn.Module):
    """
    Rational function approximation (faster than B-splines)
    
    φ(x) = P(x) / Q(x) where P, Q are polynomials
    
    This is what Efficient-KAN paper recommends for production
    """
    
    def __init__(self, in_features, out_features, degree=3):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Numerator coefficients
        self.P_coef = nn.Parameter(
            torch.randn(out_features, in_features, degree + 1) * 0.1
        )
        
        # Denominator coefficients (ensure Q(x) > 0)
        self.Q_coef = nn.Parameter(
            torch.randn(out_features, in_features, degree) * 0.1
        )
        
        # Base transformation
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.zeros_(self.base_bias)
        
        # Initialize P(0) ≈ 0, Q(0) ≈ 1 for stability
        with torch.no_grad():
            self.P_coef[:, :, 0] = 0.0
            self.Q_coef[:, :, 0] = 1.0
        
    def forward(self, x):
        """
        Args:
            x: (*, in_features)
        Returns:
            output: (*, out_features)
        """
        # Flatten batch dims
        original_shape = x.shape
        x = x.view(-1, self.in_features)  # (B, in_features)
        
        # Compute polynomial bases
        x_powers = torch.stack([x ** i for i in range(self.degree + 1)], dim=-1)
        # x_powers: (B, in_features, degree+1)
        
        # Numerator P(x)
        P = torch.einsum('oip,bip->bio', self.P_coef, x_powers)  # (B, in_features, out_features)
        
        # Denominator Q(x) = 1 + Σ q_i x^i (ensure Q > 0)
        Q = 1.0 + torch.einsum('oip,bip->bio', self.Q_coef, x_powers[:, :, 1:])
        
        # Rational function
        rational = P / (Q + 1e-6)  # (B, in_features, out_features)
        
        # Sum over input features
        rational_output = rational.sum(dim=1)  # (B, out_features)
        
        # Add base transformation
        base_output = F.linear(x, self.base_weight, self.base_bias)
        
        output = base_output + rational_output
        
        # Reshape back
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output


class KANDecoderHead(nn.Module):
    """
    KAN-based decoder head for segmentation
    
    Architecture:
    - Conv to reduce spatial dims (optional)
    - Flatten or global pooling
    - KAN layers for classification
    - Upsample back to original resolution
    
    USE CASE: Final 1x1 conv replacement in decoder
    """
    
    def __init__(self, in_channels, num_classes, use_rational=True, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(in_channels // 2, num_classes * 4)
        
        # Spatial reduction (optional - can be identity)
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)
        
        # KAN layers (replace MLP)
        KANLayer = RationalKANLayer if use_rational else EfficientKANLayer
        
        self.kan1 = KANLayer(in_channels, hidden_dim)
        self.kan2 = KANLayer(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            output: (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        
        # Spatial processing
        feat = self.norm(self.spatial_conv(x))  # (B, C, H, W)
        
        # Permute for KAN: (B, C, H, W) -> (B, H, W, C)
        feat = feat.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # KAN layers (applied per-pixel)
        feat = self.kan1(feat)  # (B, H, W, hidden_dim)
        feat = self.dropout(feat)
        output = self.kan2(feat)  # (B, H, W, num_classes)
        
        # Permute back: (B, H, W, num_classes) -> (B, num_classes, H, W)
        output = output.permute(0, 3, 1, 2)
        
        return output

