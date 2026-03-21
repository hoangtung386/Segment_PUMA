"""
FPN-style decoder with attention gates for segmentation.

Replaces dead kMaX clustering with proper attention gates on skip connections.
Multi-scale prediction heads for deep supervision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AttentionGate(nn.Module):
    """Attention gate: learns which spatial regions in skip features are relevant.

    gate = sigmoid(Wg * g + Ws * s + bias)
    output = skip * gate
    """

    def __init__(self, skip_channels, gate_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or skip_channels // 2

        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip, gate):
        """
        Args:
            skip: encoder features [B, C_skip, H, W]
            gate: decoder features (upsampled) [B, C_gate, H, W]
        Returns:
            attended skip features [B, C_skip, H, W]
        """
        s = self.W_skip(skip)
        g = self.W_gate(gate)

        # Match spatial dims if needed
        if s.shape[2:] != g.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode='bilinear', align_corners=False)

        attn = self.psi(self.relu(s + g))
        return skip * attn


class DecoderBlock(nn.Module):
    """Decoder block: upsample → attention gate on skip → concat → 2x Conv."""

    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.attention = AttentionGate(skip_channels, out_channels) if use_attention else None

        cat_channels = out_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(cat_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle size mismatch (non-power-of-2 inputs)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        if self.attention is not None:
            skip = self.attention(skip, x)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder with attention gates.

    Expects encoder features: [stem, s1, s2, s3, s4] + bottleneck
    Produces multi-scale predictions for deep supervision.
    """

    def __init__(self, num_classes=6,
                 encoder_channels=(64, 64, 128, 256, 512),
                 bottleneck_channels=1024,
                 use_attention=True):
        super().__init__()

        # encoder_channels = [stem=64, s1=64, s2=128, s3=256, s4=512]
        # Decoder goes: bottleneck → s4 → s3 → s2 → s1 → stem
        ec = encoder_channels

        self.dec5 = DecoderBlock(bottleneck_channels, ec[4], ec[4], use_attention)  # +s4 → 512
        self.dec4 = DecoderBlock(ec[4], ec[3], ec[3], use_attention)               # +s3 → 256
        self.dec3 = DecoderBlock(ec[3], ec[2], ec[2], use_attention)               # +s2 → 128
        self.dec2 = DecoderBlock(ec[2], ec[1], ec[1], use_attention)               # +s1 → 64
        self.dec1 = DecoderBlock(ec[1], ec[0], ec[0], use_attention)               # +stem → 64

        # Final upsampling to original resolution (stem is at H/2)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(ec[0], ec[0], 2, stride=2),
            nn.BatchNorm2d(ec[0]),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(ec[0], num_classes, 1)

        # Multi-scale heads for deep supervision
        self.heads = nn.ModuleList([
            nn.Conv2d(ec[4], num_classes, 1),   # H/32 scale
            nn.Conv2d(ec[3], num_classes, 1),   # H/16 scale
            nn.Conv2d(ec[2], num_classes, 1),   # H/8  scale
            nn.Conv2d(ec[1], num_classes, 1),   # H/4  scale
        ])

    def forward(self, encoder_features, bottleneck):
        """
        Args:
            encoder_features: [stem, s1, s2, s3, s4] from ResNetEncoder
            bottleneck: [B, 1024, H/64, W/64]
        Returns:
            [final_pred, scale1_pred, scale2_pred, scale3_pred, scale4_pred]
            final_pred is at full input resolution, others at decreasing scales.
        """
        stem, s1, s2, s3, s4 = encoder_features

        multiscale_preds = []

        x = self.dec5(bottleneck, s4)               # [B, 512, H/32, W/32]
        multiscale_preds.append(self.heads[0](x))

        x = self.dec4(x, s3)                        # [B, 256, H/16, W/16]
        multiscale_preds.append(self.heads[1](x))

        x = self.dec3(x, s2)                        # [B, 128, H/8, W/8]
        multiscale_preds.append(self.heads[2](x))

        x = self.dec2(x, s1)                        # [B, 64, H/4, W/4]
        multiscale_preds.append(self.heads[3](x))

        x = self.dec1(x, stem)                      # [B, 64, H/2, W/2]
        x = self.final_upsample(x)                  # [B, 64, H, W]
        final_pred = self.final(x)                   # [B, num_classes, H, W]

        return [final_pred] + multiscale_preds
