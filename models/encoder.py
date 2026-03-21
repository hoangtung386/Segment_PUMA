"""
ResNet-style encoder with SE attention for training from scratch.

Architecture per stage:
    Input → [ResBlock + SE] x N → Downsample → Output

Designed for histopathology where pretrained weights are not used.
Residual connections prevent gradient vanishing in deep networks.
SE (Squeeze-and-Excitation) provides channel attention at minimal cost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """Residual block with optional SE attention.

    Conv3x3 → BN → ReLU → Conv3x3 → BN → (+skip) → ReLU → SE
    """

    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Skip connection: match dimensions if needed
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        out = self.se(out)
        return out


class EncoderStage(nn.Module):
    """One encoder stage: N ResBlocks + output features before/after downsample."""

    def __init__(self, in_channels, out_channels, num_blocks=2, use_se=True):
        super().__init__()
        blocks = [ResBlock(in_channels, out_channels, stride=1, use_se=use_se)]
        for _ in range(1, num_blocks):
            blocks.append(ResBlock(out_channels, out_channels, stride=1, use_se=use_se))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        feat = self.blocks(x)
        return feat, self.pool(feat)


class ResNetEncoder(nn.Module):
    """4-stage ResNet encoder with SE attention.

    Stem → Stage1(64) → Stage2(128) → Stage3(256) → Stage4(512)
    Returns list of skip features at each resolution.
    """

    def __init__(self, in_channels=3, channels=(64, 128, 256, 512),
                 blocks_per_stage=2, use_se=True, dropout=0.1):
        super().__init__()

        # Stem: aggressive downsampling for large inputs
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1)
        # After stem: H/4, W/4

        self.stage1 = EncoderStage(channels[0], channels[0], blocks_per_stage, use_se)
        self.stage2 = EncoderStage(channels[0], channels[1], blocks_per_stage, use_se)
        self.stage3 = EncoderStage(channels[1], channels[2], blocks_per_stage, use_se)
        self.stage4 = EncoderStage(channels[2], channels[3], blocks_per_stage, use_se)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: list of 5 tensors [stem, s1, s2, s3, s4] at decreasing resolutions
                stem:  [B, 64,  H/4,  W/4]  (after stem+pool, before stage1 pool)
                s1:    [B, 64,  H/8,  W/8]
                s2:    [B, 128, H/16, W/16]
                s3:    [B, 256, H/32, W/32]
                s4:    [B, 512, H/64, W/64]
            bottleneck_input: s4 pooled [B, 512, H/64, W/64]
        """
        # Stem: H → H/4
        x = self.stem(x)
        stem_feat = x                      # [B, 64, H/2, W/2] before stem_pool
        x = self.stem_pool(x)              # [B, 64, H/4, W/4]

        s1_feat, x = self.stage1(x)        # feat: [B, 64,  H/4,  W/4],  pool: H/8
        x = self.dropout(x)
        s2_feat, x = self.stage2(x)        # feat: [B, 128, H/8,  W/8],  pool: H/16
        x = self.dropout(x)
        s3_feat, x = self.stage3(x)        # feat: [B, 256, H/16, W/16], pool: H/32
        x = self.dropout(x)
        s4_feat, x = self.stage4(x)        # feat: [B, 512, H/32, W/32], pool: H/64

        return [stem_feat, s1_feat, s2_feat, s3_feat, s4_feat], x
