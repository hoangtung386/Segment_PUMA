"""
Nuclei Decoder - Multi-head FPN decoder for nuclei instance segmentation.

Inspired by Hover-Net, this decoder produces three output heads:
  1. NP (Nuclei Pixel): Binary segmentation of nuclei vs background [B, 2, H, W]
  2. HV (Horizontal-Vertical): Gradient maps for instance separation [B, 2, H, W]
  3. NC (Nuclei Classification): Per-pixel class prediction [B, num_classes, H, W]

The NP and HV heads enable instance-level segmentation during post-processing,
while NC provides per-pixel classification that is aggregated per instance.

Architecture:
  - Shared decoder trunk processes features up to a mid-level
  - Three independent head branches diverge for NP, HV, NC
  - Each head has its own final convolution layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.hvt import AttentionGate


class NucleiDecoderBlock(nn.Module):
    """Decoder block with optional attention gating."""

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
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        if self.attention is not None:
            skip = self.attention(skip, x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeadBranch(nn.Module):
    """Final prediction branch for a single head (NP, HV, or NC).

    Takes mid-level decoder features and produces full-resolution predictions
    through additional convolutions and upsampling.
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2)
        self.final = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return self.final(x)


class NucleiDecoder(nn.Module):
    """Multi-head decoder for nuclei instance segmentation.

    Shared trunk:
        bottleneck -> dec5(+s4) -> dec4(+s3) -> dec3(+s2) -> dec2(+s1) -> dec1(+stem)

    Three heads diverge from dec1 output:
        - NP head: [B, 2, H, W]  (nuclei pixel: background vs foreground)
        - HV head: [B, 2, H, W]  (horizontal and vertical gradient maps)
        - NC head: [B, num_nuclei_classes, H, W]  (per-pixel nuclei classification)

    Args:
        num_nuclei_classes: Number of nuclei classes (4 for Track 1, 11 for Track 2).
        encoder_channels: Tuple of (stem, s1, s2, s3, s4) channel sizes.
        bottleneck_channels: Channel size of bottleneck output.
        use_attention: Whether to use attention gates on skip connections.
    """

    def __init__(self, num_nuclei_classes=4,
                 encoder_channels=(64, 64, 128, 256, 512),
                 bottleneck_channels=1024,
                 use_attention=True):
        super().__init__()

        ec = encoder_channels

        # Shared decoder trunk
        self.dec5 = NucleiDecoderBlock(bottleneck_channels, ec[4], ec[4], use_attention)
        self.dec4 = NucleiDecoderBlock(ec[4], ec[3], ec[3], use_attention)
        self.dec3 = NucleiDecoderBlock(ec[3], ec[2], ec[2], use_attention)
        self.dec2 = NucleiDecoderBlock(ec[2], ec[1], ec[1], use_attention)
        self.dec1 = NucleiDecoderBlock(ec[1], ec[0], ec[0], use_attention)

        # Three prediction heads (branch from dec1 features at H/2 resolution)
        # Each head: conv blocks -> upsample to full resolution -> 1x1 prediction
        self.np_head = HeadBranch(ec[0], ec[0], 2)       # nuclei pixel (bg/fg)
        self.hv_head = HeadBranch(ec[0], ec[0], 2)       # horizontal + vertical
        self.nc_head = HeadBranch(ec[0], ec[0], num_nuclei_classes)  # classification

        # Multi-scale heads for deep supervision (on NC branch)
        self.multiscale_heads = nn.ModuleList([
            nn.Conv2d(ec[4], num_nuclei_classes, 1),
            nn.Conv2d(ec[3], num_nuclei_classes, 1),
            nn.Conv2d(ec[2], num_nuclei_classes, 1),
            nn.Conv2d(ec[1], num_nuclei_classes, 1),
        ])

    def forward(self, encoder_features, bottleneck):
        """
        Args:
            encoder_features: [stem, s1, s2, s3, s4] - may be tissue-guided via TGNA.
            bottleneck: [B, C_bn, H/64, W/64] from shared bottleneck.

        Returns:
            np_pred: [B, 2, H, W] nuclei pixel logits.
            hv_pred: [B, 2, H, W] horizontal-vertical gradient predictions.
            nc_pred: [B, num_classes, H, W] nuclei classification logits.
            multiscale_preds: list of 4 multi-scale NC predictions for deep supervision.
        """
        stem, s1, s2, s3, s4 = encoder_features

        multiscale_preds = []

        d5 = self.dec5(bottleneck, s4)
        multiscale_preds.append(self.multiscale_heads[0](d5))

        d4 = self.dec4(d5, s3)
        multiscale_preds.append(self.multiscale_heads[1](d4))

        d3 = self.dec3(d4, s2)
        multiscale_preds.append(self.multiscale_heads[2](d3))

        d2 = self.dec2(d3, s1)
        multiscale_preds.append(self.multiscale_heads[3](d2))

        d1 = self.dec1(d2, stem)

        # Three heads diverge from shared decoder trunk
        np_pred = self.np_head(d1)    # [B, 2, H, W]
        hv_pred = self.hv_head(d1)    # [B, 2, H, W]
        nc_pred = self.nc_head(d1)    # [B, num_classes, H, W]

        return np_pred, hv_pred, nc_pred, multiscale_preds
