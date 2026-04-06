"""
Tissue Decoder - FPN-style decoder for semantic tissue segmentation.

Similar to the original FPNDecoder but also returns intermediate features
at each scale, which are used by TGNA to guide the nuclei decoder.

Output: tissue segmentation logits + multi-scale intermediate features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from models.decoder.hvt import AttentionGate


class TissueDecoderBlock(nn.Module):
    """Decoder block: upsample -> attention gate on skip -> concat -> 2x Conv."""

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


class TissueDecoder(nn.Module):
    """FPN decoder for tissue segmentation that exposes intermediate features.

    Architecture:
        bottleneck -> dec5(+s4) -> dec4(+s3) -> dec3(+s2) -> dec2(+s1) -> dec1(+stem)
        -> final_upsample -> tissue prediction

    Returns both the final prediction and intermediate features at each
    decoder stage, which TGNA uses to guide the nuclei decoder.

    Args:
        num_classes: Number of tissue classes (default 6).
        encoder_channels: Tuple of (stem, s1, s2, s3, s4) channel sizes.
        bottleneck_channels: Channel size of bottleneck output.
        use_attention: Whether to use attention gates on skip connections.
    """

    def __init__(self, num_classes=6,
                 encoder_channels=(64, 64, 128, 256, 512),
                 bottleneck_channels=1024,
                 use_attention=True):
        super().__init__()

        ec = encoder_channels

        self.dec5 = TissueDecoderBlock(bottleneck_channels, ec[4], ec[4], use_attention)
        self.dec4 = TissueDecoderBlock(ec[4], ec[3], ec[3], use_attention)
        self.dec3 = TissueDecoderBlock(ec[3], ec[2], ec[2], use_attention)
        self.dec2 = TissueDecoderBlock(ec[2], ec[1], ec[1], use_attention)
        self.dec1 = TissueDecoderBlock(ec[1], ec[0], ec[0], use_attention)

        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(ec[0], ec[0], 2, stride=2),
            nn.BatchNorm2d(ec[0]),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(ec[0], num_classes, 1)

        # Multi-scale heads for deep supervision
        self.heads = nn.ModuleList([
            nn.Conv2d(ec[4], num_classes, 1),
            nn.Conv2d(ec[3], num_classes, 1),
            nn.Conv2d(ec[2], num_classes, 1),
            nn.Conv2d(ec[1], num_classes, 1),
        ])

    def forward(self, encoder_features, bottleneck):
        """
        Args:
            encoder_features: [stem, s1, s2, s3, s4] from encoder.
            bottleneck: [B, C_bn, H/64, W/64] from bottleneck.

        Returns:
            tissue_pred: [B, num_classes, H, W] tissue segmentation logits.
            multiscale_preds: list of 4 multi-scale predictions.
            tissue_features: list of 5 intermediate features [d5, d4, d3, d2, d1]
                at the same spatial scales as encoder_features [s4, s3, s2, s1, stem].
                These are consumed by TGNA for cross-task fusion.
        """
        stem, s1, s2, s3, s4 = encoder_features

        tissue_features = []
        multiscale_preds = []

        d5 = self.dec5(bottleneck, s4)
        tissue_features.append(d5)
        multiscale_preds.append(self.heads[0](d5))

        d4 = self.dec4(d5, s3)
        tissue_features.append(d4)
        multiscale_preds.append(self.heads[1](d4))

        d3 = self.dec3(d4, s2)
        tissue_features.append(d3)
        multiscale_preds.append(self.heads[2](d3))

        d2 = self.dec2(d3, s1)
        tissue_features.append(d2)
        multiscale_preds.append(self.heads[3](d2))

        d1 = self.dec1(d2, stem)
        tissue_features.append(d1)

        x = self.final_upsample(d1)
        tissue_pred = self.final(x)

        # Reorder tissue_features to match encoder_features order:
        # encoder: [stem, s1, s2, s3, s4] -> spatial: [H/2, H/4, H/8, H/16, H/32]
        # tissue:  [d5,   d4, d3, d2, d1] -> spatial: [H/32, H/16, H/8, H/4, H/2]
        # Reverse to match encoder order (stem first)
        tissue_features_ordered = list(reversed(tissue_features))

        return tissue_pred, multiscale_preds, tissue_features_ordered
