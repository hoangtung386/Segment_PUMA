"""
Tissue-Guided Nuclei Attention (TGNA) Module.

Novel cross-task attention mechanism where tissue decoder features modulate
encoder skip connections before they reach the nuclei decoder. This mimics
how pathologists use tissue context (e.g., stroma, epithelium, blood vessel)
to inform nuclei classification decisions.

Mechanism at each scale:
  1. Tissue features and encoder features are projected to a shared space
  2. A gating signal is computed via sigmoid activation
  3. Encoder features are modulated: out = encoder * gate + encoder (residual)

This allows nuclei segmentation to benefit from tissue context without
hard-coupling the two tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TGNAGate(nn.Module):
    """Single-scale tissue-guided gating unit.

    Computes a spatial attention gate from tissue and encoder features,
    then applies it to modulate the encoder features for the nuclei decoder.
    """

    def __init__(self, enc_channels, tissue_channels):
        super().__init__()
        inter_ch = max(enc_channels // 4, 16)

        self.proj_enc = nn.Sequential(
            nn.Conv2d(enc_channels, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.proj_tissue = nn.Sequential(
            nn.Conv2d(tissue_channels, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.gate = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, enc_channels, 1, bias=False),
            nn.BatchNorm2d(enc_channels),
            nn.Sigmoid(),
        )

    def forward(self, enc_feat, tissue_feat):
        """
        Args:
            enc_feat: [B, C_enc, H, W] encoder skip features
            tissue_feat: [B, C_tissue, H, W] tissue decoder features at same scale

        Returns:
            modulated: [B, C_enc, H, W] tissue-guided encoder features
        """
        # Match spatial dimensions if needed
        if tissue_feat.shape[2:] != enc_feat.shape[2:]:
            tissue_feat = F.interpolate(
                tissue_feat, size=enc_feat.shape[2:],
                mode='bilinear', align_corners=False,
            )

        e = self.proj_enc(enc_feat)
        t = self.proj_tissue(tissue_feat)
        attention = self.gate(e + t)

        # Residual gating: preserve original information + tissue guidance
        return enc_feat * attention + enc_feat


class TGNA(nn.Module):
    """Tissue-Guided Nuclei Attention - multi-scale cross-task fusion.

    Applies TGNAGate at each encoder scale to create tissue-aware
    skip connections for the nuclei decoder.

    Args:
        encoder_channels: tuple of channel sizes for [stem, s1, s2, s3, s4]
        tissue_decoder_channels: tuple of channel sizes from tissue decoder
            at corresponding scales (may differ from encoder channels)
    """

    def __init__(self, encoder_channels, tissue_decoder_channels=None):
        super().__init__()
        if tissue_decoder_channels is None:
            tissue_decoder_channels = encoder_channels

        self.gates = nn.ModuleList([
            TGNAGate(enc_ch, tis_ch)
            for enc_ch, tis_ch in zip(encoder_channels, tissue_decoder_channels)
        ])

    def forward(self, encoder_features, tissue_features):
        """
        Args:
            encoder_features: list of [stem, s1, s2, s3, s4] tensors
            tissue_features: list of tissue decoder features at each scale

        Returns:
            fused_features: list of tissue-guided encoder features
        """
        fused = []
        for enc, tissue, gate in zip(encoder_features, tissue_features, self.gates):
            fused.append(gate(enc, tissue))
        return fused
