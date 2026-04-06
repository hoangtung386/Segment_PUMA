"""
PUMANet - Dual-Decoder Architecture for Panoptic Segmentation of
nUclei and tissue in advanced MelanomA.

Novel architecture contributions:
  1. Shared encoder with Mamba SSM bottleneck for global context modeling
  2. Dual task-specific decoders (tissue + nuclei) branching from shared features
  3. Tissue-Guided Nuclei Attention (TGNA) for cross-task feature fusion
  4. Multi-head nuclei decoder (NP + HV + NC) for instance segmentation

Architecture:
    Input [B, 3, H, W]
        |
    Shared ResNet Encoder (SE attention)
        |
    Shared Mamba/Conv Bottleneck
        |
    +---+---+
    |       |
    v       v
  Tissue  TGNA (tissue features guide nuclei skip connections)
  Decoder   |
    |       v
    |   Nuclei Decoder (NP + HV + NC heads)
    |       |
    v       v
  tissue  np_pred, hv_pred, nc_pred
  pred

Supports three training modes:
  - 'tissue': Only tissue decoder active
  - 'nuclei': Only nuclei decoder active (no tissue guidance)
  - 'joint':  Both decoders + TGNA cross-task fusion
"""
import torch
import torch.nn as nn

from models.encoder import ResNetEncoder
from models.bottleneck import get_bottleneck
from models.decoder.tissue_decoder import TissueDecoder
from models.decoder.nuclei_decoder import NucleiDecoder
from models.fusion import TGNA


class PUMANet(nn.Module):
    """Dual-decoder panoptic segmentation network for PUMA challenge.

    Args:
        in_channels: Input image channels (default 3 for RGB).
        num_tissue_classes: Number of tissue classes (default 6).
        num_nuclei_classes: Number of nuclei classes (4 for Track 1, 11 for Track 2).
        encoder_channels: Channel sizes for encoder stages.
        bottleneck_channels: Bottleneck feature dimension.
        blocks_per_stage: Number of ResBlocks per encoder stage.
        bottleneck_type: 'mamba' or 'standard'.
        use_se: Use Squeeze-and-Excitation in encoder.
        use_attention: Use attention gates in decoders.
        dropout: Dropout rate in encoder.
        mode: Training mode - 'tissue', 'nuclei', or 'joint'.
    """

    def __init__(self, in_channels=3,
                 num_tissue_classes=6,
                 num_nuclei_classes=4,
                 encoder_channels=(64, 128, 256, 512),
                 bottleneck_channels=1024,
                 blocks_per_stage=2,
                 bottleneck_type='mamba',
                 use_se=True,
                 use_attention=True,
                 dropout=0.1,
                 mode='joint',
                 **kwargs):
        super().__init__()

        self.mode = mode

        # --- Shared Encoder ---
        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            blocks_per_stage=blocks_per_stage,
            use_se=use_se,
            dropout=dropout,
        )

        # --- Shared Bottleneck ---
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[3], bottleneck_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = get_bottleneck(bottleneck_type, in_channels=bottleneck_channels, **kwargs)

        # Encoder output channel layout: [stem=ch0, s1=ch0, s2=ch1, s3=ch2, s4=ch3]
        enc_out_channels = (encoder_channels[0], encoder_channels[0],
                            encoder_channels[1], encoder_channels[2], encoder_channels[3])

        # --- Tissue Decoder ---
        if mode in ('tissue', 'joint'):
            self.tissue_decoder = TissueDecoder(
                num_classes=num_tissue_classes,
                encoder_channels=enc_out_channels,
                bottleneck_channels=bottleneck_channels,
                use_attention=use_attention,
            )

        # --- TGNA: Cross-task fusion (only in joint mode) ---
        if mode == 'joint':
            # Tissue decoder features are in order [stem_scale, s1_scale, ..., s4_scale]
            # with same channel dims as encoder outputs
            tissue_dec_channels = (enc_out_channels[0],  # dec1 -> stem scale
                                   enc_out_channels[1],  # dec2 -> s1 scale
                                   enc_out_channels[2],  # dec3 -> s2 scale
                                   enc_out_channels[3],  # dec4 -> s3 scale
                                   enc_out_channels[4])  # dec5 -> s4 scale
            self.tgna = TGNA(
                encoder_channels=enc_out_channels,
                tissue_decoder_channels=tissue_dec_channels,
            )

        # --- Nuclei Decoder ---
        if mode in ('nuclei', 'joint'):
            self.nuclei_decoder = NucleiDecoder(
                num_nuclei_classes=num_nuclei_classes,
                encoder_channels=enc_out_channels,
                bottleneck_channels=bottleneck_channels,
                use_attention=use_attention,
            )

    @classmethod
    def from_config(cls, config) -> 'PUMANet':
        """Create PUMANet from a TrainingConfig object."""
        return cls(
            in_channels=config.NUM_CHANNELS,
            num_tissue_classes=config.NUM_TISSUE_CLASSES,
            num_nuclei_classes=config.NUM_NUCLEI_CLASSES,
            encoder_channels=config.ENCODER_CHANNELS,
            bottleneck_channels=config.BOTTLENECK_CHANNELS,
            blocks_per_stage=config.BLOCKS_PER_STAGE,
            bottleneck_type=config.BOTTLENECK_TYPE,
            use_se=config.USE_SE,
            use_attention=config.USE_ATTENTION_GATES,
            dropout=config.DROPOUT,
            mode=config.MODE,
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input image.

        Returns:
            dict with keys depending on mode:
              - 'tissue': tissue prediction + multiscale
              - 'nuclei': np/hv/nc predictions + multiscale
              - 'joint':  all of the above
        """
        # Shared encoder
        encoder_features, encoder_out = self.encoder(x)

        # Shared bottleneck
        bottleneck = self.bottleneck_conv(encoder_out)
        bottleneck = self.bottleneck(bottleneck)

        result = {}

        # --- Tissue branch ---
        if self.mode in ('tissue', 'joint'):
            tissue_pred, tissue_ms, tissue_features = self.tissue_decoder(
                encoder_features, bottleneck,
            )
            result['tissue_pred'] = tissue_pred
            result['tissue_multiscale'] = tissue_ms

        # --- Nuclei branch ---
        if self.mode in ('nuclei', 'joint'):
            # Apply TGNA in joint mode
            if self.mode == 'joint':
                nuclei_features = self.tgna(encoder_features, tissue_features)
            else:
                nuclei_features = encoder_features

            np_pred, hv_pred, nc_pred, nuclei_ms = self.nuclei_decoder(
                nuclei_features, bottleneck,
            )
            result['np_pred'] = np_pred
            result['hv_pred'] = hv_pred
            result['nc_pred'] = nc_pred
            result['nuclei_multiscale'] = nuclei_ms

        return result

    def set_mode(self, mode):
        """Switch training mode dynamically.

        Useful for curriculum learning: train tissue first, then joint.
        """
        assert mode in ('tissue', 'nuclei', 'joint')
        self.mode = mode
