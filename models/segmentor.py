"""
CellSegmentor - ResNet encoder + FPN decoder for histopathology segmentation.

Pipeline:
1. ResNet encoder with SE attention (from scratch, no pretrained)
2. Bottleneck (Mamba SSM or standard conv)
3. FPN decoder with attention gates
4. Multi-scale predictions for deep supervision
"""
import torch
import torch.nn as nn

from models.encoder import ResNetEncoder
from models.bottleneck import get_bottleneck
from models.decoder.hvt import FPNDecoder


class CellSegmentor(nn.Module):
    def __init__(self, in_channels=3, num_classes=6,
                 encoder_channels=(64, 128, 256, 512),
                 bottleneck_channels=1024,
                 blocks_per_stage=2,
                 bottleneck_type='mamba',
                 use_se=True,
                 use_attention=True,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        # Encoder: ResNet with SE attention
        # stem outputs encoder_channels[0], stages output encoder_channels[0..3]
        # encoder features = [stem(ch0), s1(ch0), s2(ch1), s3(ch2), s4(ch3)]
        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            channels=encoder_channels,
            blocks_per_stage=blocks_per_stage,
            use_se=use_se,
            dropout=dropout,
        )

        # Bottleneck: expand to bottleneck_channels then apply Mamba/Conv
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[3], bottleneck_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, bottleneck_channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = get_bottleneck(bottleneck_type, in_channels=bottleneck_channels, **kwargs)

        # Decoder: FPN with attention gates
        # encoder outputs: [stem=ch0, s1=ch0, s2=ch1, s3=ch2, s4=ch3]
        enc_out_channels = (encoder_channels[0], encoder_channels[0],
                            encoder_channels[1], encoder_channels[2], encoder_channels[3])
        self.decoder = FPNDecoder(
            num_classes=num_classes,
            encoder_channels=enc_out_channels,
            bottleneck_channels=bottleneck_channels,
            use_attention=use_attention,
        )

    @classmethod
    def from_config(cls, config) -> 'CellSegmentor':
        """Create a CellSegmentor from a TrainingConfig object."""
        return cls(
            in_channels=config.NUM_CHANNELS,
            num_classes=config.NUM_CLASSES,
            encoder_channels=config.ENCODER_CHANNELS,
            bottleneck_channels=config.BOTTLENECK_CHANNELS,
            blocks_per_stage=config.BLOCKS_PER_STAGE,
            bottleneck_type=config.BOTTLENECK_TYPE,
            use_se=config.USE_SE,
            use_attention=config.USE_ATTENTION_GATES,
            dropout=config.DROPOUT,
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            dict with 'pred' (full resolution) and 'multiscale_preds' (4 scales)
        """
        # Encoder
        encoder_features, encoder_out = self.encoder(x)
        # encoder_features = [stem, s1, s2, s3, s4]
        # encoder_out = s4 pooled

        # Bottleneck
        bottleneck = self.bottleneck_conv(encoder_out)
        bottleneck = self.bottleneck(bottleneck)

        # Decoder
        predictions = self.decoder(encoder_features, bottleneck)
        # predictions[0] = final (full resolution)
        # predictions[1:] = multi-scale (H/32, H/16, H/8, H/4)

        return {
            'pred': predictions[0],
            'multiscale_preds': predictions[1:],
        }
