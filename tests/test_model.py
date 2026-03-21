"""Tests for model forward pass and shape correctness."""
import pytest
import torch
from models.segmentor import CellSegmentor
from models.encoder import ResNetEncoder, ResBlock, SEBlock, EncoderStage
from models.decoder.hvt import FPNDecoder, DecoderBlock, AttentionGate
from models.bottleneck import get_bottleneck


class TestSEBlock:
    def test_output_shape(self):
        block = SEBlock(channels=64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_channel_attention(self):
        block = SEBlock(channels=32, reduction=8)
        x = torch.randn(1, 32, 8, 8)
        out = block(x)
        assert out.shape == (1, 32, 8, 8)


class TestResBlock:
    def test_same_channels(self):
        block = ResBlock(64, 64, stride=1, use_se=True)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_channel_change(self):
        block = ResBlock(64, 128, stride=1, use_se=True)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_no_se(self):
        block = ResBlock(64, 64, stride=1, use_se=False)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape


class TestEncoder:
    def test_output_shapes(self):
        encoder = ResNetEncoder(in_channels=3, channels=(32, 64, 128, 256),
                                blocks_per_stage=1, use_se=True, dropout=0.0)
        x = torch.randn(1, 3, 64, 64)
        features, bottleneck_input = encoder(x)

        assert len(features) == 5  # stem, s1, s2, s3, s4
        # stem is at H/2 (before stem_pool)
        assert features[0].shape[1] == 32
        assert bottleneck_input.shape[1] == 256

    def test_feature_count(self):
        encoder = ResNetEncoder(in_channels=3, channels=(64, 128, 256, 512))
        x = torch.randn(1, 3, 128, 128)
        features, _ = encoder(x)
        assert len(features) == 5


class TestBottleneck:
    def test_standard_bottleneck(self):
        bottleneck = get_bottleneck('standard', in_channels=256)
        x = torch.randn(1, 256, 8, 8)
        out = bottleneck(x)
        assert out.shape == x.shape

    def test_unknown_bottleneck_raises(self):
        with pytest.raises(ValueError, match="Unknown bottleneck"):
            get_bottleneck('nonexistent', in_channels=256)


class TestDecoder:
    def test_attention_gate(self):
        gate = AttentionGate(skip_channels=64, gate_channels=128)
        skip = torch.randn(1, 64, 16, 16)
        g = torch.randn(1, 128, 16, 16)
        out = gate(skip, g)
        assert out.shape == skip.shape

    def test_decoder_block(self):
        block = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        x = torch.randn(1, 128, 8, 8)
        skip = torch.randn(1, 64, 16, 16)
        out = block(x, skip)
        assert out.shape == (1, 64, 16, 16)

    def test_fpn_decoder(self):
        decoder = FPNDecoder(num_classes=6,
                             encoder_channels=(32, 32, 64, 128, 256),
                             bottleneck_channels=512)
        features = [
            torch.randn(1, 32, 32, 32),   # stem
            torch.randn(1, 32, 16, 16),   # s1
            torch.randn(1, 64, 8, 8),     # s2
            torch.randn(1, 128, 4, 4),    # s3
            torch.randn(1, 256, 2, 2),    # s4
        ]
        bottleneck = torch.randn(1, 512, 1, 1)
        preds = decoder(features, bottleneck)
        assert len(preds) == 5  # 1 final + 4 multiscale


class TestCellSegmentor:
    def test_forward_pass(self):
        model = CellSegmentor(
            in_channels=3, num_classes=6,
            encoder_channels=(32, 64, 128, 256),
            bottleneck_channels=512,
            blocks_per_stage=1,
            bottleneck_type='standard',
            use_se=True,
            use_attention=True,
            dropout=0.0,
        )
        x = torch.randn(1, 3, 64, 64)
        output = model(x)

        assert 'pred' in output
        assert 'multiscale_preds' in output
        assert output['pred'].shape[0] == 1
        assert output['pred'].shape[1] == 6
        # Output should be same spatial size as input
        assert output['pred'].shape[2] == 64
        assert output['pred'].shape[3] == 64

    def test_from_config(self):
        from configs.config import TrainingConfig
        config = TrainingConfig(
            NUM_CHANNELS=3, NUM_CLASSES=4,
            ENCODER_CHANNELS=(32, 64, 128, 256),
            BOTTLENECK_CHANNELS=512,
            BLOCKS_PER_STAGE=1,
            BOTTLENECK_TYPE='standard',
            USE_SE=False,
            USE_ATTENTION_GATES=True,
            DROPOUT=0.0,
        )
        model = CellSegmentor.from_config(config)
        x = torch.randn(1, 3, 64, 64)
        output = model(x)
        assert output['pred'].shape[1] == 4

    def test_different_num_classes(self):
        for nc in [2, 4, 6, 11]:
            model = CellSegmentor(
                in_channels=3, num_classes=nc,
                encoder_channels=(32, 64, 128, 256),
                bottleneck_channels=512,
                blocks_per_stage=1,
                bottleneck_type='standard',
            )
            x = torch.randn(1, 3, 64, 64)
            output = model(x)
            assert output['pred'].shape[1] == nc
