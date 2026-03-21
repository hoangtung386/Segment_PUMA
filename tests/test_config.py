"""Tests for configuration and task config."""
import pytest
from configs.config import TrainingConfig, load_config
from configs.constants import get_task_config, IMAGENET_MEAN, IMAGENET_STD


class TestTaskConfig:
    def test_tissue_config(self):
        cfg = get_task_config('tissue')
        assert cfg.num_classes == 6
        assert len(cfg.class_names) == 6
        assert cfg.class_names[0] == 'Background'
        assert cfg.class_map is not None

    def test_nuclei_track1_config(self):
        cfg = get_task_config('nuclei', nuclei_track=1)
        assert cfg.num_classes == 4
        assert len(cfg.class_names) == 4
        assert 'TILs' in cfg.class_names

    def test_nuclei_track2_config(self):
        cfg = get_task_config('nuclei', nuclei_track=2)
        assert cfg.num_classes == 11
        assert len(cfg.class_names) == 11

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_config('unknown')

    def test_imagenet_constants(self):
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert all(0 < m < 1 for m in IMAGENET_MEAN)
        assert all(0 < s < 1 for s in IMAGENET_STD)


class TestTrainingConfig:
    def test_default_config(self):
        config = TrainingConfig()
        assert config.SEED == 42
        assert config.NUM_CLASSES == 6
        assert config.TASK == 'tissue'

    def test_resolve_task_tissue(self):
        config = TrainingConfig(TASK='tissue')
        config.resolve_task()
        assert config.NUM_CLASSES == 6

    def test_resolve_task_nuclei_track1(self):
        config = TrainingConfig(TASK='nuclei', NUCLEI_TRACK=1)
        config.resolve_task()
        assert config.NUM_CLASSES == 4

    def test_resolve_task_nuclei_track2(self):
        config = TrainingConfig(TASK='nuclei', NUCLEI_TRACK=2)
        config.resolve_task()
        assert config.NUM_CLASSES == 11

    def test_get_class_names(self):
        config = TrainingConfig(TASK='tissue')
        names = config.get_class_names()
        assert names[0] == 'Background'
        assert len(names) == 6

    def test_to_dict(self):
        config = TrainingConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert 'SEED' in d
        assert 'TASK' in d

    def test_load_config_default(self):
        config = load_config(None)
        assert isinstance(config, TrainingConfig)

    def test_load_config_missing_file(self):
        config = load_config('/nonexistent/path.yaml')
        assert isinstance(config, TrainingConfig)
