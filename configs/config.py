import os
import yaml
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from configs.constants import get_task_config


@dataclass
class TrainingConfig:
    # Basic settings
    SEED: int = 42

    # Data paths
    DATA_ROOT: str = "./dataset_PUMA"
    OUTPUT_DIR: str = "./outputs"
    CHECKPOINT_DIR: str = "./checkpoints"
    DATASET_NAME: str = "puma"

    # Task configuration
    TASK: str = "tissue"
    NUCLEI_TRACK: int = 1
    USE_CONTEXT: bool = False

    # Data split (train / val / test)
    SPLIT_RATIO: Tuple[float, float, float] = (0.7, 0.2, 0.1)

    # Model parameters
    NUM_CHANNELS: int = 3
    NUM_CLASSES: int = 6
    IMAGE_SIZE: Tuple[int, int] = (512, 512)

    # Patch-based training
    PATCH_SIZE: int = 512
    PATCHES_PER_IMAGE: int = 4

    # Encoder
    ENCODER_CHANNELS: Tuple[int, ...] = (64, 128, 256, 512)
    BLOCKS_PER_STAGE: int = 2
    USE_SE: bool = True

    # Bottleneck
    BOTTLENECK_CHANNELS: int = 1024
    BOTTLENECK_TYPE: str = "standard"     # 'mamba' or 'standard'

    # Decoder
    USE_ATTENTION_GATES: bool = True

    # Training
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 300
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4

    # DataLoader
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True

    # Training stability
    GRAD_CLIP_NORM: float = 1.0
    USE_AMP: bool = False
    DROPOUT: float = 0.1

    # Loss weights
    DICE_WEIGHT: float = 0.5
    CE_WEIGHT: float = 0.3
    FOCAL_WEIGHT: float = 0.5
    FP_PENALTY_WEIGHT: float = 0.2
    BOUNDARY_WEIGHT: float = 0.1
    CLUSTER_WEIGHT: float = 0.1

    # Class weights (None = auto-compute)
    CUSTOM_CLASS_WEIGHTS: Optional[List[float]] = None

    # W&B
    USE_WANDB: bool = True
    WANDB_PROJECT: str = "PUMA-"
    WANDB_ENTITY: Optional[str] = None
    WANDB_MODE: str = "online"

    # Scheduler
    SCHEDULER_T0: int = 10
    SCHEDULER_T_MULT: int = 2
    SCHEDULER_ETA_MIN: float = 1e-6

    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 50

    # Augmentation
    USE_AUGMENTATION: bool = True

    def resolve_task(self):
        """Auto-set NUM_CLASSES based on TASK and NUCLEI_TRACK."""
        task_cfg = get_task_config(self.TASK, self.NUCLEI_TRACK)
        self.NUM_CLASSES = task_cfg.num_classes

    def get_class_names(self) -> List[str]:
        """Get class names for the current task."""
        try:
            task_cfg = get_task_config(self.TASK, self.NUCLEI_TRACK)
            return task_cfg.class_names
        except ValueError:
            return [f'Class{i}' for i in range(self.NUM_CLASSES)]

    def create_directories(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

    def to_dict(self):
        return asdict(self)

    def print_config(self):
        print("\n" + "=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Dataset:           {self.DATASET_NAME}")
        print(f"Task:              {self.TASK}")
        if self.TASK == 'nuclei':
            print(f"Nuclei Track:      {self.NUCLEI_TRACK}")
        print(f"Num Classes:       {self.NUM_CLASSES}")
        print(f"Patch Size:        {self.PATCH_SIZE}")
        print(f"Patches/Image:     {self.PATCHES_PER_IMAGE}")
        print(f"Image Size (val):  {self.IMAGE_SIZE}")
        print(f"Encoder:           {self.ENCODER_CHANNELS}, {self.BLOCKS_PER_STAGE} blocks/stage, SE={self.USE_SE}")
        print(f"Bottleneck:        {self.BOTTLENECK_TYPE} ({self.BOTTLENECK_CHANNELS}ch)")
        print(f"Attention Gates:   {self.USE_ATTENTION_GATES}")
        print(f"Batch Size:        {self.BATCH_SIZE}")
        print(f"Learning Rate:     {self.LEARNING_RATE}")
        print(f"Epochs:            {self.NUM_EPOCHS}")
        print(f"Dropout:           {self.DROPOUT}")
        print("=" * 60 + "\n")


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    if config_path is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(config_dir, "base.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Convert lists to tuples where needed
        for key in ('IMAGE_SIZE', 'ENCODER_CHANNELS', 'SPLIT_RATIO'):
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])

        return TrainingConfig(**config_dict)

    print(f"Warning: Config {config_path} not found. Using defaults.")
    return TrainingConfig()
