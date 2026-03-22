"""Training entry point for PUMA Challenge segmentor."""
import os
import sys
import argparse

# Ensure project root is on sys.path so local packages (utils, configs, ...) are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# Device setup MUST happen before importing torch
from utils.device import early_device_setup
early_device_setup()

import torch
import torch.nn as nn
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from configs.config import load_config
from datasets.factory import create_dataloader
from models.segmentor import CellSegmentor
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Segmentor (PUMA Challenge)")
    parser.add_argument('--devices', type=str, default=None, help="CUDA devices")
    parser.add_argument('--config', type=str, default=None, help="Config YAML path")
    parser.add_argument('--task', type=str, default=None, help="'tissue' or 'nuclei'")
    parser.add_argument('--nuclei-track', type=int, default=None, help="1 or 2")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint path to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.task:
        config.TASK = args.task
    if args.nuclei_track:
        config.NUCLEI_TRACK = args.nuclei_track

    # Auto-set NUM_CLASSES from centralized task config
    config.resolve_task()

    # Reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    config.create_directories()
    config.print_config()

    if config.USE_WANDB and wandb is not None:
        wandb.init(
            project=f"{config.WANDB_PROJECT}{config.TASK}",
            entity=config.WANDB_ENTITY,
            config=config.to_dict(),
            name=f"PUMA_{config.TASK}_track{config.NUCLEI_TRACK}",
            mode=config.WANDB_MODE,
        )

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device('cuda:0')
        multi_gpu = torch.cuda.device_count() > 1
        device_ids = list(range(torch.cuda.device_count()))
    else:
        device = torch.device('cpu')
        multi_gpu = False
        device_ids = []

    train_loader = create_dataloader(config, split='train', use_augmentation=config.USE_AUGMENTATION)
    val_loader = create_dataloader(config, split='val', use_augmentation=False)

    model = CellSegmentor.from_config(config)

    if multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=config, device=device, multi_gpu=multi_gpu,
    )

    # Auto-resume: --resume flag or detect latest checkpoint
    resume_path = args.resume
    if resume_path is None:
        auto_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{config.DATASET_NAME}.pth")
        if os.path.exists(auto_ckpt):
            resume_path = auto_ckpt

    if resume_path:
        trainer.resume_from_checkpoint(resume_path)

    trainer.train(num_epochs=config.NUM_EPOCHS)


if __name__ == "__main__":
    main()
