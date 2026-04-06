"""
Training script for PUMANet V2 - Dual-decoder panoptic segmentation.

Usage:
    # Joint training (tissue + nuclei)
    python scripts/train_v2.py --config configs/puma_v2.yaml --nuclei-track 1

    # Tissue only
    python scripts/train_v2.py --config configs/puma_v2.yaml --mode tissue

    # Nuclei only
    python scripts/train_v2.py --config configs/puma_v2.yaml --mode nuclei

    # Track 2 (11 nuclei classes)
    python scripts/train_v2.py --config configs/puma_v2.yaml --nuclei-track 2

    # With Mamba bottleneck on Colab
    python scripts/train_v2.py --config configs/puma_v2_colab.yaml

    # Resume from checkpoint
    python scripts/train_v2.py --config configs/puma_v2.yaml --resume checkpoints_v2/puma_v2.pth
"""
import argparse
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import load_config
from models.puma_net import PUMANet
from datasets.puma_dataset_v2 import PUMAJointDataset, puma_collate_fn
from training.puma_trainer import PUMATrainer


def parse_args():
    parser = argparse.ArgumentParser(description='PUMANet V2 Training')
    parser.add_argument('--config', type=str, default='configs/puma_v2.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--devices', type=str, default='0',
                        help='CUDA device IDs (e.g., "0" or "0,1")')
    parser.add_argument('--mode', type=str, choices=['tissue', 'nuclei', 'joint'],
                        default=None, help='Training mode (overrides config)')
    parser.add_argument('--nuclei-track', type=int, choices=[1, 2], default=None,
                        help='Nuclei track (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    return parser.parse_args()


def setup_device(device_str):
    """Setup CUDA device(s)."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu'), False

    device_ids = [int(d) for d in device_str.split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str
    device = torch.device(f'cuda:0')

    for i in device_ids:
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"GPU {i}: {name} ({mem:.1f} GB)")

    multi_gpu = len(device_ids) > 1
    return device, multi_gpu


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.mode is not None:
        config.MODE = args.mode
    if args.nuclei_track is not None:
        config.NUCLEI_TRACK = args.nuclei_track

    # Resolve class counts
    config.resolve_v2()

    # Seed
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    # Directories
    config.create_directories()

    # Device
    device, multi_gpu = setup_device(args.devices)

    # Print config
    print("\n" + "=" * 60)
    print("PUMANet V2 TRAINING")
    print("=" * 60)
    print(f"Mode:              {config.MODE}")
    print(f"Nuclei Track:      {config.NUCLEI_TRACK}")
    print(f"Tissue Classes:    {config.NUM_TISSUE_CLASSES}")
    print(f"Nuclei Classes:    {config.NUM_NUCLEI_CLASSES}")
    print(f"Encoder:           {config.ENCODER_CHANNELS}")
    print(f"Bottleneck:        {config.BOTTLENECK_TYPE} ({config.BOTTLENECK_CHANNELS}ch)")
    print(f"Batch Size:        {config.BATCH_SIZE}")
    print(f"Patch Size:        {config.PATCH_SIZE}")
    print(f"AMP:               {config.USE_AMP}")
    print("=" * 60 + "\n")

    # W&B
    if config.USE_WANDB:
        try:
            import wandb
            wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                config=config.to_dict(),
                name=f"{config.DATASET_NAME}_{config.MODE}_track{config.NUCLEI_TRACK}",
                mode=config.WANDB_MODE,
            )
        except Exception as e:
            print(f"W&B init failed: {e}. Continuing without W&B.")
            config.USE_WANDB = False

    # Datasets
    print("Creating datasets...")
    train_dataset = PUMAJointDataset(
        dataset_root=config.DATA_ROOT,
        split='train',
        nuclei_track=config.NUCLEI_TRACK,
        mode=config.MODE,
        use_context=config.USE_CONTEXT,
        image_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        patches_per_image=config.PATCHES_PER_IMAGE,
        use_augmentation=config.USE_AUGMENTATION,
        split_ratio=config.SPLIT_RATIO,
    )

    val_dataset = PUMAJointDataset(
        dataset_root=config.DATA_ROOT,
        split='val',
        nuclei_track=config.NUCLEI_TRACK,
        mode=config.MODE,
        use_context=config.USE_CONTEXT,
        image_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        use_augmentation=False,
        split_ratio=config.SPLIT_RATIO,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        collate_fn=puma_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
        collate_fn=puma_collate_fn,
    )

    # Model
    print("Creating PUMANet model...")
    model = PUMANet.from_config(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # Multi-GPU
    if multi_gpu:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel across GPUs")

    # Trainer
    trainer = PUMATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        multi_gpu=multi_gpu,
    )

    # Resume
    resume_path = args.resume
    if resume_path is None:
        # Auto-detect latest checkpoint
        auto_path = os.path.join(config.CHECKPOINT_DIR, f'{config.DATASET_NAME}.pth')
        if os.path.exists(auto_path):
            resume_path = auto_path

    if resume_path:
        trainer.resume_from_checkpoint(resume_path)

    # Train
    trainer.train(num_epochs=config.NUM_EPOCHS)

    # Cleanup
    if config.USE_WANDB:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
