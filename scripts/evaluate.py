"""Evaluation entry point for PUMA Challenge segmentor."""
import argparse

# Device setup MUST happen before importing torch
from utils.device import early_device_setup
early_device_setup()

import torch

from configs.config import load_config
from datasets.factory import create_dataloader
from models.segmentor import CellSegmentor
from evaluation import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Segmentor")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help="Dataset split to evaluate on (default: 'test')")
    parser.add_argument('--devices', type=str, default="0")
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--nuclei-track', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Directory to save evaluation results (default: OUTPUT_DIR/evaluation)")
    parser.add_argument('--num-samples', type=int, default=5,
                        help="Number of visualization samples (default: 5)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.task:
        config.TASK = args.task
    if args.nuclei_track:
        config.NUCLEI_TRACK = args.nuclei_track

    config.resolve_task()
    config.create_directories()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    eval_loader = create_dataloader(config, split=args.split, use_augmentation=False)

    model = CellSegmentor.from_config(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    import os
    output_dir = args.output_dir or os.path.join(config.OUTPUT_DIR, "evaluation")

    evaluator = Evaluator(
        model=model,
        val_loader=eval_loader,
        device=device,
        config=config,
        num_samples=args.num_samples,
        output_dir=output_dir,
    )

    results = evaluator.run()

    print(f"\nEvaluation complete [{args.split}]")
    print(f"Task: {config.TASK}, Classes: {config.get_class_names()}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
