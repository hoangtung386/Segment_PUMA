"""Evaluation entry point for PUMA Challenge segmentor."""
import argparse

# Device setup MUST happen before importing torch
from utils.device import early_device_setup
early_device_setup()

import torch
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot

from configs.config import load_config
from datasets.factory import create_dataloader
from models.segmentor import CellSegmentor


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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    eval_loader = create_dataloader(config, split=args.split, use_augmentation=False)

    model = CellSegmentor.from_config(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction='mean')

    with torch.no_grad():
        for images, masks in tqdm(eval_loader, desc=f"Evaluating [{args.split}]"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            pred = outputs['pred']

            masks_metric = masks.unsqueeze(1) if masks.ndim == 3 else masks
            y_pred_idx = torch.argmax(pred, dim=1, keepdim=True)
            y_pred_onehot = one_hot(y_pred_idx, num_classes=config.NUM_CLASSES)
            y_target_onehot = one_hot(masks_metric, num_classes=config.NUM_CLASSES)

            dice_metric(y_pred=y_pred_onehot, y=y_target_onehot)

    final_dice = dice_metric.aggregate().item()

    # Print per-class results
    class_names = config.get_class_names()
    print(f"\nDice Score [{args.split}]: {final_dice:.4f}")
    print(f"Task: {config.TASK}, Classes: {class_names}")


if __name__ == "__main__":
    main()
