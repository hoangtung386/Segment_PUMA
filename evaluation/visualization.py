"""
Visualization utilities for segmentation results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from configs.constants import IMAGENET_MEAN, IMAGENET_STD


def visualize_predictions(model, val_loader, device, class_names,
                          output_dir, num_samples=5):
    """Visualize input / ground truth / prediction overlays."""
    model.eval()
    num_classes = len(class_names)
    # 11 colors to cover all PUMA classes (tissue=6, nuclei track2=11)
    colors = [
        'black', 'red', 'green', 'blue', 'yellow', 'purple',
        'orange', 'cyan', 'magenta', 'lime', 'pink',
    ][:num_classes]
    # Fallback: generate colors if num_classes exceeds predefined list
    while len(colors) < num_classes:
        colors.append(plt.cm.tab20(len(colors) / 20))
    cmap = ListedColormap(colors)

    samples_shown = 0

    with torch.no_grad():
        for images, masks in val_loader:
            if samples_shown >= num_samples:
                break

            images = images.to(device)
            outputs = model(images)
            pred = outputs['pred']
            preds = torch.argmax(pred, dim=1)

            # Take first sample in batch
            img_np = images[0].cpu().numpy()
            # Denormalize for display (approximate)
            if img_np.shape[0] == 3:
                mean = np.array(IMAGENET_MEAN)
                std = np.array(IMAGENET_STD)
                img_display = (img_np * std[:, None, None] + mean[:, None, None])
                img_display = np.clip(img_display.transpose(1, 2, 0), 0, 1)
            else:
                img_display = img_np[0]

            pred_np = preds[0].cpu().numpy()
            mask_np = masks[0].cpu().numpy()

            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np.squeeze(0)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_display, cmap='gray' if img_display.ndim == 2 else None)
            axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(img_display, cmap='gray' if img_display.ndim == 2 else None)
            axes[1].imshow(mask_np, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes - 1)
            axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[1].axis('off')

            axes[2].imshow(img_display, cmap='gray' if img_display.ndim == 2 else None)
            axes[2].imshow(pred_np, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes - 1)
            axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
            axes[2].axis('off')

            legend_elements = [
                Patch(facecolor=colors[i], label=class_names[i])
                for i in range(1, num_classes)
            ]
            fig.legend(handles=legend_elements, loc='lower center',
                       ncol=min(5, num_classes - 1), fontsize=12)

            plt.tight_layout()
            save_path = os.path.join(output_dir, f'prediction_{samples_shown + 1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            samples_shown += 1

    print(f"Visualizations saved to {output_dir}")


def plot_metrics_comparison(results, output_dir):
    """Plot metrics comparison across classes."""
    metrics_to_plot = ['dice', 'iou', 'precision', 'recall']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        classes = list(results.keys())
        values = [results[cls].get(metric, 0) for cls in classes]

        axes[idx].bar(classes, values, color='steelblue', alpha=0.8)
        axes[idx].set_title(metric.upper(), fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(axis='y', alpha=0.3)

        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot confusion matrix."""
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history_csv_path, output_dir):
    """Plot training history from CSV."""
    import pandas as pd

    if not os.path.exists(history_csv_path):
        print(f"History file not found: {history_csv_path}")
        return

    df = pd.read_csv(history_csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(df['epoch'], df['val_dice'], marker='o', color='green', label='Val Dice')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
