import torch
from tqdm import tqdm
import numpy as np


def compute_class_weights(dataset, num_classes=2, num_samples=None):
    """
    Compute inverse frequency weights for class imbalance handling.

    Args:
        dataset: PyTorch dataset (returns image, mask)
        num_classes: Number of classes
        num_samples: Number of samples to scan (None for all)

    Returns:
        torch.Tensor: Class weights of shape (num_classes,)
    """
    pixel_counts = torch.zeros(num_classes, dtype=torch.long)

    dataset_len = len(dataset)
    if num_samples is None or num_samples >= dataset_len:
        indices = range(dataset_len)
    else:
        indices = np.random.choice(dataset_len, min(num_samples, dataset_len), replace=False)

    for idx in tqdm(indices, desc="Computing class weights"):
        try:
            result = dataset[idx]
            mask = result[1]  # second element is mask

            if mask.ndim == 3:
                mask = mask.squeeze(0)

            unique, counts = torch.unique(mask, return_counts=True)
            for u, c in zip(unique, counts):
                if u < num_classes:
                    pixel_counts[u.long()] += c.long()

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    print(f"Raw class pixel counts: {pixel_counts.tolist()}")

    # Detect classes with zero pixels (not present in this data subset)
    zero_classes = (pixel_counts == 0).nonzero(as_tuple=True)[0].tolist()
    if zero_classes:
        print(f"Warning: Classes {zero_classes} have 0 pixels in dataset. "
              f"Setting their weights to 0 (ignored in loss).")

    pixel_counts_safe = pixel_counts.float().clone()
    # Replace zeros with 1 temporarily to avoid division explosion
    pixel_counts_safe[pixel_counts == 0] = 1.0
    total_pixels = pixel_counts.float().sum()

    weights = total_pixels / (num_classes * pixel_counts_safe)

    # Set weight=0 for classes with no pixels (CE loss will ignore them)
    for cls in zero_classes:
        weights[cls] = 0.0

    # Normalize so that present-class weights average to 1
    present_mask = weights > 0
    if present_mask.any():
        weights[present_mask] = weights[present_mask] / weights[present_mask].mean()

    print(f"Class weights: {weights.tolist()}")

    return weights.float()
