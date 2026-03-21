import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class BaseDataset(Dataset):
    """Base dataset class for segmentation tasks."""

    def __init__(self, dataset_root, split='train', transform=None):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def _build_index(self):
        raise NotImplementedError("Subclasses must implement _build_index")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")
