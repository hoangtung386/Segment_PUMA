"""
Cell segmentation dataset.

Expected directory structure:
    data_root/
        images/
            image_001.png
            image_002.png
            ...
        masks/
            image_001.png   (or .npy)
            image_002.png
            ...

Masks should be single-channel with integer class labels (0=background, 1=cell, ...).
"""
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from datasets.base import BaseDataset
from configs.constants import IMAGENET_MEAN, IMAGENET_STD

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUM_AVAILABLE = True
except ImportError:
    ALBUM_AVAILABLE = False


class CellDataset(BaseDataset):
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    def __init__(self, dataset_root, split='train', transform=None,
                 image_size=(512, 512), use_augmentation=False,
                 split_ratio=(0.8, 0.2)):
        super().__init__(dataset_root, split, transform)
        self.image_size = image_size
        self.use_augmentation = use_augmentation and (split == 'train')
        self.split_ratio = split_ratio
        self._build_index()

        if self.transform is None:
            self.transform = self._default_transform()

    def _build_index(self):
        image_dir = self.dataset_root / 'images'
        mask_dir = self.dataset_root / 'masks'

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        image_files = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS
        ])

        for img_path in image_files:
            # Try to find matching mask
            mask_path = None
            for ext in [img_path.suffix, '.png', '.npy', '.tif', '.tiff']:
                candidate = mask_dir / (img_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break

            if mask_path is not None:
                self.samples.append({
                    'image': img_path,
                    'mask': mask_path,
                })

        # Split into train/val(/test)
        n = len(self.samples)
        train_end = int(n * self.split_ratio[0])
        if len(self.split_ratio) >= 3:
            val_end = int(n * (self.split_ratio[0] + self.split_ratio[1]))
        else:
            val_end = n

        if self.split == 'train':
            self.samples = self.samples[:train_end]
        elif self.split == 'val':
            self.samples = self.samples[train_end:val_end]
        elif self.split == 'test':
            self.samples = self.samples[val_end:]

        print(f"CellDataset [{self.split}]: {len(self.samples)} samples from {self.dataset_root}")

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return np.array(img)

    def _load_mask(self, path):
        if path.suffix == '.npy':
            return np.load(path).astype(np.int64)
        mask = Image.open(path)
        return np.array(mask).astype(np.int64)

    def _default_transform(self):
        if not ALBUM_AVAILABLE:
            return None

        if self.use_augmentation:
            return A.Compose([
                A.Resize(*self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                                   rotate_limit=30, p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                ], p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                ToTensorV2(),
            ])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._load_image(sample['image'])
        mask = self._load_mask(sample['mask'])

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.transform is not None and ALBUM_AVAILABLE:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        else:
            # Manual resize and tensor conversion
            img_pil = Image.fromarray(image).resize(
                (self.image_size[1], self.image_size[0]), Image.BILINEAR)
            mask_pil = Image.fromarray(mask.astype(np.uint8)).resize(
                (self.image_size[1], self.image_size[0]), Image.NEAREST)

            image = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(np.array(mask_pil)).long()

        return image, mask
