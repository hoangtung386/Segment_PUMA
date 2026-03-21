"""
PUMA Dataset - Panoptic segmentation of nUclei and tissue in advanced MelanomA.

Supports two task modes:
  - 'tissue': Semantic tissue segmentation (6 classes)
      0=Background, 1=Tumor, 2=Stroma, 3=Epithelium, 4=Blood Vessel, 5=Necrosis
  - 'nuclei': Nuclei segmentation (varies by track)
      Track 1 (4 classes): 0=Background, 1=Tumor, 2=TILs, 3=Other
      Track 2 (11 classes): 0=Background, 1=Tumor, 2=Lymphocyte, 3=Plasma cell,
          4=Histiocyte, 5=Melanophage, 6=Neutrophil, 7=Stroma, 8=Epithelium,
          9=Endothelium, 10=Apoptosis

Features:
  - Patch-based training: random crops from full ROIs to multiply effective data
  - Stain augmentation: HED color space jitter for H&E histopathology
  - Full image validation: no cropping during eval
"""
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from datasets.base import BaseDataset
from configs.constants import (
    get_task_config, IMAGENET_MEAN, IMAGENET_STD,
)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUM_AVAILABLE = True
except ImportError:
    ALBUM_AVAILABLE = False


# Class mappings are now centralized in configs.constants.
# Use get_task_config(task, nuclei_track) to get class_map, num_classes, class_names.


def rasterize_geojson(geojson_path, class_map, image_size=(1024, 1024),
                      coord_offset=(0, 0)):
    """Rasterize GeoJSON polygon annotations into a segmentation mask."""
    mask = np.zeros(image_size, dtype=np.int32)

    with open(geojson_path, 'r') as f:
        data = json.load(f)

    for feature in data.get('features', []):
        geometry = feature.get('geometry', {})
        classification = feature.get('properties', {}).get('classification', {})
        class_name = classification.get('name', '')

        class_idx = class_map.get(class_name, None)
        if class_idx is None or class_idx == 0:
            continue

        geom_type = geometry.get('type', '')
        coords_list = geometry.get('coordinates', [])

        if geom_type == 'Polygon':
            _fill_polygon(mask, coords_list, class_idx, coord_offset)
        elif geom_type == 'MultiPolygon':
            for polygon_coords in coords_list:
                _fill_polygon(mask, polygon_coords, class_idx, coord_offset)

    return mask


def _fill_polygon(mask, coords_list, class_idx, coord_offset=(0, 0)):
    if not coords_list or not CV2_AVAILABLE:
        return

    exterior = np.array(coords_list[0], dtype=np.int32)
    if coord_offset != (0, 0):
        exterior = exterior + np.array(coord_offset, dtype=np.int32)
    cv2.fillPoly(mask, [exterior], class_idx)

    for hole in coords_list[1:]:
        hole_pts = np.array(hole, dtype=np.int32)
        if coord_offset != (0, 0):
            hole_pts = hole_pts + np.array(coord_offset, dtype=np.int32)
        cv2.fillPoly(mask, [hole_pts], 0)


# --- Stain augmentation ---

class StainAugmentation(A.ImageOnlyTransform):
    """HED color space jitter for H&E stained histopathology images.

    Converts RGB → HED (Hematoxylin-Eosin-DAB), perturbs each channel
    independently, then converts back. This simulates stain variation
    across different scanners/labs.
    """

    def __init__(self, sigma=0.05, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.sigma = sigma

    def apply(self, img, **params):
        # img is numpy uint8 [H, W, 3]
        img_float = img.astype(np.float64) / 255.0
        img_float = np.clip(img_float, 1e-6, 1.0)

        # RGB → Optical Density
        od = -np.log(img_float)

        # HED conversion matrix (Ruifrok & Johnston, 2001)
        rgb_to_hed = np.array([
            [0.65, 0.70, 0.29],
            [0.07, 0.99, 0.11],
            [0.27, 0.57, 0.78],
        ])
        try:
            hed_to_rgb = np.linalg.inv(rgb_to_hed)
        except np.linalg.LinAlgError:
            return img

        # RGB OD → HED
        od_flat = od.reshape(-1, 3)
        hed = od_flat @ rgb_to_hed.T

        # Perturb each HED channel independently
        for i in range(3):
            alpha = np.random.uniform(1 - self.sigma, 1 + self.sigma)
            beta = np.random.uniform(-self.sigma, self.sigma)
            hed[:, i] = hed[:, i] * alpha + beta

        # HED → RGB OD → RGB
        od_perturbed = hed @ hed_to_rgb.T
        rgb = np.exp(-od_perturbed).reshape(img_float.shape)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

        return rgb

    def get_transform_init_args_names(self):
        return ("sigma",)


# --- Dataset ---

class PUMADataset(BaseDataset):
    """Dataset loader for PUMA challenge with patch-based training.

    Training: extracts random patches from each ROI (multiplies data).
    Validation: uses full images (resized to image_size).
    """

    def __init__(self, dataset_root, split='train', task='tissue', nuclei_track=1,
                 use_context=False, image_size=(512, 512), patch_size=512,
                 patches_per_image=4, use_augmentation=False,
                 transform=None,
                 split_ratio=(0.7, 0.2, 0.1)):
        super().__init__(dataset_root, split, transform)

        self.task = task
        self.nuclei_track = nuclei_track
        self.use_context = use_context
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image if split == 'train' else 1
        self.use_augmentation = use_augmentation and (split == 'train')
        self.split_ratio = split_ratio  # (train, val, test)
        self.is_train = (split == 'train')

        task_cfg = get_task_config(task, nuclei_track)
        self.class_map = task_cfg.class_map
        self.num_classes = task_cfg.num_classes
        self.class_names = task_cfg.class_names

        self._build_index()

        if self.transform is None:
            self.transform = self._default_transform()

        # Cache rasterized masks to avoid re-parsing GeoJSON every patch
        self._cache = {}

    def _build_index(self):
        root = Path(self.dataset_root)

        roi_dir = root / '01_training_dataset_tif_ROIs'
        context_dir = root / '01_training_dataset_tif_context_ROIs'
        tissue_dir = root / '01_training_dataset_geojson_tissue'
        nuclei_dir = root / '01_training_dataset_geojson_nuclei'

        image_dir = context_dir if self.use_context else roi_dir

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        roi_ids = []
        for f in sorted(image_dir.iterdir()):
            if f.suffix.lower() in ('.tif', '.tiff'):
                stem = f.stem.replace('_context', '')
                roi_ids.append(stem)

        annotation_dir = tissue_dir if self.task == 'tissue' else nuclei_dir
        annotation_suffix = '_tissue.geojson' if self.task == 'tissue' else '_nuclei.geojson'

        roi_samples = []
        for roi_id in roi_ids:
            if self.use_context:
                img_path = image_dir / f'{roi_id}_context.tif'
            else:
                img_path = image_dir / f'{roi_id}.tif'

            ann_path = annotation_dir / f'{roi_id}{annotation_suffix}'

            if img_path.exists() and ann_path.exists():
                roi_samples.append({
                    'image': img_path,
                    'annotation': ann_path,
                    'roi_id': roi_id,
                })

        # Shuffle before split to mix primary/metastatic evenly.
        # Without this, alphabetical order puts all metastatic in train
        # and all primary in val (biased split).
        rng = np.random.RandomState(42)  # Fixed seed for reproducible split
        rng.shuffle(roi_samples)

        # Split: train / val / test
        n = len(roi_samples)
        r_train, r_val, r_test = self.split_ratio
        train_end = int(n * r_train)
        val_end = int(n * (r_train + r_val))

        if self.split == 'train':
            roi_samples = roi_samples[:train_end]
        elif self.split == 'val':
            roi_samples = roi_samples[train_end:val_end]
        elif self.split == 'test':
            roi_samples = roi_samples[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train', 'val', or 'test'.")

        # For training: each ROI produces patches_per_image samples
        # For validation: each ROI is 1 sample (full image)
        self.samples = []
        for roi in roi_samples:
            for patch_idx in range(self.patches_per_image):
                self.samples.append({**roi, 'patch_idx': patch_idx})

        print(f"PUMADataset [{self.split}] task={self.task}: "
              f"{len(roi_samples)} ROIs x {self.patches_per_image} patches = "
              f"{len(self.samples)} samples, {self.num_classes} classes")

    def _load_and_cache(self, sample):
        """Load image + mask, cache to avoid re-parsing GeoJSON."""
        roi_id = sample['roi_id']

        if roi_id not in self._cache:
            image = self._load_image(sample['image'])
            orig_h, orig_w = image.shape[:2]

            if self.use_context:
                roi_size = 1024
                offset_x = (orig_w - roi_size) // 2
                offset_y = (orig_h - roi_size) // 2
                coord_offset = (offset_x, offset_y)
            else:
                coord_offset = (0, 0)

            mask = rasterize_geojson(
                sample['annotation'], self.class_map,
                image_size=(orig_h, orig_w), coord_offset=coord_offset,
            )

            self._cache[roi_id] = (image, mask)

        return self._cache[roi_id]

    def _load_image(self, path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

    def _random_crop(self, image, mask):
        """Extract a random patch from image and mask."""
        h, w = image.shape[:2]
        ps = self.patch_size

        if h <= ps or w <= ps:
            return image, mask

        y = np.random.randint(0, h - ps)
        x = np.random.randint(0, w - ps)

        return image[y:y+ps, x:x+ps], mask[y:y+ps, x:x+ps]

    def _default_transform(self):
        if not ALBUM_AVAILABLE:
            return None

        if self.use_augmentation:
            aug_list = [
                # Spatial
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                                   rotate_limit=30, p=0.5),
                A.ElasticTransform(alpha=50, sigma=10, p=0.2),

                # Stain augmentation (histopathology-specific)
                StainAugmentation(sigma=0.05, p=0.5),

                # Color
                A.ColorJitter(brightness=0.15, contrast=0.15,
                              saturation=0.15, hue=0.05, p=0.3),

                # Noise
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(blur_limit=(3, 5), p=1),
                ], p=0.2),

                # Normalize + tensor
                A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                ToTensorV2(),
            ]
            return A.Compose(aug_list)
        else:
            return A.Compose([
                A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                ToTensorV2(),
            ])

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image, mask = self._load_and_cache(sample)

        # Copy to avoid modifying cache
        image = image.copy()
        mask = mask.copy()

        if self.is_train:
            # Random crop for training
            image, mask = self._random_crop(image, mask)
        else:
            # Resize full image for validation
            if ALBUM_AVAILABLE:
                resized = A.Resize(*self.image_size)(image=image, mask=mask)
                image, mask = resized['image'], resized['mask']

        if self.transform is not None and ALBUM_AVAILABLE:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask
