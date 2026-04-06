"""
PUMAJointDataset - Loads both tissue and nuclei annotations for multi-task training.

Returns per sample:
  - image:       [3, H, W]  RGB image tensor
  - tissue_mask: [H, W]     Tissue semantic segmentation labels (6 classes)
  - nuclei_mask: [H, W]     Nuclei classification labels (4 or 11 classes)
  - np_map:      [H, W]     Binary nuclei pixel map (0=bg, 1=nucleus)
  - hv_map:      [2, H, W]  Horizontal-vertical gradient maps for instances

Supports three modes via config.MODE:
  - 'tissue': Only loads tissue annotations (lighter)
  - 'nuclei': Only loads nuclei annotations + HV maps
  - 'joint':  Loads both tissue and nuclei annotations
"""
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from datasets.base import BaseDataset
from configs.constants import (
    get_task_config, TISSUE_CLASS_MAP, IMAGENET_MEAN, IMAGENET_STD,
)
from datasets.hv_utils import (
    rasterize_geojson_instances, generate_hv_map, generate_np_map,
)
from datasets.puma_dataset import rasterize_geojson, StainAugmentation

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


class PUMAJointDataset(BaseDataset):
    """Dataset for PUMANet multi-task training.

    Loads both tissue and nuclei annotations for joint training.
    Generates HV maps and NP maps for nuclei instance segmentation.
    """

    def __init__(self, dataset_root, split='train', nuclei_track=1,
                 mode='joint', use_context=False,
                 image_size=(512, 512), patch_size=512,
                 patches_per_image=4, use_augmentation=False,
                 transform=None, split_ratio=(0.7, 0.2, 0.1)):
        super().__init__(dataset_root, split, transform)

        self.nuclei_track = nuclei_track
        self.mode = mode
        self.use_context = use_context
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image if split == 'train' else 1
        self.use_augmentation = use_augmentation and (split == 'train')
        self.split_ratio = split_ratio
        self.is_train = (split == 'train')

        # Tissue config (always needed for joint mode and postprocessing)
        self.tissue_class_map = TISSUE_CLASS_MAP

        # Nuclei config
        nuclei_cfg = get_task_config('nuclei', nuclei_track)
        self.nuclei_class_map = nuclei_cfg.class_map
        self.num_nuclei_classes = nuclei_cfg.num_classes
        self.nuclei_class_names = nuclei_cfg.class_names

        self._build_index()

        if self.transform is None:
            self.transform = self._default_transform()

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

        roi_samples = []
        for roi_id in roi_ids:
            if self.use_context:
                img_path = image_dir / f'{roi_id}_context.tif'
            else:
                img_path = image_dir / f'{roi_id}.tif'

            tissue_path = tissue_dir / f'{roi_id}_tissue.geojson'
            nuclei_path = nuclei_dir / f'{roi_id}_nuclei.geojson'

            # Require both annotations for joint mode
            has_tissue = tissue_path.exists()
            has_nuclei = nuclei_path.exists()

            if not img_path.exists():
                continue
            if self.mode == 'tissue' and not has_tissue:
                continue
            if self.mode == 'nuclei' and not has_nuclei:
                continue
            if self.mode == 'joint' and not (has_tissue and has_nuclei):
                continue

            roi_samples.append({
                'image': img_path,
                'tissue_annotation': tissue_path if has_tissue else None,
                'nuclei_annotation': nuclei_path if has_nuclei else None,
                'roi_id': roi_id,
            })

        # Reproducible shuffle before split
        rng = np.random.RandomState(42)
        rng.shuffle(roi_samples)

        n = len(roi_samples)
        r_train, r_val, _ = self.split_ratio
        train_end = int(n * r_train)
        val_end = int(n * (r_train + r_val))

        if self.split == 'train':
            roi_samples = roi_samples[:train_end]
        elif self.split == 'val':
            roi_samples = roi_samples[train_end:val_end]
        elif self.split == 'test':
            roi_samples = roi_samples[val_end:]

        self.samples = []
        for roi in roi_samples:
            for patch_idx in range(self.patches_per_image):
                self.samples.append({**roi, 'patch_idx': patch_idx})

        print(f"PUMAJointDataset [{self.split}] mode={self.mode} track={self.nuclei_track}: "
              f"{len(roi_samples)} ROIs x {self.patches_per_image} patches = "
              f"{len(self.samples)} samples")

    def _load_image(self, path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

    def _load_and_cache(self, sample):
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

            cache_entry = {'image': image}

            # Tissue mask
            if self.mode in ('tissue', 'joint') and sample['tissue_annotation']:
                tissue_mask = rasterize_geojson(
                    sample['tissue_annotation'], self.tissue_class_map,
                    image_size=(orig_h, orig_w), coord_offset=coord_offset,
                )
                cache_entry['tissue_mask'] = tissue_mask

            # Nuclei: instance mask + class mask + HV maps
            if self.mode in ('nuclei', 'joint') and sample['nuclei_annotation']:
                instance_mask, class_mask = rasterize_geojson_instances(
                    sample['nuclei_annotation'], self.nuclei_class_map,
                    image_size=(orig_h, orig_w), coord_offset=coord_offset,
                )
                hv_map = generate_hv_map(instance_mask)
                np_map = generate_np_map(instance_mask)

                cache_entry['nuclei_mask'] = class_mask
                cache_entry['hv_map'] = hv_map
                cache_entry['np_map'] = np_map

            self._cache[roi_id] = cache_entry

        return self._cache[roi_id]

    def _random_crop(self, data_dict):
        """Apply same random crop to all spatial arrays in the dict."""
        image = data_dict['image']
        h, w = image.shape[:2]
        ps = self.patch_size

        if h <= ps or w <= ps:
            return data_dict

        y = np.random.randint(0, h - ps)
        x = np.random.randint(0, w - ps)

        result = {}
        for key, val in data_dict.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 2:  # [H, W] mask
                    result[key] = val[y:y+ps, x:x+ps]
                elif val.ndim == 3 and val.shape[0] in (2, 3):  # [C, H, W]
                    result[key] = val[:, y:y+ps, x:x+ps]
                elif val.ndim == 3:  # [H, W, C] image
                    result[key] = val[y:y+ps, x:x+ps]
                else:
                    result[key] = val
            else:
                result[key] = val
        return result

    def _default_transform(self):
        if not ALBUM_AVAILABLE:
            return None

        if self.use_augmentation:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                                   rotate_limit=30, p=0.5),
                A.ElasticTransform(alpha=50, sigma=10, p=0.2),
                StainAugmentation(sigma=0.05, p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15,
                              saturation=0.15, hue=0.05, p=0.3),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(blur_limit=(3, 5), p=1),
                ], p=0.2),
                A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                ToTensorV2(),
            ],
                additional_targets=self._additional_targets(),
            )
        else:
            return A.Compose(
                [
                    A.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                    ToTensorV2(),
                ],
                additional_targets=self._additional_targets(),
            )

    def _additional_targets(self):
        """Define additional targets for albumentations to keep masks in sync."""
        targets = {}
        if self.mode in ('tissue', 'joint'):
            targets['tissue_mask'] = 'mask'
        if self.mode in ('nuclei', 'joint'):
            targets['nuclei_mask'] = 'mask'
            targets['np_map'] = 'mask'
        return targets

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = self._load_and_cache(sample)

        # Deep copy to avoid modifying cache
        data = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}

        if self.is_train:
            data = self._random_crop(data)
        else:
            if ALBUM_AVAILABLE:
                h, w = self.image_size
                resize = A.Resize(h, w)
                # Resize image
                resized = resize(image=data['image'])
                data['image'] = resized['image']
                # Resize masks
                for key in ('tissue_mask', 'nuclei_mask', 'np_map'):
                    if key in data:
                        r = resize(image=data[key])
                        data[key] = r['image']
                # Resize HV map channels
                if 'hv_map' in data:
                    hv = data['hv_map']  # [2, H, W]
                    hv_resized = np.stack([
                        resize(image=hv[0])['image'],
                        resize(image=hv[1])['image'],
                    ], axis=0)
                    data['hv_map'] = hv_resized

        # Apply augmentation transform
        if self.transform is not None and ALBUM_AVAILABLE:
            aug_input = {'image': data['image']}
            if 'tissue_mask' in data:
                aug_input['tissue_mask'] = data['tissue_mask']
            if 'nuclei_mask' in data:
                aug_input['nuclei_mask'] = data['nuclei_mask']
            if 'np_map' in data:
                aug_input['np_map'] = data['np_map']

            transformed = self.transform(**aug_input)
            image = transformed['image']

            result = {'image': image}

            if 'tissue_mask' in transformed:
                result['tissue_mask'] = transformed['tissue_mask'].long()
            if 'nuclei_mask' in transformed:
                result['nuclei_mask'] = transformed['nuclei_mask'].long()
            if 'np_map' in transformed:
                result['np_map'] = transformed['np_map'].long()

            # HV map is NOT augmented by albumentations (it's a regression target)
            # We need to handle spatial transforms manually
            if 'hv_map' in data:
                hv = data['hv_map']  # [2, H, W] numpy
                result['hv_map'] = torch.from_numpy(hv).float()
        else:
            image = torch.from_numpy(data['image']).permute(2, 0, 1).float() / 255.0
            result = {'image': image}
            if 'tissue_mask' in data:
                result['tissue_mask'] = torch.from_numpy(data['tissue_mask']).long()
            if 'nuclei_mask' in data:
                result['nuclei_mask'] = torch.from_numpy(data['nuclei_mask']).long()
            if 'np_map' in data:
                result['np_map'] = torch.from_numpy(data['np_map']).long()
            if 'hv_map' in data:
                result['hv_map'] = torch.from_numpy(data['hv_map']).float()

        return result

    def __len__(self):
        return len(self.samples)


def puma_collate_fn(batch):
    """Custom collate function for PUMAJointDataset.

    Handles variable keys in batch items depending on training mode.
    """
    result = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = values

    return result
