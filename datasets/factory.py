from torch.utils.data import DataLoader

from .puma_dataset import PUMADataset
from .cell_dataset import CellDataset


def get_dataset_class(dataset_name: str):
    """Factory function to get the correct dataset class."""
    datasets = {
        'puma': PUMADataset,
        'cell': CellDataset,
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")

    return datasets[dataset_name]


def create_dataloader(config, split: str, use_augmentation: bool = False,
                      batch_size: int = None, shuffle: bool = None) -> DataLoader:
    """Create a DataLoader from config for any split.

    Args:
        config: TrainingConfig instance.
        split: 'train', 'val', or 'test'.
        use_augmentation: Whether to apply data augmentation.
        batch_size: Override config.BATCH_SIZE if provided.
        shuffle: Override default shuffle (True for train, False otherwise).

    Returns:
        DataLoader ready to use.
    """
    dataset_class = get_dataset_class(config.DATASET_NAME)

    dataset_kwargs = dict(
        dataset_root=config.DATA_ROOT,
        image_size=config.IMAGE_SIZE,
    )

    if config.DATASET_NAME == 'puma':
        dataset_kwargs.update(
            task=config.TASK,
            nuclei_track=config.NUCLEI_TRACK,
            use_context=config.USE_CONTEXT,
            split_ratio=config.SPLIT_RATIO,
            patch_size=config.PATCH_SIZE,
            patches_per_image=config.PATCHES_PER_IMAGE if split == 'train' else 1,
            use_stain_norm=getattr(config, 'USE_STAIN_NORM', True),
        )

    dataset = dataset_class(
        split=split,
        use_augmentation=use_augmentation,
        **dataset_kwargs,
    )

    bs = batch_size or config.BATCH_SIZE
    do_shuffle = shuffle if shuffle is not None else (split == 'train')

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=do_shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=(split == 'train'),
    )
