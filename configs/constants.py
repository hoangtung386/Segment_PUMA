"""
Centralized constants and task configurations.

Single source of truth for class mappings, class names, and shared constants.
All modules should import from here instead of hardcoding values.
"""
from typing import Dict, List, NamedTuple, Optional, Tuple


# --- Image normalization (ImageNet) ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- Task configuration ---

class TaskConfig(NamedTuple):
    num_classes: int
    class_names: List[str]
    class_map: Optional[Dict[str, int]]  # GeoJSON class name -> class index


# Tissue segmentation (6 classes)
TISSUE_CLASS_MAP = {
    'tissue_tumor': 1,
    'tissue_stroma': 2,
    'tissue_epithelium': 3,
    'tissue_blood_vessel': 4,
    'tissue_necrosis': 5,
    'tissue_white_background': 0,
}

TISSUE_CONFIG = TaskConfig(
    num_classes=6,
    class_names=['Background', 'Tumor', 'Stroma', 'Epithelium', 'Blood Vessel', 'Necrosis'],
    class_map=TISSUE_CLASS_MAP,
)

# Nuclei Track 1 (4 classes)
NUCLEI_TRACK1_CLASS_MAP = {
    'nuclei_tumor': 1,
    'nuclei_lymphocyte': 2,
    'nuclei_plasma_cell': 2,
    'nuclei_histiocyte': 3,
    'nuclei_melanophage': 3,
    'nuclei_neutrophil': 3,
    'nuclei_stroma': 3,
    'nuclei_epithelium': 3,
    'nuclei_endothelium': 3,
    'nuclei_apoptosis': 3,
}

NUCLEI_TRACK1_CONFIG = TaskConfig(
    num_classes=4,
    class_names=['Background', 'Tumor', 'TILs', 'Other'],
    class_map=NUCLEI_TRACK1_CLASS_MAP,
)

# Nuclei Track 2 (11 classes)
NUCLEI_TRACK2_CLASS_MAP = {
    'nuclei_tumor': 1,
    'nuclei_lymphocyte': 2,
    'nuclei_plasma_cell': 3,
    'nuclei_histiocyte': 4,
    'nuclei_melanophage': 5,
    'nuclei_neutrophil': 6,
    'nuclei_stroma': 7,
    'nuclei_epithelium': 8,
    'nuclei_endothelium': 9,
    'nuclei_apoptosis': 10,
}

NUCLEI_TRACK2_CONFIG = TaskConfig(
    num_classes=11,
    class_names=[
        'Background', 'Tumor', 'Lymphocyte', 'Plasma Cell', 'Histiocyte',
        'Melanophage', 'Neutrophil', 'Stroma', 'Epithelium', 'Endothelium', 'Apoptosis',
    ],
    class_map=NUCLEI_TRACK2_CLASS_MAP,
)


def get_task_config(task: str, nuclei_track: int = 1) -> TaskConfig:
    """Get task configuration by task name and track number.

    Args:
        task: 'tissue' or 'nuclei'
        nuclei_track: 1 or 2 (only used when task='nuclei')

    Returns:
        TaskConfig with num_classes, class_names, and class_map
    """
    if task == 'tissue':
        return TISSUE_CONFIG
    elif task == 'nuclei':
        if nuclei_track == 2:
            return NUCLEI_TRACK2_CONFIG
        return NUCLEI_TRACK1_CONFIG
    else:
        raise ValueError(f"Unknown task: {task}. Use 'tissue' or 'nuclei'.")
