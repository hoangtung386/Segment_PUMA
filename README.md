# PUMA Challenge - Cell Segmentation

Panoptic segmentation of **nUclei** and **tissue** in advanced **MelanomA** (PUMA).

A ResNet encoder-decoder segmentation model with optional Mamba SSM bottleneck and FPN decoder with attention gates for melanoma histopathology.

## Tasks

| Task | Description | Classes |
|------|-------------|---------|
| **Tissue segmentation** (Task 1) | Semantic segmentation of tissue regions | 6: Background, Tumor, Stroma, Epithelium, Blood Vessel, Necrosis |
| **Nuclei segmentation** Track 1 (Task 2) | Nuclei detection with 3 instance classes | 4: Background, Tumor, TILs, Other |
| **Nuclei segmentation** Track 2 (Task 2) | Nuclei detection with 10 instance classes | 11: Background, Tumor, Lymphocyte, Plasma Cell, Histiocyte, Melanophage, Neutrophil, Stroma, Epithelium, Endothelium, Apoptosis |

## Quick Start

### 1. Environment Setup

```bash
conda create -n puma python=3.11
conda activate puma
pip install -r requirements.txt
```

Optional (Mamba SSM bottleneck, requires CUDA):
```bash
pip install mamba-ssm
```

If Mamba is not installed, the model automatically falls back to a standard convolutional bottleneck.

### 2. Dataset

Place the PUMA dataset at `./dataset_PUMA/` with this structure:

```
dataset_PUMA/
    01_training_dataset_tif_ROIs/            *.tif (1024x1024, RGBA)
    01_training_dataset_tif_context_ROIs/     *_context.tif (5120x5120)
    01_training_dataset_geojson_tissue/       *_tissue.geojson
    01_training_dataset_geojson_nuclei/       *_nuclei.geojson
```

### 3. Train

```bash
# Tissue segmentation (default, 6 classes)
python scripts/train.py --devices 0

# Nuclei segmentation - Track 1 (4 classes)
python scripts/train.py --devices 0 --task nuclei --nuclei-track 1

# Nuclei segmentation - Track 2 (11 classes)
python scripts/train.py --devices 0 --task nuclei --nuclei-track 2

# Multi-GPU
python scripts/train.py --devices 0,1

# Custom config
python scripts/train.py --config path/to/config.yaml --devices 0
```

`NUM_CLASSES` is auto-set based on `--task` and `--nuclei-track` via `config.resolve_task()`.

### Train on Google Colab Pro

Open `train_colab.ipynb` on Colab with GPU runtime. The notebook uses `configs/colab_g4.yaml` optimized for high-VRAM GPUs (95.6 GB):

- **Batch size 32**, encoder `[128, 256, 512, 1024]`, bottleneck 2048, patch size 768
- **AMP** (mixed precision) enabled
- Checkpoints auto-saved to Google Drive every epoch
- **Auto-resume**: if Colab disconnects, re-run all cells to resume from the last checkpoint

Dataset should be placed at `MyDrive/dataset_PUMA/` on Google Drive.

```bash
# Or run manually on Colab terminal:
python scripts/train.py --devices 0 --config configs/colab_g4.yaml
```

### 4. Evaluate

```bash
# Evaluate on test set (default) - unbiased performance measurement
python scripts/evaluate.py --checkpoint checkpoints/best_puma.pth --devices 0

# Evaluate on val set (same set used during training for model selection)
python scripts/evaluate.py --checkpoint checkpoints/best_puma.pth --split val --devices 0

# With task override
python scripts/evaluate.py --checkpoint checkpoints/best_puma.pth --task nuclei --nuclei-track 1
```

**Dataset split strategy** (`SPLIT_RATIO: [0.7, 0.2, 0.1]`):
- **train (70%)**: Used for training (with augmentation, patch-based)
- **val (20%)**: Used during training for model selection (best Dice checkpoint) and early stopping
- **test (10%)**: Held out, only used by `evaluate.py` for unbiased final performance reporting

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

## Configuration

All settings are in `configs/base.yaml` (defaults defined in `configs/config.py`). Key parameters:

```yaml
# Task
TASK: "tissue"               # 'tissue' or 'nuclei'
NUCLEI_TRACK: 1              # 1 or 2 (only for nuclei task)
USE_CONTEXT: false           # true = use 5120x5120 context images

# Data
IMAGE_SIZE: [512, 512]       # Validation image size
PATCH_SIZE: 512              # Training patch size (random crop)
PATCHES_PER_IMAGE: 4         # Number of patches per ROI during training

# Model
ENCODER_CHANNELS: [64, 128, 256, 512]
BOTTLENECK_TYPE: "standard"  # 'mamba' or 'standard'
BOTTLENECK_CHANNELS: 1024
USE_SE: true                 # Squeeze-and-Excitation attention in encoder
USE_ATTENTION_GATES: true    # Attention gates on decoder skip connections

# Training
BATCH_SIZE: 8
NUM_EPOCHS: 300
LEARNING_RATE: 0.001
WEIGHT_DECAY: 0.0001
DROPOUT: 0.1
GRAD_CLIP_NORM: 1.0

# Loss weights
DICE_WEIGHT: 0.5
FOCAL_WEIGHT: 0.5
CE_WEIGHT: 0.3
BOUNDARY_WEIGHT: 0.1
FP_PENALTY_WEIGHT: 0.2
CLUSTER_WEIGHT: 0.1          # Multi-scale deep supervision

# Monitoring
USE_WANDB: true
EARLY_STOPPING_PATIENCE: 50
```

Override via CLI or custom YAML file with `--config`.

## Architecture

```
Input [B, 3, H, W]
  |
  v
ResNet Encoder (Stem + 4 stages with SE attention)
  Stem: Conv7x7(stride=2) + Conv3x3 + MaxPool -> H/4
  Stage 1: ResBlock x N (64ch)  -> H/8
  Stage 2: ResBlock x N (128ch) -> H/16
  Stage 3: ResBlock x N (256ch) -> H/32
  Stage 4: ResBlock x N (512ch) -> H/64
  |
  v
Bottleneck (Conv 512->1024 + Mamba SSM or Standard Conv)
  Mamba: flatten 2D -> 1D sequence, apply Mamba-2 layers, reshape back
  Standard: 2x (Conv3x3 + GroupNorm + ReLU) with residual
  |
  v
FPN Decoder (5 decoder blocks + attention gates)
  Each block: ConvTranspose2d -> AttentionGate(skip) -> Concat -> 2x Conv3x3
  Multi-scale prediction heads at 4 intermediate scales (deep supervision)
  Final upsample to full resolution
  |
  v
Output [B, NUM_CLASSES, H, W]
```

**Key Components:**
- **ResNet Encoder**: 4-stage encoder with SE (Squeeze-and-Excitation) channel attention, trained from scratch (no pretrained weights)
- **Bottleneck**: Mamba-2 SSM (O(N) sequence complexity) with automatic fallback to standard conv
- **FPN Decoder**: Feature Pyramid Network with attention gates on skip connections for learned feature selection
- **Deep Supervision**: Multi-scale prediction heads at 4 intermediate decoder stages
- **Loss**: Dice + Focal + CrossEntropy + boundary loss + false positive penalty + multi-scale cluster loss

## Project Structure

```
Segment_PUMA/
├── configs/
│   ├── config.py              # TrainingConfig dataclass with resolve_task(), get_class_names()
│   ├── constants.py           # Centralized class maps, class names, IMAGENET constants
│   ├── base.yaml              # Default configuration values
│   └── colab_g4.yaml          # Colab Pro G4 config (95.6 GB VRAM)
├── models/
│   ├── segmentor.py           # CellSegmentor (main model, includes from_config() factory)
│   ├── encoder.py             # ResNetEncoder with SE attention
│   ├── losses.py              # SegmentationLoss (Dice + Focal + CE + boundary + FP)
│   ├── components.py          # Reusable component exports
│   ├── bottleneck/
│   │   ├── base.py            # Abstract BaseBottleneck
│   │   ├── mamba.py           # MambaBottleneckWrapper
│   │   └── standard.py        # StandardBottleneck (conv fallback)
│   ├── decoder/
│   │   └── hvt.py             # FPNDecoder with AttentionGate + DecoderBlock
│   ├── layers/
│   │   └── mamba.py           # MambaBlock + SimplifiedSSM
│   └── experimental/
│       └── kan.py             # KAN layers (Kolmogorov-Arnold Networks, not in default pipeline)
├── datasets/
│   ├── base.py                # BaseDataset abstract class
│   ├── puma_dataset.py        # PUMADataset (GeoJSON rasterization, patch-based, stain augmentation)
│   ├── cell_dataset.py        # CellDataset (generic images/ + masks/ directory)
│   └── factory.py             # get_dataset_class() + create_dataloader() factories
├── training/
│   └── trainer.py             # Trainer (train loop, validation, checkpointing, W&B logging)
├── evaluation/
│   ├── evaluator.py           # Full evaluation pipeline
│   ├── metrics.py             # MetricCalculator (Dice, IoU, HD95, Precision, Recall + 95% CI)
│   ├── visualization.py       # Prediction overlays, confusion matrix plots
│   └── complexity.py          # Model complexity (FLOPs, params)
├── utils/
│   ├── data_utils.py          # compute_class_weights() for class imbalance
│   └── device.py              # early_device_setup() for CUDA device selection
├── scripts/
│   ├── train.py               # Training entry point
│   └── evaluate.py            # Evaluation entry point
├── train_colab.ipynb              # Colab Pro training notebook (auto-resume from Drive)
├── tests/
│   ├── test_model.py          # Model forward pass and shape tests
│   ├── test_losses.py         # Loss function tests
│   └── test_config.py         # Config and task config tests
└── dataset_PUMA/              # Dataset (not tracked in git)
```

## Key Design Patterns

### Centralized Task Config
All class names, class maps, and num_classes are defined once in `configs/constants.py`. Use `get_task_config(task, nuclei_track)` to get the configuration for any task. This avoids duplication across modules.

### Factory Methods
- **Model**: `CellSegmentor.from_config(config)` creates a model from a `TrainingConfig` object
- **DataLoader**: `create_dataloader(config, split, use_augmentation)` creates a ready-to-use DataLoader
- **Bottleneck**: `get_bottleneck(type, channels)` selects Mamba or Standard bottleneck

### Adding a New Task
1. Add the class map, class names, and `TaskConfig` entry in `configs/constants.py`
2. Add the GeoJSON parsing logic in `datasets/puma_dataset.py` if needed
3. Everything else (training, evaluation, logging) will automatically pick up the new task

## Checkpoints

Saved to `checkpoints/`:
- `best_puma.pth` - Best model (highest validation Dice)
- `puma.pth` - Latest epoch

Checkpoint contents: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `val_dice`, `best_dice`, `config`.

## Monitoring

Training metrics are logged to:
- **CSV**: `outputs/training_puma_log.csv`
- **W&B** (if enabled): batch-level and epoch-level metrics, per-class Dice, gradient norms, GPU memory, prediction visualizations

## License

See [LICENSE](LICENSE).
