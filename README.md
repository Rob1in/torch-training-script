# PyTorch Object Detection Training Script

A PyTorch-based object detection training pipeline supporting multiple model architectures with multiclass detection capabilities. Designed for RGB images using JSONL-formatted datasets with normalized bounding box annotations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (regular mode)
python src/train.py --config-name=train

# Run training with custom parameters
python src/train.py --config-name=train training.batch_size=16 training.num_epochs=50

# Evaluate a trained model
python src/eval.py dataset_dir=triangles_dataset_small run_dir=outputs/2026-01-31/20-15-26

# Convert to ONNX for deployment (FasterRCNN only)
bash convert_model.sh outputs/2026-01-31/20-15-26

# Run hyperparameter optimization (requires: pip install -e ".[sweep]")
python src/train.py --config-name=sweep --multirun
```

## Features

- **Multiclass Detection**: Train on multiple object classes simultaneously
- **RGB Images Only**: Optimized for 3-channel RGB input (no grayscale support)
- **JSONL Dataset Format**: Uses JSONL files with normalized bounding box annotations
- **Multiple Model Architectures**: Supports Faster R-CNN, SSD-Lite, EfficientNet, and Simple Detector
- **PyTorch Reference Training**: Follows PyTorch's official detection training best practices
  - SGD optimizer with momentum
  - Linear warmup + MultiStepLR scheduling
  - Per-parameter learning rates (lower LR for backbone)
  - Gradient clipping
  - Default loss weighting from torchvision
- **Automatic Class Discovery**: Can auto-discover classes from dataset or use explicit configuration
- **COCO Evaluation**: Automatic conversion from JSONL to COCO format for evaluation metrics
- **Hydra Configuration**: Flexible configuration management with Hydra

## Requirements

- **Python** >= 3.10
- See [Installation](#installation) section for package dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd torch-training-script
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format

The training pipeline expects datasets in JSONL format where each line is a JSON object containing:

```json
{
  "image_path": "path/to/image.jpg",
  "bounding_box_annotations": [
    {
      "annotation_label": "person",
      "x_min_normalized": 0.1,
      "y_min_normalized": 0.2,
      "x_max_normalized": 0.5,
      "y_max_normalized": 0.8
    },
    {
      "annotation_label": "car",
      "x_min_normalized": 0.6,
      "y_min_normalized": 0.3,
      "x_max_normalized": 0.9,
      "y_max_normalized": 0.7
    }
  ]
}
```

**Key points:**
- `image_path`: Path to the image file (can be absolute or relative to `data_dir`)
- `annotation_label`: The class name for this bounding box
- Coordinates are normalized (0.0 to 1.0) relative to image dimensions
- Images must be RGB format (3 channels)

## Configuration

### Classes Configuration

The `classes` field in `configs/config.yaml` determines which annotation labels to train on:

**Option 1: Auto-discover all classes** (default)
```yaml
classes: null  # Uses all annotation labels found in the dataset
```

**Option 2: Train on specific classes**
```yaml
classes:
  - triangle
  - person
  - car
```

**Option 3: Single class detection**
```yaml
classes:
  - person
```

The `classes` configuration:
- Is defined at the top level in `configs/config.yaml` (overrides any values in other configs)
- Determines `model.num_classes` automatically before model creation
- Filters annotations in all datasets (train, val, test)
- Creates consistent label-to-ID mappings across the pipeline

### Dataset Paths

Configure dataset paths in `configs/dataset/jsonl.yaml`:

```yaml
data:
  train_jsonl: dataset.jsonl
  train_data_dir: data
  val_jsonl: dataset.jsonl
  val_data_dir: data
  test_jsonl: dataset.jsonl
  test_data_dir: data
```

### Model Selection

Select a model in `configs/config.yaml`:

```yaml
defaults:
  - model: faster_rcnn  # Options: faster_rcnn, ssdlite, effnet, custom_detector
  - dataset: jsonl
  - _self_
```

## Supported Models

### Faster R-CNN
- **Config**: `configs/model/faster_rcnn.yaml`
- **Backbone**: MobileNetV3-Large with FPN
- **Input Size**: Configurable (default: 600x800)
- **Best for**: High accuracy, slower inference

### SSD-Lite
- **Config**: `configs/model/ssdlite.yaml`
- **Backbone**: MobileNetV3-Large
- **Input Size**: 320x320
- **Best for**: Fast inference, mobile deployment

### EfficientNet
- **Config**: `configs/model/effnet.yaml`
- **Backbone**: EfficientNet-B0
- **Input Size**: 224x224
- **Best for**: Balanced accuracy and speed

### Simple Detector
- **Config**: `configs/model/custom_detector.yaml`
- **Architecture**: Simple CNN backbone (3 conv layers + detection heads)
- **Input Size**: Configurable (default: 640x512)
- **Best for**: Baseline experiments and learning

## Training

The training pipeline supports two modes: **regular training** and **hyperparameter optimization**.

### Mode 1: Regular Training (Recommended)

Uses pre-computed hyperparameters from previous optimization runs.

**Basic usage:**
```bash
python src/train.py --config-name=train
```

**With custom parameters:**
```bash
python src/train.py --config-name=train training.batch_size=16 training.num_epochs=50
```

**With specific classes:**
Edit `configs/train.yaml` to set your classes:
```yaml
classes:
  - person
  - car
```

Then run:
```bash
python src/train.py --config-name=train
```

### Mode 2: Hyperparameter Optimization (Advanced)

Run Optuna sweeps to find optimal hyperparameters for your dataset.

**Requirements:**
```bash
pip install -e ".[sweep]"
```

**Run a sweep:**
```bash
python src/train.py --config-name=sweep --multirun
```

This will:
- Run 30 trials (configurable in `configs/sweep.yaml`)
- Optimize learning rate, weight decay, and loss weights
- Save results to Hydra's multirun output directory
- Print the best hyperparameters at the end

**Update optimization results:**
After a successful sweep, copy the best parameters to `configs/optimization_results/` for future use.

### Training Hyperparameters

The training pipeline follows **PyTorch's reference detection training** best practices:

#### Optimizer
- **Type**: SGD with Nesterov momentum
- **Learning Rate**: 0.01 (base, for batch_size=16)
- **Momentum**: 0.9
- **Weight Decay**: 0.0001 (L2 regularization)
- **Per-parameter LR**: Backbone gets 10x lower learning rate (0.001 by default)

#### Learning Rate Schedule
- **Warmup**: Linear warmup for first 500 iterations
  - Starts at 0.1% of base LR (0.00001 for base=0.01)
  - Linearly increases to base LR
- **Schedule**: MultiStepLR
  - Reduces LR by 10x at epochs [16, 22] (for 26-epoch training)
  - Adjustable via `training.lr_steps` in config

#### Gradient Clipping
- **Max Norm**: 10.0
- Prevents exploding gradients during training

#### Loss Function
- Uses **default torchvision loss weights** (no custom weighting)
- For Faster R-CNN: combines RPN + detection head losses
- For SSD-Lite: combines classification + localization losses

**Scaling for batch size:**
If you change batch size, scale the learning rate linearly:
- batch_size=8 → lr=0.005
- batch_size=16 → lr=0.01 (default)
- batch_size=32 → lr=0.02

### Understanding Output Directories

The training pipeline creates two different output directories depending on the run mode:

#### 📁 `outputs/` - Single Training Runs

Used for **regular training** (`--config-name=train`):

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/                    # Timestamp of run
        ├── .hydra/
        │   ├── config.yaml          # Full config used for this run
        │   ├── hydra.yaml           # Hydra settings
        │   └── overrides.yaml       # CLI overrides you provided
        ├── best_model.pth           # Saved checkpoint (best validation loss)
        ├── tensorboard/             # TensorBoard logs
        │   └── events.out.tfevents.*
        └── train.log                # Training logs (loss, metrics, etc.)
```

**What you'll find:**
- **`best_model.pth`**: Your trained model checkpoint (use this for evaluation)
- **`.hydra/config.yaml`**: Exact configuration used (for reproducibility)
- **`train.log`**: All training output (epochs, losses, validation metrics)
- **`tensorboard/`**: Training curves (visualize with `tensorboard --logdir outputs/`)

**Example:**
```bash
# Train once
python src/train.py --config-name=train

# Output saved to: outputs/2026-01-30/14-25-30/
```

---

#### 📁 `multirun/` - Hyperparameter Sweeps (Optuna)

Used for **hyperparameter optimization** (`--config-name=sweep --multirun`):

```
multirun/
└── YYYY-MM-DD/
    └── HH-MM-SS/                    # Timestamp of sweep start
        ├── 0/                       # Trial 0 (first hyperparameter combination)
        │   ├── .hydra/
        │   │   ├── config.yaml      # Config for this trial
        │   │   └── overrides.yaml   # Hyperparameters Optuna chose
        │   ├── tensorboard/
        │   └── train.log
        ├── 1/                       # Trial 1 (second combination)
        │   ├── .hydra/
        │   ├── tensorboard/
        │   └── train.log
        ├── 2/                       # Trial 2 (third combination)
        │   └── ...
        └── optimization_results.yaml # Best hyperparameters found
```

**What you'll find:**
- **Numbered directories (0, 1, 2, ...)**: Each trial's results
- **`.hydra/overrides.yaml`**: The hyperparameters Optuna tested for that trial
  ```yaml
  - training.learning_rate=0.0001025
  - training.weight_decay=0.0007114
  - training.loss.cls_loss_weight=0.6392
  ```
- **`optimization_results.yaml`**: Summary with best hyperparameters and their validation loss
- **No `best_model.pth`**: Sweeps don't save models by default (focused on finding best hyperparameters)

**Example:**
```bash
# Run sweep with 30 trials
python src/train.py --config-name=sweep --multirun

# Output saved to: multirun/2026-01-30/14-30-15/
#   ├── 0/  (trial 0 with learning_rate=0.001, weight_decay=1e-5)
#   ├── 1/  (trial 1 with learning_rate=0.0002, weight_decay=5e-6)
#   └── ... (28 more trials)
```

---

#### 🔍 Key Differences

| Feature | `outputs/` (Single Run) | `multirun/` (Sweep) |
|---------|------------------------|---------------------|
| **Created by** | `--config-name=train` | `--config-name=sweep --multirun` |
| **Purpose** | Train one model | Find best hyperparameters |
| **Structure** | One directory per run | One directory per trial |
| **Checkpoint** | ✅ `best_model.pth` saved | ❌ No checkpoints (hyperparameter search) |
| **Use case** | Production training | Hyperparameter tuning |
| **Training time** | Full epochs (e.g., 25) | Can use fewer epochs (e.g., 10-15) |

---

#### 💡 Typical Workflow

1. **First**: Run hyperparameter sweep to find best parameters
   ```bash
   python src/train.py --config-name=sweep --multirun
   # Check multirun/YYYY-MM-DD/HH-MM-SS/optimization_results.yaml
   ```

2. **Then**: Copy best parameters to `configs/optimization_results/`

3. **Finally**: Train production model with best hyperparameters
   ```bash
   python src/train.py --config-name=train
   # Get best_model.pth from outputs/YYYY-MM-DD/HH-MM-SS/
   ```

## Evaluation

The evaluation script (`src/eval.py`) evaluates trained models on test datasets and computes COCO metrics.

### Basic Usage

**Required arguments:**
- `dataset_dir`: Directory containing `dataset.jsonl` and `data/` folder
- `run_dir`: Training output directory (contains `.hydra/config.yaml` and `best_model.pth`)

```bash
# Evaluate a trained model
python src/eval.py \
    dataset_dir=triangles_dataset_small \
    run_dir=outputs/2026-01-31/20-15-26
```

**What happens:**
1. Loads training config from `run_dir/.hydra/config.yaml` (preserves model architecture, classes, etc.)
2. Auto-detects checkpoint at `run_dir/best_model.pth` (or use `checkpoint_path` to override)
3. Loads test dataset from `dataset_dir/dataset.jsonl` and `dataset_dir/data/`
4. Uses **Model EMA weights** if available (better evaluation performance)
5. Computes COCO metrics (mAP, AP50, AP75, etc.)
6. Saves results to `run_dir/eval_<dataset_name>_<checkpoint_name>_<format>/`

### Using Custom Checkpoint Path

You can override the checkpoint path to evaluate ONNX models or custom checkpoints:

```bash
# Evaluate an ONNX model
python src/eval.py \
    dataset_dir=triangles_dataset_small \
    run_dir=outputs/2026-01-31/20-15-26 \
    checkpoint_path=outputs/2026-01-31/20-15-26/onnx_model/model.onnx

# Evaluate a specific checkpoint
python src/eval.py \
    dataset_dir=triangles_dataset_small \
    run_dir=outputs/2026-01-31/20-15-26 \
    checkpoint_path=outputs/2026-01-31/20-15-26/checkpoint_epoch_10.pth
```

### Evaluation Outputs

Evaluation results are saved to:
```
run_dir/eval_<dataset_name>_<checkpoint_name>_<format>/
```

**Example:**
```
outputs/2026-01-31/20-15-26/
└── eval_triangles_dataset_small_best_model_pth/
    ├── faster_rcnn_predictions.json    # COCO format predictions
    ├── faster_rcnn_metrics.json        # mAP, AP50, AP75, etc.
    ├── ground_truth_coco.json         # Auto-converted COCO format ground truth
    └── visualizations/                 # Random images with predicted + ground truth boxes
        ├── 0000_detected.png
        ├── 0001_detected.png
        └── ...
```

**Output files:**
- **`{model}_predictions.json`** - Predictions in COCO format
- **`{model}_metrics.json`** - COCO evaluation metrics (mAP, AP50, AP75, etc.)
- **`ground_truth_coco.json`** - Ground truth converted to COCO format
- **`visualizations/`** - Sample images with predicted and ground truth bounding boxes

### COCO Metrics Explained

The evaluation script reports:
- **AP** (mAP @ IoU=0.50:0.95): Main metric, stricter evaluation
- **AP50** (mAP @ IoU=0.50): Common metric, more lenient
- **AP75** (mAP @ IoU=0.75): Stricter localization
- **APs, APm, APl**: AP for small, medium, large objects
- **AR** (Average Recall): Max recall given a fixed number of detections

**Automatic Processing:**
1. Converts JSONL ground truth → COCO format (if needed)
2. Scales predictions to original image dimensions
3. Evaluates using pycocotools
4. Saves results and visualizations

## ONNX Conversion

After training and evaluating your model, convert it to ONNX format for production deployment:

```bash
# Convert trained model to ONNX (FasterRCNN only)
bash convert_model.sh outputs/2026-02-02/15-15-47
```

**What this does:**
1. Converts PyTorch model to ONNX format with uint8 input support
2. Runs internal consistency tests (PyTorch vs ONNX)
3. Tests on first 5 images from training dataset
4. Saves everything to `outputs/2026-02-02/15-15-47/onnx_model/`

**Output structure:**
```
outputs/2026-02-02/15-15-47/onnx_model/
├── model.onnx                 # ONNX model (ready for deployment)
├── conversion_summary.txt     # Conversion details
└── test_results/              # Visualizations of test inferences
    ├── 0000_detected.png
    └── ...
```

**ONNX Model Specifications:**
- **Input**: `image` - uint8 tensor `[batch_size, 3, H, W]` with values 0-255
- **Outputs**:
  - `location`: Bounding boxes `[batch_size, max_detections, 4]` in (x1, y1, x2, y2) format
  - `score`: Confidence scores `[batch_size, max_detections]`
  - `category`: Class labels `[batch_size, max_detections]` (1-indexed)

**Note**: Currently only **FasterRCNN** models are supported for ONNX export.

For detailed usage and deployment examples, see `ONNX_QUICKSTART.md`.

## Project Structure

```
torch-training-script/
├── configs/
│   ├── config.yaml              # Legacy config (use train.yaml or sweep.yaml instead)
│   ├── train.yaml               # Config for regular training
│   ├── sweep.yaml               # Config for hyperparameter optimization
│   ├── dataset/
│   │   └── jsonl.yaml           # Dataset paths and transforms
│   ├── model/
│   │   ├── faster_rcnn.yaml
│   │   ├── ssdlite.yaml
│   │   ├── effnet.yaml
│   │   └── custom_detector.yaml
│   └── optimization_results/    # Pre-computed hyperparameters (may be outdated)
│       ├── faster_rcnn.yaml
│       └── ssdlite.yaml
├── src/
│   ├── train.py                 # Training script
│   ├── eval.py                  # Evaluation script
│   ├── datasets/
│   │   └── viam_dataset.py      # JSONL dataset loader
│   ├── models/
│   │   ├── faster_rcnn_detector.py
│   │   ├── ssdlite_detector.py
│   │   ├── effnet_detector.py
│   │   └── custom_detector.py
│   └── utils/
│       ├── transforms.py         # Data augmentation transforms
│       └── coco_converter.py     # JSONL to COCO converter
├── requirements.txt
└── pyproject.toml
```

## Key Implementation Details

### Multiclass Detection

- Classes are discovered from `annotation_label` fields in JSONL files
- Label-to-ID mapping is created automatically (1-based, 0 is background)
- Model `num_classes` is set automatically based on the number of classes
- All datasets (train/val/test) use the same class configuration

### RGB-Only Support

- All models assume 3-channel RGB input
- No grayscale conversion or single-channel support
- ImageNet normalization stats used by default

### Class Configuration Flow

1. `classes` is read from top-level `configs/config.yaml`
2. If `null`, classes are auto-discovered from the training dataset
3. `model.num_classes` is set to `len(classes)` before model creation
4. All datasets are created with the same `classes` list
5. COCO converter uses the same `classes` for evaluation

### Hydra Configuration Precedence

With `_self_` last in `defaults`, values in `configs/config.yaml` override values from other configs:
- `model/*.yaml` loaded first
- `dataset/jsonl.yaml` merged second
- `config.yaml` (`_self_`) merged last (highest precedence)

## Installation

### Option 1: Using requirements.txt (all dependencies)

```bash
pip install -r requirements.txt
```

### Option 2: Using pyproject.toml (selective dependencies)

Install only what you need:

```bash
# Core dependencies (minimum required)
pip install -e ".[core,config]"

# For training
pip install -e ".[train]"

# For evaluation
pip install -e ".[eval]"

# Everything (recommended)
pip install -e ".[all]"

# With development tools
pip install -e ".[all,dev]"
```

### Dependency Groups

- **core**: PyTorch, torchvision, numpy, pillow
- **config**: Hydra and OmegaConf for configuration management
- **train**: Training-specific dependencies (tqdm, torchinfo)
- **eval**: Evaluation-specific dependencies (pycocotools, matplotlib)
- **sweep**: Hyperparameter optimization (optuna, hydra-optuna-sweeper)
- **dev**: Development tools (pytest, black, flake8, mypy)
- **all**: All dependencies combined (excluding sweep and dev)

## Requirements

Key dependencies:
- **PyTorch** >= 2.0.0 - Deep learning framework
- **torchvision** >= 0.15.0 - Computer vision models and transforms
- **Hydra** >= 1.3.0 - Configuration management
- **pycocotools** >= 2.0.0 - COCO evaluation metrics
- **Pillow** >= 9.0.0 - Image processing
- **numpy** >= 1.21.0 - Numerical operations
- **matplotlib** >= 3.5.0 - Visualization (for evaluation)
- **tqdm** >= 4.64.0 - Progress bars
- **torchinfo** >= 1.8.0 - Model summary
- **tensorboard** >= 2.11.0 - Training visualization
- **optuna** >= 3.0.0 - Hyperparameter optimization (optional, install with `[sweep]`)
- **hydra-optuna-sweeper** >= 1.2.0 - Hydra integration for Optuna (optional, install with `[sweep]`)

