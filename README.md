# PyTorch Object Detection Training Script

A PyTorch-based object detection training pipeline supporting Faster R-CNN and SSD-Lite with multiclass detection capabilities. Designed for RGB images using Viam JSONL-formatted datasets.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#training)
  - [Classes Configuration](#classes-configuration)
  - [Dataset Paths](#dataset-paths)
  - [Model Selection](#model-selection)
  - [Regular Training](#regular-training)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Training Hyperparameters](#training-hyperparameters)
  - [Output Directories](#output-directories)
- [Evaluation](#evaluation)
- [ONNX Conversion](#onnx-conversion)
- [Viam Vision Service](#viam-vision-service)
- [Viam Integration Workflow](#viam-integration-workflow)
  - [Local Testing](#local-testing)
  - [Registry Deployment](#registry-deployment)
- [Project Structure](#project-structure)
- [Key Implementation Details](#key-implementation-details)
- [Key Dependencies](#key-dependencies)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download a dataset from Viam Cloud
viam dataset export --destination=./my_dataset --dataset-id=<dataset-id>

# Run training on your dataset
python src/train.py --config-name=train dataset.data.train_dir=./my_dataset

# Run training with other custom parameters
python src/train.py --config-name=train dataset.data.train_dir=./my_dataset training.batch_size=16 training.num_epochs=50

# Evaluate a trained model
python src/eval.py dataset_dir=./my_dataset run_dir=outputs/YYYY-MM-DD/HH-MM-SS

# Convert to ONNX for deployment (FasterRCNN only)
bash convert_model.sh outputs/YYYY-MM-DD/HH-MM-SS --dataset-dir ./my_dataset

# Run hyperparameter optimization (requires: pip install -e ".[sweep]")
python src/train.py --config-name=sweep --multirun
```

## Features

- **Multiclass Detection**: Train on multiple object classes simultaneously
- **RGB Images Only**: Optimized for 3-channel RGB input (no grayscale support)
- **JSONL Dataset Format**: Uses JSONL files with normalized bounding box annotations
- **Two Model Architectures**: Supports Faster R-CNN and SSD-Lite
- **PyTorch Reference Training**: Follows PyTorch's official detection training best practices
  - SGD optimizer with momentum (Adam also supported)
  - Linear warmup + MultiStepLR or CosineAnnealing scheduling
  - Optional normalization layer weight decay separation
  - Optional gradient clipping (disabled by default)
  - Default loss weighting from torchvision
- **Model EMA**: Exponential Moving Average for more stable models
- **Transfer Learning**: Pretrained COCO weights with configurable layer freezing
- **Automatic Class Discovery**: Can auto-discover classes from dataset or use explicit configuration
- **COCO Evaluation**: Automatic conversion from JSONL to COCO format for evaluation metrics
- **Hydra Configuration**: Flexible configuration management with Hydra

## Requirements

- **Python** >= 3.10
- See [Installation](#installation) section for package dependencies

## Installation

### Option 1: Using requirements.txt (all dependencies)

```bash
git clone <repository-url>
cd torch-training-script
pip install -r requirements.txt
```

### Option 2: Using pyproject.toml (selective dependencies)

Install only what you need:

```bash
# Core dependencies (minimum required)
pip install -e ".[core]"

# For training
pip install -e ".[train]"

# For evaluation
pip install -e ".[eval]"

# Everything (recommended)
pip install -e ".[all]"

# With hyperparameter optimization
pip install -e ".[sweep]"

# With development tools
pip install -e ".[all,dev]"
```

### Dependency Groups

- **core**: PyTorch, torchvision, numpy, pillow, hydra-core, omegaconf
- **train**: Training-specific dependencies (tqdm, torchinfo, tensorboard)
- **eval**: Evaluation-specific dependencies (pycocotools, matplotlib)
- **sweep**: Hyperparameter optimization (optuna, hydra-optuna-sweeper)
- **dev**: Development tools (pytest, black, flake8, mypy)
- **all**: All dependencies combined (excluding sweep and dev)

## Training

### Classes Configuration

The `classes` field in `configs/train.yaml` (or `configs/sweep.yaml`) determines which annotation labels to train on:

**Option 1: Auto-discover all classes**
```yaml
classes: null  # Uses all annotation labels found in the dataset
```

**Option 2: Train on specific classes** (default in train.yaml)
```yaml
classes:
  - triangle
  - triangle_inverted
```

**Option 3: Single class detection**
```yaml
classes:
  - person
```

The `classes` configuration:
- Is defined at the top level in `configs/train.yaml` (or `configs/sweep.yaml`)
- Determines `model.num_classes` automatically before model creation
- Filters annotations in all datasets (train, val, test)
- Creates consistent label-to-ID mappings across the pipeline

### Dataset Paths

Configure dataset paths in `configs/dataset/jsonl.yaml`. Each directory must contain a `dataset.jsonl` file and a `data/` subdirectory with images:

```yaml
data:
  train_dir: path/to/my_dataset   # Required: contains dataset.jsonl + data/
  val_dir: null                    # Optional: if null, auto-split from train_dir
```

When `val_dir` is not set, the training data is automatically split into train/val using `training.val_split` (default: 0.2).

**Note:** Test datasets are specified directly via the `dataset_dir` CLI argument to `eval.py`, not in this config file.

### Model Selection

Select a model in `configs/train.yaml`:

```yaml
defaults:
  - model: faster_rcnn  # Options: faster_rcnn, ssdlite
  - dataset: jsonl
  - _self_
```

**Faster R-CNN:**
- Config: `configs/model/faster_rcnn.yaml`
- Backbone: MobileNetV3-Large with FPN
- Input Size: Configurable (default: 800x1333)
- Best for: High accuracy, slower inference

**SSD-Lite:**
- Config: `configs/model/ssdlite.yaml`
- Backbone: MobileNetV3-Large
- Input Size: 320x320
- Best for: Fast inference, mobile deployment

### Regular Training

The training pipeline supports two modes. Regular training uses pre-computed hyperparameters.

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

### Hyperparameter Optimization

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
- Optimize learning rate, weight decay, and momentum
- Save results to Hydra's multirun output directory
- Print the best hyperparameters at the end

**Update optimization results:**
After a successful sweep, copy the best parameters to `configs/optimization_results/` for future use.

### Training Hyperparameters

The training pipeline follows **PyTorch's reference detection training** best practices:

**Optimizer:**
- Type: SGD with momentum (Adam also available via `training.optimizer`)
- Learning Rate: 0.0025 (base, for single GPU)
- Momentum: 0.9
- Weight Decay: 0.0001 (L2 regularization)
- Nesterov: Disabled by default (`training.nesterov: false`)
- Norm Weight Decay: Optional separate weight decay for normalization layers (`training.norm_weight_decay`)

**Learning Rate Schedule:**
- Warmup: Linear warmup for first 1000 iterations (epoch 0 only)
  - Starts at 0.1% of base LR (warmup_factor: 0.001)
  - Linearly increases to base LR
- Schedule: MultiStepLR (default) or CosineAnnealingLR
  - MultiStepLR: Reduces LR by 10x at epochs [16, 22] (for 26-epoch training)
  - Adjustable via `training.lr_steps` and `training.lr_gamma` in config

**Gradient Clipping:**
- Disabled by default (`training.gradient_clip: 0.0`)
- Set to a positive value (e.g., 10.0) to enable

**Loss Function:**
- Uses default torchvision loss weights (no custom weighting)
- For Faster R-CNN: combines RPN + detection head losses
- For SSD-Lite: combines classification + localization losses

**Model EMA:**
- Enabled by default (`training.use_ema: true`)
- Decay rate: 0.9998
- EMA weights are used for evaluation and saved in checkpoints

### Output Directories

The training pipeline creates two different output directories depending on the run mode:

**`outputs/` - Single Training Runs**

Used for regular training (`--config-name=train`):

```
outputs/
└── YYYY-MM-DD/
    └── HH-MM-SS/                    # Timestamp of run
        ├── .hydra/
        │   ├── config.yaml          # Full config used for this run
        │   ├── hydra.yaml           # Hydra settings
        │   └── overrides.yaml       # CLI overrides you provided
        ├── best_model.pth           # Saved checkpoint (best mAP @ IoU=0.50:0.95)
        ├── val_ground_truth_coco.json  # COCO format ground truth for validation
        ├── tensorboard/             # TensorBoard logs
        │   └── events.out.tfevents.*
        └── train.log                # Training logs (loss, metrics, etc.)
```

What you'll find:
- `best_model.pth`: Your trained model checkpoint (saved when validation mAP improves)
- `.hydra/config.yaml`: Exact configuration used (for reproducibility)
- `train.log`: All training output (epochs, losses, COCO metrics)
- `tensorboard/`: Training curves (visualize with `tensorboard --logdir outputs/`)

**`multirun/` - Hyperparameter Sweeps (Optuna)**

Used for hyperparameter optimization (`--config-name=sweep --multirun`):

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
        │   └── ...
        └── optimization_results.yaml # Best hyperparameters found
```

What you'll find:
- Numbered directories (0, 1, 2, ...): Each trial's results
- `.hydra/overrides.yaml`: The hyperparameters Optuna tested for that trial
- `optimization_results.yaml`: Summary with best hyperparameters and their validation mAP
- No `best_model.pth`: Sweeps don't save models (focused on finding best hyperparameters)

**Key Differences:**

| Feature | `outputs/` (Single Run) | `multirun/` (Sweep) |
|---------|------------------------|---------------------|
| **Created by** | `--config-name=train` | `--config-name=sweep --multirun` |
| **Purpose** | Train one model | Find best hyperparameters |
| **Checkpoint** | `best_model.pth` saved (best mAP) | No checkpoints |
| **Training time** | Full epochs (e.g., 26) | Fewer epochs (e.g., 15) |

**Typical Workflow:**

1. Run hyperparameter sweep to find best parameters
   ```bash
   python src/train.py --config-name=sweep --multirun
   ```

2. Copy best parameters to `configs/optimization_results/`

3. Train production model with best hyperparameters
   ```bash
   python src/train.py --config-name=train
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
        ├── Image_tensor([0]).png
        ├── Image_tensor([1]).png
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
1. Converts JSONL ground truth to COCO format (if needed)
2. Scales predictions to original image dimensions
3. Evaluates using pycocotools
4. Saves results and visualizations

## ONNX Conversion

After training and evaluating your model, convert it to ONNX format for production deployment:

```bash
# Convert trained model to ONNX (FasterRCNN only)
# Requires either --dataset-dir or --image-input
bash convert_model.sh outputs/2026-02-02/15-15-47 --dataset-dir triangles_dataset_small

# Convert using a specific image
bash convert_model.sh outputs/2026-02-02/15-15-47 --image-input path/to/image.jpg

# Convert and evaluate the ONNX model
bash convert_model.sh outputs/2026-02-02/15-15-47 --dataset-dir triangles_dataset_small --evaluate-converted-model
```

**What this does:**
1. Finds an image with detections from the dataset (or uses the provided image)
2. Converts PyTorch model to ONNX format with uint8 input support
3. Runs internal consistency tests (PyTorch vs ONNX on the same image)
4. Writes a `labels.txt` file for Viam Vision Service compatibility
5. Saves everything to `outputs/2026-02-02/15-15-47/onnx_model/`

**Output structure:**
```
outputs/2026-02-02/15-15-47/onnx_model/
├── model.onnx                 # ONNX model (ready for deployment)
├── labels.txt                 # Class labels for Viam Vision Service
└── conversion_summary.txt     # Conversion details
```

**Output files:**
- **`model.onnx`** - The exported ONNX model, ready for deployment
- **`labels.txt`** - Class label names, one per line, in the same order as training (line 1 = class index 1, line 2 = class index 2, etc.). Required by the Viam Vision Service to map numeric class indices back to human-readable names.
  ```
  triangle
  triangle_inverted
  ```
- **`conversion_summary.txt`** - Conversion metadata, input/output specs, and usage examples

**ONNX Model Specifications:**
- **Input**: `image` - uint8 tensor `[1, 3, H, W]` with values 0-255
- **Outputs**:
  - `location`: Bounding boxes `[N, 4]` in (x1, y1, x2, y2) format, float32
  - `score`: Confidence scores `[N]`, float32
  - `category`: Class labels `[N]`, float32 (1-indexed)

**Note**: Currently only **FasterRCNN** models are supported for ONNX export.

For detailed usage and deployment examples, see `CONVERT_MODEL_README.md`.

## Viam Vision Service

The project includes a **Viam Vision Service module** (`src/onnx_vision_service/`) that runs object detection using the exported ONNX model on a Viam machine. It uses only `onnxruntime` for inference — no PyTorch needed at runtime.

### Building the Module

```bash
# Build a standalone executable (uses PyInstaller)
bash src/onnx_vision_service/build.sh
```

This creates:
- `dist/onnx-vision-service` — standalone executable
- `dist/onnx-vision-service.tar.gz` — tarball for upload to the Viam registry

Alternatively, install just the vision-service dependencies into an existing environment:
```bash
pip install -e ".[vision-service]"
```

### Machine Configuration

See [Viam Integration Workflow](#viam-integration-workflow) for complete configuration examples (local testing and registry deployment).

### Attributes

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `model_path` | string | yes | Path to the ONNX model file (`model.onnx`) |
| `camera_name` | string | yes | Name of the camera component to get images from |
| `labels_path` | string | yes | Path to `labels.txt` (one class name per line, maps class indices to names) |
| `min_confidence` | float | no | Minimum confidence threshold for detections (default: 0.0) |

### How It Works

1. The service loads the ONNX model and reads `labels.txt` on startup
2. Input size (H, W) is auto-detected from the ONNX model metadata
3. When a detection request comes in, it:
   - Grabs an image from the configured camera
   - Resizes to the model's expected input size
   - Converts to uint8 numpy array `[1, C, H, W]`
   - Runs ONNX inference
   - Scales bounding boxes back to original image coordinates
   - Maps class indices to label names using `labels.txt`
   - Filters by `min_confidence` and returns `Detection` objects

### Supported API Methods

- **`GetDetections`** — Run detection on a provided image
- **`GetDetectionsFromCamera`** — Grab an image from the camera and run detection
- **`CaptureAllFromCamera`** — Capture image and detections in a single call

Classifications and point clouds are not supported.

## Viam Integration Workflow

This section walks through the full end-to-end workflow: from exporting a dataset from Viam Cloud to deploying a trained model on a Viam machine.

### Model training

**1. Export a dataset from Viam Cloud:**

```bash
viam dataset export --destination=./my_dataset --dataset-id=<dataset-id>
```

**2. Train a model on your dataset:**

```bash
python src/train.py --config-name=train dataset.data.train_dir=./my_dataset
```

Edit `configs/train.yaml` to set the classes you want to detect, or leave `classes: null` to auto-discover them from the dataset.

**3. Evaluate the trained model:**

```bash
python src/eval.py dataset_dir=./my_dataset run_dir=outputs/YYYY-MM-DD/HH-MM-SS
```

**4. Convert to ONNX:**

```bash
bash convert_model.sh outputs/YYYY-MM-DD/HH-MM-SS --dataset-dir ./my_dataset
```

This creates `model.onnx` and `labels.txt` in `outputs/YYYY-MM-DD/HH-MM-SS/onnx_model/`.


### Build ONNX vision service

**1. Build the vision service module:**

```bash
bash src/onnx_vision_service/build.sh
```

This produces `dist/onnx-vision-service`.

**2. Configure your Viam machine:**

Add the module, a test camera, and the vision service to your machine's JSON config. For local testing, you can use the `image_file` camera model to point at an image from your dataset:

```json
{
  "modules": [
    {
      "type": "local",
      "name": "my-onnx-module",
      "executable_path": "/path/to/dist/onnx-vision-service"
    }
  ],
  "components": [
    {
      "name": "test-camera",
      "api": "rdk:component:camera",
      "model": "rdk:builtin:image_file",
      "attributes": {
        "color_image_file_path": "/path/to/my_dataset/data/sample_image.jpeg"
      }
    }
  ],
  "services": [
    {
      "name": "my-detector",
      "namespace": "rdk",
      "type": "vision",
      "model": "viam:vision:onnx-detector",
      "attributes": {
        "model_path": "/path/to/outputs/YYYY-MM-DD/HH-MM-SS/onnx_model/model.onnx",
        "camera_name": "test-camera",
        "labels_path": "/path/to/outputs/YYYY-MM-DD/HH-MM-SS/onnx_model/labels.txt",
        "min_confidence": 0.4
      }
    }
  ]
}
```

### Registry Deployment

For production, upload your model to the Viam registry so any machine in your organization can use it without needing local file paths.

**1. Upload the model package:**

```bash
viam packages upload \
    --org-id=<org-id> \
    --name=<package-name> \
    --version=<version> \
    --type=ml_model \
    --upload=<path-to-onnx_model.tar.gz> \
    --model-framework=<framework> \
    --model-type=<model-type>
```

**2. Add the package to your machine config:**

Go to **Data -> Models** in the Viam app, find your uploaded model, and click **Copy package JSON**. Then open your machine's JSON config and paste it into the `"packages": [...]` array.

**3. Reference the package in your vision service config:**

Once the package is added, use the `${packages.ml_model.<package-name>}` variable to reference the model and labels files in your vision service attributes:

```json
{
  "name": "my-detector",
  "namespace": "rdk",
  "type": "vision",
  "model": "viam:vision:onnx-detector",
  "attributes": {
    "model_path": "${packages.ml_model.<package-name>}/model.onnx",
    "camera_name": "my-camera",
    "labels_path": "${packages.ml_model.<package-name>}/labels.txt",
    "min_confidence": 0.4
  }
}
```

This way, the machine automatically downloads the model package and resolves the paths at runtime.

## Project Structure

```
torch-training-script/
├── configs/
│   ├── train.yaml               # Config for regular training
│   ├── sweep.yaml               # Config for hyperparameter optimization
│   ├── eval.yaml                # Config for evaluation
│   ├── dataset/
│   │   └── jsonl.yaml           # Dataset paths and transforms
│   ├── model/
│   │   ├── faster_rcnn.yaml
│   │   └── ssdlite.yaml
│   └── optimization_results/    # Pre-computed hyperparameters
│       ├── faster_rcnn.yaml
│       └── ssdlite.yaml
├── src/
│   ├── train.py                 # Training script
│   ├── eval.py                  # Evaluation script
│   ├── datasets/
│   │   └── viam_dataset.py      # JSONL dataset loader
│   ├── models/
│   │   ├── faster_rcnn_detector.py
│   │   └── ssdlite_detector.py
│   ├── utils/
│   │   ├── transforms.py         # Data augmentation transforms
│   │   ├── coco_converter.py     # JSONL to COCO converter
│   │   ├── coco_eval.py          # COCO evaluation utilities
│   │   ├── freeze.py             # Transfer learning layer freezing
│   │   ├── model_ema.py          # Exponential Moving Average
│   │   ├── seed.py               # Random seed utilities
│   │   └── lr_scheduler.py       # Learning rate scheduler utilities
│   └── onnx_vision_service/      # Viam Vision Service module
│       ├── main.py               # Module entrypoint
│       ├── onnx_vision_service.py # Vision service implementation
│       ├── utils.py              # Image decoding utilities
│       └── build.sh              # Build script (PyInstaller)
├── convert_model.sh              # ONNX conversion script (shell wrapper)
├── convert_to_onnx.py            # ONNX conversion (Python)
├── compare_metrics.py            # Compare PyTorch vs ONNX metrics
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
- ImageNet normalization stats used by default (handled by model's built-in transform)

### Class Configuration Flow

1. `classes` is read from top-level config (`configs/train.yaml` or `configs/sweep.yaml`)
2. If `null`, classes are auto-discovered from the training dataset
3. `model.num_classes` is set to `len(classes)` before model creation
4. All datasets are created with the same `classes` list
5. COCO converter uses the same `classes` for evaluation

### Hydra Configuration Precedence

With `_self_` last in `defaults`, values in the top-level config (e.g., `train.yaml`) override values from sub-configs:
- `model/*.yaml` loaded first
- `dataset/jsonl.yaml` merged second
- Top-level config (`_self_`) merged last (highest precedence)

## Key Dependencies

- **PyTorch** >= 2.0.0 - Deep learning framework
- **torchvision** >= 0.15.0 - Computer vision models and transforms
- **Hydra** >= 1.3.0 - Configuration management
- **pycocotools** >= 2.0.0 - COCO evaluation metrics
- **Pillow** >= 9.0.0 - Image processing
- **numpy** >= 1.21.0 - Numerical operations
- **matplotlib** >= 3.5.0 - Visualization (for evaluation)
- **tqdm** >= 4.64.0 - Progress bars
- **torchinfo** >= 1.8.0 - Model summary
- **tensorboard** >= 2.10.0 - Training visualization
- **optuna** >= 2.10.0, < 3.0.0 - Hyperparameter optimization (optional, install with `[sweep]`)
- **hydra-optuna-sweeper** >= 1.2.0 - Hydra integration for Optuna (optional, install with `[sweep]`)
