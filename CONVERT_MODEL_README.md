# ONNX Model Conversion Guide

This guide explains how to convert trained PyTorch models to ONNX format for production deployment using `convert_model.sh`.

## Overview

The `convert_model.sh` script automates the conversion of trained FasterRCNN models to ONNX format. It handles:
1. **Model Conversion**: Converts PyTorch checkpoint to ONNX format
2. **Optional Evaluation**: Evaluates the ONNX model on a test dataset
3. **Performance Comparison**: Compares ONNX vs PyTorch model performance

## Prerequisites

- A trained model checkpoint in `outputs/` directory
- The training run directory must contain:
  - `best_model.pth` - The trained checkpoint
  - `.hydra/config.yaml` - Training configuration

## Basic Usage

### Convert Only (No Evaluation)

**With a single image:**
```bash
bash convert_model.sh outputs/2026-02-02/15-15-47 --image-input path/to/image.jpg
```

**With a dataset directory (extracts first image automatically):**
```bash
bash convert_model.sh outputs/2026-02-02/15-15-47 --dataset-dir triangles_dataset_small
```

**With a custom checkpoint:**
```bash
bash convert_model.sh outputs/2026-02-02/15-15-47 \
    --checkpoint-path outputs/2026-02-02/15-15-47/checkpoint_epoch_10.pth \
    --image-input path/to/image.jpg
```

### Convert + Evaluate

To convert and evaluate the ONNX model:
```bash
bash convert_model.sh outputs/2026-02-02/15-15-47 \
    --dataset-dir triangles_dataset_small \
    --evaluate-converted-model
```

This will:
1. Convert the model to ONNX
2. Evaluate the ONNX model on the test dataset
3. Compare ONNX vs PyTorch performance (if PyTorch evaluation exists)

## Arguments

### Required
- `<run_dir>` - Path to training output directory (e.g., `outputs/2026-02-02/15-15-47`)

### Options
- `--device cpu|cuda` - Device for conversion (default: `cpu`)
- `--checkpoint-path PATH` - Path to checkpoint file (default: auto-detects `run_dir/best_model.pth`)
- `--image-input PATH` - Path to a single image file for conversion
- `--dataset-dir DIR` - Directory with `dataset.jsonl` and `data/` folder
  - Used to extract first image if `--image-input` not provided
  - Required if `--evaluate-converted-model` is used
- `--evaluate-converted-model` - Run evaluation on converted ONNX model (requires `--dataset-dir`)

## Output Structure

After conversion, the following structure is created:

```
outputs/2026-02-02/15-15-47/
├── best_model.pth              # Original PyTorch checkpoint
├── .hydra/
│   └── config.yaml             # Training configuration
└── onnx_model/                 # ONNX conversion outputs
    ├── model.onnx              # ONNX model file (ready for deployment)
    ├── conversion_summary.txt  # Conversion details and usage instructions
    └── comparison.json         # PyTorch vs ONNX comparison (if evaluation enabled)
```

If evaluation is enabled, additional results are saved to:
```
outputs/2026-02-02/15-15-47/
└── eval_<dataset_name>_model_onnx/  # ONNX evaluation results
    ├── onnx_metrics.json            # COCO evaluation metrics
    ├── onnx_predictions.json        # Predictions in COCO format
    ├── ground_truth_coco.json       # Ground truth in COCO format
    └── visualizations/               # Sample predictions with bounding boxes
        ├── 0000_detected.png
        └── ...
```

## Evaluation Results Location

Evaluation results are saved in the training run directory with a structured naming convention:

```
run_dir/eval_<dataset_name>_<checkpoint_name>_<format>/
```

**Examples:**
- PyTorch evaluation: `outputs/18-08-48/eval_triangles_dataset_small_best_model_pth/`
- ONNX evaluation: `outputs/18-08-48/eval_triangles_dataset_small_model_onnx/`

This allows multiple evaluations on different datasets or checkpoints without overwriting results.

## Performance Comparison

When `--evaluate-converted-model` is used and both PyTorch and ONNX evaluations exist, the script automatically compares:

- **AP** (mAP @ IoU=0.50:0.95) - Main metric
- **AP50** (mAP @ IoU=0.50) - More lenient
- **AP75** (mAP @ IoU=0.75) - Stricter localization
- **AR100** (Average Recall with 100 detections)

The comparison is saved to `onnx_model/comparison.json` and includes:
- Metric differences between PyTorch and ONNX
- Dataset consistency check
- Acceptability thresholds

**Acceptability:**
- **< 0.001 difference**: Excellent - models produce nearly identical results
- **< 0.01 difference**: Good - models produce very similar results
- **≥ 0.01 difference**: Warning - significant differences detected

## ONNX Model Specifications

### Input
- **Name**: `input`
- **Type**: `float32`
- **Shape**: `[batch_size, 3, height, width]`
- **Range**: `0.0-1.0` (normalized)

### Outputs
- **`location`**: `[batch_size, N, 4]` - Bounding boxes in (x1, y1, x2, y2) format, normalized 0-1
- **`score`**: `[batch_size, N]` - Confidence scores
- **`category`**: `[batch_size, N]` - Class labels (1-indexed)

Where `N` is the number of detections (variable).

## Usage Example

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load and preprocess image (float32, range [0, 1])
img = Image.open('image.jpg').convert('RGB')
img_tensor = np.array(img).astype(np.float32) / 255.0
img_tensor = img_tensor.transpose(2, 0, 1)  # HWC -> CHW
img_batch = img_tensor[np.newaxis, ...]  # Add batch dimension

# Run inference
session = ort.InferenceSession("outputs/2026-02-02/15-15-47/onnx_model/model.onnx")
outputs = session.run(None, {'input': img_batch})
locations, scores, categories = outputs
```

## Workflow Integration

The conversion script integrates with the training workflow:

1. **Train model**: `python src/train.py --config-name=train`
   - Output: `outputs/YYYY-MM-DD/HH-MM-SS/best_model.pth`

2. **Evaluate PyTorch model** (optional):
   ```bash
   python src/eval.py dataset_dir=triangles_dataset_small run_dir=outputs/YYYY-MM-DD/HH-MM-SS
   ```
   - Output: `outputs/YYYY-MM-DD/HH-MM-SS/eval_<dataset>_best_model_pth/`

3. **Convert to ONNX**:
   ```bash
   bash convert_model.sh outputs/YYYY-MM-DD/HH-MM-SS \
       --dataset-dir triangles_dataset_small \
       --evaluate-converted-model
   ```
   - Output: `outputs/YYYY-MM-DD/HH-MM-SS/onnx_model/model.onnx`
   - Also creates ONNX evaluation and comparison if enabled

4. **Deploy**: Use `model.onnx` in production

## Troubleshooting

### Error: "Checkpoint not found"
- Ensure `best_model.pth` exists in the run directory
- Check that the run directory path is correct

### Error: "Hydra config not found"
- The run directory must contain `.hydra/config.yaml` from training
- Re-run training if the config is missing

### Error: "Either --image-input or --dataset-dir must be provided"
- Provide either a single image file or a dataset directory for conversion

### Error: "--evaluate-converted-model requires --dataset-dir"
- Evaluation requires a dataset directory with `dataset.jsonl` and `data/` folder

### ONNX evaluation metrics not found
- Ensure `--evaluate-converted-model` was used
- Check that evaluation completed successfully
- Results are in `run_dir/eval_<dataset_name>_model_onnx/`

### Comparison not available
- Run PyTorch evaluation first: `python src/eval.py dataset_dir=<dir> run_dir=<run_dir>`
- Ensure both PyTorch and ONNX evaluations used the same dataset

## Notes

- Currently only **FasterRCNN** models are supported for ONNX export
- The conversion uses the training configuration to ensure correct model architecture
- ONNX models are optimized for inference and may have slight numerical differences from PyTorch
- For detailed ONNX usage examples, see `ONNX_QUICKSTART.md`
