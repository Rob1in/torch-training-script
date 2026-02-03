#!/bin/bash
# Production script for converting trained FasterRCNN models to ONNX format
#
# This script is part of the standard training workflow:
#   1. Train model (outputs to run_dir)
#   2. Evaluate model (optional, creates faster_rcnn_metrics.json)
#   3. Convert to ONNX (this script)
#   4. Evaluate ONNX model and compare with PyTorch (optional)
#
# Usage:
#   bash convert_model.sh <run_dir> [options]
#
# Example:
#   # Basic conversion with image
#   bash convert_model.sh outputs/2026-02-02/15-15-47 --image-input path/to/image.jpg
#
#   # Conversion with dataset directory (extracts first image automatically)
#   bash convert_model.sh outputs/2026-02-02/15-15-47 --dataset-dir triangles_dataset_small
#
#   # Conversion + evaluation
#   bash convert_model.sh outputs/2026-02-02/15-15-47 --dataset-dir triangles_dataset_small --evaluate-converted-model

set -e  # Exit on error

# Configuration
DEVICE="cpu"
DATASET_DIR=""
IMAGE_INPUT=""
EVALUATE_CONVERTED_MODEL=false
CHECKPOINT_PATH=""

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_dir> [options]"
    echo ""
    echo "Arguments:"
    echo "  run_dir          Path to training output directory (e.g., outputs/2026-02-02/15-15-47)"
    echo ""
    echo "Options:"
    echo "  --device cpu|cuda              Device for conversion (default: cpu)"
    echo "  --checkpoint-path PATH          Path to checkpoint file (default: run_dir/best_model.pth)"
    echo "  --image-input PATH             Path to a single image file for conversion"
    echo "  --dataset-dir DIR              Directory with dataset.jsonl and data/ folder"
    echo "                                 (used to extract first image if --image-input not provided)"
    echo "  --evaluate-converted-model     Run evaluation on converted ONNX model"
    echo "                                 (requires --dataset-dir)"
    echo ""
    echo "Examples:"
    echo "  # Convert with single image"
    echo "  $0 outputs/2026-02-02/15-15-47 --image-input path/to/image.jpg"
    echo ""
    echo "  # Convert with dataset directory (extracts first image)"
    echo "  $0 outputs/2026-02-02/15-15-47 --dataset-dir triangles_dataset_small"
    echo ""
    echo "  # Convert + evaluate"
    echo "  $0 outputs/2026-02-02/15-15-47 --dataset-dir triangles_dataset_small --evaluate-converted-model"
    exit 1
fi

RUN_DIR="$1"
shift

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --image-input)
            IMAGE_INPUT="$2"
            shift 2
            ;;
        --dataset-dir)
            DATASET_DIR="$2"
            shift 2
            ;;
        --evaluate-converted-model)
            EVALUATE_CONVERTED_MODEL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "ONNX Model Conversion & Evaluation"
echo "=========================================="
echo ""

# Validate run_dir exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: $RUN_DIR"
    echo ""
    echo "Available runs:"
    find outputs -maxdepth 3 -name "best_model.pth" -exec dirname {} \; 2>/dev/null | head -5
    exit 1
fi

# Determine checkpoint path
if [ -z "$CHECKPOINT_PATH" ]; then
    # Auto-detect checkpoint from run_dir
    CHECKPOINT="$RUN_DIR/best_model.pth"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint not found at $CHECKPOINT"
        echo ""
        echo "The run directory should contain best_model.pth, or provide --checkpoint-path"
        exit 1
    fi
    echo "Auto-detected checkpoint: $CHECKPOINT"
else
    # Use provided checkpoint path
    CHECKPOINT="$CHECKPOINT_PATH"
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Error: Checkpoint not found at $CHECKPOINT"
        exit 1
    fi
    # Convert to absolute path if relative
    if [[ "$CHECKPOINT" != /* ]]; then
        CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
    fi
fi

# Check for hydra config
HYDRA_CONFIG="$RUN_DIR/.hydra/config.yaml"
if [ ! -f "$HYDRA_CONFIG" ]; then
    echo "Error: Hydra config not found at $HYDRA_CONFIG"
    echo ""
    echo "The run directory should contain .hydra/config.yaml from training"
    exit 1
fi

# Create output directory
ONNX_DIR="$RUN_DIR/onnx_model"
mkdir -p "$ONNX_DIR"

# Validate that we have either image-input or dataset-dir for conversion
if [ -z "$IMAGE_INPUT" ] && [ -z "$DATASET_DIR" ]; then
    echo "Error: Either --image-input or --dataset-dir must be provided for conversion"
    echo ""
    echo "Use --image-input to specify a single image file, or"
    echo "Use --dataset-dir to specify a directory with dataset.jsonl and data/ folder"
    exit 1
fi

# Validate that evaluation has dataset-dir
if [ "$EVALUATE_CONVERTED_MODEL" = true ] && [ -z "$DATASET_DIR" ]; then
    echo "Error: --evaluate-converted-model requires --dataset-dir"
    exit 1
fi

echo "Configuration:"
echo "  Run directory:     $RUN_DIR"
echo "  Checkpoint:        $CHECKPOINT"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "  (Checkpoint provided via --checkpoint-path)"
else
    echo "  (Checkpoint auto-detected from run_dir)"
fi
echo "  ONNX output dir:   $ONNX_DIR"
echo "  Device:            $DEVICE"
if [ -n "$IMAGE_INPUT" ]; then
    echo "  Image input:       $IMAGE_INPUT"
fi
if [ -n "$DATASET_DIR" ]; then
    echo "  Dataset directory: $DATASET_DIR"
fi
echo "  Evaluate model:    $EVALUATE_CONVERTED_MODEL"
echo ""
echo "=========================================="
echo ""

# Step 1: Convert model to ONNX
echo "Step 1: Converting model to ONNX format..."
echo "----------------------------------------"

# Build convert_to_onnx.py command
CONVERT_CMD="python3 convert_to_onnx.py \
    --checkpoint \"$CHECKPOINT\" \
    --config \"$HYDRA_CONFIG\" \
    --output \"$ONNX_DIR/model.onnx\" \
    --device \"$DEVICE\""

if [ -n "$IMAGE_INPUT" ]; then
    CONVERT_CMD="$CONVERT_CMD --image-input \"$IMAGE_INPUT\""
elif [ -n "$DATASET_DIR" ]; then
    CONVERT_CMD="$CONVERT_CMD --dataset-dir \"$DATASET_DIR\""
fi

eval $CONVERT_CMD

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: ONNX conversion failed!"
    exit 1
fi

echo ""
echo "✓ ONNX model created: $ONNX_DIR/model.onnx"
echo ""

# Step 2: Evaluate ONNX model (optional)
if [ "$EVALUATE_CONVERTED_MODEL" = true ]; then
    echo "=========================================="
    echo "Step 2: Evaluating ONNX model on test dataset..."
    echo "----------------------------------------"
    
    # Convert to absolute paths for eval.py
    # Note: Dataset directory validation is done in eval.py
    ABS_RUN_DIR="$(cd "$RUN_DIR" && pwd)"
    ABS_ONNX_MODEL="$(cd "$(dirname "$ONNX_DIR/model.onnx")" && pwd)/$(basename "$ONNX_DIR/model.onnx")"
    ABS_DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
    
    python3 src/eval.py \
        dataset_dir="$ABS_DATASET_DIR" \
        run_dir="$ABS_RUN_DIR" \
        checkpoint_path="$ABS_ONNX_MODEL"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "Error: ONNX evaluation failed!"
        exit 1
    fi
    
    echo ""
    echo "✓ ONNX evaluation complete"
    echo ""
else
    echo "Skipping evaluation (use --evaluate-converted-model to enable)"
    echo ""
fi

# Step 3: Compare with PyTorch model results (if available)
echo "=========================================="
echo "Step 3: Comparing PyTorch vs ONNX performance..."
echo "----------------------------------------"

# Find PyTorch and ONNX metrics (look in eval output directories)
# Format: run_dir/eval_<dataset_name>_<checkpoint_name>_<format>/<model_type>_metrics.json
if [ "$EVALUATE_CONVERTED_MODEL" = true ]; then
    DATASET_NAME="$(basename "$DATASET_DIR")"
    
    # Extract checkpoint name from checkpoint path (stem without extension)
    # For PyTorch evaluation, use the checkpoint that was used for training
    # Default to "best_model" if auto-detected, otherwise use the stem of the provided path
    if [ -z "$CHECKPOINT_PATH" ]; then
        PYTORCH_CHECKPOINT_NAME="best_model"
    else
        PYTORCH_CHECKPOINT_NAME="$(basename "$CHECKPOINT_PATH" .pth)"
    fi
    PYTORCH_EVAL_DIR="$RUN_DIR/eval_${DATASET_NAME}_${PYTORCH_CHECKPOINT_NAME}_pth"
    PYTORCH_METRICS="$PYTORCH_EVAL_DIR/faster_rcnn_metrics.json"
    
    # ONNX checkpoint name is "model" (stem of model.onnx)
    ONNX_CHECKPOINT_NAME="model"
    ONNX_EVAL_DIR="$RUN_DIR/eval_${DATASET_NAME}_${ONNX_CHECKPOINT_NAME}_onnx"
    ONNX_METRICS="$ONNX_EVAL_DIR/onnx_metrics.json"
fi

if [ "$EVALUATE_CONVERTED_MODEL" = true ] && [ -f "$PYTORCH_METRICS" ] && [ -f "$ONNX_METRICS" ]; then
    python3 compare_metrics.py "$PYTORCH_METRICS" "$ONNX_METRICS" "$ONNX_DIR/comparison.json"
else
    if [ "$EVALUATE_CONVERTED_MODEL" = false ]; then
        echo "Skipping comparison (evaluation not requested)"
    elif [ ! -f "$PYTORCH_METRICS" ]; then
        echo "ℹ️  No PyTorch evaluation results found at: $PYTORCH_METRICS"
        echo "   Run evaluation first:"
        echo "   python src/eval.py dataset_dir=$DATASET_DIR run_dir=$RUN_DIR"
        echo ""
        echo "Skipping comparison (PyTorch metrics not available)"
    elif [ ! -f "$ONNX_METRICS" ]; then
        echo "ℹ️  No ONNX evaluation results found at: $ONNX_METRICS"
        echo ""
        echo "Skipping comparison (ONNX metrics not available)"
    fi
fi

echo ""

# Create summary file
SUMMARY_FILE="$ONNX_DIR/conversion_summary.txt"
cat > "$SUMMARY_FILE" << EOF
ONNX Model Conversion Summary
Generated: $(date)
========================================

Source:
  Run directory: $RUN_DIR
  Checkpoint: $CHECKPOINT
  Config: $HYDRA_CONFIG

ONNX Model:
  Path: $ONNX_DIR/model.onnx
  Device used: $DEVICE

Input Specification:
  Name: input
  Type: float32
  Shape: [batch_size, 3, height, width]
  Range: 0.0-1.0

Output Specification:
  - location: [batch_size, N, 4] (float32, normalized 0-1)
  - score: [batch_size, N] (float32)
  - category: [batch_size, N] (float32)
  Note: N is the number of detections (variable)

Evaluation:
  Internal tests: PASSED (PyTorch vs ONNX consistency on dummy data)
EOF

if [ "$EVALUATE_CONVERTED_MODEL" = true ]; then
    cat >> "$SUMMARY_FILE" << EOF
  Dataset evaluation: COMPLETED
  Metrics: $ONNX_METRICS
  Visualizations: $ONNX_EVAL_DIR/visualizations/
  Comparison: $ONNX_DIR/comparison.json (if PyTorch results available)
EOF
else
    cat >> "$SUMMARY_FILE" << EOF
  Dataset evaluation: SKIPPED (use --evaluate-converted-model to enable)
EOF
fi

cat >> "$SUMMARY_FILE" << EOF

Model Information:
  Architecture: FasterRCNN

Usage:
  import onnxruntime as ort
  import numpy as np
  from PIL import Image
  
  # Load and preprocess image (float32, range [0, 1])
  img = Image.open('image.jpg').convert('RGB')
  img_tensor = np.array(img).astype(np.float32) / 255.0
  img_tensor = img_tensor.transpose(2, 0, 1)  # HWC -> CHW
  img_batch = img_tensor[np.newaxis, ...]  # Add batch dimension
  
  session = ort.InferenceSession("$ONNX_DIR/model.onnx")
  outputs = session.run(None, {'input': img_batch})
  locations, scores, categories = outputs

Deployment:
  The ONNX model is ready for production deployment.
  See ONNX_QUICKSTART.md for integration examples.
EOF

echo "=========================================="
echo "✓ Conversion Complete!"
echo "=========================================="
echo ""
echo "Output directory: $ONNX_DIR"
echo ""
echo "Contents:"
echo "  - model.onnx              : ONNX model file"
echo "  - conversion_summary.txt  : This summary"
if [ "$EVALUATE_CONVERTED_MODEL" = true ]; then
    echo "  Evaluation results saved to: $ONNX_EVAL_DIR"
    echo "    - onnx_metrics.json       : Evaluation metrics"
    echo "    - visualizations/        : Sample predictions"
    if [ -f "$ONNX_DIR/comparison.json" ]; then
        echo "  - comparison.json         : PyTorch vs ONNX comparison"
    fi
fi
echo ""
echo "Next steps:"
if [ "$EVALUATE_CONVERTED_MODEL" = true ]; then
    echo "  1. Review metrics: cat $ONNX_METRICS"
    echo "  2. View visualizations: open $ONNX_EVAL_DIR/visualizations"
    if [ -f "$ONNX_DIR/comparison.json" ]; then
        echo "  3. Check comparison: cat $ONNX_DIR/comparison.json"
        echo "  4. Deploy model: Use $ONNX_DIR/model.onnx in production"
    else
        echo "  3. Deploy model: Use $ONNX_DIR/model.onnx in production"
    fi
else
    echo "  1. Deploy model: Use $ONNX_DIR/model.onnx in production"
    echo "  2. Run evaluation: bash convert_model.sh $RUN_DIR --dataset-dir <dir> --evaluate-converted-model"
fi
echo "  See ONNX_QUICKSTART.md for usage examples"
echo ""
