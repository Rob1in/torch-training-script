#!/usr/bin/env python3
"""
ONNX conversion script using a single image as dummy input.
All outputs are float32 for consistency.
"""
import argparse
import logging
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, 'src')
from models.faster_rcnn_detector import FasterRCNNDetector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class FasterRCNNFloat32Wrapper(nn.Module):
    """Wrapper to ensure all outputs are float32."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W], float32 in range [0, 1]
        
        Returns:
            boxes: [N, 4] float32
            labels: [N] float32 (converted from int64)
            scores: [N] float32
        """
        # Run model
        outputs = self.model([x[0]])
        
        # Extract outputs
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        
        # Convert labels to float32 for consistency
        labels_float = labels.float()
        
        return boxes, labels_float, scores


def load_image_as_tensor(image_path: Path, target_size: tuple) -> torch.Tensor:
    """Load an image and convert it to a tensor in the format expected by the model.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (height, width) for resizing
        
    Returns:
        Tensor of shape [1, C, H, W] in float32, range [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = TVF.to_tensor(img)  # [C, H, W], float32, range [0, 1]
    img_batch = img_tensor.unsqueeze(0)  # [1, C, H, W]
    
    # Resize to target size
    input_h, input_w = target_size
    img_resized = F.interpolate(img_batch, size=(input_h, input_w), mode='bilinear', align_corners=False)
    
    return img_resized


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch FasterRCNN model to ONNX format')
    parser.add_argument('--checkpoint', required=True, help='Path to PyTorch checkpoint (.pth file)')
    parser.add_argument('--config', required=True, help='Path to Hydra config file (.yaml)')
    parser.add_argument('--output', required=True, help='Output path for ONNX model (.onnx file)')
    parser.add_argument('--device', default='cpu', help='Device for conversion (cpu or cuda)')
    parser.add_argument('--image-input', help='Path to a single image file to use as dummy input (required if dataset-dir not provided)')
    parser.add_argument('--dataset-dir', help='Directory containing dataset.jsonl and data/ folder (optional, used to extract first image if image-input not provided)')
    args = parser.parse_args()
    
    log.info("="*70)
    log.info("ONNX CONVERSION")
    log.info("="*70)
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Get input size from config
    input_h, input_w = cfg.model.transform.input_size
    log.info(f"Using input size from config: {input_h}x{input_w}")
    
    # Determine image source
    image_path = None
    if args.image_input:
        image_path = Path(args.image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        log.info(f"Using provided image: {image_path}")
    elif args.dataset_dir:
        # Try to extract first image from dataset
        dataset_dir = Path(args.dataset_dir)
        dataset_jsonl = dataset_dir / "dataset.jsonl"
        if not dataset_jsonl.exists():
            raise FileNotFoundError(f"dataset.jsonl not found in {dataset_dir}")
        
        # Read first line to get first image
        import json
        with open(dataset_jsonl, 'r') as f:
            first_line = f.readline()
            if not first_line:
                raise ValueError(f"Empty dataset.jsonl file: {dataset_jsonl}")
            sample = json.loads(first_line)
            image_path_str = sample.get('image_path', '')
            if not image_path_str:
                raise ValueError(f"No image_path in first sample of {dataset_jsonl}")
            
            # Resolve image path (matching ViamDataset logic)
            import os
            data_dir = dataset_dir / "data"
            
            if os.path.isabs(image_path_str):
                # Absolute path - use as-is
                image_path = Path(image_path_str)
            elif image_path_str.startswith(dataset_dir.name + '/'):
                # Path starts with dataset name (e.g., "triangles_dataset_small/data/...")
                # Construct path relative to current working directory or dataset_dir
                # Remove the dataset name prefix and use the rest
                path_suffix = image_path_str[len(dataset_dir.name) + 1:]  # Remove "dataset_name/"
                image_path = dataset_dir / path_suffix
            else:
                # Relative path - assume it's relative to data directory
                # Use basename to avoid path duplication issues
                image_path = data_dir / os.path.basename(image_path_str)
            
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            log.info(f"Extracted first image from dataset: {image_path}")
    else:
        raise ValueError("Either --image-input or --dataset-dir must be provided")
    
    # Load model
    log.info(f"Loading model from {args.checkpoint}")
    detector = FasterRCNNDetector(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    detector.load_state_dict(checkpoint['model_state_dict'])
    base_model = detector.model
    
    # Wrap model to ensure float32 outputs
    model = FasterRCNNFloat32Wrapper(base_model)
    log.info("✓ Model loaded and wrapped for float32 outputs")
    
    # Load image and prepare dummy input
    log.info(f"Loading image for dummy input: {image_path}")
    dummy_input = load_image_as_tensor(image_path, (input_h, input_w))
    log.info(f"✓ Dummy input: {dummy_input.shape}")
    
    # Test PyTorch forward pass
    with torch.no_grad():
        boxes, labels, scores = model(dummy_input)
        log.info(f"✓ PyTorch test: {len(boxes)} detections")
        log.info(f"  Output types: boxes={boxes.dtype}, labels={labels.dtype}, scores={scores.dtype}")
    
    # Export to ONNX
    log.info(f"\nExporting to {args.output}")
    torch.onnx.export(
        model,
        (dummy_input,),
        args.output,
        export_params=True,
        opset_version=11,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['location', 'score', 'category'],
        dynamo=False
    )
    
    size_mb = Path(args.output).stat().st_size / (1024*1024)
    log.info(f"✓ Export complete: {size_mb:.1f} MB")
    
    # Verify ONNX model
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    log.info("✓ ONNX model valid")
    
    # Verify input/output names (for Viam compatibility)
    input_names = [inp.name for inp in onnx_model.graph.input]
    output_names = [out.name for out in onnx_model.graph.output]
    log.info(f"  Input names: {input_names}")
    log.info(f"  Output names: {output_names}")
    
    # Warn if input name is not "input" (Viam requires this)
    if input_names and input_names[0] != 'input':
        log.warning(f"  ⚠️  Input name is '{input_names[0]}', but Viam requires 'input'")
        log.warning("     You may need to rename it or re-export with input_names=['input']")
    
    # Quick ONNX inference test
    log.info("\nTesting ONNX inference...")
    sess = ort.InferenceSession(args.output)
    outputs = sess.run(None, {'input': dummy_input.numpy()})
    boxes_onnx, labels_onnx, scores_onnx = outputs
    log.info(f"✓ ONNX test: {len(boxes_onnx)} detections")
    log.info(f"  Output types: boxes={boxes_onnx.dtype}, labels={labels_onnx.dtype}, scores={scores_onnx.dtype}")
    
    log.info("\n" + ("=" * 70))
    log.info("✅ SUCCESS! ONNX model created and verified")
    log.info(f"   ONNX model: {args.output}")
    log.info(f"   Size: {size_mb:.1f} MB")
    log.info("="*70)

if __name__ == '__main__':
    main()

