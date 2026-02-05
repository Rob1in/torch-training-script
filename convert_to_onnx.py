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
import torchvision.transforms.functional as TVF
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, 'src')
from models.faster_rcnn_detector import FasterRCNNDetector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class FasterRCNNVisionServiceWrapper(nn.Module):
    """Wrapper to accept uint8 input and ensure all outputs are float32.
    
    Only converts uint8 [0, 255] → float32 [0, 1].
    Normalization is handled by the model's built-in GeneralizedRCNNTransform.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, H, W], uint8 in range [0, 255]
        
        Returns:
            boxes: [N, 4] float32
            labels: [N] float32 (converted from int64)
            scores: [N] float32
        """
        # Convert uint8 [0, 255] to float32 [0, 1]
        # Model's built-in GeneralizedRCNNTransform handles normalization
        x_float = x.float() / 255.0
        
        # Run model
        outputs = self.model([x_float[0]])
        
        # Extract outputs
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        
        # Convert labels to float32 for consistency
        labels_float = labels.float()
        
        return boxes, labels_float, scores


def load_image_as_tensor(image_path: Path, target_size: tuple, as_uint8: bool = True) -> torch.Tensor:
    """Load an image and convert it to a tensor in the format expected by the model.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (height, width) for resizing
        as_uint8: If True, return uint8 tensor [0, 255]. If False, return float32 [0, 1].
        
    Returns:
        Tensor of shape [1, C, H, W] in uint8 range [0, 255] or float32 range [0, 1]
    """
    img = Image.open(image_path).convert('RGB')
    
    # Resize image first (PIL resize is cleaner for uint8)
    input_h, input_w = target_size
    img_resized = img.resize((input_w, input_h), Image.BILINEAR)
    
    if as_uint8:
        # Convert to uint8 tensor
        import numpy as np
        img_np = np.array(img_resized)  # [H, W, C], uint8
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W], uint8
        img_batch = img_tensor.unsqueeze(0)  # [1, C, H, W], uint8
    else:
        # Convert to float32 tensor [0, 1]
        img_tensor = TVF.to_tensor(img_resized)  # [C, H, W], float32, range [0, 1]
        img_batch = img_tensor.unsqueeze(0)  # [1, C, H, W]
    
    return img_batch


def _resolve_image_path(dataset_dir: Path, image_path_str: str) -> Path:
    """Resolve an image path from dataset.jsonl relative to the dataset directory."""
    import os
    data_dir = dataset_dir / "data"
    
    if os.path.isabs(image_path_str):
        return Path(image_path_str)
    elif image_path_str.startswith(dataset_dir.name + '/'):
        path_suffix = image_path_str[len(dataset_dir.name) + 1:]
        return dataset_dir / path_suffix
    else:
        return data_dir / os.path.basename(image_path_str)


def _find_image_with_detections(
    model: nn.Module,
    dataset_dir_str: str,
    target_size: tuple,
    score_threshold: float = 0.3,
    max_images: int = 50,
) -> Path:
    """Find an image from the dataset that produces detections with the PyTorch model.
    
    This ensures the ONNX export uses a meaningful dummy input, allowing us to
    verify the converted model actually produces detections.
    
    Args:
        model: The wrapped PyTorch model (FasterRCNNVisionServiceWrapper).
        dataset_dir_str: Path to dataset directory containing dataset.jsonl.
        target_size: (height, width) to resize images to.
        score_threshold: Minimum score to consider a detection valid.
        max_images: Maximum number of images to try before falling back.
        
    Returns:
        Path to an image that produces at least one detection.
        Falls back to the first image if no detections found.
    """
    import json
    
    dataset_dir = Path(dataset_dir_str)
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"dataset.jsonl not found in {dataset_dir}")
    
    log.info(f"Searching dataset for an image with detections (score > {score_threshold})...")
    
    # Read all image paths from dataset
    image_paths = []
    with open(dataset_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            image_path_str = sample.get('image_path', '')
            if not image_path_str:
                continue
            
            resolved = _resolve_image_path(dataset_dir, image_path_str)
            if resolved.exists():
                # Prefer images that have bounding box annotations (ground truth)
                has_annotations = bool(sample.get('bounding_box_annotations'))
                image_paths.append((resolved, has_annotations))
    
    if not image_paths:
        raise ValueError(f"No valid images found in {dataset_jsonl}")
    
    # Sort: images with annotations first (more likely to have detections)
    image_paths.sort(key=lambda x: x[1], reverse=True)
    
    fallback_path = image_paths[0][0]
    best_path = None
    best_score = 0.0
    best_count = 0
    
    for i, (img_path, has_annot) in enumerate(image_paths[:max_images]):
        try:
            img_tensor = load_image_as_tensor(img_path, target_size, as_uint8=True)
            with torch.no_grad():
                boxes, labels, scores = model(img_tensor)
            
            confident = scores > score_threshold
            num_detections = int(confident.sum())
            top_score = float(scores[0]) if len(scores) > 0 else 0.0
            
            if num_detections > 0 and top_score > best_score:
                best_path = img_path
                best_score = top_score
                best_count = num_detections
                log.info(f"  ✓ Image {i+1}: {img_path.name} → {num_detections} detections (top score: {top_score:.4f})")
                # Good enough — take first image with strong detections
                if top_score > 0.5:
                    break
        except Exception as e:
            log.warning(f"  ✗ Image {i+1}: {img_path.name} → error: {e}")
            continue
    
    if best_path:
        log.info(f"Selected image with {best_count} detections (top score: {best_score:.4f}): {best_path.name}")
        return best_path
    else:
        log.warning(f"No image produced detections with score > {score_threshold} in first {min(max_images, len(image_paths))} images")
        log.warning(f"Falling back to first image: {fallback_path.name}")
        return fallback_path


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
    
    # Load model first (needed to find an image with detections when using dataset-dir)
    log.info(f"Loading model from {args.checkpoint}")
    detector = FasterRCNNDetector(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    detector.load_state_dict(checkpoint['model_state_dict'])
    base_model = detector.model
    
    # Wrap model to accept uint8 input and ensure float32 outputs
    # Normalization is handled by the model's built-in GeneralizedRCNNTransform
    model = FasterRCNNVisionServiceWrapper(base_model)
    log.info("✓ Model loaded and wrapped for uint8 input (normalization handled by model's GeneralizedRCNNTransform)")
    
    # Determine image source
    image_path = None
    if args.image_input:
        image_path = Path(args.image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        log.info(f"Using provided image: {image_path}")
    elif args.dataset_dir:
        image_path = _find_image_with_detections(
            model, args.dataset_dir, (input_h, input_w), score_threshold=0.3
        )
    else:
        raise ValueError("Either --image-input or --dataset-dir must be provided")
    
    # Load image and prepare dummy input (uint8)
    log.info(f"Loading image for dummy input: {image_path}")
    dummy_input = load_image_as_tensor(image_path, (input_h, input_w), as_uint8=True)
    log.info(f"✓ Dummy input: {dummy_input.shape}, dtype: {dummy_input.dtype}")
    
    # Test PyTorch forward pass
    with torch.no_grad():
        boxes, labels, scores = model(dummy_input)
        num_confident = int((scores > 0.3).sum())
        log.info(f"✓ PyTorch test: {len(boxes)} detections ({num_confident} with score > 0.3)")
        log.info(f"  Output types: boxes={boxes.dtype}, labels={labels.dtype}, scores={scores.dtype}")
        if num_confident > 0:
            log.info(f"  Top score: {scores[0]:.4f}")
        pytorch_detections = num_confident
    
    # Export to ONNX
    log.info(f"\nExporting to {args.output}")
    torch.onnx.export(
        model,
        (dummy_input,),
        args.output,
        export_params=True,
        opset_version=11,
        do_constant_folding=False,
        input_names=['image'],
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
    
    # Warn if input name is not "image" (Viam requires this)
    if input_names and input_names[0] != 'image':
        log.warning(f"  ⚠️  Input name is '{input_names[0]}', but Viam requires 'image'")
        log.warning("     You may need to rename it or re-export with input_names=['image']")
    
    # Quick ONNX inference test on the SAME image
    log.info("\nTesting ONNX inference on same image...")
    sess = ort.InferenceSession(args.output)
    outputs = sess.run(None, {'image': dummy_input.numpy()})
    boxes_onnx, labels_onnx, scores_onnx = outputs
    onnx_confident = int((scores_onnx > 0.3).sum())
    log.info(f"✓ ONNX test: {len(boxes_onnx)} detections ({onnx_confident} with score > 0.3)")
    log.info(f"  Output types: boxes={boxes_onnx.dtype}, labels={labels_onnx.dtype}, scores={scores_onnx.dtype}")
    if len(scores_onnx) > 0:
        log.info(f"  Top score: {scores_onnx[0]:.4f}")
    
    # Compare PyTorch vs ONNX detections
    log.info("\n" + ("-" * 70))
    log.info("CONVERSION VALIDATION")
    log.info("-" * 70)
    if pytorch_detections > 0 and onnx_confident > 0:
        log.info(f"✅ Both PyTorch ({pytorch_detections}) and ONNX ({onnx_confident}) produce detections")
    elif pytorch_detections > 0 and onnx_confident == 0:
        log.error(f"❌ PyTorch produced {pytorch_detections} detections but ONNX produced NONE!")
        log.error("   The ONNX conversion may be broken. Check normalization and input preprocessing.")
        sys.exit(1)
    elif pytorch_detections == 0 and onnx_confident == 0:
        log.warning("⚠️  Neither PyTorch nor ONNX produced detections on this image.")
        log.warning("   The model may not be trained well, or the image has no objects.")
    else:
        log.warning(f"⚠️  ONNX ({onnx_confident}) produced detections but PyTorch ({pytorch_detections}) did not. Unexpected.")
    
    log.info("\n" + ("=" * 70))
    log.info("✅ SUCCESS! ONNX model created and verified")
    log.info(f"   ONNX model: {args.output}")
    log.info(f"   Size: {size_mb:.1f} MB")
    log.info("="*70)

if __name__ == '__main__':
    main()

