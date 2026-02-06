"""
Utility to convert JSONL format to COCO format for evaluation.
"""
import json
import logging
from pathlib import Path
from PIL import Image
from typing import Optional, List

log = logging.getLogger(__name__)


def jsonl_to_coco(
    jsonl_path: str,
    data_dir: str,
    output_path: str,
    classes: Optional[List[str]] = None
) -> str:
    """
    Convert JSONL format to COCO format for evaluation.
    
    Args:
        jsonl_path: Path to input JSONL file
        data_dir: Directory containing images
        output_path: Path to save COCO format JSON file
        classes: Optional list of class names to include. If None, will discover from JSONL.
    
    Returns:
        Path to the created COCO format file
    """
    jsonl_path = Path(jsonl_path)
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    log.info(f"Converting JSONL to COCO format: {jsonl_path} -> {output_path}")
    
    # First pass: discover all unique annotation labels if classes not provided
    if classes is None:
        all_labels = set()
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    bounding_boxes = data.get('bounding_box_annotations', [])
                    for bbox in bounding_boxes:
                        label = bbox.get('annotation_label')
                        if label:
                            all_labels.add(label)
                except json.JSONDecodeError:
                    continue
        classes = sorted(all_labels)
    
    # Create categories (1-based IDs, 0 is background)
    categories = [
        {
            "id": idx + 1,
            "name": class_name,
            "supercategory": "none"
        }
        for idx, class_name in enumerate(classes)
    ]
    
    # Create label to category_id mapping
    label_to_id = {class_name: idx + 1 for idx, class_name in enumerate(classes)}
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    log.info(f"Creating COCO format with {len(classes)} categories: {classes}")
    
    image_id_to_idx = {}
    annotation_id = 1
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
            
            image_path = data.get('image_path')
            if not image_path:
                log.warning(f"Skipping entry on line {line_num}: missing 'image_path'")
                continue
            
            # Resolve image path (same logic as ViamDataset)
            import os
            if os.path.isabs(image_path):
                full_path = Path(image_path)
            elif image_path.startswith(data_dir.name + '/'):
                # image_path is like "dataset_dir_name/data/file.jpg"
                # Resolve relative to data_dir's parent so it becomes an absolute path
                full_path = data_dir.parent / image_path
            else:
                full_path = data_dir / os.path.basename(image_path)
            
            if not full_path.exists():
                log.warning(f"Image not found: {full_path}, skipping")
                continue
            
            # Get image dimensions
            try:
                with Image.open(full_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                log.warning(f"Could not open image {full_path}: {e}, skipping")
                continue
            
            # Use sequential image_id starting from 0 to match dataset indices
            # Dataset uses idx (0-based) as image_id, so we match that here
            image_id = len(coco_data["images"])
            
            # Add image entry
            # Use relative path if possible, otherwise just filename
            file_name = Path(image_path).name
            
            coco_data["images"].append({
                "id": image_id,
                "file_name": file_name,
                "width": img_width,
                "height": img_height
            })
            
            image_id_to_idx[image_id] = len(coco_data["images"]) - 1
            
            # Filter annotations to only include specified classes
            bounding_boxes = data.get('bounding_box_annotations', [])
            if classes is not None:
                filtered_boxes = [
                    bbox for bbox in bounding_boxes
                    if bbox.get('annotation_label') in label_to_id
                ]
            else:
                filtered_boxes = bounding_boxes
            
            # Add annotations
            for bbox in filtered_boxes:
                x_min_norm = bbox.get('x_min_normalized')
                y_min_norm = bbox.get('y_min_normalized')
                x_max_norm = bbox.get('x_max_normalized')
                y_max_norm = bbox.get('y_max_normalized')
                
                if None in [x_min_norm, y_min_norm, x_max_norm, y_max_norm]:
                    continue
                
                # Convert normalized to pixel coordinates
                x_min = x_min_norm * img_width
                y_min = y_min_norm * img_height
                x_max = x_max_norm * img_width
                y_max = y_max_norm * img_height
                
                # Ensure valid box dimensions
                if x_max <= x_min or y_max <= y_min:
                    continue
                
                # Convert to COCO format [x, y, width, height]
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                
                # Map annotation label to category_id
                label = bbox.get('annotation_label')
                if label and label in label_to_id:
                    category_id = label_to_id[label]
                else:
                    log.warning(f"Unknown annotation label '{label}', skipping")
                    continue
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "iscrowd": 0
                })
                annotation_id += 1
    
    # Save COCO format file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    log.info(f"Converted {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to COCO format")
    return str(output_path)

