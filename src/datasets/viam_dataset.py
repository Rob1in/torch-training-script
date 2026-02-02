import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class ViamDataset(Dataset):
    """Dataset for object detection using JSONL format with normalized bounding boxes."""
    
    def __init__(
        self, 
        jsonl_path: str, 
        data_dir: str, 
        classes: Optional[List[str]] = None,
        transform=None
    ):
        """
        Args:
            jsonl_path: Path to dataset.jsonl file
            data_dir: Directory containing images (or base directory if image_path is absolute)
            classes: List of annotation labels to include (e.g., ['triangle', 'person']). 
                    If None, includes all annotations found in the JSONL file.
            transform: Optional transform to be applied to images (handled by GPUCollate)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        # Determine which classes to use
        if classes is None:
            # Auto-discover: collect all unique annotation labels
            all_labels = set()
            log.info(f"Auto-discovering classes from {jsonl_path}")
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
            log.info(f"Auto-discovered {len(classes)} classes: {classes}")
        else:
            # Use specified classes
            classes = sorted(classes)  # Sort for consistent ordering
            log.info(f"Using {len(classes)} specified classes: {classes}")
        
        # Create label to category_id mapping (1-based, 0 is background)
        self.label_to_id = {label: idx + 1 for idx, label in enumerate(classes)}
        self.id_to_label = {idx + 1: label for idx, label in enumerate(classes)}
        self.num_classes = len(classes)
        
        log.info(f"Label to ID mapping: {self.label_to_id}")
        
        # Second pass: load samples, filtering to only include specified classes
        log.info(f"Loading dataset from {jsonl_path}")
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
                
                # Filter annotations to only include specified classes
                bounding_boxes = data.get('bounding_box_annotations', [])
                filtered_boxes = [
                    bbox for bbox in bounding_boxes
                    if bbox.get('annotation_label') in self.label_to_id
                ]
                
                # Only include images with at least one annotation from specified classes
                if filtered_boxes:
                    self.samples.append({
                        'image_path': image_path,
                        'boxes': filtered_boxes
                    })
        
        log.info(f"Loaded {len(self.samples)} images with annotations from specified classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image_path']
        
        # Resolve image path
        # If image_path is absolute or starts with dataset prefix, use as-is
        # Otherwise, join with data_dir
        if os.path.isabs(image_path) or image_path.startswith(self.data_dir.name + '/'):
            full_path = Path(image_path)
        else:
            full_path = self.data_dir / os.path.basename(image_path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        # Load image (always RGB)
        image = Image.open(full_path).convert('RGB')
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Extract bounding boxes and convert from normalized to pixel coordinates
        boxes = []
        labels = []
        for bbox in sample['boxes']:
            x_min_norm = bbox.get('x_min_normalized')
            y_min_norm = bbox.get('y_min_normalized')
            x_max_norm = bbox.get('x_max_normalized')
            y_max_norm = bbox.get('y_max_normalized')
            
            if None in [x_min_norm, y_min_norm, x_max_norm, y_max_norm]:
                log.warning(f"Skipping invalid bbox in {image_path}: missing normalized coordinates")
                continue
            
            x_min = x_min_norm * img_width
            y_min = y_min_norm * img_height
            x_max = x_max_norm * img_width
            y_max = y_max_norm * img_height
            
            # Ensure valid box dimensions
            if x_max <= x_min or y_max <= y_min:
                log.warning(f"Skipping invalid bbox in {image_path}: x_max <= x_min or y_max <= y_min")
                continue
            
            # Format: [x_min, y_min, x_max, y_max]
            boxes.append([x_min, y_min, x_max, y_max])
            
            # Map annotation label to category_id (1-based, 0 is background)
            label = bbox.get('annotation_label')
            if label and label in self.label_to_id:
                category_id = self.label_to_id[label]
            else:
                log.warning(f"Unknown annotation label '{label}' in {image_path}, skipping")
                continue
            
            labels.append(category_id)
        
        # Ensure we have at least one box
        if len(boxes) == 0:
            raise ValueError(f"No valid boxes found for image {image_path}")
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        # Convert PIL image to tensor
        image = F.to_tensor(image)
        
        # Create target dictionary (image_id will be added by transforms if needed)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),  # Use dataset index as image_id
            'orig_size': torch.tensor([img_height, img_width])  # Original image dimensions [H, W]
        }
        
        # Note: Transforms are applied by GPUCollate, not here
        return image, target


