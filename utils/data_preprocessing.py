"""
Data preprocessing utilities for UAV imagery
Includes functions for preparing geospatial data for YOLOv8 training
"""

import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm


def create_yolo_dataset_structure(root_dir):
    """
    Create the YOLO dataset directory structure.
    
    Args:
        root_dir: Root directory for the dataset
    """
    root = Path(root_dir)
    
    # Create directory structure
    dirs = [
        'train/images', 'train/labels',
        'val/images', 'val/labels',
        'test/images', 'test/labels'
    ]
    
    for dir_path in dirs:
        (root / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Created YOLO dataset structure in: {root}")
    return root


def split_dataset(image_dir, label_dir, output_dir, 
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  seed=42):
    """
    Split dataset into train/val/test sets.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing YOLO format labels
        output_dir: Output directory for split dataset
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Get all image files
    image_files = sorted(Path(image_dir).glob('*.[jp][pn][g]'))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    # Shuffle
    indices = np.random.permutation(len(image_files))
    
    # Calculate split indices
    n_train = int(len(image_files) * train_ratio)
    n_val = int(len(image_files) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create output structure
    output_root = create_yolo_dataset_structure(output_dir)
    
    # Copy files
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    for split_name, split_indices in splits.items():
        print(f"\nProcessing {split_name} set ({len(split_indices)} images)...")
        
        for idx in tqdm(split_indices):
            img_file = image_files[idx]
            
            # Copy image
            dst_img = output_root / split_name / 'images' / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Copy label if exists
            label_file = Path(label_dir) / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = output_root / split_name / 'labels' / label_file.name
                shutil.copy2(label_file, dst_label)
    
    print(f"\nDataset split completed!")
    print(f"Train: {len(train_indices)} images")
    print(f"Val: {len(val_indices)} images")
    print(f"Test: {len(test_indices)} images")
    
    return output_root


def validate_yolo_labels(label_file, num_classes=2):
    """
    Validate YOLO format label file.
    
    Args:
        label_file: Path to label file
        num_classes: Number of classes in dataset
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                
                # Check format: class x_center y_center width height
                if len(parts) != 5:
                    return False
                
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:]]
                
                # Validate class ID
                if class_id < 0 or class_id >= num_classes:
                    return False
                
                # Validate bbox (should be normalized 0-1)
                if not all(0 <= x <= 1 for x in bbox):
                    return False
        
        return True
    except Exception:
        return False


def validate_dataset(dataset_dir, num_classes=2):
    """
    Validate YOLO dataset structure and labels.
    
    Args:
        dataset_dir: Root directory of YOLO dataset
        num_classes: Number of classes
    """
    dataset_root = Path(dataset_dir)
    splits = ['train', 'val', 'test']
    
    print("Validating dataset...")
    print("=" * 60)
    
    for split in splits:
        image_dir = dataset_root / split / 'images'
        label_dir = dataset_root / split / 'labels'
        
        if not image_dir.exists():
            print(f"Warning: {split} images directory not found")
            continue
        
        images = list(image_dir.glob('*.[jp][pn][g]'))
        labels = list(label_dir.glob('*.txt')) if label_dir.exists() else []
        
        print(f"\n{split.upper()} set:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # Check for missing labels
        missing_labels = []
        invalid_labels = []
        
        for img_file in images:
            label_file = label_dir / f"{img_file.stem}.txt"
            
            if not label_file.exists():
                missing_labels.append(img_file.name)
            elif not validate_yolo_labels(label_file, num_classes):
                invalid_labels.append(label_file.name)
        
        if missing_labels:
            print(f"  Missing labels: {len(missing_labels)}")
            if len(missing_labels) <= 5:
                for name in missing_labels:
                    print(f"    - {name}")
        
        if invalid_labels:
            print(f"  Invalid labels: {len(invalid_labels)}")
            if len(invalid_labels) <= 5:
                for name in invalid_labels:
                    print(f"    - {name}")
    
    print("\n" + "=" * 60)
    print("Validation complete!")


def convert_coords_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box from absolute coordinates to YOLO format.
    
    Args:
        bbox: Bounding box as (x_min, y_min, x_max, y_max)
        img_width: Image width
        img_height: Image height
        
    Returns:
        tuple: (x_center, y_center, width, height) normalized to 0-1
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize to 0-1
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height


def create_sample_data(output_dir, num_samples=10):
    """
    Create sample data for testing (synthetic images).
    
    Args:
        output_dir: Directory to save sample data
        num_samples: Number of sample images to create
    """
    output_root = Path(output_dir)
    
    # Create structure
    for split in ['train', 'val']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Create sample images and labels
    for i in range(num_samples):
        split = 'train' if i < num_samples * 0.8 else 'val'
        
        # Create a simple image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img_path = output_root / split / 'images' / f'sample_{i:03d}.jpg'
        Image.fromarray(img).save(img_path)
        
        # Create a simple label (random bbox)
        label_path = output_root / split / 'labels' / f'sample_{i:03d}.txt'
        class_id = np.random.randint(0, 2)
        x_center = np.random.uniform(0.3, 0.7)
        y_center = np.random.uniform(0.3, 0.7)
        width = np.random.uniform(0.1, 0.3)
        height = np.random.uniform(0.1, 0.3)
        
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created {num_samples} sample images in {output_dir}")
    
    return output_root


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Data preprocessing utilities')
    parser.add_argument('--action', type=str, required=True,
                        choices=['split', 'validate', 'create-sample'],
                        help='Action to perform')
    parser.add_argument('--input-images', type=str, help='Input images directory')
    parser.add_argument('--input-labels', type=str, help='Input labels directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for create-sample')
    
    args = parser.parse_args()
    
    if args.action == 'split':
        if not args.input_images or not args.input_labels or not args.output:
            parser.error('split requires --input-images, --input-labels, and --output')
        split_dataset(args.input_images, args.input_labels, args.output)
    
    elif args.action == 'validate':
        if not args.output:
            parser.error('validate requires --output')
        validate_dataset(args.output)
    
    elif args.action == 'create-sample':
        if not args.output:
            parser.error('create-sample requires --output')
        create_sample_data(args.output, args.num_samples)
