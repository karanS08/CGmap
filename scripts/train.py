"""
YOLOv8 Training Script for Crop Gap Detection
This script trains a YOLOv8 model on UAV imagery for detecting crop gaps.
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(args):
    """Train YOLOv8 model."""
    print("=" * 60)
    print("YOLOv8 Training for Crop Gap Detection")
    print("=" * 60)
    
    # Load model configuration
    if args.model_config:
        model_config = load_config(args.model_config)
        print(f"\nLoaded model config from: {args.model_config}")
    else:
        model_config = {}
    
    # Initialize model
    model_name = args.model or model_config.get('model', 'yolov8n.pt')
    print(f"\nInitializing model: {model_name}")
    model = YOLO(model_name)
    
    # Prepare training arguments
    train_args = {
        'data': args.data_config,
        'epochs': args.epochs or model_config.get('epochs', 100),
        'batch': args.batch or model_config.get('batch', 16),
        'imgsz': args.imgsz or model_config.get('imgsz', 640),
        'device': args.device or model_config.get('device', 0),
        'project': args.project or model_config.get('project', 'results'),
        'name': args.name or model_config.get('name', 'train'),
        'exist_ok': args.exist_ok,
        'verbose': True,
    }
    
    # Add optional parameters from config
    optional_params = [
        'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay',
        'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
        'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
        'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup',
        'box', 'cls', 'dfl', 'patience', 'workers', 'rect', 'cos_lr',
        'close_mosaic', 'amp', 'fraction', 'freeze', 'save', 'save_period'
    ]
    
    for param in optional_params:
        if param in model_config:
            train_args[param] = model_config[param]
    
    print(f"\nTraining configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    results = model.train(**train_args)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Results saved to: {results.save_dir if hasattr(results, 'save_dir') else train_args['project']}")
    
    # Validate the model
    if args.validate:
        print("\nRunning validation...")
        metrics = model.val()
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model, results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for crop gap detection'
    )
    
    # Required arguments
    parser.add_argument(
        '--data-config',
        type=str,
        default='configs/data.yaml',
        help='Path to data configuration YAML file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/model.yaml',
        help='Path to model configuration YAML file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model to use (e.g., yolov8n.pt, yolov8s.pt). Overrides config.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs. Overrides config.'
    )
    parser.add_argument(
        '--batch',
        type=int,
        help='Batch size. Overrides config.'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        help='Image size. Overrides config.'
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Device to use (e.g., 0, cpu). Overrides config.'
    )
    parser.add_argument(
        '--project',
        type=str,
        help='Project directory. Overrides config.'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name. Overrides config.'
    )
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Allow overwriting existing experiment'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after training'
    )
    
    args = parser.parse_args()
    
    # Verify data config exists
    if not Path(args.data_config).exists():
        raise FileNotFoundError(f"Data config not found: {args.data_config}")
    
    # Train the model
    train(args)


if __name__ == '__main__':
    main()
