"""
Evaluation script for YOLOv8 models
Computes metrics on validation or test datasets
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import json


def evaluate(args):
    """Evaluate YOLOv8 model on dataset."""
    print("=" * 60)
    print("YOLOv8 Model Evaluation")
    print("=" * 60)
    
    # Load model
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    
    # Prepare validation arguments
    val_args = {
        'data': args.data_config,
        'split': args.split,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'conf': args.conf,
        'iou': args.iou,
        'max_det': args.max_det,
        'device': args.device,
        'save_json': args.save_json,
        'save_hybrid': args.save_hybrid,
        'plots': args.plots,
        'verbose': True,
    }
    
    if args.project:
        val_args['project'] = args.project
    if args.name:
        val_args['name'] = args.name
    
    print(f"\nEvaluation configuration:")
    print(f"  Data config: {args.data_config}")
    print(f"  Split: {args.split}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    
    # Run validation
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60 + "\n")
    
    metrics = model.val(**val_args)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    results = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.p.mean()),
        'recall': float(metrics.box.r.mean()),
    }
    
    print(f"\nmAP@50: {results['mAP50']:.4f}")
    print(f"mAP@50-95: {results['mAP50-95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    
    # Print per-class metrics if available
    if hasattr(metrics.box, 'map_per_class'):
        print("\nPer-class mAP@50-95:")
        for i, map_val in enumerate(metrics.box.map_per_class):
            print(f"  Class {i}: {map_val:.4f}")
    
    # Save results to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv8 model on dataset'
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pt file)'
    )
    parser.add_argument(
        '--data-config',
        type=str,
        required=True,
        help='Path to data configuration YAML file'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for evaluation'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.6,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--max-det',
        type=int,
        default=300,
        help='Maximum detections per image'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (e.g., 0, cpu)'
    )
    
    # Output settings
    parser.add_argument(
        '--project',
        type=str,
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save results JSON file'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results to COCO JSON format'
    )
    parser.add_argument(
        '--save-hybrid',
        action='store_true',
        help='Save hybrid version of labels'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        default=True,
        help='Generate and save plots'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate(args)


if __name__ == '__main__':
    main()
