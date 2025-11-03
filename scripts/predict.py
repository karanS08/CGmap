"""
YOLOv8 Prediction Script for Crop Gap Detection
This script performs inference on UAV imagery using a trained YOLOv8 model.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np


def predict(args):
    """Run inference on images or video."""
    print("=" * 60)
    print("YOLOv8 Prediction for Crop Gap Detection")
    print("=" * 60)
    
    # Load the model
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    
    # Prepare prediction arguments
    predict_args = {
        'source': args.source,
        'conf': args.conf,
        'iou': args.iou,
        'imgsz': args.imgsz,
        'device': args.device,
        'save': args.save,
        'save_txt': args.save_txt,
        'save_conf': args.save_conf,
        'save_crop': args.save_crop,
        'show_labels': args.show_labels,
        'show_conf': args.show_conf,
        'max_det': args.max_det,
        'vid_stride': args.vid_stride,
        'line_width': args.line_width,
        'visualize': args.visualize,
        'augment': args.augment,
        'agnostic_nms': args.agnostic_nms,
        'retina_masks': args.retina_masks,
    }
    
    if args.project:
        predict_args['project'] = args.project
    if args.name:
        predict_args['name'] = args.name
    if args.classes:
        predict_args['classes'] = args.classes
    
    print(f"\nPrediction configuration:")
    print(f"  Source: {args.source}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")
    
    # Run prediction
    print("\n" + "=" * 60)
    print("Running prediction...")
    print("=" * 60 + "\n")
    
    results = model.predict(**predict_args)
    
    print("\n" + "=" * 60)
    print("Prediction completed!")
    print("=" * 60)
    
    # Print statistics
    if results:
        total_detections = sum(len(r.boxes) for r in results)
        print(f"\nTotal detections: {total_detections}")
        print(f"Images processed: {len(results)}")
        
        if args.save:
            save_dir = results[0].save_dir if hasattr(results[0], 'save_dir') else 'runs/predict'
            print(f"Results saved to: {save_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run YOLOv8 inference for crop gap detection'
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image, directory, video, or camera index (0 for webcam)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0.0-1.0)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU threshold for NMS (0.0-1.0)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (e.g., 0, cpu, 0,1,2,3)'
    )
    parser.add_argument(
        '--max-det',
        type=int,
        default=300,
        help='Maximum detections per image'
    )
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        help='Filter by class (e.g., --classes 0 1)'
    )
    
    # Visualization options
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Save images with predictions'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save results to *.txt files'
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Save confidences in --save-txt files'
    )
    parser.add_argument(
        '--save-crop',
        action='store_true',
        help='Save cropped prediction boxes'
    )
    parser.add_argument(
        '--show-labels',
        action='store_true',
        default=True,
        help='Show class labels on predictions'
    )
    parser.add_argument(
        '--show-conf',
        action='store_true',
        default=True,
        help='Show confidence scores on predictions'
    )
    parser.add_argument(
        '--line-width',
        type=int,
        help='Bounding box line width'
    )
    
    # Output options
    parser.add_argument(
        '--project',
        type=str,
        default='results',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='predict',
        help='Experiment name'
    )
    
    # Advanced options
    parser.add_argument(
        '--vid-stride',
        type=int,
        default=1,
        help='Video frame-rate stride'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize model features'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Apply test-time augmentation'
    )
    parser.add_argument(
        '--agnostic-nms',
        action='store_true',
        help='Class-agnostic NMS'
    )
    parser.add_argument(
        '--retina-masks',
        action='store_true',
        help='Use high-resolution segmentation masks'
    )
    
    args = parser.parse_args()
    
    # Run prediction
    predict(args)


if __name__ == '__main__':
    main()
