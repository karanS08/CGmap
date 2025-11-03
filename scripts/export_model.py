"""
Export YOLOv8 model to different formats for deployment
Supports: ONNX, TensorRT, CoreML, TFLite, and more
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_model(args):
    """Export YOLOv8 model to specified format."""
    print("=" * 60)
    print("YOLOv8 Model Export")
    print("=" * 60)
    
    # Load model
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    
    # Prepare export arguments
    export_args = {
        'format': args.format,
        'imgsz': args.imgsz,
        'half': args.half,
        'int8': args.int8,
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'opset': args.opset,
        'workspace': args.workspace,
        'nms': args.nms,
    }
    
    # Remove None values
    export_args = {k: v for k, v in export_args.items() if v is not None}
    
    print(f"\nExport configuration:")
    print(f"  Format: {args.format}")
    print(f"  Image size: {args.imgsz}")
    if args.half:
        print(f"  Precision: FP16")
    elif args.int8:
        print(f"  Precision: INT8")
    else:
        print(f"  Precision: FP32")
    
    # Export the model
    print("\n" + "=" * 60)
    print("Exporting model...")
    print("=" * 60 + "\n")
    
    exported_model = model.export(**export_args)
    
    print("\n" + "=" * 60)
    print("Export completed!")
    print("=" * 60)
    print(f"Exported model: {exported_model}")
    
    return exported_model


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLOv8 model to various formats'
    )
    
    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pt file)'
    )
    
    # Export format
    parser.add_argument(
        '--format',
        type=str,
        default='onnx',
        choices=[
            'torchscript', 'onnx', 'openvino', 'engine', 'coreml',
            'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle'
        ],
        help='Export format'
    )
    
    # Export settings
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size for export'
    )
    parser.add_argument(
        '--half',
        action='store_true',
        help='Export with FP16 precision'
    )
    parser.add_argument(
        '--int8',
        action='store_true',
        help='Export with INT8 precision (TensorFlow only)'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Enable dynamic axes (ONNX/TF/TensorRT)'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify ONNX model'
    )
    parser.add_argument(
        '--opset',
        type=int,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=4,
        help='TensorRT workspace size (GB)'
    )
    parser.add_argument(
        '--nms',
        action='store_true',
        help='Add NMS to CoreML/TFLite models'
    )
    
    args = parser.parse_args()
    
    # Export the model
    export_model(args)


if __name__ == '__main__':
    main()
