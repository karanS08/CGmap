"""
Complete Example Workflow for CGmap YOLOv8
Demonstrates end-to-end pipeline from data preparation to inference
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.data_preprocessing import create_sample_data, validate_dataset
from ultralytics import YOLO


def setup_sample_dataset():
    """Step 1: Create sample dataset for testing."""
    print("\n" + "=" * 70)
    print("STEP 1: Creating Sample Dataset")
    print("=" * 70)
    
    output_dir = project_root / 'data' / 'processed'
    
    print(f"\nCreating 50 sample images in {output_dir}...")
    create_sample_data(output_dir, num_samples=50)
    
    print("\nValidating dataset structure...")
    validate_dataset(output_dir, num_classes=2)
    
    print("\n✓ Sample dataset created successfully!")
    return output_dir


def train_model():
    """Step 2: Train YOLOv8 model."""
    print("\n" + "=" * 70)
    print("STEP 2: Training YOLOv8 Model")
    print("=" * 70)
    
    # Initialize model
    print("\nInitializing YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Train
    data_config = project_root / 'configs' / 'data.yaml'
    
    print(f"\nTraining on dataset: {data_config}")
    print("This is a quick demo, training for only 3 epochs...")
    
    results = model.train(
        data=str(data_config),
        epochs=3,
        imgsz=640,
        batch=8,
        device='cpu',  # Change to 0 for GPU
        project='results',
        name='example_train',
        exist_ok=True,
        verbose=True
    )
    
    print("\n✓ Training completed!")
    return model, results


def validate_model(model):
    """Step 3: Validate trained model."""
    print("\n" + "=" * 70)
    print("STEP 3: Validating Model")
    print("=" * 70)
    
    print("\nRunning validation on test set...")
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP@50: {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.p.mean():.4f}")
    print(f"  Recall: {metrics.box.r.mean():.4f}")
    
    print("\n✓ Validation completed!")
    return metrics


def run_inference(model):
    """Step 4: Run inference on sample images."""
    print("\n" + "=" * 70)
    print("STEP 4: Running Inference")
    print("=" * 70)
    
    # Predict on validation set
    val_images = project_root / 'data' / 'processed' / 'val' / 'images'
    
    if not val_images.exists():
        print(f"Warning: Validation images not found at {val_images}")
        return None
    
    print(f"\nRunning predictions on: {val_images}")
    
    results = model.predict(
        source=str(val_images),
        conf=0.25,
        save=True,
        project='results',
        name='example_predict',
        exist_ok=True
    )
    
    print(f"\nProcessed {len(results)} images")
    print(f"Results saved to: results/example_predict")
    
    print("\n✓ Inference completed!")
    return results


def export_model(model):
    """Step 5: Export model for deployment."""
    print("\n" + "=" * 70)
    print("STEP 5: Exporting Model")
    print("=" * 70)
    
    print("\nExporting model to ONNX format...")
    
    try:
        onnx_path = model.export(format='onnx', imgsz=640)
        print(f"\n✓ Model exported to: {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"\nNote: Export failed (this is normal for quick demo): {e}")
        return None


def main():
    """Run complete workflow."""
    print("\n" + "=" * 70)
    print("CGmap YOLOv8 - Complete Example Workflow")
    print("=" * 70)
    print("\nThis script demonstrates the complete pipeline:")
    print("1. Create sample dataset")
    print("2. Train YOLOv8 model")
    print("3. Validate model performance")
    print("4. Run inference on new images")
    print("5. Export model for deployment")
    print("\nNote: This is a quick demo with minimal epochs for testing.")
    
    try:
        # Step 1: Setup dataset
        dataset_dir = setup_sample_dataset()
        
        # Step 2: Train model
        model, train_results = train_model()
        
        # Step 3: Validate
        val_metrics = validate_model(model)
        
        # Step 4: Inference
        pred_results = run_inference(model)
        
        # Step 5: Export
        export_path = export_model(model)
        
        # Summary
        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nSummary:")
        print(f"  ✓ Dataset created: {dataset_dir}")
        print(f"  ✓ Model trained: results/example_train/weights/best.pt")
        print(f"  ✓ Validation mAP@50: {val_metrics.box.map50:.4f}")
        print(f"  ✓ Predictions saved: results/example_predict")
        if export_path:
            print(f"  ✓ Model exported: {export_path}")
        
        print("\nNext steps:")
        print("1. Review training results in: results/example_train/")
        print("2. Check predictions in: results/example_predict/")
        print("3. Use your own data by following QUICKSTART.md")
        print("4. Adjust hyperparameters in configs/model.yaml")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
