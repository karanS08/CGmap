# CGmap YOLOv8 Template - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Data Preparation](#data-preparation)
6. [Training](#training)
7. [Inference](#inference)
8. [Evaluation](#evaluation)
9. [Model Export](#model-export)
10. [Visualization](#visualization)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)

## Overview

CGmap is a YOLOv8-based template for crop gap detection in UAV imagery. It provides:
- Complete training and inference pipeline
- Geospatial data processing utilities
- Configurable hyperparameters
- Model evaluation and visualization tools
- Export capabilities for deployment

### Key Features
- **Framework**: YOLOv8 (Ultralytics)
- **Task**: Object Detection
- **Domain**: Agricultural UAV imagery
- **Classes**: Configurable (default: crop, gap)

## Project Structure

```
CGmap/
├── configs/                    # Configuration files
│   ├── data.yaml              # Dataset configuration
│   └── model.yaml             # Model and training configuration
│
├── data/                      # Data directory
│   ├── raw/                  # Original UAV imagery
│   ├── processed/            # Processed YOLO format data
│   │   ├── train/
│   │   │   ├── images/      # Training images
│   │   │   └── labels/      # Training labels
│   │   ├── val/
│   │   │   ├── images/      # Validation images
│   │   │   └── labels/      # Validation labels
│   │   └── test/
│   │       ├── images/      # Test images
│   │       └── labels/      # Test labels
│   └── annotations/          # Raw annotation files
│
├── models/                    # Saved model weights
│
├── scripts/                   # Main scripts
│   ├── train.py              # Training script
│   ├── predict.py            # Inference script
│   ├── evaluate.py           # Evaluation script
│   ├── export_model.py       # Model export script
│   └── visualize_results.py  # Visualization utilities
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── data_preprocessing.py # Data processing utilities
│   └── geospatial_utils.py   # Geospatial operations
│
├── results/                   # Training and inference results
│   ├── train/                # Training outputs
│   ├── val/                  # Validation outputs
│   └── test/                 # Test outputs
│
├── example_workflow.py        # Complete example workflow
├── requirements.txt           # Python dependencies
├── README.md                  # Main README
├── QUICKSTART.md             # Quick start guide
└── DOCUMENTATION.md          # This file
```

## Installation

### System Requirements
- Python 3.8 or higher
- CUDA 11.0+ (for GPU training, recommended)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ disk space

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/karanS08/CGmap.git
   cd CGmap
   ```

2. **Create virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n cgmap python=3.9
   conda activate cgmap
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "from ultralytics import YOLO; print('✓ Installation successful!')"
   ```

## Configuration

### Data Configuration (configs/data.yaml)

```yaml
# Dataset paths
path: ../data              # Dataset root directory
train: processed/train/images
val: processed/val/images
test: processed/test/images

# Classes
nc: 2                      # Number of classes
names:
  0: crop                  # Class 0
  1: gap                   # Class 1
```

**Key Parameters:**
- `path`: Root directory for dataset (relative to config or absolute)
- `train/val/test`: Paths to image directories
- `nc`: Number of classes
- `names`: Dictionary mapping class IDs to names

### Model Configuration (configs/model.yaml)

```yaml
# Model
model: yolov8n.pt         # Model variant

# Training
epochs: 100               # Number of epochs
batch: 16                 # Batch size
imgsz: 640               # Image size
device: 0                # GPU device ID or 'cpu'

# Optimizer
optimizer: auto          # Optimizer type
lr0: 0.01               # Initial learning rate
lrf: 0.01               # Final learning rate
momentum: 0.937         # SGD momentum
weight_decay: 0.0005    # Weight decay

# Data Augmentation
hsv_h: 0.015            # Hue augmentation
hsv_s: 0.7              # Saturation augmentation
hsv_v: 0.4              # Value augmentation
degrees: 0.0            # Rotation augmentation
translate: 0.1          # Translation augmentation
scale: 0.5              # Scale augmentation
flipud: 0.0             # Vertical flip probability
fliplr: 0.5             # Horizontal flip probability
mosaic: 1.0             # Mosaic augmentation probability

# Loss Weights
box: 7.5                # Box loss weight
cls: 0.5                # Class loss weight
dfl: 1.5                # DFL loss weight
```

## Data Preparation

### YOLO Format

Labels should be in YOLO format (one .txt file per image):
```
class_id x_center y_center width height
```

Where:
- `class_id`: Integer class ID (0, 1, 2, ...)
- `x_center, y_center`: Box center (normalized 0-1)
- `width, height`: Box dimensions (normalized 0-1)

### Preparing Your Data

#### Option 1: Using the Data Preprocessing Utility

```bash
# Split raw data into train/val/test
python utils/data_preprocessing.py \
    --action split \
    --input-images data/raw/images \
    --input-labels data/raw/labels \
    --output data/processed

# Validate dataset
python utils/data_preprocessing.py \
    --action validate \
    --output data/processed

# Create sample data for testing
python utils/data_preprocessing.py \
    --action create-sample \
    --output data/processed \
    --num-samples 50
```

#### Option 2: Manual Organization

1. Place images in respective directories:
   - `data/processed/train/images/`
   - `data/processed/val/images/`
   - `data/processed/test/images/`

2. Place corresponding labels:
   - `data/processed/train/labels/`
   - `data/processed/val/labels/`
   - `data/processed/test/labels/`

### Working with Geospatial Data

For large GeoTIFF orthomosaics:

```bash
# Extract tiles
python utils/geospatial_utils.py \
    --action extract-tiles \
    --input data/raw/orthomosaic.tif \
    --output data/processed/tiles \
    --tile-size 640 \
    --overlap 0.1

# Get GeoTIFF info
python utils/geospatial_utils.py \
    --action info \
    --input data/raw/orthomosaic.tif
```

## Training

### Basic Training

```bash
python scripts/train.py \
    --data-config configs/data.yaml \
    --model-config configs/model.yaml
```

### Advanced Training Options

```bash
python scripts/train.py \
    --data-config configs/data.yaml \
    --model yolov8s.pt \
    --epochs 150 \
    --batch 32 \
    --imgsz 640 \
    --device 0 \
    --project results \
    --name my_experiment \
    --validate
```

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-config` | str | configs/data.yaml | Path to data config |
| `--model-config` | str | configs/model.yaml | Path to model config |
| `--model` | str | - | Model variant (overrides config) |
| `--epochs` | int | - | Number of epochs (overrides config) |
| `--batch` | int | - | Batch size (overrides config) |
| `--imgsz` | int | - | Image size (overrides config) |
| `--device` | str | - | Device (0, 1, cpu) (overrides config) |
| `--project` | str | results | Project directory |
| `--name` | str | train | Experiment name |
| `--exist-ok` | flag | False | Overwrite existing experiment |
| `--validate` | flag | False | Run validation after training |

### Training Output

Results are saved to `results/{name}/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.csv` - Training metrics
- `results.png` - Plots of metrics
- `confusion_matrix.png` - Confusion matrix
- `F1_curve.png`, `PR_curve.png`, etc. - Performance curves

## Inference

### Basic Inference

```bash
python scripts/predict.py \
    --model results/train/weights/best.pt \
    --source data/test/images
```

### Advanced Inference

```bash
python scripts/predict.py \
    --model models/best.pt \
    --source path/to/images \
    --conf 0.3 \
    --iou 0.7 \
    --save-txt \
    --save-conf \
    --project results \
    --name test_predictions
```

### Inference on Video

```bash
python scripts/predict.py \
    --model models/best.pt \
    --source video.mp4 \
    --conf 0.25
```

### Inference Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | required | Path to model weights |
| `--source` | str | required | Image/video/directory path |
| `--conf` | float | 0.25 | Confidence threshold |
| `--iou` | float | 0.7 | IoU threshold for NMS |
| `--imgsz` | int | 640 | Image size |
| `--device` | str | 0 | Device to use |
| `--save` | flag | True | Save predictions |
| `--save-txt` | flag | False | Save as .txt files |
| `--save-conf` | flag | False | Save confidences |
| `--save-crop` | flag | False | Save cropped detections |
| `--classes` | int[] | - | Filter by class IDs |

## Evaluation

### Run Evaluation

```bash
python scripts/evaluate.py \
    --model results/train/weights/best.pt \
    --data-config configs/data.yaml \
    --split val
```

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | required | Path to model |
| `--data-config` | str | required | Data config path |
| `--split` | str | val | Dataset split (train/val/test) |
| `--imgsz` | int | 640 | Image size |
| `--batch` | int | 16 | Batch size |
| `--conf` | float | 0.001 | Confidence threshold |
| `--iou` | float | 0.6 | IoU threshold |
| `--output` | str | - | Save results to JSON |
| `--plots` | flag | True | Generate plots |

### Metrics Explained

- **mAP@50**: Mean Average Precision at IoU=0.5
- **mAP@50-95**: Mean Average Precision averaged over IoU=0.5:0.95
- **Precision**: Ratio of true positives to all predictions
- **Recall**: Ratio of true positives to all ground truths

## Model Export

### Export Formats

```bash
# ONNX (most compatible)
python scripts/export_model.py \
    --model results/train/weights/best.pt \
    --format onnx

# TensorRT (NVIDIA GPUs)
python scripts/export_model.py \
    --model results/train/weights/best.pt \
    --format engine

# CoreML (Apple devices)
python scripts/export_model.py \
    --model results/train/weights/best.pt \
    --format coreml

# TFLite (mobile devices)
python scripts/export_model.py \
    --model results/train/weights/best.pt \
    --format tflite
```

### Export with Optimizations

```bash
# FP16 precision
python scripts/export_model.py \
    --model models/best.pt \
    --format onnx \
    --half

# Dynamic batch size
python scripts/export_model.py \
    --model models/best.pt \
    --format onnx \
    --dynamic

# Simplified ONNX
python scripts/export_model.py \
    --model models/best.pt \
    --format onnx \
    --simplify
```

## Visualization

### Plot Training Results

```bash
python scripts/visualize_results.py \
    --action plot-training \
    --results-dir results/train \
    --save-dir visualizations
```

### Visualize Predictions

```bash
python scripts/visualize_results.py \
    --action visualize-predictions \
    --image-dir results/predict/images \
    --label-dir results/predict/labels \
    --class-names crop gap \
    --num-samples 10 \
    --save-dir visualizations
```

## API Reference

### Python API

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='configs/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Predict
results = model.predict(
    source='image.jpg',
    conf=0.25,
    save=True
)

# Validate
metrics = model.val()

# Export
model.export(format='onnx')
```

### Using Utilities

```python
from utils.data_preprocessing import split_dataset, validate_dataset
from utils.geospatial_utils import extract_tiles, load_geotiff

# Split dataset
split_dataset(
    image_dir='data/raw/images',
    label_dir='data/raw/labels',
    output_dir='data/processed',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
)

# Validate
validate_dataset('data/processed', num_classes=2)

# Extract tiles from GeoTIFF
tiles = extract_tiles(
    'orthomosaic.tif',
    tile_size=640,
    overlap=0.1,
    output_dir='tiles'
)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution:**
- Reduce batch size: `--batch 8` or `--batch 4`
- Reduce image size: `--imgsz 416`
- Use gradient accumulation
- Use a smaller model

#### 2. Low mAP / Poor Performance

**Solution:**
- Train longer (more epochs)
- Use larger model (yolov8m/l/x)
- Check label quality
- Add more training data
- Adjust augmentation parameters
- Try different learning rates

#### 3. Training Loss Not Decreasing

**Solution:**
- Check learning rate (try 0.001 - 0.01)
- Verify labels are correct
- Check data augmentation (too aggressive?)
- Ensure sufficient training data
- Try different optimizer

#### 4. Model Not Finding Objects

**Solution:**
- Lower confidence threshold: `--conf 0.1`
- Check if model is trained on correct classes
- Verify image preprocessing matches training
- Test on training set first

#### 5. Slow Training

**Solution:**
- Use GPU: `--device 0`
- Increase batch size: `--batch 32`
- Reduce workers if CPU-bound
- Use mixed precision: `amp: True`
- Cache images in RAM

### Getting Help

1. Check [YOLOv8 documentation](https://docs.ultralytics.com/)
2. Review [GitHub issues](https://github.com/ultralytics/ultralytics/issues)
3. Open an issue in this repository
4. Consult the [Ultralytics community](https://community.ultralytics.com/)

## Best Practices

### For Training
1. Start with pretrained weights (yolov8n.pt)
2. Use appropriate image size (640 recommended)
3. Monitor validation metrics
4. Save checkpoints regularly
5. Use early stopping (patience parameter)
6. Validate on held-out test set

### For Data
1. Balance class distribution
2. Use sufficient training data (100+ images/class minimum)
3. Ensure label quality
4. Apply appropriate augmentation
5. Split data properly (70/20/10)

### For Deployment
1. Export to appropriate format
2. Optimize for target hardware
3. Test thoroughly before deployment
4. Monitor inference speed
5. Handle edge cases

---

For additional information, see:
- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
