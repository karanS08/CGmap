# CGmap: Crop Gap Mapping with YOLOv8

A Geospatially Aware Deep Learning Framework for Crop Gap Mapping Using UAV Imagery

This repository provides a **YOLOv8-based template** for detecting and mapping crop gaps in UAV (Unmanned Aerial Vehicle) imagery. The framework is designed for agricultural monitoring and precision farming applications.

## 🚀 Features

- **YOLOv8 Integration**: State-of-the-art object detection for crop gap identification
- **Geospatial Support**: Utilities for processing georeferenced UAV imagery
- **Flexible Configuration**: YAML-based configuration for easy experimentation
- **Data Preprocessing**: Tools for preparing and splitting datasets
- **Training & Inference**: Ready-to-use scripts for model training and prediction
- **Visualization**: Built-in visualization tools for results analysis

## 📁 Project Structure

```
CGmap/
├── configs/              # Configuration files
│   ├── data.yaml        # Dataset configuration
│   └── model.yaml       # Model and training configuration
├── data/                # Dataset directory
│   ├── raw/            # Raw UAV imagery
│   ├── processed/      # Processed datasets (train/val/test)
│   └── annotations/    # Annotation files
├── models/             # Saved model weights
├── scripts/            # Training and inference scripts
│   ├── train.py       # Training script
│   └── predict.py     # Inference script
├── utils/              # Utility functions
│   ├── data_preprocessing.py   # Data preprocessing utilities
│   └── geospatial_utils.py     # Geospatial operations
├── results/            # Training and inference results
│   ├── train/         # Training outputs
│   ├── val/           # Validation outputs
│   └── test/          # Test outputs
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/karanS08/CGmap.git
   cd CGmap
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Preparation

### Dataset Format

This project uses the YOLO format for annotations:
- **Images**: Place in `data/processed/{train,val,test}/images/`
- **Labels**: Place in `data/processed/{train,val,test}/labels/`
- **Label format**: `class x_center y_center width height` (normalized 0-1)

### Prepare Your Dataset

1. **Organize raw data**
   ```bash
   # Place your UAV images in data/raw/
   # Place corresponding annotations in data/annotations/
   ```

2. **Split dataset** (using provided utility)
   ```bash
   python utils/data_preprocessing.py \
       --action split \
       --input-images data/raw/images \
       --input-labels data/raw/labels \
       --output data/processed
   ```

3. **Validate dataset**
   ```bash
   python utils/data_preprocessing.py \
       --action validate \
       --output data/processed
   ```

4. **Create sample data** (for testing)
   ```bash
   python utils/data_preprocessing.py \
       --action create-sample \
       --output data/processed \
       --num-samples 20
   ```

### Geospatial Data Processing

For georeferenced imagery (GeoTIFF):

```bash
# Extract tiles from large GeoTIFF
python utils/geospatial_utils.py \
    --action extract-tiles \
    --input data/raw/orthomosaic.tif \
    --output data/processed/tiles \
    --tile-size 640 \
    --overlap 0.1

# Get GeoTIFF information
python utils/geospatial_utils.py \
    --action info \
    --input data/raw/orthomosaic.tif
```

## 🎯 Training

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
    --name crop_gap_v1 \
    --validate
```

### Training Parameters

Edit `configs/model.yaml` to customize:
- **Model size**: `yolov8n.pt` (nano), `yolov8s.pt` (small), `yolov8m.pt` (medium), etc.
- **Epochs**: Number of training epochs
- **Batch size**: Adjust based on GPU memory
- **Image size**: Input image resolution
- **Augmentation**: Data augmentation parameters
- **Optimizer**: Learning rate, momentum, weight decay, etc.

## 🔍 Inference

### Run Predictions

```bash
python scripts/predict.py \
    --model models/best.pt \
    --source data/test/images \
    --conf 0.25 \
    --save
```

### Advanced Inference

```bash
python scripts/predict.py \
    --model models/best.pt \
    --source data/test/images \
    --conf 0.3 \
    --iou 0.7 \
    --imgsz 640 \
    --save-txt \
    --save-conf \
    --project results \
    --name predict_test
```

### Inference on Video

```bash
python scripts/predict.py \
    --model models/best.pt \
    --source path/to/video.mp4 \
    --conf 0.25
```

## 📈 Results

Training and inference results are saved in the `results/` directory:
- **Training metrics**: Loss curves, mAP, precision, recall
- **Visualizations**: Sample predictions, confusion matrix
- **Model weights**: Best and last checkpoints
- **Predictions**: Annotated images with bounding boxes

## ⚙️ Configuration

### Data Configuration (`configs/data.yaml`)

```yaml
path: ../data
train: processed/train/images
val: processed/val/images
test: processed/test/images

nc: 2  # Number of classes
names:
  0: crop
  1: gap
```

### Model Configuration (`configs/model.yaml`)

Key parameters:
- `model`: YOLOv8 variant (n/s/m/l/x)
- `epochs`: Training epochs
- `batch`: Batch size
- `imgsz`: Image size
- `optimizer`: Optimization algorithm
- `lr0`: Initial learning rate
- Augmentation parameters
- Loss weights

## 🎓 For Academic Use

This template is designed for research and academic applications. When using this framework:

1. **Cite relevant papers**: Include citations for YOLOv8/Ultralytics and your research
2. **Document experiments**: Keep track of configurations and results
3. **Version control**: Use Git to manage code and experiments
4. **Reproducibility**: Share configurations and random seeds

### Suggested Citation Format

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 📝 Model Variants

Choose based on your requirements:

| Model | Size (MB) | Speed (ms) | mAP | Use Case |
|-------|-----------|------------|-----|----------|
| YOLOv8n | 6.3 | 0.99 | 37.3 | Real-time, edge devices |
| YOLOv8s | 11.2 | 1.20 | 44.9 | Balanced performance |
| YOLOv8m | 25.9 | 1.83 | 50.2 | Higher accuracy |
| YOLOv8l | 43.7 | 2.39 | 52.9 | Research, offline |
| YOLOv8x | 68.2 | 3.53 | 53.9 | Maximum accuracy |

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in `configs/model.yaml`
   - Use smaller image size (`imgsz: 416` or `imgsz: 320`)
   - Use a smaller model variant (e.g., yolov8n instead of yolov8m)

2. **No images found**
   - Verify dataset paths in `configs/data.yaml`
   - Check that images are in the correct directories
   - Ensure image extensions are supported (.jpg, .jpeg, .png)

3. **Low mAP scores**
   - Increase training epochs
   - Adjust augmentation parameters
   - Verify label quality and format
   - Try different model variants
   - Increase dataset size

4. **Geospatial libraries not installed**
   ```bash
   pip install rasterio geopandas shapely pyproj
   ```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open-source and available under the MIT License.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Rasterio](https://rasterio.readthedocs.io/) - Geospatial raster I/O

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Format Guide](https://docs.ultralytics.com/datasets/detect/)
- [UAV Image Processing Best Practices](https://www.opendronemap.org/)
- [Precision Agriculture Resources](https://precisionag.org/)

---

**Happy Crop Gap Mapping! 🌾🚁**
