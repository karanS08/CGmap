# Quick Start Guide

Get started with CGmap YOLOv8 template in 5 minutes!

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/karanS08/CGmap.git
cd CGmap

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Sample Data

Create sample data for testing:

```bash
python utils/data_preprocessing.py \
    --action create-sample \
    --output data/processed \
    --num-samples 50
```

This creates 50 synthetic images (40 train, 10 val) to test the pipeline.

## 3. Train a Model

Train on sample data (quick test):

```bash
python scripts/train.py \
    --data-config configs/data.yaml \
    --model yolov8n.pt \
    --epochs 3 \
    --batch 8 \
    --device cpu
```

For real training with GPU:

```bash
python scripts/train.py \
    --data-config configs/data.yaml \
    --model-config configs/model.yaml
```

## 4. Run Inference

After training, test the model:

```bash
python scripts/predict.py \
    --model results/train/weights/best.pt \
    --source data/processed/val/images \
    --conf 0.25
```

## 5. Evaluate Results

Check model performance:

```bash
python scripts/evaluate.py \
    --model results/train/weights/best.pt \
    --data-config configs/data.yaml \
    --split val
```

## Next Steps

### Use Your Own Data

1. **Prepare your images and labels**
   - Images: `.jpg` or `.png` format
   - Labels: YOLO format (`.txt` files)
   - Format: `class x_center y_center width height` (normalized 0-1)

2. **Organize your data**
   ```
   data/raw/
   ├── images/
   │   ├── img001.jpg
   │   ├── img002.jpg
   │   └── ...
   └── labels/
       ├── img001.txt
       ├── img002.txt
       └── ...
   ```

3. **Split the dataset**
   ```bash
   python utils/data_preprocessing.py \
       --action split \
       --input-images data/raw/images \
       --input-labels data/raw/labels \
       --output data/processed
   ```

4. **Update configuration**
   Edit `configs/data.yaml`:
   ```yaml
   path: ../data
   train: processed/train/images
   val: processed/val/images
   
   nc: 2  # Your number of classes
   names:
     0: your_class_0
     1: your_class_1
   ```

5. **Train with your data**
   ```bash
   python scripts/train.py \
       --data-config configs/data.yaml \
       --model-config configs/model.yaml
   ```

### Optimize Performance

1. **Try different model sizes**
   - `yolov8n.pt` - Fastest, least accurate
   - `yolov8s.pt` - Balanced
   - `yolov8m.pt` - More accurate
   - `yolov8l.pt` - High accuracy
   - `yolov8x.pt` - Best accuracy, slowest

2. **Adjust hyperparameters**
   Edit `configs/model.yaml`:
   - Increase `epochs` for better convergence
   - Adjust `lr0` (learning rate) if loss plateaus
   - Modify augmentation parameters for your data

3. **Use GPU acceleration**
   ```bash
   python scripts/train.py \
       --data-config configs/data.yaml \
       --device 0  # Use GPU 0
   ```

### Export for Deployment

```bash
# Export to ONNX
python scripts/export_model.py \
    --model results/train/weights/best.pt \
    --format onnx

# Export to TensorRT (for NVIDIA GPUs)
python scripts/export_model.py \
    --model results/train/weights/best.pt \
    --format engine

# Export to CoreML (for Apple devices)
python scripts/export_model.py \
    --model results/train/weights/best.pt \
    --format coreml
```

## Common Commands Cheat Sheet

```bash
# Training
python scripts/train.py --data-config configs/data.yaml

# Inference on images
python scripts/predict.py --model models/best.pt --source path/to/images

# Inference on video
python scripts/predict.py --model models/best.pt --source video.mp4

# Evaluation
python scripts/evaluate.py --model models/best.pt --data-config configs/data.yaml

# Export model
python scripts/export_model.py --model models/best.pt --format onnx

# Validate dataset
python utils/data_preprocessing.py --action validate --output data/processed
```

## Tips

1. **Start small**: Test with a small model (yolov8n) and few epochs first
2. **Monitor training**: Check loss curves in TensorBoard or training logs
3. **Validate often**: Use `--validate` flag during training
4. **Save checkpoints**: Models are automatically saved in `results/train/weights/`
5. **Experiment tracking**: Use different `--name` for each experiment

## Troubleshooting

**Out of memory error?**
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller images: `--imgsz 416`
- Use smaller model: `yolov8n.pt`

**Low accuracy?**
- Train longer: increase `epochs`
- Use larger model: `yolov8m.pt` or `yolov8l.pt`
- Check data quality and labels
- Adjust augmentation parameters

**Training too slow?**
- Use GPU: `--device 0`
- Increase batch size: `--batch 32`
- Reduce workers if CPU-bound: `workers: 4` in config

## Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review configuration files in `configs/`
- Open an issue on GitHub

---

Happy training! 🚀
