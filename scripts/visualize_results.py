"""
Visualization utilities for YOLOv8 results
Plot training metrics, predictions, and analysis
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json


def plot_training_results(results_dir, save_dir=None):
    """
    Plot training results from YOLOv8 training.
    
    Args:
        results_dir: Directory containing training results
        save_dir: Directory to save plots (optional)
    """
    results_path = Path(results_dir)
    
    # Look for results.csv or results.png
    csv_file = results_path / 'results.csv'
    
    if not csv_file.exists():
        print(f"Results file not found: {csv_file}")
        return
    
    # Read CSV
    import pandas as pd
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv8 Training Results', fontsize=16)
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax.plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o')
    if 'train/cls_loss' in df.columns:
        ax.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', marker='s')
    if 'train/dfl_loss' in df.columns:
        ax.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: mAP
    ax = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', marker='o', linewidth=2)
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', marker='s', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Mean Average Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Precision and Recall
    ax = axes[1, 0]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', marker='o')
    if 'metrics/recall(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate
    ax = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax.plot(df['epoch'], df['lr/pg0'], label='LR', marker='o', color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        output_file = save_path / 'training_plots.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(image_dir, label_dir, class_names, num_samples=5, save_dir=None):
    """
    Visualize predictions with bounding boxes.
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing prediction labels
        class_names: List of class names
        num_samples: Number of images to visualize
        save_dir: Directory to save visualizations
    """
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    
    image_files = sorted(list(image_path.glob('*.[jp][pn][g]')))[:num_samples]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Color map for classes
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
    ]
    
    for img_file in image_files:
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Read labels
        label_file = label_path / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = f.readlines()
            
            # Draw bounding boxes
            for label in labels:
                parts = label.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    
                    # Convert to pixel coordinates
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    
                    # Draw rectangle
                    color = colors[class_id % len(colors)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    class_name = class_names[class_id] if class_id < len(class_names) else f'Class {class_id}'
                    label_text = f"{class_name} {conf:.2f}"
                    
                    # Background for text
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
                    cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display or save
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            output_file = save_path / f"vis_{img_file.name}"
            cv2.imwrite(str(output_file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved: {output_file}")
        else:
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f"Predictions: {img_file.name}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize YOLOv8 results')
    
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['plot-training', 'visualize-predictions'],
        help='Action to perform'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory containing training results'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images'
    )
    parser.add_argument(
        '--label-dir',
        type=str,
        help='Directory containing prediction labels'
    )
    parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=['crop', 'gap'],
        help='List of class names'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    if args.action == 'plot-training':
        if not args.results_dir:
            parser.error('plot-training requires --results-dir')
        plot_training_results(args.results_dir, args.save_dir)
    
    elif args.action == 'visualize-predictions':
        if not args.image_dir or not args.label_dir:
            parser.error('visualize-predictions requires --image-dir and --label-dir')
        visualize_predictions(args.image_dir, args.label_dir, args.class_names, args.num_samples, args.save_dir)


if __name__ == '__main__':
    main()
