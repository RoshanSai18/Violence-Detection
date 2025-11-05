"""
Utility Functions for Violence Detection System
Helper functions for visualization, logging, and data analysis
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import config


def setup_logging(log_file=None):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
    """
    import logging
    
    # Create logger
    logger = logging.getLogger('violence_detection')
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def visualize_video_frames(video_path, num_frames=16, save_path=None):
    """
    Visualize frames from a video in a grid
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to display
        save_path: Path to save visualization
    """
    from data_preprocessing import VideoPreprocessor
    
    preprocessor = VideoPreprocessor(
        sequence_length=num_frames,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH
    )
    
    frames = preprocessor.extract_frames(video_path, method='uniform')
    
    if frames is None:
        print(f"Could not extract frames from {video_path}")
        return
    
    # Create grid
    rows = int(np.sqrt(num_frames))
    cols = int(np.ceil(num_frames / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f'Frames from: {os.path.basename(video_path)}', fontsize=16)
    
    axes = axes.flatten() if num_frames > 1 else [axes]
    
    for i, (frame, ax) in enumerate(zip(frames, axes)):
        ax.imshow(frame)
        ax.set_title(f'Frame {i+1}')
        ax.axis('off')
    
    # Hide extra subplots
    for i in range(len(frames), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Frame visualization saved to: {save_path}")
    
    plt.show()


def visualize_augmentation(video_path, num_samples=8, save_path=None):
    """
    Visualize data augmentation effects
    
    Args:
        video_path: Path to video file
        num_samples: Number of augmented samples to show
        save_path: Path to save visualization
    """
    from data_preprocessing import VideoPreprocessor
    
    preprocessor = VideoPreprocessor()
    
    # Extract one frame
    frames = preprocessor.extract_frames(video_path, method='uniform')
    if frames is None:
        return
    
    original_frame = frames[len(frames) // 2]  # Middle frame
    
    # Create augmented versions
    augmented_samples = []
    for _ in range(num_samples - 1):
        aug_frames = preprocessor.augment_frames(np.array([original_frame]))
        augmented_samples.append(aug_frames[0])
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Data Augmentation Examples', fontsize=16)
    
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(original_frame)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Augmented
    for i, aug_frame in enumerate(augmented_samples):
        axes[i + 1].imshow(aug_frame)
        axes[i + 1].set_title(f'Augmented {i+1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Augmentation visualization saved to: {save_path}")
    
    plt.show()


def analyze_dataset_statistics(data_dir):
    """
    Analyze and visualize dataset statistics
    
    Args:
        data_dir: Path to dataset directory
    """
    from data_preprocessing import load_dataset_paths
    
    video_paths, labels = load_dataset_paths(data_dir)
    
    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    
    # Video duration analysis
    durations = []
    frame_counts = []
    fps_list = []
    
    print("Analyzing videos...")
    for video_path in video_paths[:100]:  # Sample first 100 videos
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        durations.append(duration)
        frame_counts.append(total_frames)
        fps_list.append(fps)
        
        cap.release()
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Dataset Statistics: {os.path.basename(data_dir)}', fontsize=16)
    
    # Class distribution
    axes[0, 0].bar([config.CLASSES[i] for i in unique], counts, color=['green', 'red'])
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_ylabel('Count')
    for i, (cls, cnt) in enumerate(zip(unique, counts)):
        axes[0, 0].text(i, cnt, str(cnt), ha='center', va='bottom')
    
    # Duration distribution
    axes[0, 1].hist(durations, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Video Duration Distribution')
    axes[0, 1].set_xlabel('Duration (seconds)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(durations), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(durations):.2f}s')
    axes[0, 1].legend()
    
    # Frame count distribution
    axes[1, 0].hist(frame_counts, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Frame Count Distribution')
    axes[1, 0].set_xlabel('Number of Frames')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(frame_counts), color='red', linestyle='--',
                       label=f'Mean: {np.mean(frame_counts):.0f}')
    axes[1, 0].legend()
    
    # FPS distribution
    axes[1, 1].hist(fps_list, bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('FPS Distribution')
    axes[1, 1].set_xlabel('FPS')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(fps_list), color='red', linestyle='--',
                       label=f'Mean: {np.mean(fps_list):.1f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    save_path = os.path.join(config.PLOTS_DIR, f'{os.path.basename(data_dir)}_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Dataset statistics saved to: {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset Statistics Summary")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_paths)}")
    print(f"Classes: {', '.join(config.CLASSES)}")
    print(f"Class distribution: {dict(zip([config.CLASSES[i] for i in unique], counts))}")
    print(f"\nVideo characteristics (sample of 100):")
    print(f"  Duration: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s")
    print(f"  Frame count: {np.mean(frame_counts):.0f} ± {np.std(frame_counts):.0f}")
    print(f"  FPS: {np.mean(fps_list):.1f} ± {np.std(fps_list):.1f}")
    print(f"{'='*60}\n")


def create_training_summary(history_path, save_path=None):
    """
    Create training summary visualization
    
    Args:
        history_path: Path to training history JSON
        save_path: Path to save summary
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Summary', fontsize=16)
    
    # Loss
    axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Training')
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[0, 2].plot(epochs, history['precision'], 'b-', label='Training')
    axes[0, 2].plot(epochs, history['val_precision'], 'r-', label='Validation')
    axes[0, 2].set_title('Precision')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 0].plot(epochs, history['recall'], 'b-', label='Training')
    axes[1, 0].plot(epochs, history['val_recall'], 'r-', label='Validation')
    axes[1, 0].set_title('Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 1].plot(epochs, history['auc'], 'b-', label='Training')
    axes[1, 1].plot(epochs, history['val_auc'], 'r-', label='Validation')
    axes[1, 1].set_title('AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Best metrics summary
    best_val_acc = max(history['val_accuracy'])
    best_val_acc_epoch = history['val_accuracy'].index(best_val_acc) + 1
    
    summary_text = f"Best Validation Metrics:\n"
    summary_text += f"Accuracy: {best_val_acc:.4f} (Epoch {best_val_acc_epoch})\n"
    summary_text += f"Precision: {max(history['val_precision']):.4f}\n"
    summary_text += f"Recall: {max(history['val_recall']):.4f}\n"
    summary_text += f"AUC: {max(history['val_auc']):.4f}\n"
    summary_text += f"\nFinal Metrics:\n"
    summary_text += f"Val Accuracy: {history['val_accuracy'][-1]:.4f}\n"
    summary_text += f"Val Loss: {history['val_loss'][-1]:.4f}"
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training summary saved to: {save_path}")
    
    plt.show()


def export_model_to_tflite(model_path, output_path=None, quantize=False):
    """
    Export model to TensorFlow Lite format for deployment
    
    Args:
        model_path: Path to Keras model
        output_path: Path to save TFLite model
        quantize: Whether to apply quantization
    """
    import tensorflow as tf
    from model import AttentionLayer
    
    # Load model
    model = tf.keras.models.load_model(model_path, 
                                      custom_objects={'AttentionLayer': AttentionLayer})
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Applying quantization...")
    
    tflite_model = converter.convert()
    
    # Save
    if output_path is None:
        output_path = model_path.replace('.h5', '.tflite')
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print size comparison
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\nModel export complete:")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  TFLite size: {tflite_size:.2f} MB")
    print(f"  Compression: {(1 - tflite_size/original_size)*100:.1f}%")
    print(f"  Saved to: {output_path}")


def create_readme():
    """Create README file with project documentation"""
    readme_content = """# Violence Detection System

A deep learning system for detecting violence in videos using CNN + BiLSTM architecture with MobileNet backbone.

## 🎯 Features

- **Hybrid Architecture**: MobileNetV2 for spatial features + BiLSTM for temporal modeling
- **Attention Mechanism**: Temporal attention for improved context understanding
- **High Accuracy**: Optimized for ≥90% validation accuracy
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Real-time Inference**: Support for webcam and video file processing
- **Comprehensive Evaluation**: Metrics, confusion matrix, ROC curves, and error analysis

## 📁 Project Structure

```
Violence Detection/
├── config.py                 # Configuration and hyperparameters
├── model.py                  # Model architecture (CNN + BiLSTM)
├── data_preprocessing.py     # Frame extraction and preprocessing
├── data_generator.py         # Custom data generator
├── train.py                  # Training pipeline
├── evaluate.py               # Model evaluation
├── predict.py                # Inference script
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── RWF-2000/                 # Dataset directory
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
│       ├── Fight/
│       └── NonFight/
└── outputs/                  # Output directory
    ├── models/               # Saved models
    ├── logs/                 # Training logs
    └── plots/                # Visualizations
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
python train.py
```

Optional arguments:
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--lr`: Learning rate
- `--resume`: Resume from checkpoint
- `--fine-tune`: Enable fine-tuning mode

### 3. Evaluate Model

```bash
python evaluate.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5
```

### 4. Make Predictions

Single video:
```bash
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --video path/to/video.mp4 --save-output
```

Batch prediction:
```bash
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --dir path/to/videos/ --output-csv results.csv
```

Real-time webcam:
```bash
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --webcam
```

## 📊 Model Architecture

1. **Input**: Video sequence (20 frames × 224×224×3)
2. **CNN Backbone**: MobileNetV2 (pre-trained on ImageNet)
3. **TimeDistributed**: Apply CNN to each frame
4. **BiLSTM**: 256 units with dropout (0.3)
5. **Attention Layer**: Temporal attention mechanism
6. **Dense Layers**: [512, 256] with batch normalization and dropout (0.5)
7. **Output**: Binary classification (sigmoid activation)

## 🎛️ Configuration

Key parameters in `config.py`:

- **Sequence Length**: 20 frames per video
- **Image Size**: 224×224
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Epochs**: 50

## 📈 Training Tips

1. **Initial Training**: Train with frozen MobileNet layers
2. **Fine-tuning**: Unfreeze layers and train with lower learning rate
3. **Data Augmentation**: Enabled by default for better generalization
4. **Class Weights**: Automatically calculated for imbalanced datasets
5. **Callbacks**: Early stopping, ReduceLROnPlateau, ModelCheckpoint

## 📊 Expected Performance

- **Accuracy**: ≥90%
- **Precision**: ≥88%
- **Recall**: ≥87%
- **F1-Score**: ≥87%
- **ROC-AUC**: ≥0.95

## 🔧 Troubleshooting

### GPU Memory Issues
- Reduce batch size in `config.py`
- Enable gradient accumulation
- Use mixed precision training

### Low Accuracy
- Increase training epochs
- Adjust learning rate
- Enable fine-tuning mode
- Check data augmentation

## 📝 Citation

RWF-2000 Dataset:
```
@article{cheng2021rwf,
  title={RWF-2000: An open large scale video database for violence detection},
  author={Cheng, Ming and Cai, Kunjing and Li, Ming},
  journal={arXiv preprint arXiv:1911.05913},
  year={2021}
}
```

## 📄 License

This project is licensed under the MIT License.

## 👥 Author

Deep Learning Engineer - Violence Detection System

---

For questions or issues, please open an issue on GitHub.
"""
    
    readme_path = os.path.join(config.BASE_DIR, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README.md created at: {readme_path}")


def create_requirements_txt():
    """Create requirements.txt file"""
    requirements = """# Core dependencies
tensorflow>=2.10.0
opencv-python>=4.7.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tqdm>=4.64.0

# Optional dependencies
jupyter>=1.0.0
ipykernel>=6.20.0
tensorboard>=2.10.0

# For deployment
flask>=2.2.0
"""
    
    req_path = os.path.join(config.BASE_DIR, 'requirements.txt')
    with open(req_path, 'w') as f:
        f.write(requirements)
    
    print(f"requirements.txt created at: {req_path}")


if __name__ == "__main__":
    print("Utility Functions Module")
    print("="*60)
    
    # Create README and requirements
    create_readme()
    create_requirements_txt()
    
    # Analyze datasets
    print("\nAnalyzing training dataset...")
    analyze_dataset_statistics(config.TRAIN_DIR)
    
    print("\nAnalyzing validation dataset...")
    analyze_dataset_statistics(config.VAL_DIR)
    
    print("\nUtilities initialized successfully!")
