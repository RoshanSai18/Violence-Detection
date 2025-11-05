# Violence Detection System - Complete Usage Guide

## 📋 Table of Contents

1. [Installation](#installation)
2. [Initial Setup](#initial-setup)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Inference](#inference)
6. [Advanced Usage](#advanced-usage)
7. [Tips & Best Practices](#tips--best-practices)

---

## 🔧 Installation

### Step 1: Install Python
Ensure Python 3.8+ is installed:
```powershell
python --version
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

**Optional: Create Virtual Environment**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚦 Initial Setup

### Run Setup Script
```powershell
python setup.py
```

This will:
- ✓ Check all dependencies
- ✓ Verify dataset structure
- ✓ Test GPU availability
- ✓ Validate preprocessing
- ✓ Test model building
- ✓ Analyze dataset statistics

---

## 🎓 Training

### Basic Training
```powershell
python train.py
```

### Custom Training Parameters
```powershell
# Train for 30 epochs with batch size 32
python train.py --epochs 30 --batch-size 32

# Custom learning rate
python train.py --lr 0.0001

# All parameters
python train.py --epochs 50 --batch-size 16 --lr 0.0001
```

### Resume Training from Checkpoint
```powershell
python train.py --resume outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5
```

### Fine-tuning (Transfer Learning)
```powershell
# Phase 1: Initial training (frozen MobileNet)
python train.py --epochs 30

# Phase 2: Fine-tuning (unfrozen MobileNet)
python train.py --fine-tune --lr 0.00001 --epochs 20
```

### Monitor Training with TensorBoard
Open a new terminal:
```powershell
tensorboard --logdir=outputs/logs
```
Then open: http://localhost:6006

---

## 📊 Evaluation

### Basic Evaluation
```powershell
python evaluate.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5
```

### Evaluate on Custom Dataset
```powershell
python evaluate.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --data-dir RWF-2000/val
```

### Custom Threshold
```powershell
python evaluate.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --threshold 0.6
```

### Disable Plot Saving
```powershell
python evaluate.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --no-plots
```

**Output Files:**
- `outputs/plots/violence_detection_mobilenetv2_bilstm_seq20_confusion_matrix.png`
- `outputs/plots/violence_detection_mobilenetv2_bilstm_seq20_roc_curve.png`
- `outputs/plots/violence_detection_mobilenetv2_bilstm_seq20_pr_curve.png`
- `outputs/plots/violence_detection_mobilenetv2_bilstm_seq20_evaluation_report.json`

---

## 🔮 Inference

### 1. Single Video Prediction
```powershell
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --video path/to/video.mp4
```

**With Output Video:**
```powershell
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --video path/to/video.mp4 --save-output
```

### 2. Batch Prediction
```powershell
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --dir RWF-2000/val/Fight/ --output-csv results.csv
```

### 3. Real-time Webcam Detection
```powershell
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --webcam
```
Press 'q' to quit.

### 4. Custom Threshold
```powershell
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --video test.mp4 --threshold 0.7
```

---

## 🔬 Advanced Usage

### 1. Modify Configuration

Edit `config.py` to customize:

```python
# Video processing
SEQUENCE_LENGTH = 30  # More frames = better accuracy, slower

# Model architecture
LSTM_UNITS = 512  # More units = more capacity
USE_ATTENTION = True  # Enable/disable attention

# Training
BATCH_SIZE = 8  # Reduce for GPU memory constraints
LEARNING_RATE = 5e-5  # Lower for fine-tuning
EPOCHS = 100  # More epochs for better convergence

# Data augmentation
AUGMENTATION = {
    'horizontal_flip': True,
    'rotation_range': 15,  # Increase for more augmentation
    'brightness_range': [0.7, 1.3],
}
```

### 2. Use Different MobileNet Version

In `config.py`:
```python
MOBILENET_VERSION = 'v3large'  # Options: 'v2', 'v3small', 'v3large'
```

### 3. Enable Mixed Precision Training

In `config.py`:
```python
USE_MIXED_PRECISION = True
```

### 4. Class Weighting for Imbalanced Data

Automatically enabled in `config.py`:
```python
USE_CLASS_WEIGHTS = True
```

### 5. Export to TensorFlow Lite

```python
from utils import export_model_to_tflite

export_model_to_tflite(
    'outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5',
    output_path='model.tflite',
    quantize=True
)
```

### 6. Custom Data Generator

Use balanced sampling for imbalanced datasets:
```python
from data_generator import create_data_generators

train_gen, val_gen = create_data_generators(
    batch_size=16,
    augment_train=True,
    balanced=True  # Enable balanced sampling
)
```

### 7. Optical Flow Features (Experimental)

In `config.py`:
```python
USE_OPTICAL_FLOW = True
```

### 8. Analyze Dataset

```powershell
python utils.py
```

---

## 💡 Tips & Best Practices

### Training Tips

1. **Start Small, Then Scale**
   - Begin with 10-20 epochs to test
   - Monitor validation metrics
   - Scale up if improving

2. **Two-Phase Training**
   - Phase 1: Frozen MobileNet (30 epochs)
   - Phase 2: Fine-tuning (20 epochs, lower LR)

3. **Learning Rate Schedule**
   - Initial: 1e-4 (frozen)
   - Fine-tuning: 1e-5 to 1e-6
   - Use ReduceLROnPlateau (automatic)

4. **Batch Size Selection**
   - GPU 6GB: batch_size=8
   - GPU 8GB: batch_size=16
   - GPU 12GB+: batch_size=32

5. **Monitor Overfitting**
   - Watch val_loss vs train_loss
   - If val_loss increases: reduce epochs or add dropout
   - Enable early stopping (already configured)

### Performance Optimization

1. **GPU Memory Issues**
   ```python
   # In config.py
   BATCH_SIZE = 8
   SEQUENCE_LENGTH = 15
   ```

2. **Faster Training**
   - Enable mixed precision
   - Reduce sequence length
   - Use MobileNetV2 (faster than V3)

3. **Higher Accuracy**
   - Increase sequence length (20-30)
   - More epochs (50-100)
   - Enable fine-tuning
   - Use MobileNetV3Large

### Data Preprocessing

1. **Frame Extraction Methods**
   - `uniform`: Evenly spaced (recommended)
   - `random`: Random frames
   - `consecutive`: Middle frames

2. **Normalization**
   - `tf`: [-1, 1] (recommended for MobileNet)
   - `torch`: [0, 1]
   - `caffe`: ImageNet mean subtraction

### Inference Tips

1. **Threshold Selection**
   - Default: 0.5
   - High precision: 0.6-0.7
   - High recall: 0.3-0.4

2. **Batch Processing**
   - Process multiple videos together
   - Save results to CSV for analysis

3. **Real-time Detection**
   - Reduce sequence length for speed
   - Process every N frames (not all)

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" Error
**Solution:**
```python
# In config.py
BATCH_SIZE = 4  # or even 2
SEQUENCE_LENGTH = 15
```

### Issue: Low Validation Accuracy
**Solutions:**
1. Train for more epochs
2. Enable fine-tuning
3. Increase model capacity (LSTM_UNITS)
4. Check data quality

### Issue: Model Not Learning
**Solutions:**
1. Verify dataset labels
2. Check data normalization
3. Adjust learning rate (try 1e-3 or 1e-5)
4. Ensure class balance

### Issue: Overfitting (train_acc >> val_acc)
**Solutions:**
1. Increase dropout (0.6-0.7)
2. More data augmentation
3. Reduce model capacity
4. Enable regularization

### Issue: Slow Training
**Solutions:**
1. Enable GPU
2. Reduce sequence length
3. Reduce batch size (counter-intuitive but helps with I/O)
4. Use MobileNetV2 instead of V3

---

## 📈 Expected Timeline

### Training Time (RTX 3080)
- Initial training (30 epochs): ~1.5 hours
- Fine-tuning (20 epochs): ~1 hour
- **Total**: ~2.5 hours

### Training Time (CPU)
- Initial training (30 epochs): ~15 hours
- Fine-tuning (20 epochs): ~10 hours
- **Total**: ~25 hours

### Inference Time
- Single video (GPU): ~0.5 seconds
- Single video (CPU): ~2 seconds
- Batch (100 videos, GPU): ~50 seconds
- Real-time webcam: ~20-30 FPS

---

## 📞 Getting Help

1. **Check Logs**
   - Training: `outputs/logs/violence_detection_mobilenetv2_bilstm_seq20_training.csv`
   - TensorBoard: `tensorboard --logdir=outputs/logs`

2. **Verify Setup**
   ```powershell
   python setup.py
   ```

3. **Test Components**
   ```powershell
   # Test preprocessing
   python data_preprocessing.py
   
   # Test model
   python model.py
   
   # Test generator
   python data_generator.py
   ```

---

## 🎯 Quick Command Reference

```powershell
# Setup
python setup.py

# Training
python train.py
python train.py --epochs 50 --batch-size 16
python train.py --fine-tune --lr 0.00001

# Evaluation
python evaluate.py --model outputs/models/<model>.h5

# Prediction
python predict.py --model outputs/models/<model>.h5 --video test.mp4
python predict.py --model outputs/models/<model>.h5 --webcam
python predict.py --model outputs/models/<model>.h5 --dir videos/ --output-csv results.csv

# Monitoring
tensorboard --logdir=outputs/logs

# Analysis
python utils.py
```

---

*For more details, see README.md or check the source code documentation.*
