# Violence Detection System

A state-of-the-art deep learning system for detecting violence in videos using **CNN + BiLSTM** hybrid architecture with **MobileNet** backbone.

## 🎯 Overview

This project implements a high-accuracy violence detection model trained on the **RWF-2000** dataset. The system combines spatial feature extraction via MobileNetV2 with temporal modeling using Bidirectional LSTM and attention mechanisms.

### Key Features

- ✅ **Hybrid Architecture**: MobileNetV2 + BiLSTM + Attention
- ✅ **High Accuracy**: Optimized for ≥90% validation accuracy
- ✅ **Comprehensive Pipeline**: Training, evaluation, and inference
- ✅ **Real-time Detection**: Webcam and video file support
- ✅ **Data Augmentation**: Advanced augmentation strategies
- ✅ **Detailed Metrics**: Confusion matrix, ROC curves, F1-score
- ✅ **Production Ready**: TFLite export support

## 📁 Project Structure

```
Violence Detection/
├── config.py                 # Configuration and hyperparameters
├── model.py                  # CNN + BiLSTM architecture
├── data_preprocessing.py     # Frame extraction & augmentation
├── data_generator.py         # Custom data generator
├── train.py                  # Training pipeline
├── evaluate.py               # Model evaluation
├── predict.py                # Inference script
├── utils.py                  # Utility functions
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── RWF-2000/                 # Dataset (your data)
│   ├── train/
│   │   ├── Fight/
│   │   └── NonFight/
│   └── val/
│       ├── Fight/
│       └── NonFight/
└── outputs/                  # Generated outputs
    ├── models/               # Trained models
    ├── logs/                 # Training logs
    └── plots/                # Visualizations
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Train the Model

```powershell
python train.py
```

**Training Options:**
```powershell
# Custom epochs and batch size
python train.py --epochs 50 --batch-size 16 --lr 0.0001

# Resume from checkpoint
python train.py --resume outputs/models/violence_detection_mobilenetv2_bilstm_seq20_checkpoint.h5

# Fine-tuning mode (unfreeze MobileNet)
python train.py --fine-tune --lr 0.00001
```

### 3. Evaluate the Model

```powershell
python evaluate.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5
```

### 4. Make Predictions

**Single Video:**
```powershell
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --video path/to/video.mp4 --save-output
```

**Batch Prediction:**
```powershell
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --dir RWF-2000/val/Fight/ --output-csv results.csv
```

**Real-time Webcam:**
```powershell
python predict.py --model outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5 --webcam
```

## 🏗️ Model Architecture

```
Input (20 frames × 224×224×3)
    ↓
TimeDistributed(MobileNetV2) - Spatial Feature Extraction
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (256 units) - Temporal Modeling
    ↓
Attention Layer (128 units) - Focus on Important Frames
    ↓
Dense (512) + BatchNorm + Dropout (0.5)
    ↓
Dense (256) + BatchNorm + Dropout (0.5)
    ↓
Output (Sigmoid) - Binary Classification
```

**Total Parameters**: ~3.5M  
**Trainable Parameters**: ~1.2M (initial), ~3.5M (fine-tuned)

## ⚙️ Configuration

Key hyperparameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| SEQUENCE_LENGTH | 20 | Frames per video |
| IMG_HEIGHT/WIDTH | 224 | Input image size |
| BATCH_SIZE | 16 | Training batch size |
| LEARNING_RATE | 1e-4 | Initial learning rate |
| LSTM_UNITS | 256 | BiLSTM units |
| DROPOUT_RATE | 0.5 | Dropout for dense layers |
| EPOCHS | 50 | Training epochs |
| OPTIMIZER | Adam | Optimization algorithm |

## 📊 Expected Performance

On the RWF-2000 validation set:

| Metric | Target | Description |
|--------|--------|-------------|
| **Accuracy** | ≥90% | Overall classification accuracy |
| **Precision** | ≥88% | Positive predictive value |
| **Recall** | ≥87% | True positive rate |
| **F1-Score** | ≥87% | Harmonic mean of precision/recall |
| **ROC-AUC** | ≥0.95 | Area under ROC curve |

## 🎓 Training Strategy

### Phase 1: Initial Training (Frozen MobileNet)
```powershell
python train.py --epochs 30
```
- MobileNet layers frozen
- Learn temporal patterns with BiLSTM
- Fast convergence

### Phase 2: Fine-tuning (Unfrozen MobileNet)
```powershell
python train.py --fine-tune --lr 0.00001 --epochs 20
```
- Unfreeze MobileNet layers (from layer 100)
- Lower learning rate (1e-5)
- Fine-tune spatial features

## 📈 Monitoring Training

### TensorBoard
```powershell
tensorboard --logdir=outputs/logs
```

### Training Logs
Check `outputs/logs/violence_detection_mobilenetv2_bilstm_seq20_training.csv`

## 🔍 Dataset Information

**RWF-2000** (Real World Fight) Dataset:
- **Total Videos**: 2,000
- **Classes**: Fight (Violence), NonFight (Non-violence)
- **Split**: Train (1,600), Validation (400)
- **Source**: Real-world surveillance footage

## 💡 Data Augmentation

Applied during training:
- Horizontal flip (50% probability)
- Rotation (±10 degrees)
- Brightness adjustment (0.8-1.2x)
- Width/height shift (±10%)
- Zoom (±10%)

## 🛠️ Advanced Features

### 1. Class Balancing
Automatically calculated class weights for imbalanced data.

### 2. Callbacks
- **EarlyStopping**: Stop if no improvement
- **ReduceLROnPlateau**: Adaptive learning rate
- **ModelCheckpoint**: Save best model
- **TensorBoard**: Real-time monitoring

### 3. Attention Mechanism
Learns to focus on most violent/non-violent frames in sequence.

### 4. Mixed Precision (Optional)
Enable in `config.py` for faster training on modern GPUs.

## 🐛 Troubleshooting

### GPU Memory Issues
```python
# In config.py
BATCH_SIZE = 8  # Reduce batch size
USE_MIXED_PRECISION = True
```

### Low Accuracy
1. Train for more epochs
2. Adjust learning rate
3. Enable fine-tuning
4. Check data quality

### Slow Training
1. Reduce SEQUENCE_LENGTH (e.g., 15)
2. Use smaller MobileNet variant
3. Enable GPU acceleration

## 📦 Export for Deployment

### TensorFlow Lite
```python
from utils import export_model_to_tflite

export_model_to_tflite(
    'outputs/models/violence_detection_mobilenetv2_bilstm_seq20_best.h5',
    quantize=True
)
```

## 📚 References

1. **RWF-2000 Dataset**:
   ```
   @article{cheng2021rwf,
     title={RWF-2000: An open large scale video database for violence detection},
     author={Cheng, Ming and Cai, Kunjing and Li, Ming},
     journal={arXiv preprint arXiv:1911.05913},
     year={2021}
   }
   ```

2. **MobileNetV2**:
   ```
   @inproceedings{sandler2018mobilenetv2,
     title={Mobilenetv2: Inverted residuals and linear bottlenecks},
     author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
     booktitle={CVPR},
     year={2018}
   }
   ```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review training logs

---

## 🎯 Getting Started Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify dataset structure in `RWF-2000/` folder
- [ ] Run data analysis: `python utils.py`
- [ ] Start training: `python train.py`
- [ ] Monitor with TensorBoard: `tensorboard --logdir=outputs/logs`
- [ ] Evaluate model: `python evaluate.py --model <path>`
- [ ] Test predictions: `python predict.py --model <path> --video <video>`

**Expected Training Time**: 
- GPU (RTX 3080): ~2-3 hours for 50 epochs
- CPU: ~20-30 hours for 50 epochs

**Minimum Requirements**:
- Python 3.8+
- 8GB RAM
- GPU with 6GB+ VRAM (recommended)

---