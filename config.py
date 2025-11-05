"""
Configuration file for Violence Detection System
Contains all hyperparameters, paths, and model settings
"""

import os

# ======================== PATHS ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'RWF-2000')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, MODEL_DIR, LOGS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ======================== DATA PARAMETERS ========================
# Video processing
SEQUENCE_LENGTH = 20  # Number of frames to extract per video (15-30 recommended)
IMG_HEIGHT = 224  # MobileNet input size
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Class mapping
CLASSES = ['NonFight', 'Fight']
NUM_CLASSES = len(CLASSES)

# Data split ratios (if creating custom split from train)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ======================== MODEL PARAMETERS ========================
# MobileNet backbone
MOBILENET_VERSION = 'v2'  # Options: 'v2' or 'v3small' or 'v3large'
MOBILENET_WEIGHTS = 'imagenet'
FREEZE_LAYERS = True  # Freeze MobileNet layers initially
UNFREEZE_FROM_LAYER = 100  # Layer index to start unfreezing for fine-tuning

# BiLSTM parameters
LSTM_UNITS = 256  # Number of LSTM units
LSTM_DROPOUT = 0.3
LSTM_RECURRENT_DROPOUT = 0.2
BIDIRECTIONAL = True  # Use bidirectional LSTM

# Attention mechanism
USE_ATTENTION = True  # Enable temporal attention layer
ATTENTION_UNITS = 128

# Dense layers
DENSE_UNITS = [512, 256]  # Dense layer units after LSTM
DROPOUT_RATE = 0.5  # Dropout rate for dense layers
USE_BATCH_NORM = True  # Batch normalization after dense layers

# Output layer
ACTIVATION = 'sigmoid'  # For binary classification

# ======================== TRAINING PARAMETERS ========================
# Batch size and epochs
BATCH_SIZE = 16  # Adjust based on GPU memory (16-64)
EPOCHS = 50
INITIAL_EPOCH = 0  # For resuming training

# Optimizer
OPTIMIZER = 'adam'  # Options: 'adam', 'sgd', 'rmsprop'
LEARNING_RATE = 1e-4
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999

# SGD parameters (for fine-tuning phase)
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True

# Loss function
LOSS = 'binary_crossentropy'  # For binary classification

# Class weights (for imbalanced data)
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_DICT = {0: 1.0, 1: 1.0}  # Will be calculated automatically

# ======================== DATA AUGMENTATION ========================
# Augmentation for training set
AUGMENTATION = {
    'horizontal_flip': True,
    'rotation_range': 10,  # Degrees
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'brightness_range': [0.8, 1.2],
    'zoom_range': 0.1,
    'fill_mode': 'nearest',
}

# Normalization
NORMALIZE_PIXELS = True  # Normalize to [0, 1]
PREPROCESS_MODE = 'tf'  # 'tf' for [-1, 1], 'caffe' for ImageNet mean subtraction

# ======================== CALLBACKS ========================
# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_MONITOR = 'val_loss'

# Reduce learning rate on plateau
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN_LR = 1e-7
REDUCE_LR_MONITOR = 'val_loss'

# Model checkpoint
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'
CHECKPOINT_SAVE_BEST_ONLY = True
CHECKPOINT_SAVE_WEIGHTS_ONLY = False

# TensorBoard
USE_TENSORBOARD = True

# ======================== EVALUATION METRICS ========================
METRICS = ['accuracy', 'precision', 'recall', 'auc']
COMPUTE_F1_SCORE = True

# Confusion matrix
PLOT_CONFUSION_MATRIX = True
NORMALIZE_CM = True

# ROC curve
PLOT_ROC_CURVE = True

# ======================== MULTI-GPU TRAINING ========================
USE_MULTI_GPU = False
NUM_GPUS = 1

# ======================== MIXED PRECISION ========================
USE_MIXED_PRECISION = False  # Enable for faster training on modern GPUs

# ======================== REPRODUCIBILITY ========================
RANDOM_SEED = 42

# ======================== INFERENCE ========================
CONFIDENCE_THRESHOLD = 0.5  # Classification threshold

# ======================== ADVANCED FEATURES ========================
# Optical flow (optional - computationally expensive)
USE_OPTICAL_FLOW = False

# Frame differencing
USE_FRAME_DIFFERENCE = False

# Test-time augmentation
USE_TTA = False
TTA_STEPS = 5

# Gradient accumulation (for large batch sizes with limited GPU memory)
USE_GRADIENT_ACCUMULATION = False
GRADIENT_ACCUMULATION_STEPS = 2

# ======================== LOGGING ========================
VERBOSE = 1  # Training verbosity
LOG_LEVEL = 'INFO'  # Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# ======================== MODEL NAMING ========================
MODEL_NAME = f'violence_detection_mobilenet{MOBILENET_VERSION}_bilstm_seq{SEQUENCE_LENGTH}'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}_best.h5')
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}_final.h5')
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}_checkpoint.h5')

# ======================== DATASET STATISTICS ========================
# Will be populated during data analysis
TRAIN_SAMPLES = None
VAL_SAMPLES = None
TEST_SAMPLES = None

print(f"Configuration loaded for: {MODEL_NAME}")
print(f"Sequence Length: {SEQUENCE_LENGTH} frames")
print(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LEARNING_RATE}")
