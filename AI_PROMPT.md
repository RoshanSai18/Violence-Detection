# 🤖 AI Prompt for Violence Detection System

## Complete Prompt to Generate This Project

---

### **Main Prompt:**

```
I need you to build a state-of-the-art multi-modal violence detection system for video surveillance. 
The system should achieve 92-97% accuracy by combining three different modalities: RGB frames, human 
pose estimation, and facial emotion recognition.

Requirements:

1. ARCHITECTURE:
   - Use a three-branch neural network architecture
   - Branch 1: RGB features using MobileNetV2 (pre-trained on ImageNet)
   - Branch 2: Pose features using MediaPipe Pose (33 body landmarks + engineered features)
   - Branch 3: Emotion features using DeepFace (7 emotions + variance)
   - Combine all branches using adaptive fusion with learned weights
   - Add Bidirectional LSTM layers for temporal modeling
   - Implement custom attention mechanism to focus on discriminative frames

2. FEATURE ENGINEERING:
   - For pose: Extract 120-dimensional features including:
     * 33 body landmarks (x, y, visibility) = 99 features
     * 6 joint angles (elbows, knees, shoulders)
     * Body metrics: hand distance, foot elevation, torso bend, head offset
     * Normalized versions for scale invariance
   - For emotion: Extract 8-dimensional features including:
     * 7 emotion probabilities (angry, disgust, fear, happy, sad, surprise, neutral)
     * Temporal variance as an indicator of emotional instability
   - For RGB: 1280-dimensional MobileNetV2 features per frame

3. TEMPORAL MODELING:
   - Sample 20 frames uniformly from each video
   - Use Bidirectional LSTM with 256 units for RGB branch
   - Use Bidirectional LSTM with 128 units for pose branch
   - Use Bidirectional LSTM with 64 units for emotion branch
   - Apply attention mechanism to identify key frames

4. TRAINING STRATEGY:
   - Implement feature caching to speed up training (preprocess once, train many times)
   - Use binary cross-entropy loss with class weighting
   - Adam optimizer with learning rate 1e-4
   - Batch size: 32
   - Epochs: 30 with early stopping (patience=8)
   - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
   - Regularization: Dropout (0.3-0.5), Batch Normalization, Recurrent Dropout (0.2)

5. EVALUATION:
   - Generate confusion matrix and ROC curve
   - Report accuracy, precision, recall, F1-score, and AUC
   - Visualize training history (loss, accuracy, precision, recall)
   - Analyze attention weights and feature importance
   - Compare emotion patterns between violent and non-violent videos

6. DEPLOYMENT:
   - Create a real-time inference script for webcam detection
   - Support 30-40 FPS on GPU
   - Display predictions with confidence scores
   - Provide options for video file processing and edge deployment

7. CODE REQUIREMENTS:
   - Use TensorFlow 2.x / Keras
   - Support both Google Colab and Kaggle notebooks
   - Write clean, modular, well-documented code
   - Include comprehensive error handling
   - Provide detailed comments and docstrings

8. DELIVERABLES:
   - Complete Jupyter notebook for training (Colab + Kaggle versions)
   - Real-time inference script for webcam
   - Model evaluation and visualization scripts
   - Project documentation with architecture diagrams
   - README with setup instructions and usage examples
   - Requirements.txt with all dependencies

The system should be production-ready, interpretable, and achieve state-of-the-art performance 
on video violence detection benchmarks.
```

---

## 🎯 Alternative Prompts for Specific Components

### **Prompt 1: Architecture Design**

```
Design a multi-modal deep learning architecture for violence detection that fuses three modalities:

1. RGB Branch:
   - Input: (20, 224, 224, 3) video frames
   - Use TimeDistributed MobileNetV2 (frozen, ImageNet weights)
   - Apply BiLSTM with 256 units
   - Add custom attention layer (128 units)
   - Output: 256-dim features

2. Pose Branch:
   - Input: (20, 120) pose features
   - Apply BatchNorm, Dense(128), BiLSTM(128)
   - Add attention layer (64 units)
   - Output: 128-dim features

3. Emotion Branch:
   - Input: (20, 8) emotion features
   - Apply BatchNorm, Dense(64), BiLSTM(64)
   - Use GlobalAveragePooling
   - Output: 64-dim features

4. Fusion:
   - Project all branches to 256-dim
   - Stack features
   - Learn fusion weights via softmax
   - Weighted sum for final representation

5. Classification:
   - Dense(512) → Dense(256) → Dense(1, sigmoid)
   - Add BatchNorm and Dropout(0.5) after each layer

Implement this in TensorFlow/Keras with proper regularization and training callbacks.
```

---

### **Prompt 2: Feature Extraction**

```
Implement a comprehensive feature extraction pipeline for violence detection:

1. RGB Features:
   - Load video and sample 20 frames uniformly
   - Resize to 224×224
   - Pass through MobileNetV2 (pre-trained)
   - Extract 1280-dim features per frame
   - Normalize to [0, 1]

2. Pose Features (120-dim per frame):
   - Use MediaPipe Pose to detect 33 body landmarks
   - Calculate 6 joint angles: left/right elbows, knees, shoulders
   - Compute body metrics:
     * Hand-to-hand distance (aggression indicator)
     * Foot elevation difference (kicking detection)
     * Torso bend (left and right)
     * Head offset from body center
   - Add normalized versions
   - Handle missing detections with zero-padding

3. Emotion Features (8-dim per frame):
   - Use DeepFace to detect faces and analyze emotions
   - Extract probabilities for 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
   - Calculate temporal variance across frames
   - Normalize to [0, 1]

4. Caching:
   - Save all features to compressed .npz files
   - Implement cache checking to avoid reprocessing
   - Support both training and validation sets

Create a class that handles all three modalities with robust error handling and progress tracking.
```

---

### **Prompt 3: Training Pipeline**

```
Create a complete training pipeline for a multi-modal violence detection model:

1. Data Loading:
   - Load cached features from .npz files
   - Support train/validation splits
   - Implement efficient batching

2. Model Compilation:
   - Use Adam optimizer (lr=1e-4)
   - Binary cross-entropy loss
   - Metrics: accuracy, precision, recall, AUC

3. Callbacks:
   - ModelCheckpoint: Save best model based on val_accuracy
   - EarlyStopping: Stop if no improvement for 8 epochs
   - ReduceLROnPlateau: Reduce LR by 0.5 if no improvement for 4 epochs
   - TensorBoard: Log training metrics

4. Class Balancing:
   - Compute class weights for Fight vs Non-Fight
   - Apply during training

5. Training:
   - Batch size: 32
   - Epochs: 30
   - Validation during training
   - Save training history to JSON

6. Evaluation:
   - Generate predictions on validation set
   - Create confusion matrix
   - Plot ROC curve and calculate AUC
   - Generate classification report
   - Visualize training curves

7. Model Saving:
   - Save best model (.h5)
   - Save final model (.h5)
   - Save training history (.json)
   - Create visualization plots (.png)

Implement in TensorFlow/Keras with comprehensive logging and error handling.
```

---

### **Prompt 4: Real-Time Inference**

```
Build a real-time violence detection system for live webcam feeds:

1. Model Loading:
   - Load trained multi-modal model (.h5)
   - Load feature extractors (MobileNetV2, MediaPipe, DeepFace)
   - Compile with custom attention layer

2. Video Processing:
   - Capture video from webcam using OpenCV
   - Maintain a rolling buffer of 20 frames
   - Process frames at 30 FPS

3. Feature Extraction:
   - Extract RGB features (MobileNetV2)
   - Extract pose features (MediaPipe)
   - Extract emotion features (DeepFace)
   - Handle missing detections gracefully

4. Prediction:
   - Pass features through model
   - Get violence probability (0-1)
   - Apply threshold (0.5) for classification

5. Visualization:
   - Display video feed with overlay
   - Show "FIGHT" (red) or "NON-FIGHT" (green) label
   - Display confidence percentage
   - Draw pose skeleton on frame
   - Show emotion probabilities

6. Performance:
   - Target 30-40 FPS on GPU
   - Optimize with frame skipping if needed
   - Monitor and display FPS

7. Controls:
   - 'q' to quit
   - 's' to save current frame
   - 'p' to pause/resume

Implement with threading for smooth real-time processing and professional UI.
```

---

### **Prompt 5: Kaggle/Colab Notebook**

```
Create a complete Jupyter notebook for training a violence detection model on Kaggle/Colab:

1. Setup:
   - Check GPU availability
   - Install required packages (opencv, mediapipe, deepface)
   - Auto-detect dataset location (/kaggle/input/ or /content/)

2. Dataset Handling:
   - For Kaggle: Auto-find added dataset
   - For Colab: Download via Kaggle API
   - Verify dataset structure (train/val, Fight/NonFight)
   - Print dataset statistics

3. Preprocessing:
   - Extract multi-modal features (RGB + Pose + Emotion)
   - Show progress bars (tqdm)
   - Cache features to disk
   - Estimate time remaining

4. Model Building:
   - Define custom attention layer
   - Build three-branch architecture
   - Display model summary
   - Show architecture diagram

5. Training:
   - Configure callbacks
   - Train with cached features
   - Display real-time training progress
   - Show validation metrics

6. Evaluation:
   - Plot training curves
   - Generate confusion matrix
   - Plot ROC curve
   - Show classification report

7. Model Download:
   - Package model and results
   - Create zip file
   - Auto-download to PC

8. Documentation:
   - Add markdown cells with explanations
   - Include time estimates
   - Provide troubleshooting tips
   - Add usage instructions

Make it beginner-friendly with clear instructions and extensive comments. Total runtime: 4-6 hours.
```

---

## 🔧 Prompt Templates for Debugging

### **Debug Prompt 1: Dependency Issues**

```
I'm getting dependency conflicts when installing packages for violence detection:
- mediapipe requires numpy < 2
- opencv requires numpy >= 2
- protobuf version mismatches

Create a requirements.txt with compatible versions and provide installation commands that avoid conflicts.
Also provide a pip install command sequence that works on Google Colab.
```

---

### **Debug Prompt 2: Model Training Issues**

```
My multi-modal violence detection model is experiencing:
- Overfitting after 5 epochs
- Validation accuracy stuck at 85%
- Training is very slow (10 min/epoch)

Suggest:
1. Regularization techniques to prevent overfitting
2. Data augmentation strategies
3. Hyperparameter tuning recommendations
4. Ways to speed up training

Provide specific code changes with explanations.
```

---

### **Debug Prompt 3: Feature Extraction Errors**

```
My pose/emotion feature extraction is failing with:
- MediaPipe not detecting poses in some frames
- DeepFace crashing on videos without faces
- Memory errors when processing large datasets

Implement robust error handling that:
1. Uses zero-padding for missing detections
2. Catches exceptions and continues processing
3. Logs failed videos for review
4. Manages memory efficiently

Provide updated feature extraction code.
```

---

## 🎨 Prompt for Creating Visualizations

```
Create comprehensive visualizations for a violence detection model:

1. Training History:
   - 2×2 subplot: accuracy, loss, precision, recall
   - Show train vs validation curves
   - Highlight best epoch
   - Professional styling with seaborn

2. Confusion Matrix:
   - Heatmap with annotations
   - Show Fight vs Non-Fight counts
   - Calculate and display percentages
   - Add color gradient

3. ROC Curve:
   - Plot True Positive Rate vs False Positive Rate
   - Calculate and display AUC score
   - Add diagonal reference line
   - Professional styling

4. Attention Weights:
   - Visualize which frames model focuses on
   - Show attention scores per frame
   - Overlay on sample videos
   - Compare Fight vs Non-Fight attention patterns

5. Emotion Analysis:
   - Bar charts comparing emotion distributions
   - Fight videos vs Non-Fight videos
   - Highlight dominant emotions
   - Show variance indicators

6. Feature Importance:
   - Analyze fusion weights per sample
   - Show RGB/Pose/Emotion contributions
   - Violin plots for distribution
   - Identify patterns

Save all visualizations as high-resolution PNG files and return figure objects for display.
```

---

## 📚 Prompt for Documentation

```
Create comprehensive documentation for a violence detection system:

1. README.md:
   - Project overview and key features
   - System requirements (hardware, software)
   - Installation instructions (step-by-step)
   - Quick start guide with examples
   - Usage instructions for training and inference
   - Troubleshooting common issues
   - Contributing guidelines
   - License information

2. ARCHITECTURE.md:
   - Detailed architecture explanation
   - ASCII/diagram of model structure
   - Algorithm descriptions
   - Mathematical formulas
   - Design decisions and rationale

3. API_REFERENCE.md:
   - Function signatures and parameters
   - Class documentation
   - Usage examples for each function
   - Return value descriptions
   - Exception handling

4. TUTORIAL.md:
   - Step-by-step tutorial for beginners
   - Code walkthroughs with explanations
   - Common use cases
   - Tips and best practices

5. PERFORMANCE.md:
   - Benchmark results
   - Speed/accuracy trade-offs
   - Optimization techniques
   - Hardware recommendations

Include code examples, diagrams, and clear formatting. Make it accessible to both beginners and experts.
```

---

## 🚀 Prompt for Optimization

```
Optimize a violence detection system for production deployment:

1. Model Optimization:
   - Convert to TensorFlow Lite for edge devices
   - Apply quantization (INT8, FP16)
   - Prune unnecessary weights
   - Optimize for inference speed
   - Reduce model size to <50MB

2. Code Optimization:
   - Implement batch processing for videos
   - Use multiprocessing for feature extraction
   - Cache frequently used computations
   - Optimize I/O operations
   - Profile and remove bottlenecks

3. Real-Time Performance:
   - Achieve 30+ FPS on GPU
   - 10+ FPS on CPU
   - Implement frame skipping strategies
   - Use threading for parallel processing

4. Memory Optimization:
   - Reduce memory footprint
   - Implement efficient data loading
   - Clear unused variables
   - Use generators for large datasets

5. Deployment:
   - Create Docker container
   - Build REST API with FastAPI
   - Implement model versioning
   - Add monitoring and logging
   - Write deployment documentation

Provide optimized code with performance benchmarks before/after.
```

---

## 📊 Usage Examples

### **How to Use These Prompts:**

1. **For Complete System:**
   - Use the main prompt to generate everything from scratch
   - Iterate with specific component prompts to refine

2. **For Specific Features:**
   - Use component prompts (architecture, features, training, inference)
   - Combine outputs into complete system

3. **For Debugging:**
   - Use debug prompts with specific error messages
   - Get targeted solutions

4. **For Enhancements:**
   - Use optimization, visualization, or documentation prompts
   - Improve existing implementation

---

## ✨ Tips for Effective Prompting

1. **Be Specific:**
   - Mention exact algorithms, frameworks, libraries
   - Specify dimensions, hyperparameters, architectures
   - Include expected performance metrics

2. **Provide Context:**
   - Explain the use case (surveillance, real-time, offline)
   - Mention constraints (hardware, time, accuracy)
   - Specify deployment environment

3. **Request Structure:**
   - Ask for modular, well-documented code
   - Request error handling and edge cases
   - Specify coding standards and style

4. **Include Examples:**
   - Reference similar projects or papers
   - Provide sample inputs/outputs
   - Show expected behavior

5. **Iterate:**
   - Start with high-level prompt
   - Refine with specific component prompts
   - Debug and optimize iteratively

---

**End of AI Prompt Document**
