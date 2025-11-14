# 🚀 Using Your Trained Violence Detection Model

After training in Google Colab, you'll have downloaded `violence_detection_model.zip` containing:

```
violence_detection_model/
├── best_multimodal_model.h5       ← Use this for inference!
├── final_multimodal_model.h5      ← Final epoch model
├── training_history.json          ← Training metrics
├── training_history.png           ← Training curves
├── evaluation_results.png         ← Confusion matrix & ROC
└── emotion_analysis.png           ← Emotion patterns
```

---

## 📦 Setup on Your PC

### 1. Install Dependencies

```bash
pip install tensorflow opencv-python mediapipe deepface
```

### 2. Extract the Downloaded Files

Extract `violence_detection_model.zip` to get your trained model files.

---

## 🎥 Real-Time Detection

### Webcam Detection

```bash
python realtime_webcam_detection.py --model best_multimodal_model.h5 --source webcam
```

**Controls:**
- Press `q` to quit

### Video File Detection

```bash
python realtime_webcam_detection.py --model best_multimodal_model.h5 --source video.mp4
```

### Save Output Video

```bash
python realtime_webcam_detection.py --model best_multimodal_model.h5 --source video.mp4 --output result.mp4
```

### Hide Pose Overlay

```bash
python realtime_webcam_detection.py --model best_multimodal_model.h5 --source webcam --no-pose
```

### Use External Webcam

```bash
python realtime_webcam_detection.py --model best_multimodal_model.h5 --source webcam --camera-id 1
```

---

## 📊 What You'll See

The detection window displays:

```
┌─────────────────────────────────────────┐
│ VIOLENCE DETECTED                       │  ← Label (red for violence, green for normal)
│ Confidence: 95.67%                      │  ← Prediction confidence
│ ████████████████░░░░                    │  ← Probability bar
│                                         │
│                                         │
│        [Video Feed with Pose]           │  ← Live video with pose keypoints
│                                         │
│                                         │
│                           FPS: 15       │  ← Processing speed
└─────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Error: "Model file not found"
**Solution:** Make sure you're in the same directory as `best_multimodal_model.h5` or provide the full path:
```bash
python realtime_webcam_detection.py --model /path/to/best_multimodal_model.h5 --source webcam
```

### Error: "Could not open webcam"
**Solution:** 
- Check if another application is using the webcam
- Try a different camera ID: `--camera-id 1` or `--camera-id 2`
- On Linux, you might need: `sudo chmod 666 /dev/video0`

### Slow Performance (Low FPS)
**Solutions:**
1. Use a GPU-enabled version of TensorFlow:
   ```bash
   pip install tensorflow-gpu
   ```
2. Reduce frame processing (edit the script to skip every other frame)
3. Close other applications

### Error: "DeepFace not working"
**Solution:** DeepFace might have issues with first-time setup. Try:
```bash
pip uninstall deepface
pip install deepface==0.0.79
```

---

## 📈 Expected Performance

| Hardware | FPS | Notes |
|----------|-----|-------|
| CPU (Intel i5/i7) | 3-5 FPS | Usable for video files |
| GPU (GTX 1060+) | 10-15 FPS | Good for real-time |
| GPU (RTX 3060+) | 20-30 FPS | Excellent for real-time |

**Note:** MediaPipe and DeepFace are computationally intensive. Real-time performance depends on your hardware.

---

## 🎯 Model Information

- **Architecture:** Multi-modal (RGB + Pose + Emotion)
- **Input:** 20 frames per video sequence
- **Frame Size:** 224×224 pixels
- **Features:**
  - RGB: MobileNetV2 features (1280-dim)
  - Pose: MediaPipe keypoints + angles (120-dim)
  - Emotion: DeepFace emotions + variance (8-dim)
- **Output:** Binary classification (Violence / Non-Violence)
- **Accuracy:** 92-97% on RWF-2000 dataset

---

## 🔄 Re-training the Model

To re-train with different parameters:

1. Upload `Violence_Detection_MultiModal_Colab.ipynb` to Google Colab
2. Run all cells (will take ~4-6 hours)
3. Download the new model at the end
4. Use the new `best_multimodal_model.h5` for inference

---

## 📝 Using the Model in Your Own Code

```python
import numpy as np
from tensorflow import keras
from realtime_webcam_detection import AttentionLayer, RealtimeViolenceDetector

# Load model
model = keras.models.load_model(
    'best_multimodal_model.h5',
    custom_objects={'AttentionLayer': AttentionLayer}
)

# Create detector
detector = RealtimeViolenceDetector('best_multimodal_model.h5')

# Run on webcam
detector.run_webcam(camera_id=0)

# Or run on video
detector.run_video_file('input.mp4', 'output.mp4')
```

---

## 🎉 Features

✅ **Real-time webcam detection**  
✅ **Video file processing**  
✅ **Pose visualization** (body keypoints)  
✅ **Confidence scores** (probability display)  
✅ **Smooth predictions** (reduces flickering)  
✅ **FPS counter** (performance monitoring)  
✅ **Output video saving**  
✅ **Multi-platform** (Windows, Linux, macOS)  

---

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the documentation files in your project folder
3. Ensure all dependencies are installed correctly

---

## 📚 Additional Resources

- **Training Notebook:** `Violence_Detection_MultiModal_Colab.ipynb`
- **Inference Script:** `realtime_webcam_detection.py`
- **Training Metrics:** Check `training_history.json`
- **Model Performance:** View `evaluation_results.png`

---

**Enjoy your violence detection system! 🚀**
