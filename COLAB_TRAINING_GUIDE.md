# 📋 Google Colab Training - Quick Start Guide

## ⚡ Updated Workflow (No Google Drive Required!)

Your notebook now works entirely in Google Colab's temporary storage. Here's what changed:

---

## 🔄 What's Different?

### ❌ Before (Old Version):
- Required Google Drive
- Manual dataset upload to Drive
- Persistent storage across sessions
- Complex path configuration

### ✅ Now (New Version):
- No Google Drive needed
- Dataset downloaded via Kaggle API
- Runs in single session
- Simple, streamlined workflow

---

## 🚀 Complete Workflow

### Step 1: Get Kaggle API Credentials (One-Time Setup)

1. Go to [kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`
5. Open it and note your username and key:
   ```json
   {
     "username": "your_username",
     "key": "abc123def456..."
   }
   ```

---

### Step 2: Upload Notebook to Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook
3. Select `Violence_Detection_MultiModal_Colab.ipynb`

---

### Step 3: Select GPU Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (recommended)
4. Click **Save**

---

### Step 4: Run All Cells

1. Click Runtime → **Run all**
2. When prompted for Kaggle credentials:
   - Enter your **username**
   - Enter your **API key**
3. Wait for completion (~4-6 hours total)

**Timeline:**
```
Hour 0:     Setup & dataset download (~10-15 min)
Hour 0-3:   STEP 1 - Preprocessing & caching (~2-3 hours)
Hour 3-6:   STEP 2 - Model training (~2-3 hours)
Hour 6:     Model download (~1 min)
```

---

### Step 5: Download Trained Model

At the end of the notebook (Section 14), your browser will automatically download:

```
violence_detection_model.zip (~50-100 MB)
```

Extract this file to get:
- `best_multimodal_model.h5` ← **Use this!**
- `training_history.json`
- `training_history.png`
- `evaluation_results.png`
- `emotion_analysis.png`

---

## 📊 What Happens During Training

### Section 2: Download Dataset
```
🔑 Setting up Kaggle credentials...
🔄 Downloading RWF-2000 dataset from Kaggle...
✅ Dataset downloaded to: /root/.cache/kagglehub/...
✅ Cache path: /content/violence_detection_cache
✅ Models path: /content/violence_detection_models

📊 VERIFYING DATASET STRUCTURE
✅ Train/Fight: 800 videos
✅ Train/NonFight: 800 videos
✅ Val/Fight: 200 videos
✅ Val/NonFight: 200 videos
Total videos: 2000
✅ Dataset structure is CORRECT!
```

### STEP 1: Preprocessing
```
⚡ PREPROCESSING TRAINING DATA...
Processing videos: 100%|████████████| 1600/1600 [1:45:32<00:00,  2.53s/video]
✅ Saved to cache: train_features.npz (5.7 GB)

⚡ PREPROCESSING VALIDATION DATA...
Processing videos: 100%|████████████| 400/400 [26:18<00:00,  3.95s/video]
✅ Saved to cache: val_features.npz (1.4 GB)
```

### STEP 2: Training
```
Epoch 1/30
50/50 [==============================] - 245s 5s/step - loss: 0.6234 - accuracy: 0.6562
...
Epoch 25/30
50/50 [==============================] - 218s 4s/step - loss: 0.0823 - accuracy: 0.9687
      - val_loss: 0.1234 - val_accuracy: 0.9562 ✅

✅ TRAINING COMPLETED!
Best Validation Accuracy: 95.62%
Best Validation AUC: 0.9812
```

### Section 14: Download
```
📦 Packaging trained model and results...
✅ best_multimodal_model.h5 (87.34 MB) - Best trained model
✅ training_history.json (12.45 KB) - Training metrics
✅ training_history.png (234.56 KB) - Training curves
✅ evaluation_results.png (156.78 KB) - Confusion matrix & ROC
✅ emotion_analysis.png (89.12 KB) - Emotion analysis

📥 DOWNLOADING TO YOUR PC...
✅ Download complete!
```

---

## 📁 File Locations During Training

### In Colab Storage (Temporary):
```
/root/.cache/kagglehub/
└── datasets/
    └── vulamnguyen/
        └── rwf2000/
            └── versions/1/          ← Dataset (~5-7 GB)

/content/
├── violence_detection_cache/
│   ├── train_features.npz           ← Cached features (~5-7 GB)
│   └── val_features.npz             ← Cached features (~1-2 GB)
│
└── violence_detection_models/
    ├── best_multimodal_model.h5     ← Best model
    ├── final_multimodal_model.h5    ← Final model
    ├── training_history.json
    ├── training_history.png
    ├── evaluation_results.png
    └── emotion_analysis.png
```

⚠️ **All files in Colab storage are TEMPORARY** - they'll be deleted when session ends!

### Downloaded to Your PC (Permanent):
```
Downloads/
└── violence_detection_model.zip     ← Extract this!
    ├── best_multimodal_model.h5
    ├── training_history.json
    ├── training_history.png
    ├── evaluation_results.png
    └── emotion_analysis.png
```

---

## ⏱️ Time Estimates

| Task | Duration | Can Skip? |
|------|----------|-----------|
| Setup & Kaggle credentials | ~2 min | No |
| Dataset download | ~10-15 min | No |
| **STEP 1: Preprocessing** | **~2-3 hours** | **No (first run)** |
| **STEP 2: Training** | **~2-3 hours** | **No** |
| Evaluation & visualization | ~5 min | No |
| Model download | ~1 min | No |
| **Total** | **~4-6 hours** | - |

---

## ⚠️ Important Notes

### Session Management
- **Don't close the browser tab** during training!
- **Keep your computer awake** (session may disconnect if idle)
- If disconnected, you'll need to **start over** (everything is temporary)

### GPU Quota
- Colab free tier: ~15-20 hours GPU per week
- This training uses ~4-6 hours
- Plan accordingly!

### Storage
- Colab provides ~100 GB temporary storage
- This project uses ~15-20 GB
- Should have plenty of space

---

## 🔄 If Session Disconnects

Unfortunately, if your Colab session disconnects mid-training, you'll need to:

1. Restart the runtime
2. Run all cells again from the beginning
3. Dataset will re-download (~10-15 min)
4. Preprocessing will re-run (~2-3 hours)
5. Training will restart from epoch 1

**Tips to avoid disconnection:**
- Keep browser tab active
- Don't let computer sleep
- Use Colab Pro if available (longer sessions)

---

## 🎯 After Training

### Using Your Model Locally

1. **Extract the downloaded zip**
2. **Install dependencies:**
   ```bash
   pip install tensorflow opencv-python mediapipe deepface
   ```
3. **Run real-time detection:**
   ```bash
   python realtime_webcam_detection.py --model best_multimodal_model.h5 --source webcam
   ```

See `USING_TRAINED_MODEL.md` for complete instructions!

---

## 📊 Expected Results

After training, you should see:

### Metrics:
- **Validation Accuracy:** 92-97%
- **Validation Precision:** 93-96%
- **Validation Recall:** 91-95%
- **Validation F1-Score:** 92-96%
- **Validation AUC:** 96-99%

### Files:
- Model size: ~50-100 MB
- Training history: All epoch metrics
- Visualizations: Confusion matrix, ROC curve, emotion analysis

---

## ✅ Checklist

Before starting:
- [ ] Kaggle API credentials ready
- [ ] Google Colab account active
- [ ] GPU runtime selected
- [ ] ~6 hours of free time
- [ ] Computer won't sleep
- [ ] Stable internet connection

After training:
- [ ] Model downloaded to PC
- [ ] Zip file extracted
- [ ] Dependencies installed locally
- [ ] Ready for real-time inference!

---

## 🆘 Common Issues

### "Kaggle credentials error"
**Solution:** Double-check username and API key. Get a new token if needed.

### "GPU not available"
**Solution:** 
- Runtime → Change runtime type → GPU → Save
- Restart runtime

### "Out of memory"
**Solution:**
- Reduce batch size in config (32 → 16)
- Restart runtime to clear memory

### "Dataset download stuck"
**Solution:**
- Check Kaggle account is verified
- Wait a few minutes (large dataset)
- Restart runtime if stuck >30 min

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `Violence_Detection_MultiModal_Colab.ipynb` | Main training notebook (upload to Colab) |
| `realtime_webcam_detection.py` | Real-time inference script (use locally) |
| `USING_TRAINED_MODEL.md` | Guide for using trained model |
| `QUICK_REFERENCE.md` | Complete feature checklist |
| `ACCURACY_AND_FEATURES_VERIFICATION.md` | Detailed accuracy analysis |

---

## 🎉 You're Ready!

**Just upload the notebook to Colab and click "Run all"!**

Everything else is automatic:
1. ✅ Dataset downloads automatically
2. ✅ Features preprocess and cache automatically
3. ✅ Model trains automatically
4. ✅ Results download automatically

**Total time: ~4-6 hours** ⏱️

**Expected accuracy: 92-97%** 🎯

**Ready for real-time use!** 🚀

---

**Questions?** Check the other documentation files or re-read the notebook instructions!
