"""
Quick Start Script for Violence Detection System
Runs initial setup and testing
"""

import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall missing packages with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_dataset():
    """Check if dataset is properly organized"""
    print_header("Checking Dataset")
    
    import config
    
    required_dirs = [
        (config.TRAIN_DIR, 'Train directory'),
        (os.path.join(config.TRAIN_DIR, 'Fight'), 'Train/Fight'),
        (os.path.join(config.TRAIN_DIR, 'NonFight'), 'Train/NonFight'),
        (config.VAL_DIR, 'Validation directory'),
        (os.path.join(config.VAL_DIR, 'Fight'), 'Val/Fight'),
        (os.path.join(config.VAL_DIR, 'NonFight'), 'Val/NonFight'),
    ]
    
    all_exist = True
    
    for dir_path, name in required_dirs:
        if os.path.exists(dir_path):
            # Count files
            files = [f for f in os.listdir(dir_path) 
                    if f.endswith(('.avi', '.mp4', '.mov', '.mkv'))]
            print(f"✓ {name}: {len(files)} videos")
        else:
            print(f"✗ {name}: NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Dataset structure incomplete!")
        print("\nExpected structure:")
        print("  RWF-2000/")
        print("    ├── train/")
        print("    │   ├── Fight/")
        print("    │   └── NonFight/")
        print("    └── val/")
        print("        ├── Fight/")
        print("        └── NonFight/")
        return False
    else:
        print("\n✓ Dataset structure correct!")
        return True


def check_gpu():
    """Check GPU availability"""
    print_header("Checking GPU")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Check CUDA
            print(f"\nCUDA available: {tf.test.is_built_with_cuda()}")
            print(f"GPU available: {tf.test.is_gpu_available()}")
            return True
        else:
            print("⚠ No GPU found. Training will use CPU (slower).")
            print("\nTo enable GPU:")
            print("  1. Install CUDA Toolkit")
            print("  2. Install cuDNN")
            print("  3. Install tensorflow-gpu")
            return False
    
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def test_preprocessing():
    """Test data preprocessing"""
    print_header("Testing Data Preprocessing")
    
    try:
        from data_preprocessing import VideoPreprocessor, load_dataset_paths
        import config
        
        # Load a sample video
        train_paths, train_labels = load_dataset_paths(config.TRAIN_DIR)
        
        if not train_paths:
            print("✗ No videos found in training directory")
            return False
        
        # Test frame extraction
        preprocessor = VideoPreprocessor()
        test_video = train_paths[0]
        
        print(f"Testing with: {os.path.basename(test_video)}")
        
        frames = preprocessor.extract_frames(test_video, method='uniform')
        
        if frames is not None:
            print(f"✓ Frame extraction successful: {frames.shape}")
            
            # Test normalization
            normalized = preprocessor.normalize_frames(frames)
            print(f"✓ Normalization successful: range [{normalized.min():.2f}, {normalized.max():.2f}]")
            
            return True
        else:
            print("✗ Frame extraction failed")
            return False
    
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_building():
    """Test model building"""
    print_header("Testing Model Building")
    
    try:
        from model import build_violence_detection_model, compile_model
        
        print("Building model...")
        model = build_violence_detection_model()
        
        print(f"✓ Model built successfully")
        print(f"  Total layers: {len(model.layers)}")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # Compile
        model = compile_model(model)
        print(f"✓ Model compiled successfully")
        
        # Test with dummy data
        import numpy as np
        import config
        
        dummy_input = np.random.random((1, config.SEQUENCE_LENGTH, 
                                       config.IMG_HEIGHT, config.IMG_WIDTH, 
                                       config.IMG_CHANNELS))
        
        output = model.predict(dummy_input, verbose=0)
        print(f"✓ Inference test successful: {output.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error in model building: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_dataset_analysis():
    """Run dataset analysis"""
    print_header("Dataset Analysis")
    
    try:
        from data_preprocessing import analyze_dataset
        import config
        
        print("Analyzing training set...")
        train_paths, train_labels = analyze_dataset(config.TRAIN_DIR)
        
        print("\nAnalyzing validation set...")
        val_paths, val_labels = analyze_dataset(config.VAL_DIR)
        
        return True
    
    except Exception as e:
        print(f"✗ Error in analysis: {e}")
        return False


def main():
    """Main setup function"""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  VIOLENCE DETECTION SYSTEM - QUICK START".center(78) + "║")
    print("║" + "  CNN + BiLSTM with MobileNet".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    
    results = []
    
    # 1. Check dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # 2. Check dataset
    results.append(("Dataset Structure", check_dataset()))
    
    # 3. Check GPU
    results.append(("GPU Availability", check_gpu()))
    
    # 4. Test preprocessing
    if results[1][1]:  # Only if dataset exists
        results.append(("Preprocessing", test_preprocessing()))
    
    # 5. Test model building
    if results[0][1]:  # Only if dependencies installed
        results.append(("Model Building", test_model_building()))
    
    # 6. Dataset analysis
    if results[1][1]:  # Only if dataset exists
        results.append(("Dataset Analysis", run_dataset_analysis()))
    
    # Summary
    print_header("Setup Summary")
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ Setup completed successfully!")
        print("\nYou can now start training:")
        print("  python train.py")
    else:
        print("⚠ Setup incomplete. Please fix the issues above.")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
