"""
Inference Script for Violence Detection
Predict violence in new videos using trained model
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import config
from data_preprocessing import VideoPreprocessor
import time
from pathlib import Path


def load_model_for_inference(model_path):
    """
    Load trained model for inference
    
    Args:
        model_path: Path to saved model
    
    Returns:
        Loaded model
    """
    from model import AttentionLayer
    
    print(f"Loading model from: {model_path}")
    
    custom_objects = {'AttentionLayer': AttentionLayer}
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    return model


def predict_video(model, video_path, threshold=config.CONFIDENCE_THRESHOLD,
                 return_details=False):
    """
    Predict violence in a single video
    
    Args:
        model: Trained model
        video_path: Path to video file
        threshold: Classification threshold
        return_details: Whether to return detailed information
    
    Returns:
        prediction, confidence, (optional: frames, processing_time)
    """
    start_time = time.time()
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        sequence_length=config.SEQUENCE_LENGTH,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH
    )
    
    # Extract frames
    frames = preprocessor.extract_frames(video_path, method='uniform')
    
    if frames is None:
        print(f"Failed to extract frames from {video_path}")
        return None, None, None if return_details else (None, None)
    
    # Normalize frames
    normalized_frames = preprocessor.normalize_frames(frames, mode=config.PREPROCESS_MODE)
    
    # Add batch dimension
    input_data = np.expand_dims(normalized_frames, axis=0)
    
    # Predict
    prediction_proba = model.predict(input_data, verbose=0)[0][0]
    prediction = int(prediction_proba >= threshold)
    
    processing_time = time.time() - start_time
    
    if return_details:
        return prediction, prediction_proba, frames, processing_time
    else:
        return prediction, prediction_proba


def predict_batch(model, video_paths, threshold=config.CONFIDENCE_THRESHOLD,
                 show_progress=True):
    """
    Predict violence in multiple videos
    
    Args:
        model: Trained model
        video_paths: List of video paths
        threshold: Classification threshold
        show_progress: Whether to show progress
    
    Returns:
        predictions, confidences
    """
    predictions = []
    confidences = []
    
    print(f"\nPredicting on {len(video_paths)} videos...")
    
    for i, video_path in enumerate(video_paths):
        pred, conf = predict_video(model, video_path, threshold)
        
        predictions.append(pred)
        confidences.append(conf)
        
        if show_progress and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(video_paths)} videos")
    
    return np.array(predictions), np.array(confidences)


def predict_with_visualization(model, video_path, threshold=config.CONFIDENCE_THRESHOLD,
                               save_output=False, output_dir=config.OUTPUT_DIR):
    """
    Predict and visualize results
    
    Args:
        model: Trained model
        video_path: Path to video file
        threshold: Classification threshold
        save_output: Whether to save output video
        output_dir: Directory to save output
    
    Returns:
        prediction, confidence
    """
    prediction, confidence, frames, proc_time = predict_video(
        model, video_path, threshold, return_details=True
    )
    
    if prediction is None:
        return None, None
    
    # Get label and color
    label = config.CLASSES[prediction]
    color = (0, 0, 255) if prediction == 1 else (0, 255, 0)  # Red for violence, Green for non-violence
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"Processing Time: {proc_time:.2f}s")
    print(f"{'='*60}\n")
    
    # Create visualization
    if save_output:
        output_path = os.path.join(output_dir, f'output_{os.path.basename(video_path)}')
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Read original video and add annotations
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add text overlay
            text = f"{label}: {confidence:.2%}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, color, 3, cv2.LINE_AA)
            
            # Add colored border
            cv2.rectangle(frame, (0, 0), (width-1, height-1), color, 10)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        print(f"Output video saved to: {output_path}")
    
    return prediction, confidence


def predict_realtime_webcam(model, threshold=config.CONFIDENCE_THRESHOLD,
                           window_size=30, update_interval=10):
    """
    Real-time violence detection from webcam
    
    Args:
        model: Trained model
        threshold: Classification threshold
        window_size: Number of frames to buffer
        update_interval: Frames between predictions
    """
    import collections
    
    print("Starting real-time detection...")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Frame buffer
    frame_buffer = collections.deque(maxlen=window_size)
    frame_count = 0
    
    preprocessor = VideoPreprocessor(
        sequence_length=config.SEQUENCE_LENGTH,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH
    )
    
    current_prediction = 0
    current_confidence = 0.0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Add frame to buffer
        resized_frame = cv2.resize(frame, (config.IMG_WIDTH, config.IMG_HEIGHT))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb_frame)
        
        frame_count += 1
        
        # Predict every update_interval frames
        if frame_count % update_interval == 0 and len(frame_buffer) >= config.SEQUENCE_LENGTH:
            # Get last sequence_length frames
            frames = list(frame_buffer)[-config.SEQUENCE_LENGTH:]
            frames = np.array(frames)
            
            # Normalize and predict
            normalized = preprocessor.normalize_frames(frames, mode=config.PREPROCESS_MODE)
            input_data = np.expand_dims(normalized, axis=0)
            
            prediction_proba = model.predict(input_data, verbose=0)[0][0]
            current_prediction = int(prediction_proba >= threshold)
            current_confidence = prediction_proba
        
        # Display results
        label = config.CLASSES[current_prediction]
        color = (0, 0, 255) if current_prediction == 1 else (0, 255, 0)
        
        # Add text overlay
        text = f"{label}: {current_confidence:.2%}"
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, color, 3, cv2.LINE_AA)
        
        # Add colored border
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 10)
        
        # Show frame
        cv2.imshow('Violence Detection - Press q to quit', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped")


def batch_predict_directory(model, input_dir, output_csv=None, 
                           threshold=config.CONFIDENCE_THRESHOLD):
    """
    Predict all videos in a directory
    
    Args:
        model: Trained model
        input_dir: Directory containing videos
        output_csv: Path to save results CSV
        threshold: Classification threshold
    """
    import pandas as pd
    
    # Get all video files
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    print(f"Found {len(video_files)} videos in {input_dir}")
    
    # Predict
    results = []
    
    for video_path in video_files:
        pred, conf = predict_video(model, str(video_path), threshold)
        
        if pred is not None:
            results.append({
                'filename': video_path.name,
                'path': str(video_path),
                'prediction': config.CLASSES[pred],
                'confidence': conf,
                'violence_probability': conf if pred == 1 else 1 - conf
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH PREDICTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos: {len(results)}")
    print(f"Violence detected: {sum(df['prediction'] == 'Fight')}")
    print(f"Non-violence detected: {sum(df['prediction'] == 'NonFight')}")
    print(f"Average confidence: {df['confidence'].mean():.4f}")
    print(f"{'='*80}\n")
    
    # Save to CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Violence Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to single video file')
    parser.add_argument('--dir', type=str, default=None,
                       help='Directory containing videos for batch prediction')
    parser.add_argument('--webcam', action='store_true',
                       help='Use webcam for real-time detection')
    parser.add_argument('--threshold', type=float, default=config.CONFIDENCE_THRESHOLD,
                       help='Classification threshold')
    parser.add_argument('--save-output', action='store_true',
                       help='Save output video with annotations')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Path to save batch prediction results')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_for_inference(args.model)
    
    # Run appropriate inference mode
    if args.webcam:
        # Real-time webcam detection
        predict_realtime_webcam(model, threshold=args.threshold)
    
    elif args.video:
        # Single video prediction
        predict_with_visualization(
            model, args.video, 
            threshold=args.threshold,
            save_output=args.save_output
        )
    
    elif args.dir:
        # Batch directory prediction
        batch_predict_directory(
            model, args.dir,
            output_csv=args.output_csv,
            threshold=args.threshold
        )
    
    else:
        print("Error: Please specify --video, --dir, or --webcam")
        parser.print_help()
