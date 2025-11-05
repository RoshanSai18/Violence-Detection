"""
Data Preprocessing Module for Violence Detection
Handles frame extraction, normalization, and augmentation
"""

import cv2
import numpy as np
import os
from pathlib import Path
import config
from tqdm import tqdm
import random


class VideoPreprocessor:
    """
    Video preprocessing class for extracting and processing frames
    """
    
    def __init__(self, sequence_length=config.SEQUENCE_LENGTH, 
                 img_height=config.IMG_HEIGHT, 
                 img_width=config.IMG_WIDTH):
        self.sequence_length = sequence_length
        self.img_height = img_height
        self.img_width = img_width
        
    def extract_frames(self, video_path, method='uniform'):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            method: 'uniform' - extract evenly spaced frames
                   'random' - extract random frames
                   'consecutive' - extract consecutive frames from middle
        
        Returns:
            frames: numpy array of shape (sequence_length, height, width, channels)
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"Warning: No frames found in {video_path}")
                return None
            
            # Determine which frames to extract
            if method == 'uniform':
                # Extract evenly spaced frames
                frame_indices = np.linspace(0, total_frames - 1, 
                                           self.sequence_length, dtype=int)
            elif method == 'random':
                # Extract random frames
                frame_indices = sorted(random.sample(range(total_frames), 
                                                    min(self.sequence_length, total_frames)))
            elif method == 'consecutive':
                # Extract consecutive frames from the middle
                start_idx = max(0, (total_frames - self.sequence_length) // 2)
                frame_indices = list(range(start_idx, 
                                          min(start_idx + self.sequence_length, total_frames)))
            else:
                raise ValueError(f"Unknown extraction method: {method}")
            
            # Read frames
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize frame
                    frame = cv2.resize(frame, (self.img_width, self.img_height))
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # If frame reading fails, use the last valid frame or a black frame
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
            
            cap.release()
            
            # Ensure we have the correct number of frames
            while len(frames) < self.sequence_length:
                if frames:
                    frames.append(frames[-1])  # Duplicate last frame
                else:
                    frames.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
            
            # Convert to numpy array
            frames = np.array(frames[:self.sequence_length])
            
            return frames
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None
    
    def normalize_frames(self, frames, mode='tf'):
        """
        Normalize frame pixel values
        
        Args:
            frames: numpy array of frames
            mode: 'tf' - normalize to [-1, 1]
                 'torch' - normalize to [0, 1]
                 'caffe' - ImageNet mean subtraction
        
        Returns:
            Normalized frames
        """
        frames = frames.astype(np.float32)
        
        if mode == 'tf':
            # TensorFlow mode: scale to [-1, 1]
            frames = (frames / 127.5) - 1.0
        elif mode == 'torch':
            # PyTorch mode: scale to [0, 1]
            frames = frames / 255.0
        elif mode == 'caffe':
            # Caffe mode: ImageNet mean subtraction
            mean = np.array([103.939, 116.779, 123.68])
            frames[..., 0] -= mean[0]
            frames[..., 1] -= mean[1]
            frames[..., 2] -= mean[2]
        else:
            # Default: scale to [0, 1]
            frames = frames / 255.0
        
        return frames
    
    def augment_frames(self, frames, augmentation_params=None):
        """
        Apply data augmentation to frames
        
        Args:
            frames: numpy array of frames
            augmentation_params: dict of augmentation parameters
        
        Returns:
            Augmented frames
        """
        if augmentation_params is None:
            augmentation_params = config.AUGMENTATION
        
        augmented_frames = []
        
        for frame in frames:
            # Horizontal flip
            if augmentation_params.get('horizontal_flip', False):
                if random.random() > 0.5:
                    frame = cv2.flip(frame, 1)
            
            # Rotation
            rotation_range = augmentation_params.get('rotation_range', 0)
            if rotation_range > 0:
                angle = random.uniform(-rotation_range, rotation_range)
                h, w = frame.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                frame = cv2.warpAffine(frame, M, (w, h))
            
            # Brightness adjustment
            brightness_range = augmentation_params.get('brightness_range', None)
            if brightness_range:
                alpha = random.uniform(brightness_range[0], brightness_range[1])
                frame = np.clip(frame * alpha, 0, 255).astype(np.uint8)
            
            # Width shift
            width_shift = augmentation_params.get('width_shift_range', 0)
            if width_shift > 0:
                shift_pixels = int(frame.shape[1] * random.uniform(-width_shift, width_shift))
                M = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            
            # Height shift
            height_shift = augmentation_params.get('height_shift_range', 0)
            if height_shift > 0:
                shift_pixels = int(frame.shape[0] * random.uniform(-height_shift, height_shift))
                M = np.float32([[1, 0, 0], [0, 1, shift_pixels]])
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            
            # Zoom
            zoom_range = augmentation_params.get('zoom_range', 0)
            if zoom_range > 0:
                zoom = 1 + random.uniform(-zoom_range, zoom_range)
                h, w = frame.shape[:2]
                new_h, new_w = int(h * zoom), int(w * zoom)
                frame = cv2.resize(frame, (new_w, new_h))
                # Crop or pad to original size
                if zoom > 1:
                    # Crop
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    frame = frame[start_h:start_h+h, start_w:start_w+w]
                else:
                    # Pad
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    frame = cv2.copyMakeBorder(frame, pad_h, h-new_h-pad_h, 
                                              pad_w, w-new_w-pad_w, 
                                              cv2.BORDER_CONSTANT, value=0)
            
            augmented_frames.append(frame)
        
        return np.array(augmented_frames)
    
    def compute_optical_flow(self, frames):
        """
        Compute optical flow between consecutive frames
        
        Args:
            frames: numpy array of frames (RGB)
        
        Returns:
            Optical flow features
        """
        flow_frames = []
        
        for i in range(len(frames) - 1):
            # Convert to grayscale
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                               pyr_scale=0.5, levels=3, 
                                               winsize=15, iterations=3,
                                               poly_n=5, poly_sigma=1.2, flags=0)
            
            # Convert flow to RGB for visualization/processing
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_rgb = np.zeros_like(frames[i])
            flow_rgb[..., 0] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            flow_rgb[..., 1] = angle * 180 / np.pi / 2
            flow_rgb[..., 2] = 255
            flow_rgb = cv2.cvtColor(flow_rgb.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            flow_frames.append(flow_rgb)
        
        # Add the last frame's flow (duplicate last flow)
        if flow_frames:
            flow_frames.append(flow_frames[-1])
        
        return np.array(flow_frames)
    
    def compute_frame_difference(self, frames):
        """
        Compute frame differences
        
        Args:
            frames: numpy array of frames
        
        Returns:
            Frame difference features
        """
        diff_frames = []
        
        for i in range(len(frames) - 1):
            diff = cv2.absdiff(frames[i], frames[i + 1])
            diff_frames.append(diff)
        
        # Add the last frame's difference (duplicate last difference)
        if diff_frames:
            diff_frames.append(diff_frames[-1])
        
        return np.array(diff_frames)


def load_dataset_paths(data_dir, classes=['NonFight', 'Fight']):
    """
    Load all video paths and labels from dataset directory
    
    Args:
        data_dir: Path to data directory (train/val/test)
        classes: List of class names
    
    Returns:
        video_paths: List of video file paths
        labels: List of corresponding labels
    """
    video_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        # Get all video files
        video_files = [f for f in os.listdir(class_dir) 
                      if f.endswith(('.avi', '.mp4', '.mov', '.mkv'))]
        
        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            video_paths.append(video_path)
            labels.append(class_idx)
        
        print(f"Found {len(video_files)} videos in {class_name}")
    
    return video_paths, labels


def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        labels: List of labels
    
    Returns:
        class_weight: Dictionary of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', 
                                         classes=unique_classes, 
                                         y=labels)
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"Class weights: {class_weight_dict}")
    
    return class_weight_dict


def analyze_dataset(data_dir):
    """
    Analyze dataset and print statistics
    
    Args:
        data_dir: Path to dataset directory
    """
    print(f"\n{'='*50}")
    print(f"Dataset Analysis: {os.path.basename(data_dir)}")
    print(f"{'='*50}")
    
    video_paths, labels = load_dataset_paths(data_dir)
    
    print(f"\nTotal videos: {len(video_paths)}")
    print(f"Class distribution:")
    
    for class_idx, class_name in enumerate(config.CLASSES):
        count = labels.count(class_idx)
        percentage = (count / len(labels)) * 100 if labels else 0
        print(f"  {class_name}: {count} ({percentage:.2f}%)")
    
    # Sample video analysis
    if video_paths:
        print(f"\nSample video analysis:")
        preprocessor = VideoPreprocessor()
        
        for i in range(min(3, len(video_paths))):
            cap = cv2.VideoCapture(video_paths[i])
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            print(f"  Video {i+1}: {total_frames} frames, {fps} FPS, {duration:.2f}s")
    
    print(f"{'='*50}\n")
    
    return video_paths, labels


if __name__ == "__main__":
    # Test preprocessing
    print("Testing data preprocessing...")
    
    # Analyze train dataset
    train_paths, train_labels = analyze_dataset(config.TRAIN_DIR)
    
    # Analyze validation dataset
    val_paths, val_labels = analyze_dataset(config.VAL_DIR)
    
    # Calculate class weights
    if config.USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(train_labels)
    
    # Test frame extraction
    if train_paths:
        print("\nTesting frame extraction...")
        preprocessor = VideoPreprocessor()
        
        # Extract frames from first video
        frames = preprocessor.extract_frames(train_paths[0], method='uniform')
        
        if frames is not None:
            print(f"Extracted frames shape: {frames.shape}")
            
            # Test normalization
            normalized = preprocessor.normalize_frames(frames, mode='tf')
            print(f"Normalized frames range: [{normalized.min():.2f}, {normalized.max():.2f}]")
            
            # Test augmentation
            augmented = preprocessor.augment_frames(frames)
            print(f"Augmented frames shape: {augmented.shape}")
            
            print("\nPreprocessing test completed successfully!")
        else:
            print("Frame extraction failed!")
