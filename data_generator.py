"""
Custom Data Generator for Video Classification
Efficiently loads and processes video batches during training
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import config
from data_preprocessing import VideoPreprocessor, load_dataset_paths
import random


class VideoDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for video classification
    Generates batches of video sequences for training/validation
    """
    
    def __init__(self, video_paths, labels, batch_size=config.BATCH_SIZE,
                 sequence_length=config.SEQUENCE_LENGTH,
                 img_height=config.IMG_HEIGHT,
                 img_width=config.IMG_WIDTH,
                 num_classes=config.NUM_CLASSES,
                 shuffle=True,
                 augment=False,
                 preprocess_mode='tf',
                 use_optical_flow=False,
                 use_frame_difference=False):
        """
        Initialize the data generator
        
        Args:
            video_paths: List of video file paths
            labels: List of corresponding labels
            batch_size: Number of samples per batch
            sequence_length: Number of frames per video
            img_height: Image height
            img_width: Image width
            num_classes: Number of classes
            shuffle: Whether to shuffle data after each epoch
            augment: Whether to apply data augmentation
            preprocess_mode: Normalization mode ('tf', 'torch', 'caffe')
            use_optical_flow: Whether to compute optical flow
            use_frame_difference: Whether to compute frame differences
        """
        self.video_paths = np.array(video_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.preprocess_mode = preprocess_mode
        self.use_optical_flow = use_optical_flow
        self.use_frame_difference = use_frame_difference
        
        self.preprocessor = VideoPreprocessor(sequence_length, img_height, img_width)
        self.indices = np.arange(len(self.video_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        
        Args:
            index: Batch index
        
        Returns:
            X: Batch of video sequences
            y: Batch of labels
        """
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.video_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch data
        batch_paths = self.video_paths[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Generate data
        X, y = self.__data_generation(batch_paths, batch_labels)
        
        return X, y
    
    def on_epoch_end(self):
        """Updates indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __data_generation(self, batch_paths, batch_labels):
        """
        Generate batch data
        
        Args:
            batch_paths: Batch of video paths
            batch_labels: Batch of labels
        
        Returns:
            X: Batch of processed videos
            y: Batch of labels
        """
        X = []
        y = []
        
        for video_path, label in zip(batch_paths, batch_labels):
            try:
                # Extract frames
                frames = self.preprocessor.extract_frames(video_path, method='uniform')
                
                if frames is None:
                    # Skip invalid videos
                    continue
                
                # Apply augmentation if enabled
                if self.augment:
                    frames = self.preprocessor.augment_frames(frames)
                
                # Add optical flow or frame difference features if enabled
                if self.use_optical_flow:
                    flow = self.preprocessor.compute_optical_flow(frames)
                    frames = np.concatenate([frames, flow], axis=-1)
                
                if self.use_frame_difference:
                    diff = self.preprocessor.compute_frame_difference(frames)
                    frames = np.concatenate([frames, diff], axis=-1)
                
                # Normalize frames
                frames = self.preprocessor.normalize_frames(frames, mode=self.preprocess_mode)
                
                X.append(frames)
                y.append(label)
                
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                continue
        
        # Convert to numpy arrays
        if len(X) == 0:
            # Return empty batch if all videos failed
            X = np.zeros((1, self.sequence_length, self.img_height, 
                         self.img_width, config.IMG_CHANNELS))
            y = np.zeros((1,))
        else:
            X = np.array(X)
            y = np.array(y)
        
        # Convert labels to categorical if multi-class
        if self.num_classes > 2:
            y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        return X, y


class BalancedVideoDataGenerator(VideoDataGenerator):
    """
    Balanced data generator that ensures equal samples from each class
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Calculate samples per class per batch
        self.samples_per_class = self.batch_size // len(self.class_indices)
        
    def __getitem__(self, index):
        """
        Generate one balanced batch of data
        """
        batch_indices = []
        
        # Sample equal number from each class
        for class_label, class_idx_list in self.class_indices.items():
            if self.shuffle:
                sampled_indices = random.sample(class_idx_list, 
                                              min(self.samples_per_class, len(class_idx_list)))
            else:
                start = (index * self.samples_per_class) % len(class_idx_list)
                sampled_indices = class_idx_list[start:start + self.samples_per_class]
            
            batch_indices.extend(sampled_indices)
        
        # Get batch data
        batch_paths = self.video_paths[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Generate data
        X, y = self.__data_generation(batch_paths, batch_labels)
        
        return X, y


def create_data_generators(train_dir=config.TRAIN_DIR,
                          val_dir=config.VAL_DIR,
                          batch_size=config.BATCH_SIZE,
                          augment_train=True,
                          balanced=False):
    """
    Create train and validation data generators
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        augment_train: Whether to augment training data
        balanced: Whether to use balanced sampling
    
    Returns:
        train_generator: Training data generator
        val_generator: Validation data generator
    """
    # Load dataset paths
    train_paths, train_labels = load_dataset_paths(train_dir)
    val_paths, val_labels = load_dataset_paths(val_dir)
    
    print(f"\nDataset loaded:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    
    # Choose generator class
    GeneratorClass = BalancedVideoDataGenerator if balanced else VideoDataGenerator
    
    # Create train generator
    train_generator = GeneratorClass(
        video_paths=train_paths,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
        preprocess_mode=config.PREPROCESS_MODE,
        use_optical_flow=config.USE_OPTICAL_FLOW,
        use_frame_difference=config.USE_FRAME_DIFFERENCE
    )
    
    # Create validation generator
    val_generator = VideoDataGenerator(
        video_paths=val_paths,
        labels=val_labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        preprocess_mode=config.PREPROCESS_MODE,
        use_optical_flow=config.USE_OPTICAL_FLOW,
        use_frame_difference=config.USE_FRAME_DIFFERENCE
    )
    
    print(f"Data generators created:")
    print(f"  Train batches per epoch: {len(train_generator)}")
    print(f"  Validation batches per epoch: {len(val_generator)}")
    
    return train_generator, val_generator


if __name__ == "__main__":
    # Test data generator
    print("Testing data generator...")
    
    # Create generators
    train_gen, val_gen = create_data_generators(
        batch_size=4,
        augment_train=True,
        balanced=False
    )
    
    # Test batch generation
    print("\nGenerating test batch...")
    X_batch, y_batch = train_gen[0]
    
    print(f"Batch shape: {X_batch.shape}")
    print(f"Labels shape: {y_batch.shape}")
    print(f"Batch data range: [{X_batch.min():.2f}, {X_batch.max():.2f}]")
    print(f"Labels: {y_batch}")
    
    print("\nData generator test completed successfully!")
