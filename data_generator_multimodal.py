"""
Multi-Modal Data Generator for Violence Detection
Generates batches with RGB frames, pose features, and emotion features
"""

import os
import numpy as np
import cv2
from tensorflow import keras
from sklearn.utils import shuffle
import config
from pose_emotion_preprocessing import PoseEmotionPreprocessor


class MultiModalDataGenerator(keras.utils.Sequence):
    """
    Custom data generator for multi-modal violence detection
    
    Generates batches containing:
    - RGB frames (batch_size, sequence_length, 224, 224, 3)
    - Pose features (batch_size, sequence_length, 120)
    - Emotion features (batch_size, sequence_length, 8)
    - Labels (batch_size, 1)
    """
    
    def __init__(
        self,
        video_paths,
        labels,
        batch_size=config.BATCH_SIZE,
        sequence_length=config.SEQUENCE_LENGTH,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        augment=False,
        shuffle_data=True,
        use_advanced_pose=True,
        use_multi_person_emotion=True
    ):
        """
        Initialize multi-modal data generator
        
        Args:
            video_paths: List of video file paths
            labels: List of labels (0 or 1)
            batch_size: Batch size
            sequence_length: Number of frames per video
            img_height: Image height
            img_width: Image width
            augment: Whether to apply data augmentation
            shuffle_data: Whether to shuffle data after each epoch
            use_advanced_pose: Use advanced pose features (120-dim)
            use_multi_person_emotion: Use multi-person emotion aggregation
        """
        self.video_paths = np.array(video_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.shuffle_data = shuffle_data
        self.use_advanced_pose = use_advanced_pose
        self.use_multi_person_emotion = use_multi_person_emotion
        
        # Initialize pose-emotion preprocessor
        self.preprocessor = PoseEmotionPreprocessor()
        
        # Shuffle initially
        if self.shuffle_data:
            self._shuffle()
        
        print(f"\n{'='*60}")
        print("Multi-Modal Data Generator Initialized")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.video_paths)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batches per epoch: {len(self)}")
        print(f"Sequence length: {self.sequence_length} frames")
        print(f"Image size: {self.img_height}x{self.img_width}")
        print(f"Augmentation: {'ON' if self.augment else 'OFF'}")
        print(f"Advanced pose: {'ON' if self.use_advanced_pose else 'OFF'}")
        print(f"Multi-person emotion: {'ON' if self.use_multi_person_emotion else 'OFF'}")
        print(f"{'='*60}\n")
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        
        Returns:
            Tuple of (inputs_dict, labels) where:
            - inputs_dict = {'frames': [...], 'pose': [...], 'emotion': [...]}
            - labels = [...]
        """
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.video_paths))
        
        batch_paths = self.video_paths[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        # Generate batch
        return self._generate_batch(batch_paths, batch_labels)
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        if self.shuffle_data:
            self._shuffle()
    
    def _shuffle(self):
        """Shuffle video paths and labels together"""
        self.video_paths, self.labels = shuffle(
            self.video_paths,
            self.labels,
            random_state=np.random.randint(10000)
        )
    
    def _generate_batch(self, batch_paths, batch_labels):
        """
        Generate one batch of multi-modal data
        
        Args:
            batch_paths: List of video paths in batch
            batch_labels: List of labels in batch
        
        Returns:
            (inputs_dict, labels_array)
        """
        batch_size = len(batch_paths)
        
        # Initialize arrays
        frames_batch = np.zeros(
            (batch_size, self.sequence_length, self.img_height, self.img_width, 3),
            dtype=np.float32
        )
        pose_batch = np.zeros(
            (batch_size, self.sequence_length, 120 if self.use_advanced_pose else 99),
            dtype=np.float32
        )
        emotion_batch = np.zeros(
            (batch_size, self.sequence_length, 8),
            dtype=np.float32
        )
        labels_batch = np.zeros((batch_size, 1), dtype=np.float32)
        
        # Process each video
        for i, video_path in enumerate(batch_paths):
            try:
                # Extract features
                features = self.preprocessor.extract_enhanced_features(
                    video_path=video_path,
                    num_frames=self.sequence_length,
                    target_size=(self.img_height, self.img_width),
                    use_advanced_pose=self.use_advanced_pose,
                    use_multi_person_emotion=self.use_multi_person_emotion
                )
                
                frames = features['frames']
                pose = features['pose']
                emotion = features['emotion']
                
                # Apply augmentation to frames if needed
                if self.augment:
                    frames = self._augment_frames(frames)
                
                # Normalize frames
                frames = frames / 255.0  # Normalize to [0, 1]
                
                # Store in batch
                frames_batch[i] = frames
                pose_batch[i] = pose
                emotion_batch[i] = emotion
                labels_batch[i] = batch_labels[i]
                
            except Exception as e:
                # Handle errors gracefully - use zeros for failed samples
                print(f"\nWarning: Failed to process {video_path}: {str(e)}")
                # Arrays already initialized to zeros
                labels_batch[i] = batch_labels[i]
        
        # Create inputs dictionary
        inputs = {
            'frames': frames_batch,
            'pose': pose_batch,
            'emotion': emotion_batch
        }
        
        return inputs, labels_batch
    
    def _augment_frames(self, frames):
        """
        Apply data augmentation to frames
        
        Args:
            frames: Array of shape (sequence_length, height, width, 3)
        
        Returns:
            Augmented frames
        """
        augmented = frames.copy()
        
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            augmented = np.flip(augmented, axis=2)
        
        # Random brightness adjustment (±20%)
        if np.random.random() > 0.5:
            brightness_factor = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random rotation (±10 degrees)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-10, 10)
            augmented = self._rotate_frames(augmented, angle)
        
        # Random zoom (90%-110%)
        if np.random.random() > 0.5:
            zoom_factor = np.random.uniform(0.9, 1.1)
            augmented = self._zoom_frames(augmented, zoom_factor)
        
        return augmented
    
    def _rotate_frames(self, frames, angle):
        """Rotate all frames by given angle"""
        rotated = np.zeros_like(frames)
        h, w = frames.shape[1], frames.shape[2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        for i in range(len(frames)):
            rotated[i] = cv2.warpAffine(
                frames[i],
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
        
        return rotated
    
    def _zoom_frames(self, frames, zoom_factor):
        """Zoom all frames by given factor"""
        zoomed = np.zeros_like(frames)
        h, w = frames.shape[1], frames.shape[2]
        
        for i in range(len(frames)):
            # Calculate crop size
            crop_h = int(h / zoom_factor)
            crop_w = int(w / zoom_factor)
            
            # Calculate crop region
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            
            # Crop and resize
            cropped = frames[i][start_h:start_h+crop_h, start_w:start_w+crop_w]
            zoomed[i] = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed
    
    def get_sample_output(self):
        """
        Get a sample batch for testing
        
        Returns:
            (inputs_dict, labels_array)
        """
        return self.__getitem__(0)


def create_multimodal_generators(
    train_paths,
    train_labels,
    val_paths,
    val_labels,
    batch_size=config.BATCH_SIZE,
    augment_train=True,
    use_advanced_pose=True,
    use_multi_person_emotion=True
):
    """
    Create training and validation multi-modal generators
    
    Args:
        train_paths: List of training video paths
        train_labels: List of training labels
        val_paths: List of validation video paths
        val_labels: List of validation labels
        batch_size: Batch size
        augment_train: Whether to augment training data
        use_advanced_pose: Use advanced pose features
        use_multi_person_emotion: Use multi-person emotion aggregation
    
    Returns:
        (train_generator, val_generator)
    """
    print("\n" + "="*80)
    print("Creating Multi-Modal Data Generators")
    print("="*80)
    
    # Training generator (with augmentation)
    train_gen = MultiModalDataGenerator(
        video_paths=train_paths,
        labels=train_labels,
        batch_size=batch_size,
        augment=augment_train,
        shuffle_data=True,
        use_advanced_pose=use_advanced_pose,
        use_multi_person_emotion=use_multi_person_emotion
    )
    
    # Validation generator (no augmentation)
    val_gen = MultiModalDataGenerator(
        video_paths=val_paths,
        labels=val_labels,
        batch_size=batch_size,
        augment=False,
        shuffle_data=False,
        use_advanced_pose=use_advanced_pose,
        use_multi_person_emotion=use_multi_person_emotion
    )
    
    print("\n" + "="*80)
    print("GENERATOR SUMMARY")
    print("="*80)
    print(f"Training samples: {len(train_paths)}")
    print(f"Training batches: {len(train_gen)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Validation batches: {len(val_gen)}")
    print(f"Batch size: {batch_size}")
    print("="*80 + "\n")
    
    return train_gen, val_gen


if __name__ == "__main__":
    """Test the multi-modal data generator"""
    print("Testing Multi-Modal Data Generator...")
    print("="*60)
    
    # Get sample video paths
    dataset_dir = config.DATASET_DIR
    train_fight_dir = os.path.join(dataset_dir, 'train', 'Fight')
    train_nonfight_dir = os.path.join(dataset_dir, 'train', 'NonFight')
    
    # Collect paths
    fight_videos = [
        os.path.join(train_fight_dir, f)
        for f in os.listdir(train_fight_dir)[:5]
        if f.endswith(('.avi', '.mp4'))
    ]
    
    nonfight_videos = [
        os.path.join(train_nonfight_dir, f)
        for f in os.listdir(train_nonfight_dir)[:5]
        if f.endswith(('.avi', '.mp4'))
    ]
    
    # Create labels
    all_paths = fight_videos + nonfight_videos
    all_labels = [1] * len(fight_videos) + [0] * len(nonfight_videos)
    
    print(f"\nTest dataset:")
    print(f"  Fight videos: {len(fight_videos)}")
    print(f"  Non-fight videos: {len(nonfight_videos)}")
    print(f"  Total: {len(all_paths)}")
    
    # Create generator
    generator = MultiModalDataGenerator(
        video_paths=all_paths,
        labels=all_labels,
        batch_size=2,
        augment=True,
        use_advanced_pose=True,
        use_multi_person_emotion=True
    )
    
    print(f"\nGenerator batches: {len(generator)}")
    
    # Test batch generation
    print("\nGenerating test batch...")
    inputs, labels = generator[0]
    
    print(f"\nBatch shapes:")
    print(f"  Frames: {inputs['frames'].shape}")
    print(f"  Pose: {inputs['pose'].shape}")
    print(f"  Emotion: {inputs['emotion'].shape}")
    print(f"  Labels: {labels.shape}")
    
    print(f"\nValue ranges:")
    print(f"  Frames: [{inputs['frames'].min():.3f}, {inputs['frames'].max():.3f}]")
    print(f"  Pose: [{inputs['pose'].min():.3f}, {inputs['pose'].max():.3f}]")
    print(f"  Emotion: [{inputs['emotion'].min():.3f}, {inputs['emotion'].max():.3f}]")
    
    print("\n✓ Multi-Modal Data Generator test completed!")
