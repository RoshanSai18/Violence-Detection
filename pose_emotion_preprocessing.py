"""
Enhanced Preprocessing with Pose Detection and Emotion Recognition
Increases violence detection accuracy by adding human pose and facial emotion features
"""

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import os
import config
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class PoseEmotionPreprocessor:
    """
    Advanced preprocessor that extracts pose keypoints and facial emotions
    alongside video frames to enhance violence detection
    
    Expected accuracy improvement: 5-10% over baseline RGB-only model
    """
    
    def __init__(self, sequence_length=config.SEQUENCE_LENGTH, 
                 img_height=config.IMG_HEIGHT, 
                 img_width=config.IMG_WIDTH,
                 enable_pose=True,
                 enable_emotion=True):
        self.sequence_length = sequence_length
        self.img_height = img_height
        self.img_width = img_width
        self.enable_pose = enable_pose
        self.enable_emotion = enable_emotion
        
        # Initialize MediaPipe Pose
        if self.enable_pose:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("✓ MediaPipe Pose detector initialized")
        
        # Emotion detection setup
        if self.enable_emotion:
            # DeepFace supports multiple backends: opencv, ssd, dlib, mtcnn, retinaface
            self.emotion_backend = 'opencv'  # Fastest for real-time
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            print("✓ DeepFace emotion detector initialized")
    
    def extract_pose_keypoints(self, frame):
        """
        Extract 33 pose landmarks using MediaPipe
        
        Returns:
            pose_features: 99-dimensional vector (33 landmarks × 3 coordinates)
                          or zeros if no pose detected
        """
        try:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose_detector.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract normalized coordinates (x, y, z) for 33 landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                return np.array(landmarks, dtype=np.float32)  # Shape: (99,)
            else:
                # No pose detected - return zeros
                return np.zeros(99, dtype=np.float32)
                
        except Exception as e:
            # print(f"Pose extraction error: {e}")
            return np.zeros(99, dtype=np.float32)
    
    def extract_pose_features_advanced(self, frame):
        """
        Extract advanced pose features including:
        - Joint angles (elbows, knees, shoulders)
        - Body pose (standing, sitting, lying)
        - Movement intensity
        - Limb positions
        
        Returns:
            enhanced_features: 120-dimensional feature vector
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                features = []
                
                # Basic keypoints (99 features)
                for lm in landmarks:
                    features.extend([lm.x, lm.y, lm.z])
                
                # Calculate joint angles (more discriminative for violence)
                # Right arm angle (shoulder-elbow-wrist)
                right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
                right_elbow = np.array([landmarks[14].x, landmarks[14].y])
                right_wrist = np.array([landmarks[16].x, landmarks[16].y])
                right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Left arm angle
                left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
                left_elbow = np.array([landmarks[13].x, landmarks[13].y])
                left_wrist = np.array([landmarks[15].x, landmarks[15].y])
                left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Leg angles (kicks detection)
                right_hip = np.array([landmarks[24].x, landmarks[24].y])
                right_knee = np.array([landmarks[26].x, landmarks[26].y])
                right_ankle = np.array([landmarks[28].x, landmarks[28].y])
                right_leg_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
                
                left_hip = np.array([landmarks[23].x, landmarks[23].y])
                left_knee = np.array([landmarks[25].x, landmarks[25].y])
                left_ankle = np.array([landmarks[27].x, landmarks[27].y])
                left_leg_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
                
                # Body orientation (important for fighting stance)
                shoulder_hip_angle = self._calculate_angle(
                    np.array([landmarks[11].x, landmarks[11].y]),  # Left shoulder
                    np.array([landmarks[23].x, landmarks[23].y]),  # Left hip
                    np.array([landmarks[25].x, landmarks[25].y])   # Left knee
                )
                
                # Distance features (punching/kicking range)
                # Hand-to-hand distance
                hand_distance = np.linalg.norm(
                    np.array([landmarks[16].x, landmarks[16].y]) - 
                    np.array([landmarks[15].x, landmarks[15].y])
                )
                
                # Foot elevation (kicks)
                right_foot_height = landmarks[28].y
                left_foot_height = landmarks[27].y
                
                # Torso bend (dodging, falling)
                nose = np.array([landmarks[0].x, landmarks[0].y])
                mid_hip = np.array([
                    (landmarks[23].x + landmarks[24].x) / 2,
                    (landmarks[23].y + landmarks[24].y) / 2
                ])
                torso_angle = np.arctan2(nose[1] - mid_hip[1], nose[0] - mid_hip[0])
                
                # Arm spread (aggression indicator)
                arm_spread = np.linalg.norm(
                    np.array([landmarks[16].x, landmarks[16].y]) - 
                    np.array([landmarks[15].x, landmarks[15].y])
                )
                
                # Head position relative to body
                head_body_distance = np.linalg.norm(nose - mid_hip)
                
                # Append calculated features
                additional_features = [
                    right_arm_angle, left_arm_angle,
                    right_leg_angle, left_leg_angle,
                    shoulder_hip_angle, hand_distance,
                    right_foot_height, left_foot_height,
                    torso_angle, arm_spread, head_body_distance
                ]
                
                features.extend(additional_features)
                
                # Pad to fixed size (120 features)
                while len(features) < 120:
                    features.append(0.0)
                
                return np.array(features[:120], dtype=np.float32)
            else:
                return np.zeros(120, dtype=np.float32)
                
        except Exception as e:
            # print(f"Advanced pose extraction error: {e}")
            return np.zeros(120, dtype=np.float32)
    
    def _calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points (in degrees)"""
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def extract_emotion_features(self, frame):
        """
        Extract facial emotion using DeepFace
        
        Returns:
            emotion_vector: 7-dimensional probability distribution
                           [angry, disgust, fear, happy, sad, surprise, neutral]
        """
        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (224, 224))
            
            # Analyze emotions
            analysis = DeepFace.analyze(
                small_frame,
                actions=['emotion'],
                enforce_detection=False,  # Continue even if face not detected
                detector_backend=self.emotion_backend,
                silent=True
            )
            
            # Extract emotion probabilities
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            emotions = analysis['emotion']
            emotion_vector = [
                emotions.get('angry', 0),
                emotions.get('disgust', 0),
                emotions.get('fear', 0),
                emotions.get('happy', 0),
                emotions.get('sad', 0),
                emotions.get('surprise', 0),
                emotions.get('neutral', 0)
            ]
            
            # Normalize to sum=1
            emotion_sum = sum(emotion_vector) + 1e-6
            emotion_vector = [e / emotion_sum for e in emotion_vector]
            
            return np.array(emotion_vector, dtype=np.float32)
            
        except Exception as e:
            # Return neutral emotion if detection fails
            return np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    
    def extract_multi_person_emotions(self, frame):
        """
        Detect emotions for multiple people in the frame
        Aggregates emotions to capture group dynamics
        
        Returns:
            aggregated_emotion: 8-dimensional vector (mean emotions + variance)
        """
        try:
            # Detect all faces
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            all_emotions = []
            
            for (x, y, w, h) in faces[:5]:  # Limit to 5 faces for performance
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    emotion = self.extract_emotion_features(face_roi)
                    all_emotions.append(emotion)
            
            if all_emotions:
                # Mean emotion (average mood)
                mean_emotion = np.mean(all_emotions, axis=0)
                
                # Emotion variance (high variance = conflicting emotions = possible violence)
                emotion_variance = np.var(all_emotions, axis=0).mean()
                
                # Concatenate mean + variance
                features = np.concatenate([mean_emotion, [emotion_variance]])
                return features.astype(np.float32)
            else:
                # No faces detected
                return np.zeros(8, dtype=np.float32)
                
        except Exception as e:
            return np.zeros(8, dtype=np.float32)
    
    def extract_enhanced_features(self, video_path):
        """
        Extract RGB frames + Pose + Emotion features from video
        
        Returns:
            features_dict: {
                'frames': (seq_len, H, W, 3),
                'pose': (seq_len, 120),
                'emotion': (seq_len, 8)
            }
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frame indices uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        
        frames = []
        pose_features = []
        emotion_features = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame_resized = cv2.resize(frame, (self.img_width, self.img_height))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Extract pose
                if self.enable_pose:
                    pose = self.extract_pose_features_advanced(frame)
                    pose_features.append(pose)
                
                # Extract emotion
                if self.enable_emotion:
                    emotion = self.extract_multi_person_emotions(frame)
                    emotion_features.append(emotion)
            else:
                # Use last valid frame or zeros
                if frames:
                    frames.append(frames[-1])
                    if self.enable_pose:
                        pose_features.append(pose_features[-1])
                    if self.enable_emotion:
                        emotion_features.append(emotion_features[-1])
                else:
                    frames.append(np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
                    if self.enable_pose:
                        pose_features.append(np.zeros(120, dtype=np.float32))
                    if self.enable_emotion:
                        emotion_features.append(np.zeros(8, dtype=np.float32))
        
        cap.release()
        
        # Ensure correct sequence length
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8))
            if self.enable_pose:
                pose_features.append(pose_features[-1] if pose_features else np.zeros(120, dtype=np.float32))
            if self.enable_emotion:
                emotion_features.append(emotion_features[-1] if emotion_features else np.zeros(8, dtype=np.float32))
        
        features_dict = {
            'frames': np.array(frames[:self.sequence_length]),
            'pose': np.array(pose_features[:self.sequence_length]) if self.enable_pose else None,
            'emotion': np.array(emotion_features[:self.sequence_length]) if self.enable_emotion else None
        }
        
        return features_dict
    
    def normalize_frames(self, frames, mode='tf'):
        """
        Normalize frame pixel values
        
        Args:
            frames: numpy array of frames
            mode: 'tf' - normalize to [-1, 1]
                 'torch' - normalize to [0, 1]
        
        Returns:
            Normalized frames
        """
        frames = frames.astype(np.float32)
        
        if mode == 'tf':
            frames = (frames / 127.5) - 1.0
        else:
            frames = frames / 255.0
        
        return frames
    
    def compute_pose_velocity(self, pose_sequence):
        """
        Calculate velocity of key joints (useful for detecting sudden movements)
        
        Args:
            pose_sequence: (seq_len, 120) pose features
        
        Returns:
            velocity_features: (seq_len, 120) velocities
        """
        velocities = []
        
        for i in range(len(pose_sequence)):
            if i == 0:
                velocity = np.zeros_like(pose_sequence[0])
            else:
                velocity = pose_sequence[i] - pose_sequence[i-1]
            velocities.append(velocity)
        
        return np.array(velocities)
    
    def visualize_pose_emotion(self, frame, pose_landmarks=None, emotion=None):
        """
        Overlay pose skeleton and emotion on frame for visualization
        
        Args:
            frame: Input frame
            pose_landmarks: MediaPipe pose landmarks
            emotion: Emotion dictionary
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw pose skeleton
        if pose_landmarks and self.enable_pose:
            self.mp_drawing.draw_landmarks(
                annotated,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2
                )
            )
        
        # Draw emotion text
        if emotion and self.enable_emotion:
            if isinstance(emotion, np.ndarray):
                emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                dominant_idx = np.argmax(emotion[:7])
                dominant_emotion = emotion_labels[dominant_idx]
                confidence = emotion[dominant_idx]
            else:
                dominant_emotion = max(emotion, key=emotion.get)
                confidence = emotion[dominant_emotion]
            
            text = f"{dominant_emotion}: {confidence:.2f}"
            cv2.putText(annotated, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return annotated


def test_pose_emotion_extraction():
    """Test the enhanced preprocessing pipeline"""
    print("\n" + "="*60)
    print("Testing Pose + Emotion Extraction Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = PoseEmotionPreprocessor(
        sequence_length=20,
        enable_pose=True,
        enable_emotion=True
    )
    
    # Test on first video
    from data_preprocessing import load_dataset_paths
    
    train_paths, train_labels = load_dataset_paths(config.TRAIN_DIR)
    
    if train_paths:
        test_video = train_paths[0]
        print(f"\nProcessing video: {os.path.basename(test_video)}")
        
        # Extract features
        features = preprocessor.extract_enhanced_features(test_video)
        
        if features:
            print(f"\n✓ Extracted features:")
            print(f"  Frames shape: {features['frames'].shape}")
            if features['pose'] is not None:
                print(f"  Pose shape: {features['pose'].shape}")
                print(f"  Pose sample (first frame, first 10): {features['pose'][0][:10]}")
            if features['emotion'] is not None:
                print(f"  Emotion shape: {features['emotion'].shape}")
                print(f"  Emotion sample (first frame): {features['emotion'][0]}")
            
            print("\n✓ Feature extraction successful!")
        else:
            print("\n✗ Feature extraction failed!")
    else:
        print("No videos found in training directory!")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    test_pose_emotion_extraction()
