"""
Real-Time Violence Detection from Webcam or Video Files
Uses trained multi-modal model (RGB + Pose + Emotion)

Usage:
    # Webcam detection
    python realtime_webcam_detection.py --model best_multimodal_model.h5 --source webcam
    
    # Video file detection
    python realtime_webcam_detection.py --model best_multimodal_model.h5 --source video.mp4
    
    # Save output video
    python realtime_webcam_detection.py --model best_multimodal_model.h5 --source video.mp4 --output result.mp4
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import mediapipe as mp
from deepface import DeepFace
import time
import argparse
import os


class AttentionLayer(keras.layers.Layer):
    """Custom Attention Layer (required for loading the trained model)"""
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], self.units),
                                initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(self.units,),
                                initializer='zeros', trainable=True)
        self.u = self.add_weight(name='attention_context', shape=(self.units,),
                                initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)
        weighted_input = x * attention_weights
        return tf.reduce_sum(weighted_input, axis=1)
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class RealtimeViolenceDetector:
    def __init__(self, model_path, sequence_length=20, img_size=(224, 224)):
        """
        Initialize real-time violence detector
        
        Args:
            model_path: Path to saved .h5 model
            sequence_length: Number of frames to analyze (default: 20)
            img_size: Frame resize dimensions (default: 224x224)
        """
        print("🔄 Loading model...")
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print("✅ Model loaded successfully!")
        
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        # Frame buffers (FIFO queues)
        self.frame_buffer = deque(maxlen=sequence_length)
        self.pose_buffer = deque(maxlen=sequence_length)
        self.emotion_buffer = deque(maxlen=sequence_length)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        
    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
    
    def extract_pose_features(self, frame):
        """Extract 120-dim pose features from frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Keypoints (33 × 3 = 99)
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.visibility])
            
            # Joint angles (6 angles)
            left_elbow = self._calculate_angle(
                [landmarks[11].x, landmarks[11].y],
                [landmarks[13].x, landmarks[13].y],
                [landmarks[15].x, landmarks[15].y]
            )
            right_elbow = self._calculate_angle(
                [landmarks[12].x, landmarks[12].y],
                [landmarks[14].x, landmarks[14].y],
                [landmarks[16].x, landmarks[16].y]
            )
            left_knee = self._calculate_angle(
                [landmarks[23].x, landmarks[23].y],
                [landmarks[25].x, landmarks[25].y],
                [landmarks[27].x, landmarks[27].y]
            )
            right_knee = self._calculate_angle(
                [landmarks[24].x, landmarks[24].y],
                [landmarks[26].x, landmarks[26].y],
                [landmarks[28].x, landmarks[28].y]
            )
            left_shoulder = self._calculate_angle(
                [landmarks[13].x, landmarks[13].y],
                [landmarks[11].x, landmarks[11].y],
                [landmarks[23].x, landmarks[23].y]
            )
            right_shoulder = self._calculate_angle(
                [landmarks[14].x, landmarks[14].y],
                [landmarks[12].x, landmarks[12].y],
                [landmarks[24].x, landmarks[24].y]
            )
            
            # Body metrics
            hand_distance = np.sqrt(
                (landmarks[15].x - landmarks[16].x)**2 +
                (landmarks[15].y - landmarks[16].y)**2
            )
            foot_elevation = abs(landmarks[27].y - landmarks[28].y)
            torso_bend_left = abs(landmarks[11].y - landmarks[23].y)
            torso_bend_right = abs(landmarks[12].y - landmarks[24].y)
            
            center_x = np.mean([landmarks[11].x, landmarks[12].x, landmarks[23].x, landmarks[24].x])
            center_y = np.mean([landmarks[11].y, landmarks[12].y, landmarks[23].y, landmarks[24].y])
            head_offset_x = landmarks[0].x - center_x
            head_offset_y = landmarks[0].y - center_y
            
            # Combine all features (120-dim)
            features = keypoints + [
                left_elbow, right_elbow, left_knee, right_knee,
                left_shoulder, right_shoulder, hand_distance, foot_elevation,
                torso_bend_left, torso_bend_right, center_x, center_y,
                head_offset_x, head_offset_y,
                left_elbow/180, right_elbow/180, left_knee/180, right_knee/180,
                left_shoulder/180, right_shoulder/180, hand_distance * 10
            ]
            
            return np.array(features[:120]), results.pose_landmarks
        else:
            return np.zeros(120), None
    
    def extract_emotion_features(self, frame):
        """Extract 8-dim emotion features from frame"""
        try:
            analysis = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            emotions = analysis['emotion']
            emotion_vector = [
                emotions.get('angry', 0.0) / 100.0,
                emotions.get('disgust', 0.0) / 100.0,
                emotions.get('fear', 0.0) / 100.0,
                emotions.get('happy', 0.0) / 100.0,
                emotions.get('sad', 0.0) / 100.0,
                emotions.get('surprise', 0.0) / 100.0,
                emotions.get('neutral', 0.0) / 100.0
            ]
            variance = np.var(emotion_vector)
            return np.array(emotion_vector + [variance])
        except:
            return np.zeros(8)
    
    def process_frame(self, frame):
        """Process single frame and add to buffers"""
        # Resize frame
        resized = cv2.resize(frame, self.img_size)
        normalized = resized.astype(np.float32) / 255.0
        
        # Extract features
        pose_features, pose_landmarks = self.extract_pose_features(frame)
        emotion_features = self.extract_emotion_features(frame)
        
        # Add to buffers
        self.frame_buffer.append(normalized)
        self.pose_buffer.append(pose_features)
        self.emotion_buffer.append(emotion_features)
        
        return pose_landmarks
    
    def predict(self):
        """Make prediction on current buffer"""
        if len(self.frame_buffer) < self.sequence_length:
            return None  # Not enough frames yet
        
        # Prepare inputs
        frames = np.array(list(self.frame_buffer))
        poses = np.array(list(self.pose_buffer))
        emotions = np.array(list(self.emotion_buffer))
        
        # Add batch dimension
        frames = np.expand_dims(frames, axis=0)
        poses = np.expand_dims(poses, axis=0)
        emotions = np.expand_dims(emotions, axis=0)
        
        # Predict
        prediction = self.model.predict(
            [frames, poses, emotions],
            verbose=0
        )[0][0]
        
        # Smooth prediction
        self.prediction_buffer.append(prediction)
        smoothed_prediction = np.mean(self.prediction_buffer)
        
        return smoothed_prediction
    
    def run_webcam(self, camera_id=0, show_pose=True):
        """
        Run real-time detection on webcam
        
        Args:
            camera_id: Webcam device ID (0 for default)
            show_pose: Whether to draw pose landmarks
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("🎥 Starting webcam... Press 'q' to quit")
        
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            pose_landmarks = self.process_frame(frame)
            
            # Make prediction
            prediction = self.predict()
            
            # Draw pose landmarks
            if show_pose and pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Display prediction
            if prediction is not None:
                label = "VIOLENCE DETECTED" if prediction > 0.5 else "Normal"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
                
                # Draw background rectangle
                cv2.rectangle(frame, (10, 10), (500, 100), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(frame, label, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw probability bar
                bar_width = int(400 * prediction)
                cv2.rectangle(frame, (10, 110), (410, 130), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, 110), (10 + bar_width, 130), color, -1)
                cv2.putText(frame, f"{prediction:.2%}", (420, 125),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {current_fps}", (frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Real-Time Violence Detection', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam stopped")
    
    def run_video_file(self, video_path, output_path=None):
        """
        Run detection on video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"🎥 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        frame_count = 0
        violence_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            pose_landmarks = self.process_frame(frame)
            prediction = self.predict()
            
            # Draw pose
            if pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
            
            # Display prediction
            if prediction is not None:
                label = "VIOLENCE" if prediction > 0.5 else "Normal"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)
                
                if prediction > 0.5:
                    violence_frames += 1
                
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
                cv2.putText(frame, label, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, f"Conf: {confidence:.2%}", (20, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Progress
            progress = (frame_count / total_frames) * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", (width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Display (optional - comment out for faster processing)
            cv2.imshow('Processing...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Summary
        violence_percentage = (violence_frames / frame_count) * 100
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Total frames: {frame_count}")
        print(f"Violence detected in: {violence_frames} frames ({violence_percentage:.2f}%)")
        if output_path:
            print(f"Output saved to: {output_path}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Real-Time Violence Detection')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--source', type=str, default='webcam', 
                       help='Source: "webcam" or path to video file')
    parser.add_argument('--output', type=str, default=None, 
                       help='Path to save output video (for video files only)')
    parser.add_argument('--camera-id', type=int, default=0, 
                       help='Camera device ID (default: 0)')
    parser.add_argument('--no-pose', action='store_true',
                       help='Hide pose landmarks overlay')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Initialize detector
    print("="*70)
    print("🎯 REAL-TIME VIOLENCE DETECTION SYSTEM")
    print("="*70)
    detector = RealtimeViolenceDetector(args.model)
    
    # Run detection
    if args.source == 'webcam':
        print("\n📹 Running webcam detection...")
        detector.run_webcam(camera_id=args.camera_id, show_pose=not args.no_pose)
    else:
        if not os.path.exists(args.source):
            print(f"❌ Error: Video file not found: {args.source}")
            return
        print(f"\n📹 Running video file detection...")
        detector.run_video_file(args.source, args.output)


if __name__ == "__main__":
    main()
