"""
Multi-Modal Violence Detection Model
Combines RGB frames, Pose keypoints, and Emotion features
Expected accuracy improvement: 5-10% over baseline (from ~90% to ~95-97%)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import config


class AttentionLayer(layers.Layer):
    """
    Self-attention mechanism for temporal features
    """
    
    def __init__(self, units=128, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # Compute attention scores
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(ait, axis=1)
        
        # Apply attention weights
        attention_weights = tf.expand_dims(attention_weights, -1)
        weighted_input = x * attention_weights
        
        # Sum over time dimension
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


def build_multimodal_violence_model(
    sequence_length=config.SEQUENCE_LENGTH,
    img_height=config.IMG_HEIGHT,
    img_width=config.IMG_WIDTH,
    img_channels=config.IMG_CHANNELS,
    pose_dim=120,
    emotion_dim=8,
    lstm_units=256,
    use_attention=True,
    fusion_type='adaptive'
):
    """
    Build multi-modal violence detection model
    
    Args:
        sequence_length: Number of frames per video
        img_height: Image height
        img_width: Image width
        img_channels: Number of image channels
        pose_dim: Dimension of pose features (120)
        emotion_dim: Dimension of emotion features (8)
        lstm_units: Number of LSTM units
        use_attention: Whether to use attention mechanism
        fusion_type: 'concat' or 'adaptive'
    
    Returns:
        Multi-modal Keras model
    """
    
    # ===================== INPUT LAYERS =====================
    # RGB frames input
    frame_input = layers.Input(
        shape=(sequence_length, img_height, img_width, img_channels),
        name='frames'
    )
    
    # Pose keypoints input
    pose_input = layers.Input(
        shape=(sequence_length, pose_dim),
        name='pose'
    )
    
    # Emotion features input
    emotion_input = layers.Input(
        shape=(sequence_length, emotion_dim),
        name='emotion'
    )
    
    # ===================== RGB BRANCH (MobileNet + BiLSTM) =====================
    print("Building RGB branch...")
    
    # MobileNetV2 for spatial features
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, img_channels),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    mobilenet.trainable = False  # Freeze initially
    
    # Apply MobileNet to each frame
    x_rgb = layers.TimeDistributed(mobilenet, name='mobilenet_features')(frame_input)
    x_rgb = layers.Dropout(0.3, name='rgb_dropout1')(x_rgb)
    
    # BiLSTM for temporal modeling
    x_rgb = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='rgb_bilstm'
    )(x_rgb)
    x_rgb = layers.Dropout(0.4, name='rgb_dropout2')(x_rgb)
    
    # Attention or pooling
    if use_attention:
        rgb_output = AttentionLayer(units=128, name='rgb_attention')(x_rgb)
    else:
        rgb_output = layers.GlobalAveragePooling1D(name='rgb_gap')(x_rgb)
    
    rgb_output = layers.Dense(256, activation='relu', name='rgb_dense')(rgb_output)
    rgb_output = layers.BatchNormalization(name='rgb_bn')(rgb_output)
    rgb_output = layers.Dropout(0.5, name='rgb_dropout3')(rgb_output)
    
    # ===================== POSE BRANCH (BiLSTM for body movements) =====================
    print("Building Pose branch...")
    
    # Normalize pose features
    x_pose = layers.BatchNormalization(name='pose_bn1')(pose_input)
    
    # Dense projection
    x_pose = layers.TimeDistributed(
        layers.Dense(128, activation='relu'),
        name='pose_dense1'
    )(x_pose)
    x_pose = layers.Dropout(0.3, name='pose_dropout1')(x_pose)
    
    # BiLSTM for temporal pose dynamics
    x_pose = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='pose_bilstm'
    )(x_pose)
    x_pose = layers.Dropout(0.4, name='pose_dropout2')(x_pose)
    
    # Attention or pooling
    if use_attention:
        pose_output = AttentionLayer(units=64, name='pose_attention')(x_pose)
    else:
        pose_output = layers.GlobalAveragePooling1D(name='pose_gap')(x_pose)
    
    pose_output = layers.Dense(128, activation='relu', name='pose_dense2')(pose_output)
    pose_output = layers.BatchNormalization(name='pose_bn2')(pose_output)
    pose_output = layers.Dropout(0.5, name='pose_dropout3')(pose_output)
    
    # ===================== EMOTION BRANCH (BiLSTM for emotion dynamics) =====================
    print("Building Emotion branch...")
    
    # Normalize emotion probabilities
    x_emotion = layers.BatchNormalization(name='emotion_bn1')(emotion_input)
    
    # Dense projection
    x_emotion = layers.TimeDistributed(
        layers.Dense(64, activation='relu'),
        name='emotion_dense1'
    )(x_emotion)
    x_emotion = layers.Dropout(0.3, name='emotion_dropout1')(x_emotion)
    
    # BiLSTM for emotion evolution
    x_emotion = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        name='emotion_bilstm'
    )(x_emotion)
    x_emotion = layers.Dropout(0.4, name='emotion_dropout2')(x_emotion)
    
    # Global pooling
    emotion_output = layers.GlobalAveragePooling1D(name='emotion_gap')(x_emotion)
    
    emotion_output = layers.Dense(64, activation='relu', name='emotion_dense2')(emotion_output)
    emotion_output = layers.BatchNormalization(name='emotion_bn2')(emotion_output)
    emotion_output = layers.Dropout(0.5, name='emotion_dropout3')(emotion_output)
    
    # ===================== FEATURE FUSION =====================
    print(f"Fusing features ({fusion_type} fusion)...")
    
    if fusion_type == 'adaptive':
        # Adaptive fusion with learned weights
        # Project all features to same dimension
        rgb_proj = layers.Dense(256, activation='relu', name='rgb_projection')(rgb_output)
        pose_proj = layers.Dense(256, activation='relu', name='pose_projection')(pose_output)
        emotion_proj = layers.Dense(256, activation='relu', name='emotion_projection')(emotion_output)
        
        # Stack features
        stacked = tf.stack([rgb_proj, pose_proj, emotion_proj], axis=1, name='feature_stack')
        
        # Attention-based fusion (learn importance of each modality)
        fusion_weights = layers.Dense(3, activation='softmax', name='fusion_weights')(
            layers.Flatten()(stacked)
        )
        fusion_weights = layers.Reshape((3, 1), name='fusion_reshape')(fusion_weights)
        
        # Weighted features
        weighted_features = layers.Multiply(name='weighted_multiply')([stacked, fusion_weights])
        fused = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='fusion_sum')(weighted_features)
        
    else:
        # Simple concatenation
        fused = layers.Concatenate(name='feature_fusion')([
            rgb_output,
            pose_output,
            emotion_output
        ])
    
    # ===================== CLASSIFICATION HEAD =====================
    print("Building classification head...")
    
    # Dense layers
    x = layers.Dense(512, activation='relu', name='fc1')(fused)
    x = layers.BatchNormalization(name='fc_bn1')(x)
    x = layers.Dropout(0.5, name='fc_dropout1')(x)
    
    x = layers.Dense(256, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization(name='fc_bn2')(x)
    x = layers.Dropout(0.5, name='fc_dropout2')(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='violence_output')(x)
    
    # ===================== CREATE MODEL =====================
    model = Model(
        inputs=[frame_input, pose_input, emotion_input],
        outputs=output,
        name='multimodal_violence_detector'
    )
    
    return model


def compile_multimodal_model(model, learning_rate=config.LEARNING_RATE):
    """
    Compile multi-modal model with optimizer and metrics
    
    Args:
        model: Keras model
        learning_rate: Learning rate
    
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    metrics = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    print(f"\n{'='*60}")
    print("Model compiled successfully!")
    print(f"Optimizer: Adam (LR={learning_rate})")
    print(f"Loss: binary_crossentropy")
    print(f"Metrics: {[m if isinstance(m, str) else m.name for m in metrics]}")
    print(f"{'='*60}\n")
    
    return model


def get_multimodal_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "="*80)
    print("MULTI-MODAL VIOLENCE DETECTION MODEL")
    print("="*80)
    
    model.summary()
    
    print("\n" + "="*80)
    print("MODEL STATISTICS")
    print("="*80)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    print("\n" + "="*80)
    print("INPUT SPECIFICATIONS")
    print("="*80)
    print(f"RGB Frames: {model.input[0].shape}")
    print(f"Pose Features: {model.input[1].shape}")
    print(f"Emotion Features: {model.input[2].shape}")
    print(f"Output: {model.output.shape}")
    print("="*80 + "\n")


def unfreeze_mobilenet_for_finetuning(model, unfreeze_from_layer=100):
    """
    Unfreeze MobileNet layers for fine-tuning
    
    Args:
        model: Complete multi-modal model
        unfreeze_from_layer: Layer index to start unfreezing from
    """
    # Find TimeDistributed layer containing MobileNet
    for layer in model.layers:
        if 'time_distributed' in layer.name.lower() or 'mobilenet' in layer.name.lower():
            try:
                mobilenet = layer.layer if hasattr(layer, 'layer') else layer
                
                # Unfreeze layers
                mobilenet.trainable = True
                
                # Freeze early layers
                for sub_layer in mobilenet.layers[:unfreeze_from_layer]:
                    sub_layer.trainable = False
                
                trainable_count = sum([1 for l in mobilenet.layers if l.trainable])
                print(f"\n✓ Unfroze MobileNet layers from index {unfreeze_from_layer}")
                print(f"  Total trainable layers: {trainable_count}/{len(mobilenet.layers)}")
                
                break
            except:
                pass


if __name__ == "__main__":
    print("Building Multi-Modal Violence Detection Model...")
    print("="*60)
    
    # Build model
    model = build_multimodal_violence_model(
        sequence_length=20,
        img_height=224,
        img_width=224,
        img_channels=3,
        pose_dim=120,
        emotion_dim=8,
        lstm_units=256,
        use_attention=True,
        fusion_type='adaptive'
    )
    
    # Compile model
    model = compile_multimodal_model(model, learning_rate=1e-4)
    
    # Print summary
    get_multimodal_model_summary(model)
    
    # Test with dummy data
    print("Testing model with random input...")
    import numpy as np
    
    dummy_frames = np.random.random((2, 20, 224, 224, 3))
    dummy_pose = np.random.random((2, 20, 120))
    dummy_emotion = np.random.random((2, 20, 8))
    
    output = model.predict({
        'frames': dummy_frames,
        'pose': dummy_pose,
        'emotion': dummy_emotion
    }, verbose=0)
    
    print(f"\nTest prediction shape: {output.shape}")
    print(f"Test prediction values: {output.flatten()}")
    
    print("\n✓ Model building and testing completed successfully!")
