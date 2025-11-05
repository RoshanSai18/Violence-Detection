"""
Model Architecture: CNN + BiLSTM for Violence Detection
Uses MobileNet for spatial feature extraction and BiLSTM for temporal modeling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large
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


def build_cnn_feature_extractor(input_shape=(224, 224, 3),
                                mobilenet_version='v2',
                                weights='imagenet',
                                freeze_layers=True):
    """
    Build CNN feature extractor using MobileNet
    
    Args:
        input_shape: Input image shape
        mobilenet_version: 'v2', 'v3small', or 'v3large'
        weights: Pre-trained weights ('imagenet' or None)
        freeze_layers: Whether to freeze MobileNet layers
    
    Returns:
        Feature extractor model
    """
    # Select MobileNet version
    if mobilenet_version == 'v2':
        base_model = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling='avg'
        )
    elif mobilenet_version == 'v3small':
        base_model = MobileNetV3Small(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling='avg',
            minimalistic=False
        )
    elif mobilenet_version == 'v3large':
        base_model = MobileNetV3Large(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            pooling='avg',
            minimalistic=False
        )
    else:
        raise ValueError(f"Unknown MobileNet version: {mobilenet_version}")
    
    # Freeze base model layers if specified
    if freeze_layers:
        base_model.trainable = False
        print(f"MobileNet{mobilenet_version} layers frozen")
    else:
        base_model.trainable = True
        print(f"MobileNet{mobilenet_version} layers trainable")
    
    return base_model


def build_violence_detection_model(
    sequence_length=config.SEQUENCE_LENGTH,
    img_height=config.IMG_HEIGHT,
    img_width=config.IMG_WIDTH,
    img_channels=config.IMG_CHANNELS,
    mobilenet_version=config.MOBILENET_VERSION,
    lstm_units=config.LSTM_UNITS,
    lstm_dropout=config.LSTM_DROPOUT,
    lstm_recurrent_dropout=config.LSTM_RECURRENT_DROPOUT,
    use_bidirectional=config.BIDIRECTIONAL,
    use_attention=config.USE_ATTENTION,
    attention_units=config.ATTENTION_UNITS,
    dense_units=config.DENSE_UNITS,
    dropout_rate=config.DROPOUT_RATE,
    use_batch_norm=config.USE_BATCH_NORM,
    num_classes=config.NUM_CLASSES,
    activation=config.ACTIVATION
):
    """
    Build complete CNN + BiLSTM model for violence detection
    
    Args:
        sequence_length: Number of frames per video
        img_height: Image height
        img_width: Image width
        img_channels: Number of image channels
        mobilenet_version: MobileNet version
        lstm_units: Number of LSTM units
        lstm_dropout: LSTM dropout rate
        lstm_recurrent_dropout: LSTM recurrent dropout rate
        use_bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to use attention mechanism
        attention_units: Number of attention units
        dense_units: List of dense layer units
        dropout_rate: Dropout rate for dense layers
        use_batch_norm: Whether to use batch normalization
        num_classes: Number of output classes
        activation: Output activation function
    
    Returns:
        Complete model
    """
    # Input layer
    input_layer = layers.Input(shape=(sequence_length, img_height, img_width, img_channels))
    
    # Build CNN feature extractor
    cnn_base = build_cnn_feature_extractor(
        input_shape=(img_height, img_width, img_channels),
        mobilenet_version=mobilenet_version,
        weights=config.MOBILENET_WEIGHTS,
        freeze_layers=config.FREEZE_LAYERS
    )
    
    # TimeDistributed CNN for processing each frame
    # This applies the CNN to each frame independently
    x = layers.TimeDistributed(cnn_base, name='time_distributed_cnn')(input_layer)
    
    # Add dropout after CNN features
    x = layers.Dropout(0.3, name='cnn_dropout')(x)
    
    # BiLSTM layer for temporal modeling
    if use_bidirectional:
        lstm_layer = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                dropout=lstm_dropout,
                recurrent_dropout=lstm_recurrent_dropout,
                return_sequences=True if use_attention else False,
                name='lstm'
            ),
            name='bidirectional_lstm'
        )
    else:
        lstm_layer = layers.LSTM(
            lstm_units,
            dropout=lstm_dropout,
            recurrent_dropout=lstm_recurrent_dropout,
            return_sequences=True if use_attention else False,
            name='lstm'
        )
    
    x = lstm_layer(x)
    
    # Attention mechanism (optional)
    if use_attention:
        x = AttentionLayer(units=attention_units, name='attention')(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        
        if use_batch_norm:
            x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
        
        x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        output_layer = layers.Dense(1, activation=activation, name='output')(x)
    else:
        # Multi-class classification
        output_layer = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer, name='violence_detection_model')
    
    return model


def unfreeze_mobilenet_layers(model, unfreeze_from_layer=config.UNFREEZE_FROM_LAYER):
    """
    Unfreeze MobileNet layers for fine-tuning
    
    Args:
        model: The complete model
        unfreeze_from_layer: Layer index to start unfreezing from
    """
    # Find TimeDistributed layer containing MobileNet
    for layer in model.layers:
        if 'time_distributed' in layer.name.lower():
            mobilenet = layer.layer
            
            # Unfreeze layers
            mobilenet.trainable = True
            
            # Freeze early layers
            for sub_layer in mobilenet.layers[:unfreeze_from_layer]:
                sub_layer.trainable = False
            
            print(f"Unfroze MobileNet layers from index {unfreeze_from_layer}")
            print(f"Total trainable layers: {sum([1 for l in mobilenet.layers if l.trainable])}")
            
            break


def get_model_summary(model):
    """
    Print model summary and layer information
    
    Args:
        model: Keras model
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    
    model.summary()
    
    print("\n" + "="*80)
    print("LAYER DETAILS")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    for layer in model.layers:
        params = layer.count_params()
        total_params += params
        
        if layer.trainable:
            trainable_params += params
        
        print(f"{layer.name:30s} | Trainable: {layer.trainable:5} | Params: {params:,}")
    
    print("="*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*80 + "\n")


def compile_model(model, 
                 optimizer=config.OPTIMIZER,
                 learning_rate=config.LEARNING_RATE,
                 loss=config.LOSS):
    """
    Compile the model with optimizer and loss
    
    Args:
        model: Keras model
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        loss: Loss function
    
    Returns:
        Compiled model
    """
    # Create optimizer
    if optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=config.ADAM_BETA_1,
            beta_2=config.ADAM_BETA_2
        )
    elif optimizer.lower() == 'sgd':
        opt = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=config.SGD_MOMENTUM,
            nesterov=config.SGD_NESTEROV
        )
    elif optimizer.lower() == 'rmsprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Define metrics
    metrics = [
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
    
    # Compile model
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    
    print(f"\nModel compiled with {optimizer.upper()} optimizer (LR={learning_rate})")
    print(f"Loss: {loss}")
    print(f"Metrics: {[m if isinstance(m, str) else m.name for m in metrics]}")
    
    return model


if __name__ == "__main__":
    # Test model building
    print("Building violence detection model...")
    
    # Build model
    model = build_violence_detection_model()
    
    # Print model summary
    get_model_summary(model)
    
    # Compile model
    model = compile_model(model)
    
    # Test model input/output
    print("\nTesting model with random input...")
    import numpy as np
    
    dummy_input = np.random.random((2, config.SEQUENCE_LENGTH, 
                                   config.IMG_HEIGHT, config.IMG_WIDTH, 
                                   config.IMG_CHANNELS))
    
    output = model.predict(dummy_input, verbose=0)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    print("\nModel building test completed successfully!")
