"""
Training Script for Violence Detection Model
Handles complete training pipeline with callbacks and monitoring
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import json

import config
from model import build_violence_detection_model, compile_model, get_model_summary, unfreeze_mobilenet_layers
from data_generator import create_data_generators
from data_preprocessing import calculate_class_weights, load_dataset_paths


def set_seed(seed=config.RANDOM_SEED):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def setup_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Mixed precision training (optional)
            if config.USE_MIXED_PRECISION:
                policy = keras.mixed_precision.Policy('mixed_float16')
                keras.mixed_precision.set_global_policy(policy)
                print("Mixed precision training enabled")
        
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Training on CPU.")


def create_callbacks(model_name=config.MODEL_NAME):
    """
    Create training callbacks
    
    Returns:
        List of callbacks
    """
    callbacks_list = []
    
    # Model checkpoint - save best model
    checkpoint_path = os.path.join(config.MODEL_DIR, f'{model_name}_best.h5')
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config.CHECKPOINT_MONITOR,
        mode=config.CHECKPOINT_MODE,
        save_best_only=config.CHECKPOINT_SAVE_BEST_ONLY,
        save_weights_only=config.CHECKPOINT_SAVE_WEIGHTS_ONLY,
        verbose=1
    )
    callbacks_list.append(checkpoint)
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.EARLY_STOPPING_MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=True,
        verbose=1
    )
    callbacks_list.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=config.REDUCE_LR_MONITOR,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.REDUCE_LR_MIN_LR,
        verbose=1
    )
    callbacks_list.append(reduce_lr)
    
    # TensorBoard
    if config.USE_TENSORBOARD:
        log_dir = os.path.join(config.LOGS_DIR, 
                              f'{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
        callbacks_list.append(tensorboard)
        print(f"TensorBoard logs: {log_dir}")
    
    # CSV Logger
    csv_path = os.path.join(config.LOGS_DIR, f'{model_name}_training.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_path, append=True)
    callbacks_list.append(csv_logger)
    
    # Custom callback for printing metrics
    class MetricsCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nEpoch {epoch + 1} metrics:")
            print(f"  Train Loss: {logs.get('loss', 0):.4f} | Train Acc: {logs.get('accuracy', 0):.4f}")
            print(f"  Val Loss: {logs.get('val_loss', 0):.4f} | Val Acc: {logs.get('val_accuracy', 0):.4f}")
            print(f"  Val Precision: {logs.get('val_precision', 0):.4f} | Val Recall: {logs.get('val_recall', 0):.4f}")
            print(f"  Val AUC: {logs.get('val_auc', 0):.4f}")
    
    callbacks_list.append(MetricsCallback())
    
    return callbacks_list


def train_model(resume_from=None, fine_tune=False):
    """
    Main training function
    
    Args:
        resume_from: Path to model checkpoint to resume from
        fine_tune: Whether to perform fine-tuning (unfreeze MobileNet layers)
    
    Returns:
        Trained model and training history
    """
    print("\n" + "="*80)
    print("VIOLENCE DETECTION MODEL TRAINING")
    print("="*80 + "\n")
    
    # Set random seed
    set_seed()
    
    # Setup GPU
    setup_gpu()
    
    # Create data generators
    print("\nCreating data generators...")
    train_generator, val_generator = create_data_generators(
        batch_size=config.BATCH_SIZE,
        augment_train=True,
        balanced=False
    )
    
    # Calculate class weights
    if config.USE_CLASS_WEIGHTS:
        print("\nCalculating class weights...")
        train_paths, train_labels = load_dataset_paths(config.TRAIN_DIR)
        class_weights = calculate_class_weights(train_labels)
    else:
        class_weights = None
    
    # Build or load model
    if resume_from and os.path.exists(resume_from):
        print(f"\nLoading model from {resume_from}...")
        model = keras.models.load_model(resume_from, custom_objects={'AttentionLayer': AttentionLayer})
        print("Model loaded successfully!")
    else:
        print("\nBuilding new model...")
        model = build_violence_detection_model()
        get_model_summary(model)
        
        # Compile model
        model = compile_model(
            model,
            optimizer=config.OPTIMIZER,
            learning_rate=config.LEARNING_RATE,
            loss=config.LOSS
        )
    
    # Fine-tuning phase
    if fine_tune:
        print("\n" + "="*80)
        print("FINE-TUNING MODE: Unfreezing MobileNet layers")
        print("="*80)
        
        unfreeze_mobilenet_layers(model, config.UNFREEZE_FROM_LAYER)
        
        # Recompile with lower learning rate for fine-tuning
        model = compile_model(
            model,
            optimizer='sgd',
            learning_rate=config.LEARNING_RATE / 10,
            loss=config.LOSS
        )
    
    # Create callbacks
    callbacks = create_callbacks()
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Optimizer: {config.OPTIMIZER}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Class weights: {class_weights}")
    print("="*80 + "\n")
    
    # Train model
    print("Starting training...\n")
    
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=config.EPOCHS,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=config.VERBOSE,
            initial_epoch=config.INITIAL_EPOCH
        )
        
        print("\nTraining completed successfully!")
        
        # Save final model
        final_model_path = os.path.join(config.MODEL_DIR, f'{config.MODEL_NAME}_final.h5')
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save training history
        history_path = os.path.join(config.LOGS_DIR, f'{config.MODEL_NAME}_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            json.dump(history_dict, f, indent=4)
        print(f"Training history saved to: {history_path}")
        
        return model, history
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        
        # Save interrupted model
        interrupt_path = os.path.join(config.MODEL_DIR, f'{config.MODEL_NAME}_interrupted.h5')
        model.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
        
        return model, None
    
    except Exception as e:
        print(f"\n\nTraining failed with error: {str(e)}")
        raise


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Training history object
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    if history is None:
        print("No training history to plot.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision Over Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall Over Epochs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Violence Detection Model')
    parser.add_argument('--resume', type=str, default=None, help='Path to model checkpoint to resume from')
    parser.add_argument('--fine-tune', action='store_true', help='Enable fine-tuning mode')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Import AttentionLayer for model loading
    from model import AttentionLayer
    
    # Train model
    model, history = train_model(
        resume_from=args.resume,
        fine_tune=args.fine_tune
    )
    
    # Plot training history
    if history:
        plot_path = os.path.join(config.PLOTS_DIR, f'{config.MODEL_NAME}_training_history.png')
        plot_training_history(history, save_path=plot_path)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
