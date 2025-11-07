"""
Training Script for Multi-Modal Violence Detection Model
Combines RGB frames, Pose features, and Emotion features
"""

import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from model_multimodal import (
    build_multimodal_violence_model,
    compile_multimodal_model,
    get_multimodal_model_summary,
    unfreeze_mobilenet_for_finetuning
)
from data_generator_multimodal import create_multimodal_generators
from sklearn.model_selection import train_test_split


def load_dataset_paths(dataset_dir):
    """
    Load video paths and labels from dataset directory
    
    Args:
        dataset_dir: Root directory of RWF-2000 dataset
    
    Returns:
        train_paths, train_labels, val_paths, val_labels
    """
    print("\n" + "="*80)
    print("Loading Dataset Paths")
    print("="*80)
    
    # Training data
    train_fight_dir = os.path.join(dataset_dir, 'train', 'Fight')
    train_nonfight_dir = os.path.join(dataset_dir, 'train', 'NonFight')
    
    # Validation data
    val_fight_dir = os.path.join(dataset_dir, 'val', 'Fight')
    val_nonfight_dir = os.path.join(dataset_dir, 'val', 'NonFight')
    
    # Collect training paths
    train_fight = [
        os.path.join(train_fight_dir, f)
        for f in os.listdir(train_fight_dir)
        if f.endswith(('.avi', '.mp4'))
    ]
    train_nonfight = [
        os.path.join(train_nonfight_dir, f)
        for f in os.listdir(train_nonfight_dir)
        if f.endswith(('.avi', '.mp4'))
    ]
    
    # Collect validation paths
    val_fight = [
        os.path.join(val_fight_dir, f)
        for f in os.listdir(val_fight_dir)
        if f.endswith(('.avi', '.mp4'))
    ]
    val_nonfight = [
        os.path.join(val_nonfight_dir, f)
        for f in os.listdir(val_nonfight_dir)
        if f.endswith(('.avi', '.mp4'))
    ]
    
    # Create labels (1 = Fight, 0 = Non-Fight)
    train_paths = train_fight + train_nonfight
    train_labels = [1] * len(train_fight) + [0] * len(train_nonfight)
    
    val_paths = val_fight + val_nonfight
    val_labels = [1] * len(val_fight) + [0] * len(val_nonfight)
    
    print(f"\nTraining Set:")
    print(f"  Fight videos: {len(train_fight)}")
    print(f"  Non-Fight videos: {len(train_nonfight)}")
    print(f"  Total: {len(train_paths)}")
    print(f"  Class balance: {len(train_fight)/len(train_paths)*100:.1f}% / {len(train_nonfight)/len(train_paths)*100:.1f}%")
    
    print(f"\nValidation Set:")
    print(f"  Fight videos: {len(val_fight)}")
    print(f"  Non-Fight videos: {len(val_nonfight)}")
    print(f"  Total: {len(val_paths)}")
    print(f"  Class balance: {len(val_fight)/len(val_paths)*100:.1f}% / {len(val_nonfight)/len(val_paths)*100:.1f}%")
    
    print("="*80 + "\n")
    
    return train_paths, train_labels, val_paths, val_labels


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced dataset
    
    Args:
        labels: List of labels
    
    Returns:
        Dictionary of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    class_weights = {
        int(cls): total / (len(unique) * count)
        for cls, count in zip(unique, counts)
    }
    
    print(f"\nClass weights: {class_weights}")
    
    return class_weights


def setup_callbacks(model_save_dir):
    """
    Setup training callbacks
    
    Args:
        model_save_dir: Directory to save model checkpoints
    
    Returns:
        List of callbacks
    """
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Model checkpoint - save best model
    checkpoint_path = os.path.join(model_save_dir, 'best_multimodal_model.h5')
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
    
    # TensorBoard logging
    log_dir = os.path.join(model_save_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # CSV logger
    csv_path = os.path.join(model_save_dir, 'training_history.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_path, append=True)
    
    callbacks = [checkpoint, early_stop, reduce_lr, tensorboard, csv_logger]
    
    print(f"\n{'='*60}")
    print("Callbacks Setup:")
    print(f"  - Model checkpoint: {checkpoint_path}")
    print(f"  - Early stopping: patience=8")
    print(f"  - Reduce LR: factor=0.5, patience=4")
    print(f"  - TensorBoard logs: {log_dir}")
    print(f"  - CSV logs: {csv_path}")
    print(f"{'='*60}\n")
    
    return callbacks


def train_multimodal_model(
    dataset_dir=config.DATASET_DIR,
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    use_advanced_pose=True,
    use_multi_person_emotion=True,
    fusion_type='adaptive',
    fine_tune_epoch=None
):
    """
    Complete training pipeline for multi-modal violence detection
    
    Args:
        dataset_dir: Path to RWF-2000 dataset
        batch_size: Batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        use_advanced_pose: Use advanced pose features (120-dim)
        use_multi_person_emotion: Use multi-person emotion aggregation
        fusion_type: 'concat' or 'adaptive'
        fine_tune_epoch: Epoch to start fine-tuning MobileNet (None = no fine-tuning)
    
    Returns:
        Trained model and training history
    """
    print("\n" + "="*80)
    print("MULTI-MODAL VIOLENCE DETECTION - TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Advanced pose: {use_advanced_pose}")
    print(f"Multi-person emotion: {use_multi_person_emotion}")
    print(f"Fusion type: {fusion_type}")
    print("="*80 + "\n")
    
    # 1. Load dataset paths
    train_paths, train_labels, val_paths, val_labels = load_dataset_paths(dataset_dir)
    
    # 2. Compute class weights
    class_weights = compute_class_weights(train_labels)
    
    # 3. Create data generators
    train_gen, val_gen = create_multimodal_generators(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        batch_size=batch_size,
        augment_train=True,
        use_advanced_pose=use_advanced_pose,
        use_multi_person_emotion=use_multi_person_emotion
    )
    
    # 4. Build model
    print("\n" + "="*80)
    print("Building Multi-Modal Model")
    print("="*80)
    
    model = build_multimodal_violence_model(
        sequence_length=config.SEQUENCE_LENGTH,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        img_channels=config.IMG_CHANNELS,
        pose_dim=120 if use_advanced_pose else 99,
        emotion_dim=8,
        lstm_units=256,
        use_attention=True,
        fusion_type=fusion_type
    )
    
    # 5. Compile model
    model = compile_multimodal_model(model, learning_rate=learning_rate)
    
    # 6. Print summary
    get_multimodal_model_summary(model)
    
    # 7. Setup callbacks
    model_save_dir = config.MODEL_SAVE_DIR
    callbacks = setup_callbacks(model_save_dir)
    
    # 8. Train model
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # 9. Fine-tuning (if specified)
    if fine_tune_epoch is not None and fine_tune_epoch < epochs:
        print("\n" + "="*80)
        print(f"Fine-Tuning MobileNet (from epoch {fine_tune_epoch})")
        print("="*80)
        
        # Unfreeze MobileNet
        unfreeze_mobilenet_for_finetuning(model, unfreeze_from_layer=100)
        
        # Recompile with lower learning rate
        fine_tune_lr = learning_rate / 10
        model = compile_multimodal_model(model, learning_rate=fine_tune_lr)
        
        # Continue training
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            initial_epoch=fine_tune_epoch,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Merge histories
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
    
    # 10. Save final model
    final_model_path = os.path.join(model_save_dir, 'final_multimodal_model.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # 11. Save training history
    history_path = os.path.join(model_save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types
        history_dict = {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    print(f"✓ Training history saved: {history_path}")
    
    # 12. Print final metrics
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal Metrics:")
    print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    if 'val_auc' in history.history:
        print(f"  Validation AUC: {history.history['val_auc'][-1]:.4f}")
    if 'val_precision' in history.history:
        print(f"  Validation Precision: {history.history['val_precision'][-1]:.4f}")
    if 'val_recall' in history.history:
        print(f"  Validation Recall: {history.history['val_recall'][-1]:.4f}")
    
    print(f"\nBest Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Best Validation AUC: {max(history.history.get('val_auc', [0])):.4f}")
    print("="*80 + "\n")
    
    return model, history


if __name__ == "__main__":
    """Run training"""
    
    # Check GPU availability
    print("\n" + "="*80)
    print("System Check")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print("="*80 + "\n")
    
    # Train model
    model, history = train_multimodal_model(
        dataset_dir=config.DATASET_DIR,
        batch_size=16,  # Reduce if GPU memory limited
        epochs=30,
        learning_rate=1e-4,
        use_advanced_pose=True,
        use_multi_person_emotion=True,
        fusion_type='adaptive',
        fine_tune_epoch=20  # Start fine-tuning after epoch 20
    )
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"Model saved at: {config.MODEL_SAVE_DIR}")
