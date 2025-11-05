"""
Model Evaluation Script
Comprehensive evaluation with metrics, confusion matrix, ROC curve, and analysis
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

import config
from data_generator import create_data_generators
from data_preprocessing import load_dataset_paths


def load_trained_model(model_path):
    """
    Load a trained model
    
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
    return model


def evaluate_model(model, data_generator, threshold=config.CONFIDENCE_THRESHOLD):
    """
    Evaluate model on dataset
    
    Args:
        model: Trained model
        data_generator: Data generator
        threshold: Classification threshold
    
    Returns:
        y_true, y_pred, y_pred_proba
    """
    print("\nEvaluating model...")
    
    y_true = []
    y_pred_proba = []
    
    # Predict on all batches
    for i in range(len(data_generator)):
        X_batch, y_batch = data_generator[i]
        
        # Get predictions
        predictions = model.predict(X_batch, verbose=0)
        
        y_true.extend(y_batch)
        y_pred_proba.extend(predictions.flatten())
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(data_generator)} batches")
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"Evaluation completed on {len(y_true)} samples")
    
    return y_true, y_pred, y_pred_proba


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate comprehensive metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, matthews_corrcoef
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    return metrics


def print_evaluation_results(metrics, y_true, y_pred):
    """
    Print evaluation results
    
    Args:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1-Score:           {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"  Matthews Corr Coef: {metrics['mcc']:.4f}")
    
    print(f"\n{'-'*80}")
    print("Classification Report:")
    print("-"*80)
    print(classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4))
    
    print("="*80 + "\n")


def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=True):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=config.CLASSES, 
                yticklabels=config.CLASSES,
                cbar_kws={'label': 'Percentage' if normalize else 'Count'})
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, pad=20)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to: {save_path}")
    
    plt.show()


def plot_prediction_distribution(y_true, y_pred_proba, save_path=None):
    """
    Plot prediction probability distribution
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    # Separate probabilities by class
    violence_probs = y_pred_proba[y_true == 1]
    non_violence_probs = y_pred_proba[y_true == 0]
    
    plt.hist(non_violence_probs, bins=50, alpha=0.6, label='Non-Violence (True)', 
             color='green', edgecolor='black')
    plt.hist(violence_probs, bins=50, alpha=0.6, label='Violence (True)', 
             color='red', edgecolor='black')
    
    plt.axvline(x=config.CONFIDENCE_THRESHOLD, color='blue', linestyle='--', 
                linewidth=2, label=f'Threshold ({config.CONFIDENCE_THRESHOLD})')
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction distribution saved to: {save_path}")
    
    plt.show()


def analyze_errors(y_true, y_pred, y_pred_proba, video_paths=None, top_n=10):
    """
    Analyze misclassified samples
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        video_paths: List of video paths (optional)
        top_n: Number of top errors to display
    """
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    # Find misclassified indices
    errors = np.where(y_true != y_pred)[0]
    
    print(f"\nTotal errors: {len(errors)} / {len(y_true)} ({len(errors)/len(y_true)*100:.2f}%)")
    
    # False positives
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    print(f"False Positives: {len(fp_indices)} (predicted violence, actually non-violence)")
    
    # False negatives
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    print(f"False Negatives: {len(fn_indices)} (predicted non-violence, actually violence)")
    
    # Top confident errors
    if len(errors) > 0:
        error_confidences = np.abs(y_pred_proba[errors] - 0.5)
        top_error_indices = errors[np.argsort(error_confidences)[-top_n:]][::-1]
        
        print(f"\nTop {min(top_n, len(top_error_indices))} Most Confident Errors:")
        print("-" * 80)
        
        for i, idx in enumerate(top_error_indices):
            true_label = config.CLASSES[int(y_true[idx])]
            pred_label = config.CLASSES[int(y_pred[idx])]
            confidence = y_pred_proba[idx]
            
            path_info = f" | Path: {video_paths[idx]}" if video_paths else ""
            print(f"{i+1}. True: {true_label:12s} | Pred: {pred_label:12s} | "
                  f"Confidence: {confidence:.4f}{path_info}")
    
    print("="*80 + "\n")


def save_evaluation_report(metrics, save_dir=config.PLOTS_DIR):
    """
    Save evaluation report to file
    
    Args:
        metrics: Dictionary of metrics
        save_dir: Directory to save report
    """
    report_path = os.path.join(save_dir, f'{config.MODEL_NAME}_evaluation_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation report saved to: {report_path}")


def comprehensive_evaluation(model_path, data_dir=config.VAL_DIR, 
                            save_plots=True, batch_size=config.BATCH_SIZE):
    """
    Perform comprehensive model evaluation
    
    Args:
        model_path: Path to trained model
        data_dir: Directory containing evaluation data
        save_plots: Whether to save plots
        batch_size: Batch size for evaluation
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80 + "\n")
    
    # Load model
    model = load_trained_model(model_path)
    
    # Create data generator
    from data_generator import VideoDataGenerator
    
    video_paths, labels = load_dataset_paths(data_dir)
    
    eval_generator = VideoDataGenerator(
        video_paths=video_paths,
        labels=labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        preprocess_mode=config.PREPROCESS_MODE
    )
    
    # Evaluate model
    y_true, y_pred, y_pred_proba = evaluate_model(model, eval_generator)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    
    # Print results
    print_evaluation_results(metrics, y_true, y_pred)
    
    # Plot confusion matrix
    if config.PLOT_CONFUSION_MATRIX and save_plots:
        cm_path = os.path.join(config.PLOTS_DIR, f'{config.MODEL_NAME}_confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, save_path=cm_path, 
                            normalize=config.NORMALIZE_CM)
    
    # Plot ROC curve
    if config.PLOT_ROC_CURVE and save_plots:
        roc_path = os.path.join(config.PLOTS_DIR, f'{config.MODEL_NAME}_roc_curve.png')
        plot_roc_curve(y_true, y_pred_proba, save_path=roc_path)
    
    # Plot Precision-Recall curve
    if save_plots:
        pr_path = os.path.join(config.PLOTS_DIR, f'{config.MODEL_NAME}_pr_curve.png')
        plot_precision_recall_curve(y_true, y_pred_proba, save_path=pr_path)
    
    # Plot prediction distribution
    if save_plots:
        dist_path = os.path.join(config.PLOTS_DIR, f'{config.MODEL_NAME}_pred_distribution.png')
        plot_prediction_distribution(y_true, y_pred_proba, save_path=dist_path)
    
    # Error analysis
    analyze_errors(y_true, y_pred, y_pred_proba, video_paths=video_paths, top_n=10)
    
    # Save evaluation report
    if save_plots:
        save_evaluation_report(metrics)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80 + "\n")
    
    return metrics, y_true, y_pred, y_pred_proba


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Violence Detection Model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default=config.VAL_DIR,
                       help='Directory containing evaluation data')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for evaluation')
    parser.add_argument('--threshold', type=float, default=config.CONFIDENCE_THRESHOLD,
                       help='Classification threshold')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable saving plots')
    
    args = parser.parse_args()
    
    # Override config
    if args.threshold:
        config.CONFIDENCE_THRESHOLD = args.threshold
    
    # Run evaluation
    metrics, y_true, y_pred, y_pred_proba = comprehensive_evaluation(
        model_path=args.model,
        data_dir=args.data_dir,
        save_plots=not args.no_plots,
        batch_size=args.batch_size
    )
