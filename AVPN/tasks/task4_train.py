"""
Task 4: Noise Compensation - Matched Training and Evaluation
Train CNN on clean+noisy data and evaluate on all conditions.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import json
from pathlib import Path

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Configuration
SCRIPT_DIR = Path(__file__).parent.parent  # Go up to TestingAOu-locase/
FEATURES_DIR = SCRIPT_DIR / "features"
RESULTS_DIR = SCRIPT_DIR / "results/task4"
MODEL_DIR = SCRIPT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_features_file(npz_path):
    """Load features from a single .npz file."""
    npz_path = Path(npz_path)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find features file: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Load metadata - find corresponding JSON file
    base_name = str(npz_path).replace('.npz', '')
    json_path = Path(base_name + '.json')
    
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find metadata JSON: {json_path}")
    
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    return X, y, metadata


def normalize_features(X_train, X_val, X_test):
    """Normalize using training set statistics."""
    # Compute mean and std from training set
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    
    # Apply to all sets
    X_train_norm = (X_train - train_mean) / (train_std + 1e-8)
    X_val_norm = (X_val - train_mean) / (train_std + 1e-8)
    X_test_norm = (X_test - train_mean) / (train_std + 1e-8)
    
    return X_train_norm, X_val_norm, X_test_norm, train_mean, train_std


def split_data(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """Split data into train/val/test sets with stratification."""
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, stratify=y
    )
    
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_seed, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_baseline_cnn(input_shape, num_classes):
    """Build the baseline CNN architecture (same as before)."""
    model = keras.Sequential([
        # Conv layer
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((3, 3)),
        
        # Flatten and dense
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def plot_confusion_matrix(y_true, y_pred, labels, output_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_training_history(history, output_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def evaluate_on_condition(model, X, y, label_to_idx, unique_labels, condition_name, output_dir):
    """Evaluate model on a specific condition and save results."""
    # Convert string labels to indices
    y_idx = np.array([label_to_idx[label] for label in y])
    y_categorical = to_categorical(y_idx, num_classes=len(unique_labels))
    
    # Predict
    y_pred_probs = model.predict(X, verbose=0)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_idx, y_pred_idx)
    
    # Plot confusion matrix
    cm_path = output_dir / f"confusion_{condition_name}.png"
    plot_confusion_matrix(y_idx, y_pred_idx, unique_labels, cm_path, 
                         title=f"Confusion Matrix - {condition_name}")
    
    # Save classification report
    report = classification_report(y_idx, y_pred_idx, target_names=unique_labels)
    report_path = output_dir / f"classification_report_{condition_name}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {condition_name}\n")
        f.write("="*70 + "\n")
        f.write(report)
    print(f"  Saved: {report_path}")
    
    return accuracy


def main_baseline():
    """
    Strategy 1: Baseline (CLEAN only training)
    Train on clean data, evaluate on clean and noisy.
    """
    print("\n" + "="*70)
    print("TASK 4 - STRATEGY 1: BASELINE (CLEAN ONLY)")
    print("="*70)
    
    # Create output directory
    baseline_dir = RESULTS_DIR / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    # Load clean features
    print("\nLoading clean features...")
    X_clean, y_clean, metadata = load_features_file(FEATURES_DIR / "features.npz")
    print(f"  Clean data: {X_clean.shape}")
    
    # Split data
    print("\nSplitting data (80/10/10)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_clean, y_clean)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize
    print("\nNormalizing features...")
    X_train_norm, X_val_norm, X_test_norm, train_mean, train_std = normalize_features(
        X_train, X_val, X_test
    )
    print(f"  Train mean: {train_mean:.4f}, std: {train_std:.4f}")
    
    # Build model
    unique_labels = metadata['labels']
    label_to_idx = metadata['label_to_idx']
    num_classes = len(unique_labels)
    input_shape = X_train.shape[1:]
    
    print(f"\nBuilding baseline CNN...")
    print(f"  Input shape: {input_shape}")
    print(f"  Output classes: {num_classes}")
    
    model = build_baseline_cnn(input_shape, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare labels
    y_train_idx = np.array([label_to_idx[label] for label in y_train])
    y_val_idx = np.array([label_to_idx[label] for label in y_val])
    y_test_idx = np.array([label_to_idx[label] for label in y_test])
    
    y_train_cat = to_categorical(y_train_idx, num_classes)
    y_val_cat = to_categorical(y_val_idx, num_classes)
    y_test_cat = to_categorical(y_test_idx, num_classes)
    
    # Train
    print("\nTraining baseline model...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(baseline_dir / 'baseline_best.keras', monitor='val_loss', save_best_only=True, verbose=0)
    ]
    
    history = model.fit(
        X_train_norm, y_train_cat,
        validation_data=(X_val_norm, y_val_cat),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history plot
    plot_training_history(history, baseline_dir / 'training_history_baseline.png')
    
    # Evaluate on ALL clean data (following lab4 methodology)
    print("\n" + "="*70)
    print("BASELINE MODEL EVALUATION")
    print("="*70)
    
    # Normalize ALL clean data using training stats
    X_clean_all_norm = (X_clean - train_mean) / (train_std + 1e-8)
    
    acc_clean = evaluate_on_condition(
        model, X_clean_all_norm, y_clean, label_to_idx, unique_labels,
        "clean", baseline_dir
    )
    print(f"\nClean Accuracy (all 800 samples): {acc_clean:.4f}")
    
    # Now evaluate on noisy conditions using ALL DATA (following lab4 methodology)
    results = {'CLEAN': acc_clean}
    
    for snr in [20, 10, 0]:
        print(f"\nEvaluating on SNR {snr} dB...")
        X_noisy, y_noisy, _ = load_features_file(FEATURES_DIR / f"noisy_SNR{snr}.npz")
        
        # Normalize using clean training stats
        X_noisy_all_norm = (X_noisy - train_mean) / (train_std + 1e-8)
        
        acc_noisy = evaluate_on_condition(
            model, X_noisy_all_norm, y_noisy, label_to_idx, unique_labels,
            f"SNR{snr}", baseline_dir
        )
        results[f'SNR{snr}'] = acc_noisy
        print(f"  SNR {snr} dB Accuracy (all 800 samples): {acc_noisy:.4f}")
    
    # Save model
    model.save(MODEL_DIR / 'task4_baseline.keras')
    print(f"\nModel saved: {MODEL_DIR / 'task4_baseline.keras'}")
    
    return results


def main_matched():
    """
    Strategy 2: Matched training
    Train on CLEAN + all noisy conditions, evaluate separately.
    """
    print("\n" + "="*70)
    print("TASK 4 - STRATEGY 2: MATCHED TRAINING (CLEAN + NOISY)")
    print("="*70)
    
    # Create output directory
    matched_dir = RESULTS_DIR / "matched"
    matched_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all features
    print("\nLoading all features...")
    X_clean, y_clean, metadata_clean = load_features_file(FEATURES_DIR / "features.npz")
    print(f"  Clean: {X_clean.shape}")
    
    X_list = [X_clean]
    y_list = [y_clean]
    
    for snr in [20, 10, 0]:
        X_noisy, y_noisy, _ = load_features_file(FEATURES_DIR / f"noisy_SNR{snr}.npz")
        print(f"  SNR {snr} dB: {X_noisy.shape}")
        X_list.append(X_noisy)
        y_list.append(y_noisy)
    
    # Concatenate all data
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print(f"\n✓ Combined dataset: {X_all.shape}")
    
    # Split combined data
    print("\nSplitting combined data (80/10/10)...")
    X_train, X_val, X_test_combined, y_train, y_val, y_test_combined = split_data(X_all, y_all)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test_combined.shape}")
    
    # Normalize
    print("\nNormalizing features...")
    X_train_norm, X_val_norm, X_test_norm, train_mean, train_std = normalize_features(
        X_train, X_val, X_test_combined
    )
    print(f"  Train mean: {train_mean:.4f}, std: {train_std:.4f}")
    
    # Build model
    unique_labels = metadata_clean['labels']
    label_to_idx = metadata_clean['label_to_idx']
    num_classes = len(unique_labels)
    input_shape = X_train.shape[1:]
    
    print(f"\nBuilding baseline CNN...")
    print(f"  Input shape: {input_shape}")
    print(f"  Output classes: {num_classes}")
    
    model = build_baseline_cnn(input_shape, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare labels
    y_train_idx = np.array([label_to_idx[label] for label in y_train])
    y_val_idx = np.array([label_to_idx[label] for label in y_val])
    y_test_idx = np.array([label_to_idx[label] for label in y_test_combined])
    
    y_train_cat = to_categorical(y_train_idx, num_classes)
    y_val_cat = to_categorical(y_val_idx, num_classes)
    y_test_cat = to_categorical(y_test_idx, num_classes)
    
    # Train
    print("\nTraining matched model...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(matched_dir / 'matched_best.keras', monitor='val_loss', save_best_only=True, verbose=0)
    ]
    
    history = model.fit(
        X_train_norm, y_train_cat,
        validation_data=(X_val_norm, y_val_cat),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history plot
    plot_training_history(history, matched_dir / 'training_history_matched.png')
    
    # Evaluate on each condition separately using ALL DATA (following lab4 methodology)
    print("\n" + "="*70)
    print("MATCHED MODEL EVALUATION")
    print("="*70)
    
    results = {}
    
    # Evaluate on ALL clean data
    print("\nEvaluating on CLEAN...")
    X_clean_only, y_clean_only, _ = load_features_file(FEATURES_DIR / "features.npz")
    X_clean_all_norm = (X_clean_only - train_mean) / (train_std + 1e-8)
    acc_clean = evaluate_on_condition(
        model, X_clean_all_norm, y_clean_only, label_to_idx, unique_labels,
        "clean_matched", matched_dir
    )
    results['CLEAN'] = acc_clean
    print(f"  Clean Accuracy (all 800 samples): {acc_clean:.4f}")
    
    # Evaluate on each noisy condition using ALL DATA
    for snr in [20, 10, 0]:
        print(f"\nEvaluating on SNR {snr} dB...")
        X_noisy, y_noisy, _ = load_features_file(FEATURES_DIR / f"noisy_SNR{snr}.npz")
        X_noisy_all_norm = (X_noisy - train_mean) / (train_std + 1e-8)
        
        acc_noisy = evaluate_on_condition(
            model, X_noisy_all_norm, y_noisy, label_to_idx, unique_labels,
            f"SNR{snr}_matched", matched_dir
        )
        results[f'SNR{snr}'] = acc_noisy
        print(f"  SNR {snr} dB Accuracy (all 800 samples): {acc_noisy:.4f}")
    
    # Save model
    model.save(MODEL_DIR / 'task4_matched.keras')
    print(f"\nModel saved: {MODEL_DIR / 'task4_matched.keras'}")
    
    return results


def plot_comparison(baseline_results, matched_results, output_path):
    """Plot accuracy comparison between baseline and matched training."""
    conditions = ['CLEAN', 'SNR20', 'SNR10', 'SNR0']
    baseline_accs = [baseline_results[c] for c in conditions]
    matched_accs = [matched_results[c] for c in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline (Clean only)', alpha=0.8)
    bars2 = ax.bar(x + width/2, matched_accs, width, label='Matched (Clean + Noisy)', alpha=0.8)
    
    ax.set_xlabel('Condition')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task 4: Noise Compensation - Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved: {output_path}")


def main():
    """Run both strategies and compare."""
    print("\n" + "="*70)
    print("TASK 4: NOISE COMPENSATION EVALUATION")
    print("="*70)
    print("Strategy 1: Baseline (train on clean only)")
    print("Strategy 2: Matched training (train on clean + noisy)")
    print("="*70)
    
    # Run baseline
    baseline_results = main_baseline()
    
    # Run matched
    matched_results = main_matched()
    
    # Create comparison plot
    plot_comparison(baseline_results, matched_results, RESULTS_DIR / 'compare_accuracy.png')
    
    # Print final summary table
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Condition':<15} {'Baseline':<12} {'Matched':<12} {'Improvement':<12}")
    print("-"*70)
    
    for condition in ['CLEAN', 'SNR20', 'SNR10', 'SNR0']:
        baseline_acc = baseline_results[condition]
        matched_acc = matched_results[condition]
        improvement = matched_acc - baseline_acc
        print(f"{condition:<15} {baseline_acc:>10.4f}   {matched_acc:>10.4f}   {improvement:>+10.4f}")
    
    print("="*70)
    print(f"\n✓ All outputs saved to: {RESULTS_DIR}/")
    print("  - Confusion matrices for each condition")
    print("  - Training history plots")
    print("  - Accuracy comparison plot")
    print("  - Classification reports")
    print("  - Trained models")
    print("="*70)


if __name__ == "__main__":
    main()
