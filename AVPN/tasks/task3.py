# -*- coding: utf-8 -*-
"""
task3.py
========
Task 3: Training Baseline CNN for Speaker Recognition

This script trains the baseline CNN architecture following specifications:
- Input: (N, 120, max_frames, 1) from feature extraction
- Architecture: Conv2D(64,3×3) → MaxPool(3×3) → Flatten → Dense(256) → Softmax
- Adam optimizer, lr=1e-3
- 80/10/10 train/val/test split
- Early stopping on validation loss

Outputs:
- confusion_matrix.png
- trained_model.keras
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
import json
from pathlib import Path

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Add parent directory to path for helper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our custom module
from helpers.feature_utils import load_features, normalize_features


def build_baseline_cnn(input_shape, num_classes):
    """
    Build the baseline CNN architecture.
    
    Specification:
    - Conv2D(64 filters, 3×3 kernel, ReLU activation)
    - MaxPooling2D(3×3 pool size)
    - Flatten
    - Dense(256, ReLU)
    - Dense(num_classes, Softmax)
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input (n_mfcc, max_frames, 1)
    num_classes : int
        Number of output classes
        
    Returns
    -------
    keras.Model
        Compiled model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Convolutional layer: 64 filters, 3×3 kernel
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                      name='conv2d_64_3x3'),
        
        # Max pooling: 3×3 pool size
        layers.MaxPooling2D(pool_size=(3, 3), name='maxpool_3x3'),
        
        # Flatten to 1D
        layers.Flatten(name='flatten'),
        
        # Dense layer: 256 units
        layers.Dense(256, activation='relu', name='dense_256'),
        
        # Output layer: Softmax
        layers.Dense(num_classes, activation='softmax', name='output_softmax')
    ], name='baseline_cnn')
    
    return model


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """
    Plot and save confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (indices)
    y_pred : np.ndarray
        Predicted labels (indices)
    labels : list
        Label names
    output_path : str or Path
        Output file path
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Baseline CNN', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Confusion matrix saved to {output_path}")


def save_model_summary(model, output_path):
    """
    Save model architecture summary to text file.
    
    Parameters
    ----------
    model : keras.Model
        The model to summarize
    output_path : str or Path
        Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f"✓ Model summary saved to {output_path}")


def main():
    """
    Main training pipeline.
    
    Steps:
    1. Load features from features.npz
    2. Split into 80/10/10 train/val/test
    3. Normalize per-dataset (train stats to val/test)
    4. Build baseline CNN
    5. Train with early stopping
    6. Evaluate and save outputs
    """
    print("="*70)
    print("TRAINING PIPELINE - Baseline CNN")
    print("="*70)
    
    # Configuration
    FEATURES_NPZ = Path("features/features.npz")
    FEATURES_JSON = Path("features/features.json")
    OUTPUT_DIR = Path("results/task3")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)
    
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 30
    PATIENCE = 15
    
    # Load features directly (not using load_features due to double .npz extension)
    print(f"\n1. Loading features from {FEATURES_NPZ}...")
    data = np.load(FEATURES_NPZ, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Load metadata
    with open(FEATURES_JSON, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n✓ Loaded features:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Num samples: {metadata['num_samples']}")
    print(f"  Num classes: {metadata['num_classes']}")
    print(f"  n_mfcc: {metadata['n_mfcc']}")
    print(f"  max_frames: {metadata['max_frames']}")
    
    # Get label mapping
    labels = metadata['labels']
    label_to_idx = metadata['label_to_idx']
    num_classes = metadata['num_classes']
    
    # Convert string labels to indices
    y_indices = np.array([label_to_idx[label] for label in y])
    
    # 80/10/10 split
    print(f"\n2. Splitting data (80/10/10)...")
    
    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_indices, test_size=0.2, random_state=42, stratify=y_indices
    )
    
    # Second split: 10% val, 10% test (split the 20% temp equally)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n✓ Split complete:")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Normalize per-dataset
    print(f"\n3. Normalizing features (per-dataset)...")
    X_train_norm, X_val_norm, X_test_norm, train_mean, train_std = normalize_features(
        X_train, X_val, X_test
    )
    
    print(f"\n✓ Normalization complete:")
    print(f"  Train mean: {train_mean:.4f}, std: {train_std:.4f}")
    print(f"  Shapes after normalization:")
    print(f"    Train: {X_train_norm.shape}")
    print(f"    Val:   {X_val_norm.shape}")
    print(f"    Test:  {X_test_norm.shape}")
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Build model
    print(f"\n4. Building baseline CNN...")
    input_shape = X_train_norm.shape[1:]  # (n_mfcc, max_frames, 1)
    print(f"  Input shape: {input_shape}")
    
    model = build_baseline_cnn(input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n✓ Model built and compiled:")
    model.summary()
    
    # Save model summary
    summary_path = OUTPUT_DIR / "model_summary.txt"
    save_model_summary(model, summary_path)
    
    # Setup callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train model
    print(f"\n5. Training model...")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Early stopping patience: {PATIENCE}")
    print("\n" + "="*70)
    
    history = model.fit(
        X_train_norm, y_train_cat,
        validation_data=(X_val_norm, y_val_cat),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n" + "="*70)
    print("[OK] Training complete!")
    
    # Evaluate on ALL data (following lab4 methodology - evaluate on full dataset)
    print(f"\n6. Evaluating on ALL data (800 samples)...")
    
    # Normalize ALL data using training statistics
    X_all_norm = (X - train_mean) / (train_std + 1e-8)
    y_all_cat = keras.utils.to_categorical(y_indices, num_classes)
    
    test_loss, test_acc = model.evaluate(X_all_norm, y_all_cat, verbose=0)
    
    print(f"\n[OK] Evaluation Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc*100:.2f}% (evaluated on all 800 samples)")
    
    # Get predictions on ALL data
    y_pred_probs = model.predict(X_all_norm, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y_indices, y_pred, target_names=labels, digits=4))
    
    # Plot confusion matrix
    print(f"\n7. Generating confusion matrix...")
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_indices, y_pred, labels, cm_path)
    
    # Save model
    model_path = MODEL_DIR / "task3_baseline.keras"
    model.save(model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Plot training history
    print(f"\n8. Plotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = OUTPUT_DIR / "training_history.png"
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved to {history_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"\nOutputs saved to {OUTPUT_DIR}/:")
    print(f"  - model_summary.txt")
    print(f"  - confusion_matrix.png")
    print(f"  - training_history.png")
    print(f"\nModel saved to models/:")
    print(f"  - task3_baseline.keras")
    print("="*70)


if __name__ == "__main__":
    main()
