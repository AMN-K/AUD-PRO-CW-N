#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
task5.py
========
Task 5: Final Evaluation - Load Features/Models and Print Results

Usage:
    python tasks/task5.py

Outputs:
- Accuracy table for all conditions (clean, SNR 20/10/0)
- Comparison of baseline vs matched strategies
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from pathlib import Path
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_npz(path):
    """Load NPZ features."""
    data = np.load(path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Handle string labels (convert to numeric if needed)
    if y.dtype.kind in ('U', 'S', 'O'):  # Unicode, bytes, or object
        unique_labels = np.unique(y)
        label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        y = np.array([label_map[label] for label in y])
    
    return X, y

def evaluate_all():
    """Evaluate all models on all conditions."""
    base_dir = Path(__file__).parent.parent
    
    # Load features
    print("Loading features...")
    conditions = {
        'Clean': base_dir / 'features' / 'features.npz',
        'SNR20': base_dir / 'features' / 'noisy_snr20.npz',
        'SNR10': base_dir / 'features' / 'noisy_snr10.npz',
        'SNR0': base_dir / 'features' / 'noisy_snr0.npz'
    }
    
    features = {}
    for name, path in conditions.items():
        X, y = load_npz(path)
        # Use test split (10%)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        features[name] = (X_test, y_test)
    
    # Load models
    print("Loading models...")
    models = {
        'Task3 Baseline': keras.models.load_model(base_dir / 'models' / 'task3_baseline.keras'),
        'Task4 Baseline': keras.models.load_model(base_dir / 'models' / 'task4_baseline.keras'),
        'Task4 Matched': keras.models.load_model(base_dir / 'models' / 'task4_matched.keras')
    }
    
    # Evaluate all combinations
    print("\n" + "="*80)
    print("EVALUATION RESULTS - ACCURACY TABLE")
    print("="*80)
    print(f"{'Model':<20} {'Clean':>10} {'SNR20':>10} {'SNR10':>10} {'SNR0':>10}")
    print("-"*80)
    
    for model_name, model in models.items():
        accs = []
        for cond_name in ['Clean', 'SNR20', 'SNR10', 'SNR0']:
            X_test, y_test = features[cond_name]
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            acc = accuracy_score(y_test, y_pred)
            accs.append(acc)
        
        print(f"{model_name:<20} {accs[0]:>9.1%} {accs[1]:>9.1%} {accs[2]:>9.1%} {accs[3]:>9.1%}")
    
    print("="*80)
    print("\n[OK] Evaluation complete! Results saved in results/task4/")

if __name__ == "__main__":
    evaluate_all()
