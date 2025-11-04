#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_test.py
============
Demo/Testing Script - Test individual audio files with trained model

This script allows you to test the model on individual files where
the filename does NOT match the actual speaker (e.g., test000.wav contains Joshua's voice).

Usage:
    # Test single file
    python tasks/demo_test.py --audio Audio/test000.wav --true-label joshua
    
    # Test multiple files with manual labels
    python tasks/demo_test.py --batch --audio-folder Audio/ --labels-file test_labels.txt
    
    # Test all files matching pattern
    python tasks/demo_test.py --pattern "Audio/test*.wav" --true-label joshua

The script will:
1. Extract MFCC features from the audio file
2. Load the trained model
3. Predict the speaker name
4. Compare with the true label (if provided)
5. Show confidence scores
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import argparse
import numpy as np
import json
from pathlib import Path
from tensorflow import keras
import glob

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.feature_utils import (
    load_wav,
    trim_silence,
    frame_signal,
    mfcc_from_frames,
    add_deltas,
    pad_features
)


def load_metadata(features_dir):
    """Load label mappings from features metadata."""
    metadata_path = features_dir / "features.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    label_to_idx = metadata['label_to_idx']
    idx_to_label = {int(v): str(k) for k, v in label_to_idx.items()}
    max_frames = metadata['max_frames']
    
    return label_to_idx, idx_to_label, max_frames


def extract_features_from_audio(audio_path, max_frames, sr=16000, win_ms=25, hop_ms=10, 
                               n_fft=512, keep_mfcc=40):
    """
    Extract MFCC features from a single audio file.
    
    Args:
        audio_path: Path to .wav file
        max_frames: Target frame count for padding
        sr: Sample rate (default 16000 Hz)
        win_ms: Window length in ms (default 25 ms)
        hop_ms: Hop length in ms (default 10 ms)
        n_fft: FFT size (default 512)
        keep_mfcc: Number of MFCC coefficients (default 40)
        
    Returns:
        features: Array of shape (num_windows, 120, max_frames, 1).
        starts:   1D array with the starting frame index for each window.
    """
    # Load audio (mono, 16kHz)
    signal = load_wav(str(audio_path), target_sr=sr)
    trimmed = trim_silence(signal, top_db=35.0, min_duration=0.3, sr=sr)
    if trimmed.size >= int(0.4 * sr):
        signal = trimmed
    
    # Frame the signal (25ms window, 10ms hop, Hamming)
    frames = frame_signal(signal, sr, win_ms=win_ms, hop_ms=hop_ms)
    
    # Extract 40 MFCCs
    mfcc = mfcc_from_frames(frames, sr, n_fft=n_fft, keep_mfcc=keep_mfcc)
    
    # Add Δ and ΔΔ → 120 features (40 + 40 + 40)
    feat_120 = add_deltas(mfcc, order=2, width=2)  # Shape: (num_frames, 120)

    num_frames = feat_120.shape[0]

    windows = []
    starts = []
    if num_frames <= max_frames:
        feat_padded = pad_features(feat_120, max_frames)
        windows.append(feat_padded)
        starts.append(0)
    else:
        frame_energy = np.linalg.norm(feat_120, axis=1)
        max_energy = np.max(frame_energy)
        energy_threshold = 0.3 * max_energy if max_energy > 0 else 0
        active = np.where(frame_energy >= energy_threshold)[0]

        first_start = 0
        center_start = max((active[0] + active[-1]) // 2 - max_frames // 2, 0) if active.size else 0
        last_start = max(num_frames - max_frames, 0)

        candidate_starts = {first_start, center_start, last_start}
        if active.size:
            candidate_starts.add(max(active[0] - max_frames // 4, 0))
            candidate_starts.add(max(active[-1] - max_frames // 2, 0))

        for start_idx in sorted(candidate_starts):
            start_idx = max(0, min(start_idx, num_frames - max_frames))
            window_feat = feat_120[start_idx:start_idx + max_frames]
            if window_feat.shape[0] < max_frames:
                window_feat = pad_features(window_feat, max_frames)
            windows.append(window_feat)
            starts.append(start_idx)

    window_arrays = [window.T[..., np.newaxis].astype(np.float32) for window in windows]
    return np.stack(window_arrays, axis=0), np.array(starts, dtype=np.int32)


def normalize_features(features, train_mean, train_std):
    """Normalize using training set statistics."""
    return (features - train_mean) / (train_std + 1e-8)


def predict_speaker(audio_path, model, label_to_idx, idx_to_label, max_frames, 
                   train_mean=None, train_std=None):
    """
    Predict speaker from audio file.
    
    Args:
        audio_path: Path to audio file
        model: Trained Keras model
        label_to_idx: Label to index mapping
        idx_to_label: Index to label mapping
        max_frames: Maximum number of frames
        train_mean: Training set mean for normalization (optional)
        train_std: Training set std for normalization (optional)
        
    Returns:
        predicted_name: Speaker name
        confidence: Prediction confidence (0-1)
        all_probs: Probability for each speaker
    """
    feature_batch, frame_starts = extract_features_from_audio(audio_path, max_frames)

    if train_mean is not None and train_std is not None:
        feature_batch = normalize_features(feature_batch, train_mean, train_std)

    predictions = model.predict(feature_batch, verbose=0)

    mean_probs = predictions.mean(axis=0)
    best_window_idx = int(np.argmax(predictions.max(axis=1)))
    best_probs = predictions[best_window_idx]
    blended_probs = 0.7 * best_probs + 0.3 * mean_probs

    # Check early windows (close to start of detected speech)
    earliest_start = int(frame_starts.min())
    early_mask = frame_starts <= earliest_start + max_frames // 4
    early_indices = np.where(early_mask)[0]

    override_probs = None
    if early_indices.size:
        early_scores = predictions[early_indices]
        early_best_idx = early_indices[int(np.argmax(early_scores.max(axis=1)))]
        early_probs = predictions[early_best_idx]
        early_label = int(np.argmax(early_probs))
        early_conf = float(early_probs[early_label])
        if early_conf >= 0.6 and early_label != int(np.argmax(blended_probs)):
            override_probs = 0.7 * early_probs + 0.3 * blended_probs

    final_probs = override_probs if override_probs is not None else blended_probs

    predicted_idx = int(np.argmax(final_probs))
    confidence = float(final_probs[predicted_idx])
    predicted_name = idx_to_label[predicted_idx]
    
    all_probs = {idx_to_label[i]: float(final_probs[i]) for i in range(len(final_probs))}
    
    return predicted_name, confidence, all_probs


def test_single_file(audio_path, model_path, true_label=None, show_top_n=5):
    """
    Test a single audio file.
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model
        true_label: True speaker name (optional)
        show_top_n: Show top N predictions
    """
    base_dir = Path(__file__).parent.parent
    features_dir = base_dir / "features"
    
    # Load metadata
    label_to_idx, idx_to_label, max_frames = load_metadata(features_dir)
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load training statistics for normalization
    train_data = np.load(features_dir / "features.npz")
    X_train = train_data['X']
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    
    # Predict
    print(f"\nTesting audio file: {audio_path}")
    predicted_name, confidence, all_probs = predict_speaker(
        audio_path, model, label_to_idx, idx_to_label, max_frames,
        train_mean, train_std
    )
    
    # Results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Audio file:      {Path(audio_path).name}")
    if true_label:
        print(f"True speaker:    {true_label}")
    print(f"Predicted:       {predicted_name}")
    print(f"Confidence:      {confidence*100:.2f}%")
    
    if true_label:
        if predicted_name.lower() == true_label.lower():
            print(f"Result:          [Yes] CORRECT")
        else:
            print(f"Result:          [WRONG] INCORRECT")
    
    # Show top N predictions
    print(f"\nTop {show_top_n} predictions:")
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (name, prob) in enumerate(sorted_probs[:show_top_n], 1):
        marker = " <--" if name == predicted_name else ""
        print(f"  {i}. {name:15s} {prob*100:6.2f}%{marker}")
    
    print("="*70)
    
    return predicted_name, confidence


def test_batch_files(pattern, model_path, true_label=None):
    """
    Test multiple audio files matching a pattern.
    
    Args:
        pattern: Glob pattern for audio files (e.g., "Audio/test*.wav")
        model_path: Path to trained model
        true_label: True speaker name for all files (optional)
    """
    base_dir = Path(__file__).parent.parent
    features_dir = base_dir / "features"
    
    # Get all matching files
    if not Path(pattern).is_absolute():
        pattern = str(base_dir / pattern)
    
    audio_files = sorted(glob.glob(pattern))
    
    if not audio_files:
        print(f"[ERROR] No files found matching pattern: {pattern}")
        return
    
    print(f"\nFound {len(audio_files)} files matching pattern: {pattern}")
    
    # Load metadata
    label_to_idx, idx_to_label, max_frames = load_metadata(features_dir)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path)
    
    # Load training statistics
    train_data = np.load(features_dir / "features.npz")
    X_train = train_data['X']
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    
    # Test each file
    results = []
    print("\n" + "="*90)
    print(f"{'File':<25} {'True Label':<15} {'Predicted':<15} {'Confidence':<12} {'Correct?'}")
    print("="*90)
    
    correct = 0
    for audio_path in audio_files:
        predicted_name, confidence, _ = predict_speaker(
            audio_path, model, label_to_idx, idx_to_label, max_frames,
            train_mean, train_std
        )
        
        filename = Path(audio_path).name
        
        if true_label:
            is_correct = predicted_name.lower() == true_label.lower()
            correct_mark = "[Yes]" if is_correct else "[WRONG]"
            if is_correct:
                correct += 1
        else:
            true_label_display = "N/A"
            correct_mark = "N/A"
        
        true_label_display = true_label if true_label else "N/A"
        
        print(f"{filename:<25} {true_label_display:<15} {predicted_name:<15} "
              f"{confidence*100:>5.1f}%       {correct_mark}")
        
        results.append({
            'file': filename,
            'true': true_label,
            'predicted': predicted_name,
            'confidence': confidence,
            'correct': is_correct if true_label else None
        })
    
    print("="*90)
    
    if true_label:
        accuracy = correct / len(audio_files) * 100
        print(f"\nAccuracy: {correct}/{len(audio_files)} ({accuracy:.1f}%)")
    
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test speech recognizer on individual audio files'
    )
    parser.add_argument('--audio', type=str, help='Path to single audio file')
    parser.add_argument('--pattern', type=str, help='Glob pattern for batch testing (e.g., "Audio/test*.wav")')
    parser.add_argument('--model', type=str, default='models/task4_matched.keras',
                       help='Path to trained model (default: task4_matched.keras)')
    parser.add_argument('--true-label', type=str, help='True speaker name')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Show top N predictions (default: 5)')
    
    args = parser.parse_args()
    
    # Determine base directory
    base_dir = Path(__file__).parent.parent
    
    # Resolve model path
    model_path = base_dir / args.model
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    # Single file test
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.is_absolute():
            audio_path = base_dir / audio_path
        
        if not audio_path.exists():
            print(f"[ERROR] Audio file not found: {audio_path}")
            return
        
        test_single_file(audio_path, model_path, args.true_label, args.top_n)
    
    # Batch test
    elif args.pattern:
        test_batch_files(args.pattern, model_path, args.true_label)
    
    else:
        parser.print_help()
        print("\n[ERROR] Please provide either --audio or --pattern")


if __name__ == "__main__":
    main()



# python tasks/demo_test.py --pattern "Audio_test/*.wav" --true-label charlie

# python tasks/demo_test.py --audio "Audio_test/test_000.wav" --true-label ahmed --top-n 10