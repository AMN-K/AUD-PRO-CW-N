# -*- coding: utf-8 -*-
"""
task2.py
========
Task 2: Feature Extraction for Speaker Recognition

This script extracts MFCC features from all audio files following the specification:
- 16 kHz sample rate
- 25ms window, 10ms hop
- 512 FFT, Hamming window
- 40 Mel filters, 40 MFCC coefficients
- Delta and Delta-Delta → 120 total features (40 + 40 + 40)

NO TRAINING - Only feature extraction and saving.
"""

import sys
from pathlib import Path

# Add parent directory to path for helper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
import json

# Import our custom module
from helpers.feature_utils import (
    load_wav,
    frame_signal,
    mel_filterbank,
    mfcc_from_frames,
    add_deltas,
    pad_features,
    save_features
)


def extract_name_from_filename(filename):
    """
    Extract speaker name from filename.
    
    Expected format: speakerName###.wav (e.g., ahmed000.wav, john001.wav)
    
    Parameters
    ----------
    filename : str
        The filename to parse
        
    Returns
    -------
    str
        The speaker name (lowercase, alphabetic prefix)
    """
    stem = Path(filename).stem
    # Remove trailing digits
    name = ''.join([c for c in stem if c.isalpha()]).lower()
    return name


def main():
    """
    Main feature extraction pipeline.
    
    Steps:
    1. Load all audio files from Audio/ folder
    2. Extract 120-dim features per file (40 MFCC + Δ + ΔΔ)
    3. Determine dynamic max_frames from dataset
    4. Pad/truncate all features to max_frames
    5. Save features.npz and metadata.json
    """
    print("="*70)
    print("FEATURE EXTRACTION PIPELINE - Task 2")
    print("="*70)
    print("Specification:")
    print("  - Sample Rate: 16 kHz")
    print("  - Window: 25 ms, Hop: 10 ms")
    print("  - FFT Size: 512, Window: Hamming")
    print("  - Mel Filters: 40")
    print("  - MFCC: 40 coefficients + Δ + ΔΔ = 120 features")
    print("="*70)
    
    # Configuration
    AUDIO_DIR = Path("Audio")
    OUTPUT_DIR = Path("features")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    SR = 16000
    WIN_MS = 25
    HOP_MS = 10
    N_FFT = 512
    N_MELS = 40
    KEEP_MFCC = 40
    
    # Find all WAV files
    audio_files = sorted(list(AUDIO_DIR.glob("*.wav")))
    print(f"\n✓ Found {len(audio_files)} audio files in {AUDIO_DIR}/")
    
    if len(audio_files) == 0:
        print(f"ERROR: No .wav files found in {AUDIO_DIR}/")
        return
    
    # Extract features for all files
    features_list = []
    labels_list = []
    
    print("\nExtracting features...")
    for audio_path in tqdm(audio_files, desc="Processing"):
        # 1. Load and resample to 16 kHz
        x = load_wav(str(audio_path), target_sr=SR)
        
        # 2. Frame the signal (25ms window, 10ms hop, Hamming)
        frames = frame_signal(x, SR, win_ms=WIN_MS, hop_ms=HOP_MS)
        
        # 3. Extract 40 MFCCs
        mfcc = mfcc_from_frames(frames, SR, n_fft=N_FFT, keep_mfcc=KEEP_MFCC)
        
        # 4. Add Δ and ΔΔ → 120 features
        feat_120 = add_deltas(mfcc, order=2, width=2)
        
        # 5. Store features and label
        features_list.append(feat_120)
        label = extract_name_from_filename(audio_path.name)
        labels_list.append(label)
    
    print(f"\n✓ Extracted features for {len(features_list)} files")
    print(f"  Feature shape per file (before padding): (num_frames, 120)")
    print(f"  Example: {features_list[0].shape}")
    
    # Determine dynamic max_frames from dataset
    all_frame_counts = [feat.shape[0] for feat in features_list]
    max_frames = max(all_frame_counts)
    min_frames = min(all_frame_counts)
    avg_frames = np.mean(all_frame_counts)
    
    print(f"\n✓ Frame count statistics:")
    print(f"  Min frames: {min_frames}")
    print(f"  Max frames: {max_frames}")
    print(f"  Avg frames: {avg_frames:.1f}")
    print(f"\n  → Using max_frames = {max_frames} (dynamic from dataset)")
    
    # Pad/truncate all features to max_frames
    print("\nPadding features to uniform shape...")
    padded_features = []
    for feat in tqdm(features_list, desc="Padding"):
        padded = pad_features(feat, max_frames)
        padded_features.append(padded)
    
    # Stack into numpy array: (N, 120, max_frames)
    X = np.array(padded_features)
    
    # Transpose to (N, n_mfcc, max_frames) for CNN input
    # Already in correct shape: (N, max_frames, 120) → need (N, 120, max_frames)
    X = X.transpose(0, 2, 1)
    
    # Add channel dimension for CNN: (N, 120, max_frames, 1)
    X = X[..., np.newaxis]
    
    # Convert labels to numpy array
    y = np.array(labels_list)
    
    # Get unique classes
    unique_labels = sorted(list(set(y)))
    num_classes = len(unique_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"\n✓ Padded and reshaped features")
    print(f"  Final shape: {X.shape}")
    print(f"  Expected: (N, n_mfcc=120, max_frames={max_frames}, 1)")
    print(f"\n✓ Detected {num_classes} speaker classes:")
    print(f"  {unique_labels}")
    
    # Prepare metadata
    metadata = {
        'num_samples': len(X),
        'num_classes': num_classes,
        'n_mfcc': 120,
        'max_frames': int(max_frames),
        'sample_rate': SR,
        'window_ms': WIN_MS,
        'hop_ms': HOP_MS,
        'n_fft': N_FFT,
        'n_mels': N_MELS,
        'keep_mfcc': KEEP_MFCC,
        'labels': unique_labels,
        'label_to_idx': label_to_idx
    }
    
    # Save features
    output_path = OUTPUT_DIR / "features"
    print(f"\nSaving features to {output_path}.npz...")
    save_features(X, y, metadata, str(output_path))
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*70)
    print(f"Output saved to: {OUTPUT_DIR}/")
    print(f"  - features.npz (X, y arrays)")
    print(f"  - metadata.json (config and label mapping)")
    print("\nNext step: Run tasks/task3.py to train the baseline CNN")
    print("="*70)


if __name__ == "__main__":
    main()
