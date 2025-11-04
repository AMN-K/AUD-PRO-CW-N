"""
task2_noisy.py
==============
Extract features for noisy datasets (Task 4).
Supports custom audio directories and output paths for different SNR levels.
"""

import sys
from pathlib import Path

# Add parent directory to path for helper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
import json
import argparse

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
        Speaker name in lowercase
    """
    stem = Path(filename).stem
    # Remove trailing digits
    name = ''.join([c for c in stem if c.isalpha()]).lower()
    return name


def extract_features_from_directory(audio_dir, output_path, description=""):
    """
    Extract features from all audio files in a directory.
    
    Parameters
    ----------
    audio_dir : str or Path
        Directory containing audio files
    output_path : str or Path
        Path where features will be saved (e.g., 'features/noisy_SNR20.npz')
    description : str
        Description of the dataset (e.g., 'SNR20', 'Clean')
    """
    audio_dir = Path(audio_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"FEATURE EXTRACTION - {description}")
    print("="*70)
    print(f"Audio directory: {audio_dir}")
    print(f"Output: {output_path}")
    print("="*70)
    
    # Configuration (matching specification)
    SR = 16000
    WIN_MS = 25
    HOP_MS = 10
    N_FFT = 512
    N_MELS = 40
    KEEP_MFCC = 40
    
    # Get all audio files
    audio_files = sorted(list(audio_dir.glob("*.wav")))
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    print(f"\nFound {len(audio_files)} audio files")
    print(f"Processing with {KEEP_MFCC} MFCC + Δ + ΔΔ = 120 features\n")
    
    # Extract features
    all_features = []
    labels_list = []
    
    for audio_file in tqdm(audio_files, desc="Extracting features"):
        # Load audio
        signal = load_wav(str(audio_file), target_sr=SR)
        
        # Frame the signal
        frames = frame_signal(signal, SR, WIN_MS, HOP_MS, N_FFT)
        
        # Extract MFCCs (mfcc_from_frames handles mel filterbank internally)
        mfcc = mfcc_from_frames(frames, SR, n_fft=N_FFT, keep_mfcc=KEEP_MFCC)
        
        # Add delta and delta-delta
        features_with_deltas = add_deltas(mfcc)
        
        # Store
        all_features.append(features_with_deltas)
        
        # Extract label
        label = extract_name_from_filename(audio_file.name)
        labels_list.append(label)
    
    # Determine max_frames dynamically
    frame_counts = [f.shape[0] for f in all_features]
    max_frames = max(frame_counts)
    min_frames = min(frame_counts)
    mean_frames = np.mean(frame_counts)
    
    print(f"\n✓ Extracted features from {len(all_features)} files")
    print(f"  Frame statistics: min={min_frames}, max={max_frames}, mean={mean_frames:.1f}")
    print(f"  Using max_frames = {max_frames} for padding")
    
    # Pad all features to max_frames
    padded_features = []
    for features in all_features:
        padded = pad_features(features, max_frames)
        padded_features.append(padded)
    
    # Stack into numpy array: (N, max_frames, 120)
    X = np.array(padded_features)
    
    # Transpose to (N, 120, max_frames) for CNN input
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
    print(f"\n✓ Detected {num_classes} speaker classes")
    
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
        'label_to_idx': label_to_idx,
        'description': description,
        'audio_directory': str(audio_dir)
    }
    
    # Save features
    print(f"\nSaving features to {output_path}...")
    save_features(X, y, metadata, str(output_path))
    
    print("\n" + "="*70)
    print(f"✓ Feature extraction complete for {description}")
    print(f"  Output: {output_path}")
    print(f"  Shape: {X.shape}, Classes: {num_classes}")
    print("="*70 + "\n")
    
    return X.shape, num_classes


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Extract MFCC features for speaker recognition")
    parser.add_argument(
        '--audio_dir',
        type=str,
        default='Audio',
        help='Directory containing audio files (default: Audio)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='features/features',
        help='Output path for features file (default: features/features)'
    )
    parser.add_argument(
        '--description',
        type=str,
        default='Clean',
        help='Description of the dataset (default: Clean)'
    )
    
    args = parser.parse_args()
    
    extract_features_from_directory(
        audio_dir=args.audio_dir,
        output_path=args.output,
        description=args.description
    )


if __name__ == "__main__":
    main()
