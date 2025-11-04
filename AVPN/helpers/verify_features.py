#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick verification of extracted features"""

import numpy as np
import json
from pathlib import Path

# Load features (using relative paths from script location)
base_dir = Path(__file__).parent.parent
features_path = base_dir / "features" / "features.npz"
metadata_path = base_dir / "features" / "features.json"

print("="*70)
print("FEATURE EXTRACTION VERIFICATION")
print("="*70)

# Load NPZ
data = np.load(features_path)
X = data['X']
y = data['y']

print(f"\n✓ Loaded {features_path}")
print(f"  Arrays: {list(data.keys())}")
print(f"\n  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Load metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"\n✓ Loaded {metadata_path}")
print(f"\nMetadata:")
print(f"  Num samples: {metadata['num_samples']}")
print(f"  Num classes: {metadata['num_classes']}")
print(f"  n_mfcc: {metadata['n_mfcc']}")
print(f"  max_frames: {metadata['max_frames']}")
print(f"  Sample rate: {metadata['sample_rate']} Hz")
print(f"  Window: {metadata['window_ms']} ms")
print(f"  Hop: {metadata['hop_ms']} ms")
print(f"  FFT size: {metadata['n_fft']}")
print(f"  Mel filters: {metadata['n_mels']}")

print(f"\n✓ Detected classes ({len(metadata['labels'])} total):")
for i, label in enumerate(metadata['labels']):
    print(f"  {i:2d}. {label}")

print("\n" + "="*70)
print("SPECIFICATION COMPLIANCE CHECK")
print("="*70)

checks = [
    ("Sample rate", metadata['sample_rate'] == 16000, "16 kHz"),
    ("Window length", metadata['window_ms'] == 25, "25 ms"),
    ("Hop length", metadata['hop_ms'] == 10, "10 ms"),
    ("FFT size", metadata['n_fft'] == 512, "512"),
    ("Mel filters", metadata['n_mels'] == 40, "40"),
    ("MFCC coeffs", metadata['n_mfcc'] == 120, "120 (40+Δ+ΔΔ)"),
    ("Input shape", X.shape == (799, 120, 99, 1), "(N, 120, max_frames, 1)")
]

all_pass = True
for check_name, passed, expected in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} {check_name:20s} → {expected}")
    all_pass = all_pass and passed

print("\n" + "="*70)
if all_pass:
    print("✅ ALL SPECIFICATION CHECKS PASSED!")
    print("Ready to proceed with training: python train_model.py")
else:
    print("❌ SOME CHECKS FAILED - Review specification")
print("="*70)
