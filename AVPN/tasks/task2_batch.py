"""
Batch extract features for all noisy conditions (SNR 20, 10, 0).
"""

import sys
from pathlib import Path

# Add parent directory to path for helper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.task2_noisy import extract_features_from_directory


def main():
    """Extract features for all noisy conditions."""
    
    # Use relative paths from the project root
    base_dir = Path(__file__).parent.parent
    audio_noisy_dir = base_dir / "Audio_noisy"
    features_dir = base_dir / "features"
    
    snr_levels = [20, 10, 0]
    
    print("="*70)
    print("BATCH FEATURE EXTRACTION FOR NOISY DATASETS")
    print("="*70)
    print(f"Processing {len(snr_levels)} SNR levels: {snr_levels}")
    print("="*70 + "\n")
    
    results = {}
    
    for snr in snr_levels:
        audio_dir = audio_noisy_dir / f"SNR{snr}"
        output_path = features_dir / f"noisy_SNR{snr}.npz"
        description = f"Noisy SNR {snr}dB"
        
        shape, num_classes = extract_features_from_directory(
            audio_dir=audio_dir,
            output_path=output_path,
            description=description
        )
        
        results[snr] = {'shape': shape, 'num_classes': num_classes}
    
    print("\n" + "="*70)
    print("ALL FEATURE EXTRACTIONS COMPLETE")
    print("="*70)
    for snr, info in results.items():
        print(f"  SNR {snr:2d} dB: shape={info['shape']}, classes={info['num_classes']}")
    print("="*70)


if __name__ == "__main__":
    main()
