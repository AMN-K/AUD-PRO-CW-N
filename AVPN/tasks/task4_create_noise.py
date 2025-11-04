"""
Generate noisy datasets at different SNR levels for Task 4.
Creates SNR20, SNR10, and SNR0 versions of the clean audio dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path for helper imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.noise_utils import batch_mix


def main():
    """Generate noisy datasets using the noise bank."""
    
    # Configuration
    clean_audio_dir = "Audio"
    noise_dir = "Audio_noise"
    output_dir = "Audio_noisy"
    snr_levels = [20, 10, 0]  # SNR levels in dB
    sample_rate = 16000
    
    print("=== Task 4: Generating Noisy Datasets ===\n")
    print(f"Clean audio: {clean_audio_dir}")
    print(f"Noise bank: {noise_dir}")
    print(f"Output: {output_dir}")
    print(f"SNR levels: {snr_levels} dB")
    print(f"Sample rate: {sample_rate} Hz\n")
    
    # Check directories exist
    if not Path(clean_audio_dir).exists():
        raise FileNotFoundError(f"Clean audio directory not found: {clean_audio_dir}")
    
    if not Path(noise_dir).exists():
        raise FileNotFoundError(f"Noise directory not found: {noise_dir}")
    
    # Generate noisy datasets
    stats = batch_mix(
        input_dir_speech=clean_audio_dir,
        input_dir_noise=noise_dir,
        out_dir=output_dir,
        snr_list=snr_levels,
        sample_rate=sample_rate
    )
    
    print("\n" + "="*60)
    print("✓ Noisy dataset generation complete!")
    print(f"  Total files processed: {stats['files_processed']}")
    print(f"  SNR levels: {stats['snr_levels']}")
    print(f"  Output directories:")
    for snr in snr_levels:
        snr_dir = Path(output_dir) / f"SNR{int(snr)}"
        n_files = len(list(snr_dir.glob("*.wav")))
        print(f"    - {snr_dir}: {n_files} files")
    print("="*60)


if __name__ == "__main__":
    main()
