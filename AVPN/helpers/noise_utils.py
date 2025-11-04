"""
Task 4: Noise Compensation Utilities
Provides functions for adding noise at controlled SNR levels and batch processing.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Tuple
import json
from scipy import signal as sp_signal


def compute_power(x: np.ndarray) -> float:
    """
    Compute signal power as mean of squares.
    
    Args:
        x: Audio signal array
        
    Returns:
        Power (mean of squared values)
    """
    return np.mean(x ** 2)


def mix_to_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """
    Mix speech with noise at a target SNR level.
    
    Formula:
        Ps = mean(speech**2)
        Pn = mean(noise**2)
        alpha = sqrt(Ps / (Pn * 10**(target_snr_db/10)))
        y = speech + alpha * noise[:len(speech)]
    
    Args:
        speech: Clean speech signal
        noise: Noise signal (will be tiled/trimmed to match speech length)
        target_snr_db: Target SNR in decibels
        
    Returns:
        Noisy speech signal
    """
    # Ensure noise matches speech length
    if len(noise) < len(speech):
        # Tile noise to cover speech length
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise_extended = np.tile(noise, repeats)[:len(speech)]
    else:
        # Trim noise to match speech length
        noise_extended = noise[:len(speech)]
    
    # Compute powers
    Ps = compute_power(speech)
    Pn = compute_power(noise_extended)
    
    # Avoid division by zero
    if Pn < 1e-10:
        print(f"Warning: Noise power too low ({Pn}), adding minimal noise")
        return speech.copy()
    
    # Calculate scaling factor
    alpha = np.sqrt(Ps / (Pn * 10 ** (target_snr_db / 10)))
    
    # Mix speech and scaled noise
    y = speech + alpha * noise_extended
    
    return y


def verify_snr(noisy: np.ndarray, clean: np.ndarray) -> float:
    """
    Compute the actual SNR between noisy and clean signals.
    
    SNR_dB = 10 * log10(P_signal / P_noise)
    where P_noise = mean((noisy - clean)**2)
    
    Args:
        noisy: Noisy signal
        clean: Clean signal
        
    Returns:
        Measured SNR in dB
    """
    # Ensure same length
    min_len = min(len(noisy), len(clean))
    noisy = noisy[:min_len]
    clean = clean[:min_len]
    
    # Compute signal and noise powers
    P_signal = compute_power(clean)
    noise_component = noisy - clean
    P_noise = compute_power(noise_component)
    
    # Avoid division by zero
    if P_noise < 1e-10:
        return float('inf')
    
    # Calculate SNR
    snr_db = 10 * np.log10(P_signal / P_noise)
    
    return snr_db


def batch_mix(
    input_dir_speech: str,
    input_dir_noise: str,
    out_dir: str,
    snr_list: List[float] = [20, 10, 0],
    sample_rate: int = 16000
) -> dict:
    """
    Create noisy versions of all speech files at multiple SNR levels.
    
    For each clean WAV file, produces noisy versions at each SNR in snr_list.
    Saves under: out_dir/SNR{snr}/ subdirectories.
    Handles stereo-to-mono conversion and resampling automatically.
    
    Args:
        input_dir_speech: Directory containing clean speech WAV files
        input_dir_noise: Directory containing noise WAV files
        out_dir: Output directory for noisy files
        snr_list: List of target SNR levels in dB
        sample_rate: Expected sample rate (for verification)
        
    Returns:
        Dictionary with statistics about processing
    """
    speech_dir = Path(input_dir_speech)
    noise_dir = Path(input_dir_noise)
    output_dir = Path(out_dir)
    
    # Create output directories for each SNR level
    for snr in snr_list:
        snr_dir = output_dir / f"SNR{int(snr)}"
        snr_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all noise files
    noise_files = list(noise_dir.glob("*.wav"))
    if not noise_files:
        raise ValueError(f"No noise files found in {noise_dir}")
    
    print(f"Loading {len(noise_files)} noise file(s)...")
    noise_signals = []
    for noise_file in noise_files:
        noise, sr = sf.read(noise_file)
        
        # Convert stereo to mono if needed
        if noise.ndim > 1:
            noise = np.mean(noise, axis=1)
        
        # Resample if needed
        if sr != sample_rate:
            n_samples = int(len(noise) * sample_rate / sr)
            noise = sp_signal.resample(noise, n_samples)
            print(f"  Resampled {noise_file.name} from {sr} Hz to {sample_rate} Hz")
        
        noise_signals.append(noise)
    
    # Concatenate all noise into one long signal
    combined_noise = np.concatenate(noise_signals)
    print(f"Combined noise length: {len(combined_noise)} samples ({len(combined_noise)/sample_rate:.2f}s)")
    
    # Process each speech file
    speech_files = list(speech_dir.glob("*.wav"))
    if not speech_files:
        raise ValueError(f"No speech files found in {speech_dir}")
    
    print(f"\nProcessing {len(speech_files)} speech files at {len(snr_list)} SNR levels...")
    
    stats = {
        'total_files': len(speech_files),
        'snr_levels': snr_list,
        'snr_errors': {snr: [] for snr in snr_list},
        'files_processed': 0,
        'resampled': 0,
        'stereo_converted': 0
    }
    
    for i, speech_file in enumerate(speech_files):
        # Load clean speech
        speech, sr = sf.read(speech_file)
        
        # Convert stereo to mono if needed
        if speech.ndim > 1:
            speech = np.mean(speech, axis=1)
            stats['stereo_converted'] += 1
        
        # Resample if needed
        if sr != sample_rate:
            n_samples = int(len(speech) * sample_rate / sr)
            speech = sp_signal.resample(speech, n_samples)
            stats['resampled'] += 1
        
        # Get a random segment of noise for this file
        if len(combined_noise) > len(speech):
            start_idx = np.random.randint(0, len(combined_noise) - len(speech))
            noise_segment = combined_noise[start_idx:start_idx + len(speech)]
        else:
            noise_segment = combined_noise
        
        # Create noisy version at each SNR level
        for snr in snr_list:
            # Mix speech with noise
            noisy = mix_to_snr(speech, noise_segment, target_snr_db=snr)
            
            # Verify actual SNR
            actual_snr = verify_snr(noisy, speech)
            snr_error = abs(actual_snr - snr)
            stats['snr_errors'][snr].append(snr_error)
            
            # Save noisy file
            output_path = output_dir / f"SNR{int(snr)}" / speech_file.name
            sf.write(output_path, noisy, sample_rate)
        
        stats['files_processed'] += 1
        
        # Progress indicator
        if (i + 1) % 50 == 0 or (i + 1) == len(speech_files):
            print(f"  Processed {i + 1}/{len(speech_files)} files")
    
    # Calculate and display SNR statistics
    print(f"\n=== Processing Summary ===")
    print(f"Files processed: {stats['files_processed']}/{stats['total_files']}")
    print(f"Files resampled: {stats['resampled']}")
    print(f"Stereo->Mono conversions: {stats['stereo_converted']}")
    
    print("\n=== SNR Verification ===")
    for snr in snr_list:
        errors = stats['snr_errors'][snr]
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        within_tolerance = sum(1 for e in errors if e <= 0.5)
        print(f"SNR {snr:2.0f} dB: Mean error = {mean_error:.3f} dB, "
              f"Max error = {max_error:.3f} dB, "
              f"Within ±0.5 dB: {within_tolerance}/{len(errors)} "
              f"({100*within_tolerance/len(errors):.1f}%)")
    
    # Save statistics
    stats_file = output_dir / "mixing_stats.json"
    # Convert numpy types to Python types for JSON serialization
    stats_serializable = {
        'total_files': stats['total_files'],
        'files_processed': stats['files_processed'],
        'resampled': stats['resampled'],
        'stereo_converted': stats['stereo_converted'],
        'snr_levels': [float(s) for s in stats['snr_levels']],
        'snr_errors': {
            str(snr): [float(e) for e in errors]
            for snr, errors in stats['snr_errors'].items()
        }
    }
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    print(f"\nStatistics saved to {stats_file}")
    print(f"Noisy files saved to {output_dir}/SNR{{20,10,0}}/")
    
    return stats


if __name__ == "__main__":
    # Example usage and testing
    print("=== Noise Utils Module ===")
    print("Testing basic functionality...\n")
    
    # Generate test signals
    duration = 2.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a simple speech-like signal (sine wave)
    speech = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Create noise (white noise)
    noise = np.random.randn(len(speech)) * 0.1
    
    # Test SNR mixing
    print("Testing SNR mixing:")
    for target_snr in [20, 10, 0, -5]:
        noisy = mix_to_snr(speech, noise, target_snr)
        actual_snr = verify_snr(noisy, speech)
        error = abs(actual_snr - target_snr)
        status = "✓" if error <= 0.5 else "✗"
        print(f"  Target: {target_snr:5.1f} dB, Actual: {actual_snr:5.1f} dB, "
              f"Error: {error:.3f} dB {status}")
    
    print("\nAll tests completed!")
