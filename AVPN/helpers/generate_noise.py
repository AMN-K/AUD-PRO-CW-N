"""
Generate synthetic noise files for Task 4 noise compensation experiments.
Creates white noise and pink noise files for mixing with clean speech.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_white_noise(duration_sec: float, sample_rate: int = 16000, amplitude: float = 0.1) -> np.ndarray:
    """Generate white noise signal."""
    n_samples = int(duration_sec * sample_rate)
    noise = np.random.randn(n_samples) * amplitude
    return noise


def generate_pink_noise(duration_sec: float, sample_rate: int = 16000, amplitude: float = 0.1) -> np.ndarray:
    """
    Generate pink noise (1/f noise) using the Voss-McCartney algorithm.
    Pink noise has equal energy per octave.
    """
    n_samples = int(duration_sec * sample_rate)
    
    # Number of random sources
    n_rows = 16
    
    # Generate random values
    array = np.random.randn(n_rows, n_samples)
    
    # Apply exponential decay to each row
    weights = np.exp(-np.log(2) * np.arange(n_rows) / n_rows)
    weights = weights.reshape(-1, 1)
    
    # Sum weighted random values
    pink = np.sum(array * weights, axis=0)
    
    # Normalize and scale
    pink = pink / np.std(pink) * amplitude
    
    return pink


def generate_babble_noise(duration_sec: float, sample_rate: int = 16000, n_voices: int = 5, amplitude: float = 0.1) -> np.ndarray:
    """
    Generate babble noise by mixing multiple frequency-modulated signals.
    Simulates background speech/crowd noise.
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples)
    
    babble = np.zeros(n_samples)
    
    # Mix multiple "voices" with different fundamental frequencies
    base_freqs = [80, 120, 180, 220, 280][:n_voices]
    
    for f0 in base_freqs:
        # Add harmonics with random modulation
        for harmonic in range(1, 6):
            freq = f0 * harmonic
            # Random amplitude and phase modulation
            mod_freq = np.random.uniform(0.5, 3.0)
            phase_mod = np.random.uniform(0, 2*np.pi)
            amp_mod = np.random.uniform(0.3, 1.0)
            
            signal = amp_mod * np.sin(2 * np.pi * freq * t + 
                                     0.3 * np.sin(2 * np.pi * mod_freq * t + phase_mod))
            babble += signal
    
    # Normalize and scale
    babble = babble / np.std(babble) * amplitude
    
    return babble


def main():
    """Generate noise files for the experiment."""
    
    # Configuration
    sample_rate = 16000
    duration = 60  # 60 seconds of noise for each type
    output_dir = Path("Audio_noise")
    output_dir.mkdir(exist_ok=True)
    
    print("=== Generating Noise Files ===")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} seconds")
    print(f"Output directory: {output_dir}\n")
    
    # Generate white noise
    print("Generating white noise...")
    white = generate_white_noise(duration, sample_rate, amplitude=0.15)
    white_path = output_dir / "white_noise.wav"
    sf.write(white_path, white, sample_rate)
    print(f"  Saved: {white_path} ({len(white)} samples)")
    
    # Generate pink noise
    print("Generating pink noise...")
    pink = generate_pink_noise(duration, sample_rate, amplitude=0.15)
    pink_path = output_dir / "pink_noise.wav"
    sf.write(pink_path, pink, sample_rate)
    print(f"  Saved: {pink_path} ({len(pink)} samples)")
    
    # Generate babble noise
    print("Generating babble noise...")
    babble = generate_babble_noise(duration, sample_rate, n_voices=5, amplitude=0.15)
    babble_path = output_dir / "babble_noise.wav"
    sf.write(babble_path, babble, sample_rate)
    print(f"  Saved: {babble_path} ({len(babble)} samples)")
    
    print("\n✓ All noise files generated successfully!")
    print(f"\nNoise bank ready for Task 4 experiments in: {output_dir}/")


if __name__ == "__main__":
    main()
