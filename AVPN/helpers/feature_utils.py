"""
Task 2: MFCC Feature Extraction Module

Implements lecture-aligned short-time audio analysis for speaker recognition.

Specification:
- Sample rate: 16 kHz (mono, float32)
- Frame length: 25 ms (~400 samples @ 16kHz)
- Hop length: 10 ms (~160 samples @ 16kHz)
- Window: Hamming
- FFT size: 512
- Mel filters: 40 triangular filters
- MFCC: 40 coefficients + Δ + ΔΔ = 120 features total

Author: Refactored for assignment compliance
Date: November 2, 2025
"""

import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal.windows import hamming as hamming_window
import json
from pathlib import Path
import librosa

# Constants
DEFAULT_SR = 16000
DEFAULT_WIN_MS = 25
DEFAULT_HOP_MS = 10
DEFAULT_N_FFT = 512
DEFAULT_N_MELS = 40


def load_wav(path: str, target_sr: int = DEFAULT_SR) -> np.ndarray:
    """
    Load WAV file as mono float32, resample if needed.
    
    Args:
        path: Path to WAV file
        target_sr: Target sample rate (default 16000 Hz)
        
    Returns:
        signal: 1D numpy array, mono, float32, normalized to [-1, 1]
    """
    # Load audio using soundfile
    signal, sr = sf.read(path, dtype='float32')
    
    # Convert to mono if stereo
    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
    
    return signal


def trim_silence(signal: np.ndarray,
                 top_db: float = 35.0,
                 min_duration: float = 0.0,
                 sr: int = DEFAULT_SR) -> np.ndarray:
    """Trim leading/trailing sections that are far below peak energy.

    Args:
        signal: Input waveform (mono)
        top_db: Threshold in decibels relative to peak (default 35 dB)
        min_duration: Minimum duration to keep after trimming (seconds)
        sr: Sample rate used to interpret ``min_duration``

    Returns:
        Trimmed waveform. Falls back to original if trimming removes
        everything or drops below ``min_duration``.
    """
    if signal.size == 0:
        return signal

    try:
        trimmed, idx = librosa.effects.trim(signal, top_db=top_db)
    except Exception:
        # If librosa fails (e.g. short clip), return original
        return signal

    if trimmed.size == 0:
        return signal

    if min_duration > 0:
        if trimmed.shape[0] < int(min_duration * sr):
            return signal

    return trimmed.astype(np.float32, copy=False)


def frame_signal(x: np.ndarray, sr: int, win_ms: float = DEFAULT_WIN_MS,
                 hop_ms: float = DEFAULT_HOP_MS, 
                 window: str = "hamming") -> np.ndarray:
    """
    Frame signal into overlapping windows.
    
    Args:
        x: Input signal (1D array)
        sr: Sample rate
        win_ms: Window length in milliseconds (default 25 ms)
        hop_ms: Hop length in milliseconds (default 10 ms)
        window: Window type (default "hamming")
        
    Returns:
        frames: 2D array (num_frames, frame_length)
    """
    frame_size = int(win_ms * sr / 1000)  # Convert ms to samples
    frame_step = int(hop_ms * sr / 1000)
    
    # Calculate number of frames
    signal_length = len(x)
    num_frames = int(np.ceil((signal_length - frame_size) / frame_step)) + 1
    
    # Zero padding to ensure all frames are equal length
    pad_signal_length = num_frames * frame_step + frame_size
    padded_signal = np.append(x, np.zeros(pad_signal_length - signal_length))
    
    # Create frames
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_size
        frame = padded_signal[start:end]
        
        # Apply window
        if window == "hamming":
            frame = frame * hamming_window(frame_size)
        
        frames[i] = frame
    
    return frames


def mel_filterbank(sr: int, n_fft: int = DEFAULT_N_FFT, 
                   n_mels: int = DEFAULT_N_MELS,
                   fmin: float = 0, fmax: float = None) -> np.ndarray:
    """
    Create Mel-scaled triangular filterbank matrix.
    
    Args:
        sr: Sample rate
        n_fft: FFT size (default 512)
        n_mels: Number of Mel filters (default 40)
        fmin: Minimum frequency (default 0 Hz)
        fmax: Maximum frequency (default sr/2)
        
    Returns:
        filterbank: 2D array (n_mels, n_fft//2 + 1)
    """
    if fmax is None:
        fmax = sr / 2.0
    
    # Helper functions for Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)
    
    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595.0) - 1)
    
    # Create Mel-spaced points
    low_mel = hz_to_mel(fmin)
    high_mel = hz_to_mel(fmax)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Convert Hz to FFT bin numbers
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Create filterbank
    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    
    for i in range(1, n_mels + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        
        # Rising slope
        for j in range(left, center):
            if center > left:
                fbank[i - 1, j] = (j - left) / (center - left)
        
        # Falling slope
        for j in range(center, right):
            if right > center:
                fbank[i - 1, j] = (right - j) / (right - center)
    
    return fbank


def mfcc_from_frames(frames: np.ndarray, sr: int,
                     n_fft: int = DEFAULT_N_FFT,
                     n_mels: int = DEFAULT_N_MELS,
                     use_log_energy: bool = False,
                     keep_mfcc: int = 40) -> np.ndarray:
    """
    Compute MFCC features from framed signal.
    
    Pipeline: FFT → Mel filterbank → Log → DCT → Keep coefficients
    
    Args:
        frames: Framed signal (num_frames, frame_length)
        sr: Sample rate
        n_fft: FFT size (default 512)
        n_mels: Number of Mel filters (default 40)
        use_log_energy: Prepend log energy as coefficient 0 (default False)
        keep_mfcc: Number of MFCC coefficients to keep (default 40)
        
    Returns:
        mfcc: 2D array (num_frames, keep_mfcc)
    """
    # Get Mel filterbank
    mel_fbanks = mel_filterbank(sr, n_fft, n_mels)
    
    mfcc_features = []
    
    for frame in frames:
        # FFT (already windowed in frame_signal)
        fft_frame = np.fft.fft(frame, n=n_fft)
        mag_spec = np.abs(fft_frame[:n_fft // 2 + 1])
        
        # Apply Mel filterbank
        mel_energies = np.dot(mel_fbanks, mag_spec)
        
        # Log (for human perception)
        log_mel_energies = np.log(mel_energies + 1e-8)
        
        # DCT (Discrete Cosine Transform)
        mfcc_vector = dct(log_mel_energies, type=2, norm='ortho')
        
        # Keep only first 'keep_mfcc' coefficients
        mfcc_frame = mfcc_vector[:keep_mfcc]
        
        # Optionally add log energy
        if use_log_energy:
            energy = np.sum(frame ** 2)
            log_energy = np.log(energy + 1e-8)
            mfcc_frame = np.insert(mfcc_frame, 0, log_energy)
        
        mfcc_features.append(mfcc_frame)
    
    return np.array(mfcc_features)


def add_deltas(mfcc: np.ndarray, order: int = 2, width: int = 2) -> np.ndarray:
    """
    Compute temporal derivatives (Δ and ΔΔ).
    
    Args:
        mfcc: Static MFCC features (num_frames, n_mfcc)
        order: Derivative order (1=Δ, 2=Δ+ΔΔ) (default 2)
        width: Window width for derivative computation (default 2)
        
    Returns:
        augmented: Concatenated [static | Δ | ΔΔ] along feature axis
                   Shape: (num_frames, n_mfcc * (order + 1))
                   Example: 40 MFCC → (frames, 120) for order=2
    """
    def compute_delta(features, N=2):
        """Compute delta features using regression formula."""
        num_frames, num_features = features.shape
        delta_feat = np.zeros_like(features)
        
        # Pad at boundaries
        padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
        
        # Denominator for regression formula
        denom = 2 * sum([n**2 for n in range(1, N + 1)])
        
        # Compute delta for each frame
        for t in range(num_frames):
            delta_feat[t] = sum([
                n * (padded[t + N + n] - padded[t + N - n]) 
                for n in range(1, N + 1)
            ]) / denom
        
        return delta_feat
    
    # Start with static features
    result = [mfcc]
    
    # Compute delta (Δ)
    if order >= 1:
        delta = compute_delta(mfcc, N=width)
        result.append(delta)
    
    # Compute delta-delta (ΔΔ)
    if order >= 2:
        delta_delta = compute_delta(delta, N=width)
        result.append(delta_delta)
    
    # Concatenate along feature axis
    return np.hstack(result)


def pad_features(feat_2d: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Pad or truncate feature matrix to max_frames along time axis.
    
    Args:
        feat_2d: Feature matrix (num_frames, n_features)
        max_frames: Target number of frames
        
    Returns:
        padded: Shape (max_frames, n_features)
                Pads with zeros if num_frames < max_frames
                Truncates if num_frames > max_frames
    """
    num_frames, n_features = feat_2d.shape
    
    if num_frames < max_frames:
        # Pad with zeros
        padding = np.zeros((max_frames - num_frames, n_features))
        return np.vstack([feat_2d, padding])
    elif num_frames > max_frames:
        # Truncate
        return feat_2d[:max_frames, :]
    else:
        # Already correct size
        return feat_2d


def normalize_features(X_train: np.ndarray, X_val: np.ndarray = None,
                      X_test: np.ndarray = None) -> tuple:
    """
    Normalize features using training set statistics.
    
    Args:
        X_train: Training features (N, n_mfcc, T, 1)
        X_val: Validation features (optional)
        X_test: Test features (optional)
        
    Returns:
        (X_train_norm, X_val_norm, X_test_norm, mean, std)
        All normalized using X_train mean/std
    """
    # Compute statistics from training data
    mean = np.mean(X_train)
    std = np.std(X_train)
    
    # Normalize training data
    X_train_norm = (X_train - mean) / (std + 1e-8)
    
    # Normalize validation data using training stats
    X_val_norm = None
    if X_val is not None:
        X_val_norm = (X_val - mean) / (std + 1e-8)
    
    # Normalize test data using training stats
    X_test_norm = None
    if X_test is not None:
        X_test_norm = (X_test - mean) / (std + 1e-8)
    
    return X_train_norm, X_val_norm, X_test_norm, mean, std


def save_features(X: np.ndarray, y: np.ndarray, metadata: dict, out_path: str):
    """
    Save features, labels, and metadata to disk.
    
    Args:
        X: Features (N, n_mfcc, T)
        y: Labels (N,) as strings
        metadata: Dict with keys: n_mfcc, max_frames, mean, std, classes, sr
        out_path: Base path (will create .npz and .json)
    """
    # Create output directory if it doesn't exist
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Strip .npz extension if already present to avoid .npz.npz
    base = str(out_path)
    if base.endswith('.npz'):
        base = base[:-4]
    
    # Save features and labels as NPZ
    np.savez(f'{base}.npz', X=X, y=y)
    
    # Save metadata as JSON
    with open(f'{base}.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved: {base}.npz")
    print(f"Saved: {base}.json")


def load_features(base_path: str) -> tuple:
    """
    Load features from disk.
    
    Args:
        base_path: Base path (without extension)
        
    Returns:
        (X, y, metadata)
    """
    # Load NPZ
    data = np.load(f'{base_path}.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Load metadata
    with open(f'{base_path}.json', 'r') as f:
        metadata = json.load(f)
    
    return X, y, metadata


if __name__ == "__main__":
    # Quick test
    print("Feature Extraction Module")
    print(f"Default config: {DEFAULT_SR} Hz, {DEFAULT_WIN_MS}ms win, {DEFAULT_HOP_MS}ms hop")
    print(f"FFT: {DEFAULT_N_FFT}, Mel filters: {DEFAULT_N_MELS}")
    print(f"Features: 40 MFCC + Δ + ΔΔ = 120 total")
