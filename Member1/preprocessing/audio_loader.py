"""
Audio Loading and Normalization Module

Handles loading audio files from various formats and normalizing them
to a consistent sample rate and amplitude for feature extraction.
"""

import os
from typing import Tuple, Optional, Union
import numpy as np
import librosa


def load_audio(
    path: str,
    sr: int = 22050,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and normalize to target sample rate.
    
    Parameters
    ----------
    path : str
        Path to the audio file (supports wav, mp3, flac, ogg, m4a)
    sr : int, optional
        Target sample rate in Hz (default: 22050)
    mono : bool, optional
        Whether to convert to mono (default: True)
    duration : float, optional
        Duration in seconds to load (default: None = full file)
    offset : float, optional
        Start reading from this time in seconds (default: 0.0)
        
    Returns
    -------
    y : np.ndarray
        Audio time series (1D for mono, 2D for stereo)
    sr : int
        Sample rate of the audio
        
    Raises
    ------
    FileNotFoundError
        If the audio file does not exist
    ValueError
        If the file format is not supported or file is corrupted
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    # Supported formats
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    ext = os.path.splitext(path)[1].lower()
    
    if ext not in supported_extensions:
        raise ValueError(f"Unsupported audio format: {ext}. Supported: {supported_extensions}")
    
    try:
        # Load with librosa (handles resampling automatically)
        y, loaded_sr = librosa.load(
            path,
            sr=sr,
            mono=mono,
            duration=duration,
            offset=offset
        )
        
        return y, loaded_sr
        
    except Exception as e:
        raise ValueError(f"Error loading audio file {path}: {str(e)}")


def validate_audio(
    y: np.ndarray,
    sr: int,
    min_duration: float = 1.0,
    max_silence_ratio: float = 0.9,
    clipping_threshold: float = 0.99
) -> dict:
    """
    Validate audio quality and return diagnostic information.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    min_duration : float, optional
        Minimum required duration in seconds (default: 1.0)
    max_silence_ratio : float, optional
        Maximum allowed ratio of silent samples (default: 0.9)
    clipping_threshold : float, optional
        Threshold above which samples are considered clipped (default: 0.99)
        
    Returns
    -------
    dict
        Validation results including:
        - is_valid: Overall validity
        - duration: Duration in seconds
        - is_silent: Whether audio is mostly silent
        - is_clipped: Whether audio shows clipping
        - issues: List of detected issues
    """
    issues = []
    
    # Calculate duration
    duration = len(y) / sr
    
    # Check minimum duration
    if duration < min_duration:
        issues.append(f"Audio too short: {duration:.2f}s < {min_duration:.2f}s")
    
    # Check for silence (using RMS energy)
    rms = np.sqrt(np.mean(y**2))
    silence_threshold = 0.001
    silence_ratio = np.mean(np.abs(y) < silence_threshold)
    
    is_silent = silence_ratio > max_silence_ratio
    if is_silent:
        issues.append(f"Audio is mostly silent: {silence_ratio*100:.1f}% silent samples")
    
    # Check for clipping
    clipping_ratio = np.mean(np.abs(y) > clipping_threshold)
    is_clipped = clipping_ratio > 0.01  # More than 1% clipped
    
    if is_clipped:
        issues.append(f"Audio shows clipping: {clipping_ratio*100:.2f}% samples clipped")
    
    # Check for NaN or Inf values
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        issues.append("Audio contains NaN or Inf values")
    
    return {
        'is_valid': len(issues) == 0,
        'duration': duration,
        'sample_rate': sr,
        'num_samples': len(y),
        'rms_energy': float(rms),
        'is_silent': is_silent,
        'silence_ratio': float(silence_ratio),
        'is_clipped': is_clipped,
        'clipping_ratio': float(clipping_ratio),
        'issues': issues
    }


def normalize_amplitude(
    y: np.ndarray,
    method: str = 'peak',
    target_db: float = -3.0
) -> np.ndarray:
    """
    Normalize audio amplitude.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    method : str, optional
        Normalization method: 'peak', 'rms', or 'lufs' (default: 'peak')
    target_db : float, optional
        Target level in dB (default: -3.0 for headroom)
        
    Returns
    -------
    np.ndarray
        Normalized audio time series
    """
    if len(y) == 0:
        return y
    
    # Convert target dB to linear scale
    target_amplitude = 10 ** (target_db / 20.0)
    
    if method == 'peak':
        # Peak normalization
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y * (target_amplitude / peak)
            
    elif method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y = y * (target_amplitude / rms)
            # Clip to prevent clipping after RMS normalization
            y = np.clip(y, -1.0, 1.0)
            
    elif method == 'lufs':
        # Simplified LUFS-like normalization (K-weighted)
        # For full LUFS, would need pyloudnorm library
        # This is a simplified version using weighted RMS
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y = y * (target_amplitude / rms)
            y = np.clip(y, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return y


def trim_silence(
    y: np.ndarray,
    top_db: float = 20.0,
    frame_length: int = 2048,
    hop_length: int = 512
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim leading and trailing silence from audio.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    top_db : float, optional
        Threshold in dB below peak (default: 20.0)
    frame_length : int, optional
        Frame length for energy computation (default: 2048)
    hop_length : int, optional
        Hop length for frames (default: 512)
        
    Returns
    -------
    y_trimmed : np.ndarray
        Trimmed audio
    indices : tuple
        (start_sample, end_sample) of the non-silent region
    """
    y_trimmed, indices = librosa.effects.trim(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    return y_trimmed, indices


def resample_audio(
    y: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to a different sample rate.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    orig_sr : int
        Original sample rate
    target_sr : int
        Target sample rate
        
    Returns
    -------
    np.ndarray
        Resampled audio
    """
    if orig_sr == target_sr:
        return y
    
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def get_audio_info(path: str) -> dict:
    """
    Get metadata about an audio file without loading it fully.
    
    Parameters
    ----------
    path : str
        Path to the audio file
        
    Returns
    -------
    dict
        Audio file information including duration, sample rate, channels
    """
    import soundfile as sf
    
    try:
        info = sf.info(path)
        return {
            'path': path,
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype
        }
    except Exception:
        # Fallback for formats soundfile doesn't support
        y, sr = librosa.load(path, sr=None, duration=0.1)
        duration = librosa.get_duration(path=path)
        return {
            'path': path,
            'duration': duration,
            'sample_rate': sr,
            'channels': 1 if y.ndim == 1 else y.shape[0],
            'format': 'unknown',
            'subtype': 'unknown'
        }
