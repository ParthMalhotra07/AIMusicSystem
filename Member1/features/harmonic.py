"""
Harmonic Feature Extraction Module

Extracts features that capture pitch and harmony:
- Chroma Features: 12-bin pitch class distribution
- Chromagram CQT: Constant-Q chroma features
- Tonnetz: Tonal centroid features (6D harmonic space)
"""

from typing import Optional
import numpy as np
import librosa


def extract_chroma(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_chroma: int = 12
) -> np.ndarray:
    """
    Extract chroma features (pitch class profile).
    
    Chroma features represent which of the 12 pitch classes
    (C, C#, D, ..., B) are present at each time frame. This
    captures the harmonic content regardless of octave.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    n_fft : int, optional
        FFT window size (default: 2048)
    n_chroma : int, optional
        Number of chroma bins (default: 12)
        
    Returns
    -------
    np.ndarray
        Chromagram of shape (n_chroma, n_frames)
    """
    return librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        n_chroma=n_chroma
    )


def extract_chroma_cqt(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_chroma: int = 12,
    fmin: Optional[float] = None,
    n_octaves: int = 7
) -> np.ndarray:
    """
    Extract Constant-Q chromagram.
    
    CQT-based chroma features are more robust to tuning variations
    and provide better frequency resolution at low frequencies.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    n_chroma : int, optional
        Number of chroma bins (default: 12)
    fmin : float, optional
        Minimum frequency (default: C1 ~32.7Hz)
    n_octaves : int, optional
        Number of octaves (default: 7)
        
    Returns
    -------
    np.ndarray
        CQT chromagram of shape (n_chroma, n_frames)
    """
    if fmin is None:
        fmin = librosa.note_to_hz('C1')
    
    return librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_chroma=n_chroma,
        fmin=fmin,
        n_octaves=n_octaves
    )


def extract_chroma_cens(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_chroma: int = 12,
    n_octaves: int = 7
) -> np.ndarray:
    """
    Extract Chroma Energy Normalized Statistics (CENS).
    
    CENS features are robust to dynamics, tempo, and articulation,
    making them ideal for music similarity and cover song detection.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    n_chroma : int, optional
        Number of chroma bins (default: 12)
    n_octaves : int, optional
        Number of octaves (default: 7)
        
    Returns
    -------
    np.ndarray
        CENS chromagram of shape (n_chroma, n_frames)
    """
    return librosa.feature.chroma_cens(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_chroma=n_chroma,
        n_octaves=n_octaves
    )


def extract_tonnetz(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract Tonnetz (tonal centroid) features.
    
    Tonnetz represents tonal content in a 6-dimensional space based
    on perfect fifths, minor thirds, and major thirds. This captures
    harmonic relationships and chord progressions.
    
    The 6 dimensions represent:
    - Fifths (y = C → G → D → ...)
    - Minor thirds (y = C → Eb → Gb → ...)
    - Major thirds (y = C → E → G# → ...)
    - And their complements
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    np.ndarray
        Tonnetz features of shape (6, n_frames)
    """
    # First compute harmonic component for better results
    y_harmonic = librosa.effects.harmonic(y)
    
    return librosa.feature.tonnetz(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length
    )


def estimate_key(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> dict:
    """
    Estimate the musical key of the audio.
    
    Uses chroma features and Krumhansl-Schmuckler key-finding algorithm.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    dict
        Key estimation including:
        - key: Key name (e.g., 'C major', 'A minor')
        - correlation: Confidence score
    """
    # Krumhansl-Kessler major and minor profiles
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Get mean chroma vector
    chroma = extract_chroma(y, sr, hop_length)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Normalize
    chroma_mean = chroma_mean / np.sum(chroma_mean)
    
    best_key = None
    best_mode = None
    best_corr = -1
    
    # Try all keys and modes
    for i in range(12):
        # Rotate profiles to test each key
        major_rotated = np.roll(MAJOR_PROFILE, i)
        minor_rotated = np.roll(MINOR_PROFILE, i)
        
        # Normalize profiles
        major_rotated = major_rotated / np.sum(major_rotated)
        minor_rotated = minor_rotated / np.sum(minor_rotated)
        
        # Compute correlations
        major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
        minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
        
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = KEY_NAMES[i]
            best_mode = 'major'
        
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = KEY_NAMES[i]
            best_mode = 'minor'
    
    return {
        'key': f"{best_key} {best_mode}",
        'key_name': best_key,
        'mode': best_mode,
        'correlation': float(best_corr)
    }


def extract_harmonic_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048
) -> dict:
    """
    Extract comprehensive harmonic features.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    hop_length : int
        Hop length
    n_fft : int
        FFT window size
        
    Returns
    -------
    dict
        Dictionary with all harmonic features
    """
    return {
        'chroma': extract_chroma(y, sr, hop_length, n_fft),
        'chroma_cqt': extract_chroma_cqt(y, sr, hop_length),
        'tonnetz': extract_tonnetz(y, sr, hop_length),
        'key': estimate_key(y, sr, hop_length)
    }


def extract_all_harmonic_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048
) -> dict:
    """
    Extract all harmonic features for the pipeline.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    hop_length : int
        Hop length
    n_fft : int
        FFT window size
        
    Returns
    -------
    dict
        Dictionary containing chroma and tonnetz matrices
    """
    return {
        'chroma': extract_chroma(y, sr, hop_length, n_fft),
        'tonnetz': extract_tonnetz(y, sr, hop_length),
    }
