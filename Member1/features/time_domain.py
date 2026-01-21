"""
Time-Domain Feature Extraction Module

Extracts the "vital signs" of audio signals:
- Short-Time Energy (STE): Frame-level power measurement
- Inter-Beat Interval (IBI): Beat timing variability
- Dynamic Range: Peak-to-quiet ratio in dB
- Onset Rate: Rhythmic density (onsets per second)
- RMS Energy: Root-mean-square energy (moved from timbral)
"""

from typing import Tuple, Dict
import numpy as np
import librosa


def extract_short_time_energy(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract Short-Time Energy (STE).
    
    STE measures the signal's power at each frame, capturing
    the dynamic evolution of energy throughout the track.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    frame_length : int, optional
        Frame length (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    np.ndarray
        STE of shape (1, n_frames)
    """
    # Calculate energy per frame
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames ** 2, axis=0)
    
    return energy.reshape(1, -1)


def extract_rms_energy(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract RMS (root-mean-square) energy.
    
    RMS energy represents the "loudness" of the signal at each frame.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    frame_length : int, optional
        Frame length (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    np.ndarray
        RMS energy of shape (1, n_frames)
    """
    return librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )


def extract_inter_beat_interval(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract Inter-Beat Interval (IBI) statistics.
    
    IBI measures the time between consecutive beats. The variability
    in IBI indicates tempo stability and groove feel.
    
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
        IBI array in seconds
    """
    # Get beat frames
    _, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length
    )
    
    # Convert to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    
    # Calculate inter-beat intervals
    if len(beat_times) > 1:
        ibi = np.diff(beat_times)
    else:
        ibi = np.array([0.5])  # Default to 120 BPM equivalent
    
    return ibi


def extract_dynamic_range(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
    percentile_low: float = 10,
    percentile_high: float = 90
) -> float:
    """
    Extract Dynamic Range in decibels.
    
    Dynamic range measures the difference between the loudest and
    quietest parts of the audio, indicating compression vs expressiveness.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    frame_length : int, optional
        Frame length (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
    percentile_low : float, optional
        Low percentile for noise floor (default: 10)
    percentile_high : float, optional
        High percentile for peak (default: 90)
        
    Returns
    -------
    float
        Dynamic range in dB
    """
    # Get RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Avoid log of zero
    rms = np.maximum(rms, 1e-10)
    
    # Convert to dB
    rms_db = 20 * np.log10(rms)
    
    # Calculate dynamic range as difference between percentiles
    peak_level = np.percentile(rms_db, percentile_high)
    quiet_level = np.percentile(rms_db, percentile_low)
    
    return float(peak_level - quiet_level)


def extract_onset_rate(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> float:
    """
    Extract Onset Rate (onsets per second).
    
    Onset rate quantifies rhythmic density - how many note attacks
    occur per second. Higher values indicate busier, more complex rhythms.
    
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
    float
        Onsets per second
    """
    # Get onset times
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length
    )
    
    # Calculate duration in seconds
    duration = len(y) / sr
    
    # Calculate onset rate
    if duration > 0:
        onset_rate = len(onset_frames) / duration
    else:
        onset_rate = 0.0
    
    return float(onset_rate)


def extract_zero_crossing_rate(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract zero-crossing rate.
    
    The zero-crossing rate is the rate at which the signal changes
    sign. Higher values indicate noisier or more percussive sounds.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    frame_length : int, optional
        Frame length (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    np.ndarray
        Zero-crossing rate of shape (1, n_frames)
    """
    return librosa.feature.zero_crossing_rate(
        y,
        frame_length=frame_length,
        hop_length=hop_length
    )


def extract_all_time_domain_features(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict[str, np.ndarray]:
    """
    Extract all time-domain features in one pass.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    n_fft : int
        FFT window size (used as frame_length)
    hop_length : int
        Hop length
        
    Returns
    -------
    dict
        Dictionary containing all time-domain features
    """
    return {
        'short_time_energy': extract_short_time_energy(y, n_fft, hop_length),
        'rms_energy': extract_rms_energy(y, n_fft, hop_length),
        'zero_crossing_rate': extract_zero_crossing_rate(y, n_fft, hop_length),
        'inter_beat_interval': extract_inter_beat_interval(y, sr, hop_length).reshape(1, -1),
        'dynamic_range': np.array([[extract_dynamic_range(y, n_fft, hop_length)]]),
        'onset_rate': np.array([[extract_onset_rate(y, sr, hop_length)]]),
    }
