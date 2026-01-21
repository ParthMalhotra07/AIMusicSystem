"""
Rhythmic Feature Extraction Module

Extracts features that capture rhythm and "danceability":
- Tempo (BPM): Beats per minute
- Beat Frames: Locations of beats
- Beat Strength: Intensity of beats
- Onset Strength: Attack/transient detection
- Tempogram: Local tempo estimate over time
"""

from typing import Tuple, Optional
import numpy as np
import librosa


def extract_tempo(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    start_bpm: float = 120.0
) -> float:
    """
    Estimate the tempo (BPM) of an audio signal.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    start_bpm : float, optional
        Initial tempo estimate (default: 120.0)
        
    Returns
    -------
    float
        Estimated tempo in BPM
    """
    # Get tempo - librosa 0.11 returns array directly
    tempo = librosa.feature.tempo(
        y=y,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm
    )
    
    # librosa returns an array, extract scalar
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    
    return tempo


def extract_beat_frames(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    start_bpm: float = 120.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect beat positions in the audio.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    start_bpm : float, optional
        Initial tempo estimate (default: 120.0)
        
    Returns
    -------
    tempo : float
        Estimated tempo in BPM
    beat_frames : np.ndarray
        Frame indices of detected beats
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm
    )
    
    # Convert tempo to scalar if needed
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    
    return tempo, beat_frames


def extract_beat_times(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    """
    Get beat times in seconds.
    
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
        Beat times in seconds
    """
    _, beat_frames = extract_beat_frames(y, sr, hop_length)
    return librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)


def extract_beat_strength(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract beat strength at each beat position.
    
    Beat strength indicates how "strong" or pronounced each beat is.
    Useful for differentiating between music with strong vs. subtle beats.
    
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
        Beat strength values at each beat
    """
    # Get onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length
    )
    
    # Get beat frames
    _, beat_frames = extract_beat_frames(y, sr, hop_length)
    
    # Sample onset strength at beat positions
    if len(beat_frames) > 0:
        # Ensure beat frames are within bounds
        beat_frames = beat_frames[beat_frames < len(onset_env)]
        beat_strength = onset_env[beat_frames]
    else:
        beat_strength = np.array([0.0])
    
    return beat_strength


def extract_onset_strength(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_fft: int = 2048,
    aggregate: np.ufunc = np.mean
) -> np.ndarray:
    """
    Extract onset strength envelope.
    
    The onset strength tracks how much the signal changes over time,
    highlighting note attacks and transients.
    
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
    aggregate : callable, optional
        Aggregation function (default: np.mean)
        
    Returns
    -------
    np.ndarray
        Onset strength envelope
    """
    return librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft,
        aggregate=aggregate
    )


def extract_onset_times(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    backtrack: bool = True
) -> np.ndarray:
    """
    Detect onset times (note attacks) in seconds.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    backtrack : bool, optional
        Backtrack to before attack (default: True)
        
    Returns
    -------
    np.ndarray
        Onset times in seconds
    """
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack
    )
    
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def extract_tempogram(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    win_length: int = 384
) -> np.ndarray:
    """
    Compute the tempogram (local tempo estimate over time).
    
    A tempogram shows how the tempo varies throughout the song,
    useful for detecting tempo changes and complex rhythms.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    win_length : int, optional
        Window length for tempogram (default: 384)
        
    Returns
    -------
    np.ndarray
        Tempogram matrix
    """
    # First get onset envelope
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length
    )
    
    return librosa.feature.tempogram(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length
    )


def extract_rhythm_patterns(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> dict:
    """
    Extract comprehensive rhythm features.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    hop_length : int
        Hop length
        
    Returns
    -------
    dict
        Dictionary containing tempo, beat tracking metrics, and onset info
    """
    tempo, beat_frames = extract_beat_frames(y, sr, hop_length)
    beat_strength = extract_beat_strength(y, sr, hop_length)
    onset_strength = extract_onset_strength(y, sr, hop_length)
    
    # Calculate beat regularity (variance in inter-beat intervals)
    if len(beat_frames) > 1:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        inter_beat_intervals = np.diff(beat_times)
        beat_regularity = 1.0 / (1.0 + np.std(inter_beat_intervals))
    else:
        beat_regularity = 0.0
    
    return {
        'tempo': tempo,
        'beat_count': len(beat_frames),
        'beat_regularity': beat_regularity,
        'beat_strength_mean': float(np.mean(beat_strength)) if len(beat_strength) > 0 else 0.0,
        'beat_strength_std': float(np.std(beat_strength)) if len(beat_strength) > 0 else 0.0,
        'onset_strength_mean': float(np.mean(onset_strength)),
        'onset_strength_std': float(np.std(onset_strength)),
    }


def extract_all_rhythmic_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> dict:
    """
    Extract all rhythmic features in one pass.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    hop_length : int
        Hop length
        
    Returns
    -------
    dict
        Dictionary containing all rhythmic features
    """
    onset_env = extract_onset_strength(y, sr, hop_length)
    
    return {
        'tempo': np.array([[extract_tempo(y, sr, hop_length)]]),  # Shape (1, 1)
        'onset_strength': onset_env.reshape(1, -1),  # Shape (1, n_frames)
        'beat_strength': extract_beat_strength(y, sr, hop_length).reshape(1, -1),
    }
