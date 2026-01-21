"""
Spectral Transformation Module

Converts time-domain audio signals into frequency-domain representations
using STFT, Mel spectrograms, and power spectrograms.
"""

from typing import Optional, Tuple
import numpy as np
import librosa


def compute_stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: Optional[int] = None,
    window: str = 'hann',
    center: bool = True
) -> np.ndarray:
    """
    Compute Short-Time Fourier Transform.
    
    Converts time-domain signal to frequency-domain, showing which
    frequencies are present at each point in time.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    n_fft : int, optional
        FFT window size (default: 2048)
    hop_length : int, optional
        Number of samples between frames (default: 512)
    win_length : int, optional
        Window length (default: n_fft)
    window : str, optional
        Window function (default: 'hann')
    center : bool, optional
        Center the frames (default: True)
        
    Returns
    -------
    np.ndarray
        Complex STFT matrix of shape (1 + n_fft/2, n_frames)
    """
    if win_length is None:
        win_length = n_fft
    
    stft_matrix = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center
    )
    
    return stft_matrix


def compute_magnitude_spectrogram(
    y: np.ndarray = None,
    stft_matrix: np.ndarray = None,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Compute magnitude spectrogram from audio or STFT.
    
    Parameters
    ----------
    y : np.ndarray, optional
        Audio time series (if stft_matrix not provided)
    stft_matrix : np.ndarray, optional
        Pre-computed STFT matrix
    n_fft : int, optional
        FFT window size (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    np.ndarray
        Magnitude spectrogram
    """
    if stft_matrix is None:
        if y is None:
            raise ValueError("Either y or stft_matrix must be provided")
        stft_matrix = compute_stft(y, n_fft=n_fft, hop_length=hop_length)
    
    return np.abs(stft_matrix)


def compute_power_spectrogram(
    y: np.ndarray = None,
    stft_matrix: np.ndarray = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    ref: float = 1.0,
    to_db: bool = True
) -> np.ndarray:
    """
    Compute power spectrogram in decibels.
    
    Parameters
    ----------
    y : np.ndarray, optional
        Audio time series (if stft_matrix not provided)
    stft_matrix : np.ndarray, optional
        Pre-computed STFT matrix
    n_fft : int, optional
        FFT window size (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
    ref : float, optional
        Reference power for dB conversion (default: 1.0)
    to_db : bool, optional
        Convert to decibels (default: True)
        
    Returns
    -------
    np.ndarray
        Power spectrogram (in dB if to_db=True)
    """
    magnitude = compute_magnitude_spectrogram(
        y=y,
        stft_matrix=stft_matrix,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    power = magnitude ** 2
    
    if to_db:
        power = librosa.power_to_db(power, ref=ref)
    
    return power


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    to_db: bool = True
) -> np.ndarray:
    """
    Compute Mel-scale spectrogram.
    
    The Mel scale mimics human perception of pitch, where higher
    frequencies are compressed. This is ideal for music analysis.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    n_fft : int, optional
        FFT window size (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
    n_mels : int, optional
        Number of Mel bands (default: 128)
    fmin : float, optional
        Minimum frequency in Hz (default: 0.0)
    fmax : float, optional
        Maximum frequency in Hz (default: sr/2)
    power : float, optional
        Exponent for the magnitude (default: 2.0)
    to_db : bool, optional
        Convert to decibels (default: True)
        
    Returns
    -------
    np.ndarray
        Mel spectrogram of shape (n_mels, n_frames)
    """
    if fmax is None:
        fmax = sr / 2.0
    
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=power
    )
    
    if to_db:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec


def compute_mel_filterbank(
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Create a Mel filterbank matrix.
    
    Parameters
    ----------
    sr : int, optional
        Sample rate (default: 22050)
    n_fft : int, optional
        FFT window size (default: 2048)
    n_mels : int, optional
        Number of Mel bands (default: 128)
    fmin : float, optional
        Minimum frequency (default: 0.0)
    fmax : float, optional
        Maximum frequency (default: sr/2)
        
    Returns
    -------
    np.ndarray
        Mel filterbank of shape (n_mels, 1 + n_fft/2)
    """
    return librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )


def get_spectrogram_times(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512
) -> np.ndarray:
    """
    Get time values for spectrogram frames.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    np.ndarray
        Time values in seconds for each spectrogram frame
    """
    return librosa.times_like(
        librosa.stft(y, hop_length=hop_length),
        sr=sr,
        hop_length=hop_length
    )


def get_spectrogram_frequencies(
    sr: int = 22050,
    n_fft: int = 2048
) -> np.ndarray:
    """
    Get frequency values for spectrogram bins.
    
    Parameters
    ----------
    sr : int, optional
        Sample rate (default: 22050)
    n_fft : int, optional
        FFT window size (default: 2048)
        
    Returns
    -------
    np.ndarray
        Frequency values in Hz for each bin
    """
    return librosa.fft_frequencies(sr=sr, n_fft=n_fft)


def harmonic_percussive_separation(
    y: np.ndarray,
    margin: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate audio into harmonic and percussive components.
    
    Useful for extracting cleaner features from each component.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    margin : float, optional
        Separation margin (default: 3.0)
        
    Returns
    -------
    y_harmonic : np.ndarray
        Harmonic component (pitched sounds)
    y_percussive : np.ndarray
        Percussive component (drums, transients)
    """
    return librosa.effects.hpss(y, margin=margin)
