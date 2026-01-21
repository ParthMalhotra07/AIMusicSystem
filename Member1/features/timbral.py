"""
Timbral Feature Extraction Module

Extracts features that capture the "texture" of sound:
- MFCCs: Mel-frequency cepstral coefficients
- Spectral Centroid: "Brightness" of sound
- Spectral Rolloff: Frequency concentration
- Spectral Contrast: Peak-valley differences in spectrum
- Spectral Bandwidth: Frequency spread
- Zero Crossing Rate: Noisiness indicator
"""

from typing import Optional
import numpy as np
import librosa


def extract_mfccs(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    Extract Mel-frequency cepstral coefficients.
    
    MFCCs are the most important timbral features, representing
    the "shape" of the spectral envelope. They capture what makes
    a piano sound different from a guitar.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    n_mfcc : int, optional
        Number of MFCCs to return (default: 13)
    n_fft : int, optional
        FFT window size (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
    n_mels : int, optional
        Number of Mel bands (default: 128)
    fmin : float, optional
        Minimum frequency (default: 0.0)
    fmax : float, optional
        Maximum frequency (default: sr/2)
        
    Returns
    -------
    np.ndarray
        MFCC matrix of shape (n_mfcc, n_frames)
    """
    mfccs = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    return mfccs


def extract_delta_mfccs(
    mfccs: np.ndarray,
    order: int = 1,
    width: int = 9
) -> np.ndarray:
    """
    Compute delta (derivative) MFCCs.
    
    Delta MFCCs capture how the timbral features change over time,
    useful for detecting transitions and dynamics.
    
    Parameters
    ----------
    mfccs : np.ndarray
        MFCC matrix
    order : int, optional
        Order of derivative (1 for velocity, 2 for acceleration)
    width : int, optional
        Window width for derivative computation (default: 9)
        
    Returns
    -------
    np.ndarray
        Delta MFCCs of same shape as input
    """
    return librosa.feature.delta(mfccs, order=order, width=width)


def extract_spectral_centroid(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract spectral centroid (brightness).
    
    The spectral centroid represents the "center of mass" of the
    spectrum. Higher values indicate brighter sounds (more high
    frequency content).
    
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
        
    Returns
    -------
    np.ndarray
        Spectral centroid of shape (1, n_frames)
    """
    return librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )


def extract_spectral_rolloff(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    roll_percent: float = 0.85
) -> np.ndarray:
    """
    Extract spectral rolloff frequency.
    
    The rolloff is the frequency below which a specified fraction
    (default 85%) of the total spectral energy is contained.
    Indicates how much high-frequency content is present.
    
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
    roll_percent : float, optional
        Fraction of energy (default: 0.85)
        
    Returns
    -------
    np.ndarray
        Spectral rolloff of shape (1, n_frames)
    """
    return librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        roll_percent=roll_percent
    )


def extract_spectral_contrast(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 6,
    fmin: float = 200.0
) -> np.ndarray:
    """
    Extract spectral contrast.
    
    Spectral contrast measures the difference between peaks and
    valleys in the spectrum across frequency bands. Higher values
    indicate more "peaky" sounds (like single notes), lower values
    indicate more uniform/noisy sounds.
    
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
    n_bands : int, optional
        Number of frequency bands (default: 6)
    fmin : float, optional
        Minimum frequency (default: 200.0)
        
    Returns
    -------
    np.ndarray
        Spectral contrast of shape (n_bands + 1, n_frames)
    """
    return librosa.feature.spectral_contrast(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands,
        fmin=fmin
    )


def extract_spectral_bandwidth(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    p: float = 2.0
) -> np.ndarray:
    """
    Extract spectral bandwidth.
    
    The bandwidth measures the spread of the spectrum around its
    centroid. Wider bandwidth indicates more frequency diversity.
    
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
    p : float, optional
        Power for computing bandwidth (default: 2.0)
        
    Returns
    -------
    np.ndarray
        Spectral bandwidth of shape (1, n_frames)
    """
    return librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        p=p
    )


def extract_spectral_flatness(
    y: np.ndarray = None,
    S: np.ndarray = None,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract spectral flatness (tonality coefficient).
    
    Spectral flatness indicates whether the spectrum is more
    noise-like (flat, value ~1) or more tone-like (peaked, value ~0).
    
    Parameters
    ----------
    y : np.ndarray, optional
        Audio time series
    S : np.ndarray, optional
        Pre-computed spectrogram
    n_fft : int, optional
        FFT window size (default: 2048)
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    np.ndarray
        Spectral flatness of shape (1, n_frames)
    """
    return librosa.feature.spectral_flatness(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length
    )


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


def extract_spectral_flux(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract Spectral Flux.
    
    Spectral flux measures how rapidly the spectrum changes over time.
    High values indicate rapid changes (transients), low values indicate
    stable sounds.
    
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
        
    Returns
    -------
    np.ndarray
        Spectral flux of shape (1, n_frames)
    """
    # Compute spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Compute differences between consecutive frames
    flux = np.zeros(S.shape[1])
    for i in range(1, S.shape[1]):
        diff = S[:, i] - S[:, i - 1]
        # Sum of positive differences (half-wave rectification)
        flux[i] = np.sum(np.maximum(diff, 0))
    
    return flux.reshape(1, -1)


def extract_all_timbral_features(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 13,
    n_mels: int = 128
) -> dict:
    """
    Extract all timbral features in one pass.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    n_fft : int
        FFT window size
    hop_length : int
        Hop length
    n_mfcc : int
        Number of MFCCs
    n_mels : int
        Number of Mel bands
        
    Returns
    -------
    dict
        Dictionary containing all timbral features
    """
    # Extract MFCCs
    mfccs = extract_mfccs(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                          hop_length=hop_length, n_mels=n_mels)
    
    # Extract Delta and Delta-Delta MFCCs
    delta_mfccs = extract_delta_mfccs(mfccs, order=1)
    delta2_mfccs = extract_delta_mfccs(mfccs, order=2)
    
    return {
        'mfcc': mfccs,
        'mfcc_delta': delta_mfccs,
        'mfcc_delta2': delta2_mfccs,
        'spectral_centroid': extract_spectral_centroid(y, sr, n_fft, hop_length),
        'spectral_rolloff': extract_spectral_rolloff(y, sr, n_fft, hop_length),
        'spectral_contrast': extract_spectral_contrast(y, sr, n_fft, hop_length),
        'spectral_bandwidth': extract_spectral_bandwidth(y, sr, n_fft, hop_length),
        'spectral_flatness': extract_spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length),
        'spectral_flux': extract_spectral_flux(y, sr, n_fft, hop_length),
    }
