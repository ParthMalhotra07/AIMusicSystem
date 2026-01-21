"""
Groove and Dynamics Feature Extraction Module

Extracts features that capture the "feel" and production quality:
- Fundamental Frequency (F0): Pitch baseline
- Key Strength: Confidence of key detection
- Harmonic Energy Ratio: Harmonic vs percussive content
- Rhythmic Complexity: Variance in onset intervals
- Syncopation Index: Off-beat emphasis
- Perceived Loudness (LUFS): Human perception-weighted loudness
- Crest Factor: Peak-to-RMS ratio (transient content)
- Attack/Decay Times: Note envelope characteristics
"""

from typing import Dict, Tuple, Optional
import numpy as np
import librosa


def extract_fundamental_frequency(
    y: np.ndarray,
    sr: int = 22050,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract Fundamental Frequency (F0) using PYIN algorithm.
    
    F0 represents the pitch baseline of the audio, useful for
    identifying vocal range and melodic characteristics.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    fmin : float, optional
        Minimum frequency (default: 50.0 Hz)
    fmax : float, optional
        Maximum frequency (default: 2000.0 Hz)
    hop_length : int, optional
        Hop length (default: 512)
        
    Returns
    -------
    np.ndarray
        F0 values in Hz (NaN for unvoiced frames)
    """
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length
        )
        # Replace NaN with 0 for unvoiced frames
        f0 = np.nan_to_num(f0, nan=0.0)
    except Exception:
        # Fallback for edge cases
        n_frames = 1 + len(y) // hop_length
        f0 = np.zeros(n_frames)
    
    return f0.reshape(1, -1)


def extract_harmonic_energy_ratio(
    y: np.ndarray,
    sr: int = 22050
) -> float:
    """
    Extract Harmonic-to-Percussive Energy Ratio.
    
    This ratio indicates whether the audio is more melodic/harmonic
    (vocals, strings) or more percussive (drums, clicks).
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
        
    Returns
    -------
    float
        Ratio of harmonic to total energy (0 to 1)
    """
    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Calculate energies
    harmonic_energy = np.sum(y_harmonic ** 2)
    total_energy = np.sum(y ** 2)
    
    if total_energy > 0:
        ratio = harmonic_energy / total_energy
    else:
        ratio = 0.5
    
    return float(np.clip(ratio, 0.0, 1.0))


def extract_rhythmic_complexity(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> float:
    """
    Extract Rhythmic Complexity.
    
    Measures the variance in onset intervals, indicating how
    irregular or complex the rhythm is.
    
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
        Rhythmic complexity score (normalized)
    """
    # Get onset times
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    if len(onset_times) > 2:
        intervals = np.diff(onset_times)
        # Coefficient of variation as complexity measure
        complexity = np.std(intervals) / (np.mean(intervals) + 1e-10)
    else:
        complexity = 0.0
    
    return float(np.clip(complexity, 0.0, 2.0))  # Cap at 2.0


def extract_syncopation_index(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> float:
    """
    Extract Syncopation Index.
    
    Measures the off-beat emphasis by comparing onset strength
    at beat vs. off-beat positions. Higher values indicate more
    syncopated, groovy rhythms.
    
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
        Syncopation index (0 = on-beat, 1 = highly syncopated)
    """
    # Get onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Get beat frames
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length
    )
    
    if len(beat_frames) < 2:
        return 0.0
    
    # Calculate positions between beats (off-beat positions)
    off_beat_frames = []
    for i in range(len(beat_frames) - 1):
        mid_frame = (beat_frames[i] + beat_frames[i + 1]) // 2
        if mid_frame < len(onset_env):
            off_beat_frames.append(mid_frame)
    
    if not off_beat_frames:
        return 0.0
    
    # Compare onset strength at beats vs off-beats
    beat_frames_valid = beat_frames[beat_frames < len(onset_env)]
    
    if len(beat_frames_valid) == 0:
        return 0.0
    
    on_beat_strength = np.mean(onset_env[beat_frames_valid])
    off_beat_strength = np.mean(onset_env[off_beat_frames])
    
    # Syncopation = relative strength of off-beats
    total_strength = on_beat_strength + off_beat_strength + 1e-10
    syncopation = off_beat_strength / total_strength
    
    return float(np.clip(syncopation, 0.0, 1.0))


def extract_perceived_loudness(
    y: np.ndarray,
    sr: int = 22050
) -> float:
    """
    Extract Perceived Loudness approximation (pseudo-LUFS).
    
    Approximates LUFS by weighting frequencies according to
    human hearing sensitivity (simplified K-weighting).
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
        
    Returns
    -------
    float
        Perceived loudness in approximate LUFS
    """
    # Apply simplified K-weighting using high-pass filter
    # K-weighting emphasizes frequencies around 1-4 kHz
    
    # Simple approximation: apply pre-emphasis filter
    y_weighted = librosa.effects.preemphasis(y, coef=0.97)
    
    # Calculate mean square
    mean_square = np.mean(y_weighted ** 2)
    
    if mean_square > 0:
        # Convert to LUFS-like scale
        lufs = -0.691 + 10 * np.log10(mean_square + 1e-10)
    else:
        lufs = -70.0  # Silence
    
    return float(lufs)


def extract_crest_factor(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512
) -> float:
    """
    Extract Crest Factor.
    
    Crest factor is the ratio of peak to RMS amplitude, indicating
    the amount of transient/dynamic content (punchiness).
    
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
    float
        Crest factor (typically 3-20, higher = more punchy)
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    peak = np.max(np.abs(y))
    mean_rms = np.mean(rms)
    
    if mean_rms > 0:
        crest = peak / mean_rms
    else:
        crest = 1.0
    
    return float(crest)


def extract_attack_decay_times(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Attack and Decay Times.
    
    Attack time: how quickly the sound reaches peak amplitude
    Decay time: how quickly the sound fades after the peak
    
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
    attack_times : np.ndarray
        Attack times in seconds
    decay_times : np.ndarray
        Decay times in seconds
    """
    # Get onset frames
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=True
    )
    
    # Get RMS envelope
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    attack_times = []
    decay_times = []
    
    time_per_frame = hop_length / sr
    
    for i, onset in enumerate(onset_frames[:-1]):
        # Define segment end as next onset or end of rms
        end_frame = min(onset_frames[i + 1] if i + 1 < len(onset_frames) else len(rms), len(rms))
        
        if end_frame <= onset:
            continue
        
        segment = rms[onset:end_frame]
        
        if len(segment) < 2:
            continue
        
        # Find peak within segment
        peak_idx = np.argmax(segment)
        peak_val = segment[peak_idx]
        
        if peak_val <= 0:
            continue
        
        # Attack time: time to reach 90% of peak from onset
        threshold = 0.9 * peak_val
        attack_frames = 0
        for j in range(peak_idx + 1):
            if segment[j] >= threshold:
                attack_frames = j
                break
        attack_times.append(attack_frames * time_per_frame)
        
        # Decay time: time to fall to 10% of peak after peak
        threshold = 0.1 * peak_val
        decay_frames = len(segment) - peak_idx
        for j in range(peak_idx, len(segment)):
            if segment[j] <= threshold:
                decay_frames = j - peak_idx
                break
        decay_times.append(decay_frames * time_per_frame)
    
    if not attack_times:
        attack_times = [0.01]
    if not decay_times:
        decay_times = [0.1]
    
    return np.array(attack_times), np.array(decay_times)


def extract_beat_strength_distribution(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract Beat Strength Distribution.
    
    Returns the onset strength values at detected beat positions,
    providing insight into beat intensity variation.
    
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
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length
    )
    
    # Sample onset strength at beat positions
    beat_frames_valid = beat_frames[beat_frames < len(onset_env)]
    
    if len(beat_frames_valid) > 0:
        beat_strength = onset_env[beat_frames_valid]
    else:
        beat_strength = np.array([0.0])
    
    return beat_strength.reshape(1, -1)


def extract_all_groove_features(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict[str, np.ndarray]:
    """
    Extract all groove and dynamics features in one pass.
    
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
        
    Returns
    -------
    dict
        Dictionary containing all groove features
    """
    attack_times, decay_times = extract_attack_decay_times(y, sr, hop_length)
    
    return {
        'fundamental_frequency': extract_fundamental_frequency(y, sr, hop_length=hop_length),
        'harmonic_energy_ratio': np.array([[extract_harmonic_energy_ratio(y, sr)]]),
        'rhythmic_complexity': np.array([[extract_rhythmic_complexity(y, sr, hop_length)]]),
        'syncopation_index': np.array([[extract_syncopation_index(y, sr, hop_length)]]),
        'perceived_loudness': np.array([[extract_perceived_loudness(y, sr)]]),
        'crest_factor': np.array([[extract_crest_factor(y, n_fft, hop_length)]]),
        'attack_time': attack_times.reshape(1, -1),
        'decay_time': decay_times.reshape(1, -1),
        'beat_strength_dist': extract_beat_strength_distribution(y, sr, hop_length),
    }
