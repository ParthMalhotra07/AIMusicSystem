"""
Structural Feature Extraction Module

Extracts features that capture the song's overall "shape" and evolution:
- Section Count: Number of detected structural segments
- Section Duration Statistics: Mean/std of section lengths
- Repetition Score: Self-similarity (chorus/hook detection)
- Novelty Curve: Change intensity over time (transitions)
"""

from typing import Dict, Tuple
import numpy as np
import librosa


def compute_self_similarity_matrix(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_mfcc: int = 20
) -> np.ndarray:
    """
    Compute self-similarity matrix based on MFCCs.
    
    The self-similarity matrix shows how similar different parts
    of the song are to each other, revealing structure.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    n_mfcc : int, optional
        Number of MFCCs (default: 20)
        
    Returns
    -------
    np.ndarray
        Self-similarity matrix
    """
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    
    # Normalize
    mfcc = librosa.util.normalize(mfcc, axis=0)
    
    # Compute similarity matrix (cosine similarity)
    similarity = np.dot(mfcc.T, mfcc)
    
    return similarity


def extract_novelty_curve(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract Novelty Curve.
    
    The novelty curve indicates how much the audio "changes" at each
    point in time, useful for detecting transitions and section boundaries.
    
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
        Novelty curve values
    """
    # Use spectral flux as novelty measure
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    
    # Compute spectral flux (difference between consecutive frames)
    novelty = np.zeros(S.shape[1])
    for i in range(1, S.shape[1]):
        diff = S[:, i] - S[:, i - 1]
        novelty[i] = np.sum(np.maximum(diff, 0))  # Only positive changes
    
    # Normalize
    if np.max(novelty) > 0:
        novelty = novelty / np.max(novelty)
    
    return novelty


def detect_sections(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    n_sections: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect structural sections using novelty-based segmentation.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int, optional
        Sample rate (default: 22050)
    hop_length : int, optional
        Hop length (default: 512)
    n_sections : int, optional
        Target number of sections (default: 10)
        
    Returns
    -------
    boundaries : np.ndarray
        Section boundary times in seconds
    durations : np.ndarray
        Section durations in seconds
    """
    # Get novelty curve
    novelty = extract_novelty_curve(y, sr, hop_length)
    
    # Find peaks in novelty curve (section boundaries)
    # Use adaptive threshold based on novelty statistics
    threshold = np.mean(novelty) + 0.5 * np.std(novelty)
    
    # Find local maxima above threshold
    peaks = []
    for i in range(1, len(novelty) - 1):
        if (novelty[i] > novelty[i - 1] and 
            novelty[i] > novelty[i + 1] and
            novelty[i] > threshold):
            peaks.append(i)
    
    # Limit to reasonable number of sections
    if len(peaks) > n_sections * 2:
        # Keep strongest peaks
        peak_strengths = [novelty[p] for p in peaks]
        sorted_indices = np.argsort(peak_strengths)[::-1][:n_sections * 2]
        peaks = sorted([peaks[i] for i in sorted_indices])
    
    # Convert to times
    if peaks:
        boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        
        # Add start and end
        total_duration = len(y) / sr
        boundary_times = np.concatenate([[0], boundary_times, [total_duration]])
        
        # Calculate durations
        durations = np.diff(boundary_times)
    else:
        # No clear sections detected
        total_duration = len(y) / sr
        boundary_times = np.array([0, total_duration])
        durations = np.array([total_duration])
    
    return boundary_times, durations


def extract_repetition_score(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> float:
    """
    Extract Repetition Score.
    
    Measures how self-similar the song is, indicating the presence
    of choruses, hooks, and repeated patterns.
    
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
        Repetition score (0 = no repetition, 1 = highly repetitive)
    """
    # Compute self-similarity matrix
    sim_matrix = compute_self_similarity_matrix(y, sr, hop_length)
    
    # Calculate off-diagonal similarity (excludes trivial main diagonal)
    n = sim_matrix.shape[0]
    
    if n < 2:
        return 0.0
    
    # Create mask to exclude main diagonal band
    mask = np.ones_like(sim_matrix, dtype=bool)
    band_width = max(1, n // 20)  # Exclude ~5% around diagonal
    for i in range(n):
        for j in range(max(0, i - band_width), min(n, i + band_width + 1)):
            mask[i, j] = False
    
    # Calculate mean off-diagonal similarity
    off_diag_values = sim_matrix[mask]
    
    if len(off_diag_values) > 0:
        repetition = np.mean(off_diag_values)
    else:
        repetition = 0.0
    
    return float(np.clip(repetition, 0.0, 1.0))


def extract_section_count(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> int:
    """
    Extract the number of detected structural sections.
    
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
    int
        Number of sections
    """
    _, durations = detect_sections(y, sr, hop_length)
    return len(durations)


def extract_all_structural_features(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512
) -> Dict[str, np.ndarray]:
    """
    Extract all structural features in one pass.
    
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
        Dictionary containing all structural features
    """
    # Get sections
    _, durations = detect_sections(y, sr, hop_length)
    
    # Get novelty curve
    novelty = extract_novelty_curve(y, sr, hop_length)
    
    return {
        'section_count': np.array([[len(durations)]]),
        'section_duration': durations.reshape(1, -1),  # Variable length, will be aggregated
        'repetition_score': np.array([[extract_repetition_score(y, sr, hop_length)]]),
        'novelty_curve': novelty.reshape(1, -1),  # Will be aggregated to stats
    }
