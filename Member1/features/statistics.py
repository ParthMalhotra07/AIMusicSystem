"""
Statistical Aggregation Module

Computes statistical summaries of time-varying features to create
fixed-length feature vectors regardless of audio duration.

This is the key to solving the "different length songs" problem:
- A 3-minute song and a 5-minute song will have the same vector length
- We compute mean, std, skewness, kurtosis across all frames
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from scipy import stats


def compute_statistics(
    feature_matrix: np.ndarray,
    statistics: Tuple[str, ...] = ('mean', 'std', 'skew', 'kurtosis')
) -> np.ndarray:
    """
    Compute statistical summaries of a feature matrix.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_features, n_frames)
    statistics : tuple, optional
        Statistics to compute (default: mean, std, skew, kurtosis)
        
    Returns
    -------
    np.ndarray
        Flattened array of statistics with shape (n_features * n_stats,)
    """
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)
    
    n_features, n_frames = feature_matrix.shape
    
    if n_frames == 0:
        return np.zeros(n_features * len(statistics))
    
    results = []
    
    for stat in statistics:
        if stat == 'mean':
            results.append(np.mean(feature_matrix, axis=1))
        elif stat == 'std':
            results.append(np.std(feature_matrix, axis=1))
        elif stat == 'var':
            results.append(np.var(feature_matrix, axis=1))
        elif stat == 'skew':
            # Skewness measures asymmetry of the distribution
            results.append(stats.skew(feature_matrix, axis=1))
        elif stat == 'kurtosis':
            # Kurtosis measures "peakedness" of the distribution
            results.append(stats.kurtosis(feature_matrix, axis=1))
        elif stat == 'min':
            results.append(np.min(feature_matrix, axis=1))
        elif stat == 'max':
            results.append(np.max(feature_matrix, axis=1))
        elif stat == 'median':
            results.append(np.median(feature_matrix, axis=1))
        elif stat == 'percentile_25':
            results.append(np.percentile(feature_matrix, 25, axis=1))
        elif stat == 'percentile_75':
            results.append(np.percentile(feature_matrix, 75, axis=1))
        elif stat == 'range':
            results.append(np.ptp(feature_matrix, axis=1))  # peak to peak
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    
    # Stack and flatten: (n_stats, n_features) -> (n_features * n_stats,)
    result = np.stack(results, axis=0)  # (n_stats, n_features)
    
    # Replace any NaN or Inf with 0
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result.T.flatten()  # Interleave: feat1_mean, feat1_std, feat1_skew, ...


def aggregate_features(
    features_dict: Dict[str, np.ndarray],
    statistics: Tuple[str, ...] = ('mean', 'std', 'skew', 'kurtosis')
) -> np.ndarray:
    """
    Aggregate a dictionary of features into a single vector.
    
    Parameters
    ----------
    features_dict : dict
        Dictionary mapping feature names to feature matrices
    statistics : tuple, optional
        Statistics to compute for each feature
        
    Returns
    -------
    np.ndarray
        Concatenated feature vector
    """
    aggregated = []
    
    for name, feature_matrix in features_dict.items():
        if feature_matrix is None:
            continue
            
        # Handle scalar features (like tempo)
        if np.isscalar(feature_matrix):
            aggregated.append(np.array([feature_matrix]))
            continue
        
        # Ensure 2D
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        
        # Special case: single-frame features (like tempo)
        if feature_matrix.shape[1] == 1:
            aggregated.append(feature_matrix.flatten())
        else:
            # Compute statistics over time
            agg = compute_statistics(feature_matrix, statistics)
            aggregated.append(agg)
    
    return np.concatenate(aggregated)


def get_feature_names(
    feature_structure: Dict[str, int],
    statistics: Tuple[str, ...] = ('mean', 'std', 'skew', 'kurtosis')
) -> List[str]:
    """
    Generate names for the aggregated feature vector.
    
    Parameters
    ----------
    feature_structure : dict
        Dictionary mapping feature names to their dimensions
        Example: {'mfcc': 13, 'spectral_centroid': 1}
    statistics : tuple, optional
        Statistics computed for each feature
        
    Returns
    -------
    list
        List of feature names
    """
    names = []
    
    for feature_name, n_dims in feature_structure.items():
        if n_dims == 1:
            # Single dimension features like tempo
            names.append(feature_name)
        else:
            # Multi-dimensional features with statistics
            for dim in range(n_dims):
                for stat in statistics:
                    names.append(f"{feature_name}_{dim}_{stat}")
    
    return names


def global_pooling(
    feature_matrix: np.ndarray,
    method: str = 'mean_std'
) -> np.ndarray:
    """
    Apply global pooling to convert variable-length features to fixed length.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_features, n_frames)
    method : str, optional
        Pooling method: 'mean', 'max', 'mean_std', 'all_stats'
        
    Returns
    -------
    np.ndarray
        Pooled features
    """
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)
    
    if method == 'mean':
        return np.mean(feature_matrix, axis=1)
    elif method == 'max':
        return np.max(feature_matrix, axis=1)
    elif method == 'mean_std':
        mean = np.mean(feature_matrix, axis=1)
        std = np.std(feature_matrix, axis=1)
        return np.concatenate([mean, std])
    elif method == 'all_stats':
        return compute_statistics(feature_matrix, ('mean', 'std', 'skew', 'kurtosis'))
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def temporal_summarization(
    feature_matrix: np.ndarray,
    n_segments: int = 10
) -> np.ndarray:
    """
    Summarize features by dividing the audio into segments.
    
    This captures temporal evolution (beginning vs middle vs end)
    while still producing a fixed-length output.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_features, n_frames)
    n_segments : int, optional
        Number of temporal segments (default: 10)
        
    Returns
    -------
    np.ndarray
        Segment-wise means of shape (n_features * n_segments,)
    """
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(1, -1)
    
    n_features, n_frames = feature_matrix.shape
    
    # Divide into segments
    segment_size = max(1, n_frames // n_segments)
    
    segment_means = []
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else n_frames
        segment = feature_matrix[:, start:end]
        segment_means.append(np.mean(segment, axis=1))
    
    return np.stack(segment_means, axis=1).flatten()


def delta_features(
    feature_matrix: np.ndarray,
    order: int = 1,
    width: int = 9
) -> np.ndarray:
    """
    Compute delta (derivative) features.
    
    Delta features capture how features change over time,
    useful for detecting musical transitions.
    
    Parameters
    ----------
    feature_matrix : np.ndarray
        Feature matrix of shape (n_features, n_frames)
    order : int, optional
        Order of derivative (1 = velocity, 2 = acceleration)
    width : int, optional
        Window width for derivative computation
        
    Returns
    -------
    np.ndarray
        Delta features of same shape as input
    """
    import librosa
    return librosa.feature.delta(feature_matrix, order=order, width=width)
