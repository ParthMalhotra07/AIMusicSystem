"""
Feature Schema Module

Provides structured documentation of the feature vector,
including names, descriptions, and value ranges.

This serves as the "contract" between Member 1 (Feature Engineering)
and Member 2 (Clustering) / Member 3 (UI/Recommendations).
"""

from typing import List, Dict, Tuple


# Complete feature schema with descriptions
FEATURE_DEFINITIONS = {
    # Timbral Features
    'mfcc': {
        'full_name': 'Mel-Frequency Cepstral Coefficients',
        'dimensions': 13,
        'description': 'Captures the shape of the spectral envelope. The "fingerprint" of how instruments sound.',
        'value_range': 'Typically [-50, 50] before scaling',
        'musical_meaning': 'Distinguishes piano from guitar, violin from synth, etc.',
    },
    'spectral_centroid': {
        'full_name': 'Spectral Centroid',
        'dimensions': 1,
        'description': 'Center of mass of the spectrum. Higher = brighter sound.',
        'value_range': '[0, sr/2] Hz',
        'musical_meaning': 'Bright sounds like hi-hats vs dark sounds like bass.',
    },
    'spectral_rolloff': {
        'full_name': 'Spectral Rolloff',
        'dimensions': 1,
        'description': 'Frequency below which 85% of spectral energy is contained.',
        'value_range': '[0, sr/2] Hz',
        'musical_meaning': 'High rolloff = lots of high frequency content.',
    },
    'spectral_contrast': {
        'full_name': 'Spectral Contrast',
        'dimensions': 7,
        'description': 'Difference between peaks and valleys in each frequency band.',
        'value_range': '[0, ~50] dB',
        'musical_meaning': 'Tonal sounds (high contrast) vs noisy sounds (low contrast).',
    },
    'spectral_bandwidth': {
        'full_name': 'Spectral Bandwidth',
        'dimensions': 1,
        'description': 'Standard deviation of the spectrum around the centroid.',
        'value_range': '[0, sr/2] Hz',
        'musical_meaning': 'Wide bandwidth = rich, complex sound. Narrow = pure tone.',
    },
    'spectral_flatness': {
        'full_name': 'Spectral Flatness',
        'dimensions': 1,
        'description': 'Ratio of geometric mean to arithmetic mean of spectrum.',
        'value_range': '[0, 1]',
        'musical_meaning': '1 = noise, 0 = pure tone.',
    },
    'zero_crossing_rate': {
        'full_name': 'Zero-Crossing Rate',
        'dimensions': 1,
        'description': 'Rate at which the signal changes sign.',
        'value_range': '[0, 1]',
        'musical_meaning': 'High for noisy/percussive sounds, low for smooth sounds.',
    },
    'rms_energy': {
        'full_name': 'RMS Energy',
        'dimensions': 1,
        'description': 'Root-mean-square energy of the signal.',
        'value_range': '[0, 1]',
        'musical_meaning': 'Loudness/intensity over time.',
    },
    
    # Rhythmic Features
    'tempo': {
        'full_name': 'Tempo',
        'dimensions': 1,
        'description': 'Estimated beats per minute.',
        'value_range': '[40, 240] BPM typically',
        'musical_meaning': 'Fast/slow. Crucial for danceability.',
    },
    'onset_strength': {
        'full_name': 'Onset Strength',
        'dimensions': 1,
        'description': 'Strength of note attacks/transients.',
        'value_range': '[0, ~10]',
        'musical_meaning': 'Sharp attacks (drums) vs smooth attacks (strings).',
    },
    'beat_strength': {
        'full_name': 'Beat Strength',
        'dimensions': 1,
        'description': 'Prominence of beats.',
        'value_range': '[0, ~10]',
        'musical_meaning': 'Strong, danceable beats vs subtle rhythm.',
    },
    
    # Harmonic Features
    'chroma': {
        'full_name': 'Chroma Features',
        'dimensions': 12,
        'description': '12-bin pitch class distribution (C, C#, D, ..., B).',
        'value_range': '[0, 1]',
        'musical_meaning': 'Which notes are present, regardless of octave.',
    },
    'tonnetz': {
        'full_name': 'Tonnetz (Tonal Centroid)',
        'dimensions': 6,
        'description': '6D representation of harmonic relationships.',
        'value_range': '[-1, 1]',
        'musical_meaning': 'Captures chord progressions and harmonic complexity.',
    },
}


# Statistics applied to time-varying features
STATISTICS_DEFINITIONS = {
    'mean': 'Average value over time',
    'std': 'Standard deviation - how much the feature varies',
    'skew': 'Asymmetry of the distribution',
    'kurtosis': 'Peakedness - presence of extreme values',
}


def get_feature_names(
    statistics: Tuple[str, ...] = ('mean', 'std', 'skew', 'kurtosis')
) -> List[str]:
    """
    Generate the complete list of feature names.
    
    Parameters
    ----------
    statistics : tuple
        Statistics computed for each feature
        
    Returns
    -------
    list
        Ordered list of feature names
    """
    names = []
    
    # Timbral features
    for i in range(13):  # MFCCs
        for stat in statistics:
            names.append(f'mfcc_{i}_{stat}')
    
    for feature in ['spectral_centroid', 'spectral_rolloff', 
                    'spectral_bandwidth', 'spectral_flatness',
                    'zero_crossing_rate', 'rms_energy']:
        for stat in statistics:
            names.append(f'{feature}_{stat}')
    
    for i in range(7):  # Spectral contrast
        for stat in statistics:
            names.append(f'spectral_contrast_{i}_{stat}')
    
    # Rhythmic features
    names.append('tempo')
    
    for feature in ['onset_strength', 'beat_strength']:
        for stat in statistics:
            names.append(f'{feature}_{stat}')
    
    # Harmonic features
    for i in range(12):  # Chroma
        for stat in statistics:
            names.append(f'chroma_{i}_{stat}')
    
    for i in range(6):  # Tonnetz
        for stat in statistics:
            names.append(f'tonnetz_{i}_{stat}')
    
    return names


def get_feature_descriptions() -> Dict[str, str]:
    """
    Get descriptions for all features.
    
    Returns
    -------
    dict
        Mapping of feature names to descriptions
    """
    descriptions = {}
    
    for feature_name, info in FEATURE_DEFINITIONS.items():
        descriptions[feature_name] = info['description']
    
    return descriptions


def get_feature_category(feature_name: str) -> str:
    """
    Get the category of a feature based on its name.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature
        
    Returns
    -------
    str
        Category: 'timbral', 'rhythmic', or 'harmonic'
    """
    timbral_keywords = ['mfcc', 'spectral', 'zero_crossing', 'rms']
    rhythmic_keywords = ['tempo', 'beat', 'onset']
    harmonic_keywords = ['chroma', 'tonnetz']
    
    if any(kw in feature_name for kw in timbral_keywords):
        return 'timbral'
    elif any(kw in feature_name for kw in rhythmic_keywords):
        return 'rhythmic'
    elif any(kw in feature_name for kw in harmonic_keywords):
        return 'harmonic'
    else:
        return 'unknown'


def get_feature_index_ranges() -> Dict[str, Tuple[int, int]]:
    """
    Get the index ranges for each feature category.
    
    Returns
    -------
    dict
        Mapping of category to (start_index, end_index)
    """
    # With 4 statistics per feature:
    # MFCCs: 13 dims × 4 stats = 52 features (indices 0-51)
    # Spectral: 6 features × 4 stats = 24 features (indices 52-75)
    # Spectral Contrast: 7 dims × 4 stats = 28 features (indices 76-103)
    # Tempo: 1 feature (index 104)
    # Onset/Beat: 2 features × 4 stats = 8 features (indices 105-112)
    # Chroma: 12 dims × 4 stats = 48 features (indices 113-160)
    # Tonnetz: 6 dims × 4 stats = 24 features (indices 161-184)
    
    return {
        'mfcc': (0, 52),
        'spectral': (52, 76),
        'spectral_contrast': (76, 104),
        'tempo': (104, 105),
        'rhythm': (105, 113),
        'chroma': (113, 161),
        'tonnetz': (161, 185),
    }


def get_total_dimensions(
    statistics: Tuple[str, ...] = ('mean', 'std', 'skew', 'kurtosis')
) -> int:
    """
    Calculate the total number of features.
    
    Parameters
    ----------
    statistics : tuple
        Statistics used per feature
        
    Returns
    -------
    int
        Total feature vector length
    """
    n_stats = len(statistics)
    
    # Timbral
    dims = 13 * n_stats  # MFCCs
    dims += 6 * n_stats  # other spectral features
    dims += 7 * n_stats  # spectral contrast
    
    # Rhythmic
    dims += 1  # tempo (no stats, single value)
    dims += 2 * n_stats  # onset/beat strength
    
    # Harmonic
    dims += 12 * n_stats  # chroma
    dims += 6 * n_stats  # tonnetz
    
    return dims
