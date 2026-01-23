"""
Data Loading Module for Member 2
Loads and preprocesses audio features from Member 1's output
"""

from .loader import (
    FeatureDataset,
    load_features,
    load_from_parquet,
    load_from_csv,
    load_from_npy,
    find_latest_feature_file,
    load_multiple_feature_files
)

from .validator import (
    FeatureValidator,
    quick_validate
)

from .preprocessor import (
    FeaturePreprocessor,
    auto_preprocess
)

__all__ = [
    # Dataset class
    'FeatureDataset',

    # Main loaders
    'load_features',
    'load_from_parquet',
    'load_from_csv',
    'load_from_npy',
    'find_latest_feature_file',
    'load_multiple_feature_files',

    # Validation
    'FeatureValidator',
    'quick_validate',

    # Preprocessing
    'FeaturePreprocessor',
    'auto_preprocess',
]

__version__ = '1.0.0'
__author__ = 'Member 2 - Embedding & Clustering Engineer'
