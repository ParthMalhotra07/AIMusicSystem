"""
Feature Data Loader Module
Loads audio features exported by Member 1 in various formats (Parquet, CSV, NPY)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import glob

from .validator import FeatureValidator, quick_validate
from .preprocessor import FeaturePreprocessor, auto_preprocess


class FeatureDataset:
    """
    Unified interface for loading and accessing audio features from Member 1.
    """

    def __init__(
        self,
        features: np.ndarray,
        file_paths: List[str],
        metadata: Optional[pd.DataFrame] = None
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (N x D)
        file_paths : list
            List of audio file paths
        metadata : pd.DataFrame, optional
            Metadata (tempo, key, duration, etc.)
        """
        self.features = features
        self.file_paths = file_paths
        self.metadata = metadata

        # Derived attributes
        self.n_samples = len(features)
        self.n_features = features.shape[1]
        self.song_names = [Path(p).stem for p in file_paths]

    def __len__(self) -> int:
        """Get number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample by index.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        dict
            Sample data
        """
        sample = {
            'features': self.features[idx],
            'file_path': self.file_paths[idx],
            'song_name': self.song_names[idx],
            'index': idx
        }

        if self.metadata is not None:
            sample['metadata'] = self.metadata.iloc[idx].to_dict()

        return sample

    def get_feature_matrix(self) -> np.ndarray:
        """Get the full feature matrix."""
        return self.features

    def get_file_paths(self) -> List[str]:
        """Get list of file paths."""
        return self.file_paths

    def get_song_names(self) -> List[str]:
        """Get list of song names."""
        return self.song_names

    def get_metadata(self) -> Optional[pd.DataFrame]:
        """Get metadata DataFrame."""
        return self.metadata

    def summary(self):
        """Print dataset summary."""
        print("=" * 60)
        print("📦 Feature Dataset Summary")
        print("=" * 60)
        print(f"Number of songs: {self.n_samples}")
        print(f"Feature dimension: {self.n_features}")
        print(f"Shape: {self.features.shape}")
        print(f"\nFeature statistics:")
        print(f"  Mean: {self.features.mean():.4f}")
        print(f"  Std:  {self.features.std():.4f}")
        print(f"  Min:  {self.features.min():.4f}")
        print(f"  Max:  {self.features.max():.4f}")

        if self.metadata is not None:
            print(f"\nMetadata columns: {list(self.metadata.columns)}")

            if 'tempo_bpm' in self.metadata.columns:
                tempo_vals = self.metadata['tempo_bpm'].dropna()
                if len(tempo_vals) > 0:
                    print(f"  Tempo range: {tempo_vals.min():.1f} - {tempo_vals.max():.1f} BPM")

            if 'duration_seconds' in self.metadata.columns:
                dur_vals = self.metadata['duration_seconds'].dropna()
                if len(dur_vals) > 0:
                    print(f"  Duration range: {dur_vals.min():.1f} - {dur_vals.max():.1f} seconds")

        print("=" * 60)


def load_from_parquet(
    filepath: str,
    validate: bool = True,
    verbose: bool = True
) -> FeatureDataset:
    """
    Load features from Parquet file (Member 1's default format).

    Parameters
    ----------
    filepath : str
        Path to parquet file
    validate : bool
        Run validation checks
    verbose : bool
        Print loading info

    Returns
    -------
    FeatureDataset
        Loaded dataset
    """
    if verbose:
        print(f"📂 Loading features from: {filepath}")

    # Load parquet file
    df = pd.read_parquet(filepath)

    # Extract file paths
    if 'file_path' not in df.columns:
        raise ValueError("Parquet file missing 'file_path' column")

    file_paths = df['file_path'].tolist()

    # Extract metadata columns
    metadata_cols = ['tempo_bpm', 'key', 'duration_seconds']
    available_metadata_cols = [col for col in metadata_cols if col in df.columns]

    if available_metadata_cols:
        metadata = df[['file_path'] + available_metadata_cols].copy()
    else:
        metadata = None

    # Extract feature columns (everything except metadata and non-numeric)
    exclude_cols = ['file_path', 'file_name'] + available_metadata_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float32', 'float64', 'int32', 'int64']]

    features = df[feature_cols].astype(np.float32).values

    if verbose:
        print(f"✓ Loaded {len(features)} songs with {features.shape[1]} features")

    # Validate
    if validate:
        validator = FeatureValidator(expected_dim=features.shape[1])
        is_valid, results = validator.validate_all(features, file_paths, verbose=verbose)

        if not is_valid:
            warnings.warn("Validation failed - data may have quality issues")

    return FeatureDataset(features, file_paths, metadata)


def load_from_csv(
    filepath: str,
    validate: bool = True,
    verbose: bool = True
) -> FeatureDataset:
    """
    Load features from CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file
    validate : bool
        Run validation checks
    verbose : bool
        Print loading info

    Returns
    -------
    FeatureDataset
        Loaded dataset
    """
    if verbose:
        print(f"📂 Loading features from: {filepath}")

    # Load CSV
    df = pd.read_csv(filepath)

    # Extract file paths
    if 'file_path' not in df.columns:
        raise ValueError("CSV file missing 'file_path' column")

    file_paths = df['file_path'].tolist()

    # Extract metadata
    metadata_cols = ['tempo_bpm', 'key', 'duration_seconds']
    available_metadata_cols = [col for col in metadata_cols if col in df.columns]

    if available_metadata_cols:
        metadata = df[['file_path'] + available_metadata_cols].copy()
    else:
        metadata = None

    # Extract features (only numeric columns)
    exclude_cols = ['file_path', 'file_name'] + available_metadata_cols
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float32', 'float64', 'int32', 'int64']]

    features = df[feature_cols].astype(np.float32).values

    if verbose:
        print(f"✓ Loaded {len(features)} songs with {features.shape[1]} features")

    # Validate
    if validate:
        validator = FeatureValidator(expected_dim=features.shape[1])
        is_valid, results = validator.validate_all(features, file_paths, verbose=verbose)

        if not is_valid:
            warnings.warn("Validation failed - data may have quality issues")

    return FeatureDataset(features, file_paths, metadata)


def load_from_npy(
    features_path: str,
    file_paths_path: Optional[str] = None,
    validate: bool = True,
    verbose: bool = True
) -> FeatureDataset:
    """
    Load features from NumPy binary file.

    Parameters
    ----------
    features_path : str
        Path to .npy file with features
    file_paths_path : str, optional
        Path to text file with file paths (one per line)
        If None, will look for <features_path>_paths.txt
    validate : bool
        Run validation checks
    verbose : bool
        Print loading info

    Returns
    -------
    FeatureDataset
        Loaded dataset
    """
    if verbose:
        print(f"📂 Loading features from: {features_path}")

    # Load features
    features = np.load(features_path)

    if features.ndim != 2:
        raise ValueError(f"Expected 2D array, got {features.ndim}D")

    # Load file paths
    if file_paths_path is None:
        # Try to find companion paths file
        base_path = Path(features_path).with_suffix('')
        file_paths_path = f"{base_path}_paths.txt"

    if Path(file_paths_path).exists():
        with open(file_paths_path, 'r') as f:
            file_paths = [line.strip() for line in f]

        if len(file_paths) != len(features):
            raise ValueError(
                f"Mismatch: {len(features)} features but {len(file_paths)} paths"
            )
    else:
        # Generate dummy paths
        warnings.warn(f"File paths not found at {file_paths_path}, using dummy names")
        file_paths = [f"song_{i:04d}" for i in range(len(features))]

    if verbose:
        print(f"✓ Loaded {len(features)} songs with {features.shape[1]} features")

    # Validate
    if validate:
        validator = FeatureValidator(expected_dim=features.shape[1])
        is_valid, results = validator.validate_all(features, file_paths, verbose=verbose)

        if not is_valid:
            warnings.warn("Validation failed - data may have quality issues")

    return FeatureDataset(features, file_paths, metadata=None)


def load_features(
    filepath: str,
    format: Optional[str] = None,
    validate: bool = True,
    preprocess: bool = True,
    scaling_method: str = 'zscore',
    verbose: bool = True
) -> Tuple[FeatureDataset, Optional[FeaturePreprocessor]]:
    """
    Universal feature loader - automatically detects format.

    Parameters
    ----------
    filepath : str
        Path to feature file
    format : str, optional
        File format: 'parquet', 'csv', 'npy'. If None, auto-detect from extension
    validate : bool
        Run validation checks
    preprocess : bool
        Apply preprocessing (scaling)
    scaling_method : str
        Scaling method if preprocess=True
    verbose : bool
        Print loading info

    Returns
    -------
    dataset : FeatureDataset
        Loaded dataset (with preprocessed features if preprocess=True)
    preprocessor : FeaturePreprocessor or None
        Fitted preprocessor (if preprocess=True), else None
    """
    filepath = Path(filepath)

    # Auto-detect format
    if format is None:
        format = filepath.suffix.lower().lstrip('.')

    # Load based on format
    if format == 'parquet':
        dataset = load_from_parquet(str(filepath), validate, verbose)
    elif format == 'csv':
        dataset = load_from_csv(str(filepath), validate, verbose)
    elif format in ['npy', 'npz']:
        dataset = load_from_npy(str(filepath), None, validate, verbose)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Preprocessing
    preprocessor = None
    if preprocess:
        if verbose:
            print(f"\n🔧 Preprocessing features...")

        processed_features, preprocessor = auto_preprocess(
            dataset.features,
            scaling_method=scaling_method,
            check_scaling=True,
            verbose=verbose
        )

        # Update dataset with preprocessed features
        dataset.features = processed_features

    return dataset, preprocessor


def find_latest_feature_file(
    directory: str,
    format: str = 'parquet',
    pattern: str = 'audio_features_*'
) -> Optional[str]:
    """
    Find the most recent feature file in a directory.

    Parameters
    ----------
    directory : str
        Directory to search
    format : str
        File format extension
    pattern : str
        Glob pattern for filename

    Returns
    -------
    str or None
        Path to latest file, or None if not found
    """
    search_pattern = f"{directory}/{pattern}.{format}"
    files = glob.glob(search_pattern)

    if not files:
        return None

    # Sort by modification time
    latest = max(files, key=lambda p: Path(p).stat().st_mtime)
    return latest


def load_multiple_feature_files(
    filepaths: List[str],
    validate: bool = True,
    preprocess: bool = True,
    scaling_method: str = 'zscore',
    verbose: bool = True
) -> Tuple[FeatureDataset, Optional[FeaturePreprocessor]]:
    """
    Load and concatenate multiple feature files.

    Parameters
    ----------
    filepaths : list
        List of feature file paths
    validate : bool
        Run validation checks
    preprocess : bool
        Apply preprocessing
    scaling_method : str
        Scaling method
    verbose : bool
        Print info

    Returns
    -------
    dataset : FeatureDataset
        Combined dataset
    preprocessor : FeaturePreprocessor or None
        Fitted preprocessor
    """
    if verbose:
        print(f"📂 Loading {len(filepaths)} feature files...")

    all_features = []
    all_paths = []
    all_metadata = []

    for filepath in filepaths:
        dataset, _ = load_features(
            filepath,
            validate=False,  # Validate only at the end
            preprocess=False,  # Preprocess only at the end
            verbose=False
        )

        all_features.append(dataset.features)
        all_paths.extend(dataset.file_paths)

        if dataset.metadata is not None:
            all_metadata.append(dataset.metadata)

    # Concatenate
    combined_features = np.vstack(all_features)

    combined_metadata = None
    if all_metadata:
        combined_metadata = pd.concat(all_metadata, ignore_index=True)

    if verbose:
        print(f"✓ Combined {len(combined_features)} songs")

    # Create combined dataset
    dataset = FeatureDataset(combined_features, all_paths, combined_metadata)

    # Validate combined data
    if validate:
        validator = FeatureValidator(expected_dim=combined_features.shape[1])
        is_valid, results = validator.validate_all(
            combined_features,
            all_paths,
            verbose=verbose
        )

        if not is_valid:
            warnings.warn("Validation failed on combined data")

    # Preprocess
    preprocessor = None
    if preprocess:
        if verbose:
            print(f"\n🔧 Preprocessing combined features...")

        processed_features, preprocessor = auto_preprocess(
            dataset.features,
            scaling_method=scaling_method,
            verbose=verbose
        )

        dataset.features = processed_features

    return dataset, preprocessor


if __name__ == '__main__':
    # Example usage
    print("Feature Loader - Example Usage\n")
    print("=" * 60)

    # Example 1: Load from parquet
    print("\nExample 1: Load from Parquet")
    print("-" * 60)
    print("dataset, preprocessor = load_features(")
    print("    '../Member1/output/audio_features_20240115_120000.parquet',")
    print("    preprocess=True,")
    print("    scaling_method='zscore'")
    print(")")
    print("dataset.summary()")

    # Example 2: Load from CSV
    print("\n\nExample 2: Load from CSV")
    print("-" * 60)
    print("dataset = load_from_csv('../Member1/output/features.csv')")

    # Example 3: Find and load latest
    print("\n\nExample 3: Auto-find latest file")
    print("-" * 60)
    print("latest_file = find_latest_feature_file('../Member1/output')")
    print("if latest_file:")
    print("    dataset, preprocessor = load_features(latest_file)")

    # Example 4: Access data
    print("\n\nExample 4: Access data")
    print("-" * 60)
    print("# Get feature matrix")
    print("X = dataset.get_feature_matrix()  # Shape: (N, 170)")
    print()
    print("# Get song names")
    print("names = dataset.get_song_names()")
    print()
    print("# Get single sample")
    print("sample = dataset[0]")
    print("print(sample['song_name'], sample['features'].shape)")

    print("\n" + "=" * 60)
