"""
Feature Preprocessing Module
Handles normalization, scaling, and feature transformations for embedding models
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import warnings


class FeaturePreprocessor:
    """Preprocesses features for embedding/clustering models."""

    def __init__(
        self,
        scaling_method: str = 'zscore',
        feature_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Initialize preprocessor.

        Parameters
        ----------
        scaling_method : str
            Scaling method: 'zscore', 'minmax', or 'none'
        feature_range : tuple
            Target range for minmax scaling
        """
        self.scaling_method = scaling_method
        self.feature_range = feature_range

        # Store fitted parameters
        self.is_fitted = False
        self.mean_ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None

    def fit(self, features: np.ndarray) -> 'FeaturePreprocessor':
        """
        Fit preprocessor parameters on training data.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (N x D)

        Returns
        -------
        self
        """
        if self.scaling_method == 'zscore':
            self.mean_ = np.mean(features, axis=0)
            self.std_ = np.std(features, axis=0)

            # Avoid division by zero
            self.std_[self.std_ == 0] = 1.0

        elif self.scaling_method == 'minmax':
            self.min_ = np.min(features, axis=0)
            self.max_ = np.max(features, axis=0)

            # Avoid division by zero
            range_ = self.max_ - self.min_
            range_[range_ == 0] = 1.0
            self.max_[range_ == 0] = self.min_[range_ == 0] + 1.0

        self.is_fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (N x D)

        Returns
        -------
        transformed : np.ndarray
            Scaled features
        """
        if not self.is_fitted and self.scaling_method != 'none':
            raise ValueError("Preprocessor must be fitted before transform")

        if self.scaling_method == 'zscore':
            transformed = (features - self.mean_) / self.std_

        elif self.scaling_method == 'minmax':
            # Scale to [0, 1] first
            normalized = (features - self.min_) / (self.max_ - self.min_)

            # Scale to target range
            min_val, max_val = self.feature_range
            transformed = normalized * (max_val - min_val) + min_val

        elif self.scaling_method == 'none':
            transformed = features.copy()

        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        return transformed

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (N x D)

        Returns
        -------
        transformed : np.ndarray
            Scaled features
        """
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, scaled_features: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling transformation.

        Parameters
        ----------
        scaled_features : np.ndarray
            Scaled feature matrix

        Returns
        -------
        original : np.ndarray
            Features in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")

        if self.scaling_method == 'zscore':
            original = scaled_features * self.std_ + self.mean_

        elif self.scaling_method == 'minmax':
            # Reverse range scaling
            min_val, max_val = self.feature_range
            normalized = (scaled_features - min_val) / (max_val - min_val)

            # Reverse minmax scaling
            original = normalized * (self.max_ - self.min_) + self.min_

        elif self.scaling_method == 'none':
            original = scaled_features.copy()

        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        return original

    def check_if_already_scaled(self, features: np.ndarray) -> Dict[str, bool]:
        """
        Check if features are already scaled (by Member 1).

        Parameters
        ----------
        features : np.ndarray
            Feature matrix

        Returns
        -------
        dict
            Detection results
        """
        mean = np.mean(features)
        std = np.std(features)
        min_val = np.min(features)
        max_val = np.max(features)

        # Check for z-score scaling (mean ≈ 0, std ≈ 1)
        is_zscore = abs(mean) < 0.1 and abs(std - 1.0) < 0.2

        # Check for minmax scaling (min ≈ 0, max ≈ 1)
        is_minmax = (abs(min_val) < 0.1 and abs(max_val - 1.0) < 0.1)

        return {
            'appears_zscore': is_zscore,
            'appears_minmax': is_minmax,
            'mean': float(mean),
            'std': float(std),
            'min': float(min_val),
            'max': float(max_val)
        }

    def handle_missing_values(
        self,
        features: np.ndarray,
        method: str = 'mean'
    ) -> np.ndarray:
        """
        Handle missing values in features.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        method : str
            Imputation method: 'mean', 'median', 'zero', or 'drop'

        Returns
        -------
        filled : np.ndarray
            Features with missing values handled
        """
        if not np.any(np.isnan(features)):
            return features

        if method == 'drop':
            # This should be handled by validator
            raise ValueError("Use validator.remove_problematic_samples() instead")

        filled = features.copy()

        if method == 'mean':
            col_means = np.nanmean(filled, axis=0)
            inds = np.where(np.isnan(filled))
            filled[inds] = np.take(col_means, inds[1])

        elif method == 'median':
            col_medians = np.nanmedian(filled, axis=0)
            inds = np.where(np.isnan(filled))
            filled[inds] = np.take(col_medians, inds[1])

        elif method == 'zero':
            filled[np.isnan(filled)] = 0.0

        else:
            raise ValueError(f"Unknown imputation method: {method}")

        return filled

    def split_features_by_category(
        self,
        features: np.ndarray,
        category_ranges: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Split features into categories (timbral, rhythmic, harmonic, etc.).

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (N x 170)
        category_ranges : dict, optional
            Dictionary mapping category names to (start, end) indices
            If None, uses default ranges from Member 1's schema

        Returns
        -------
        dict
            Dictionary of category name -> feature submatrix
        """
        if category_ranges is None:
            # Default ranges from Member 1's feature schema
            # These are approximate - should be verified
            category_ranges = {
                'time_domain': (0, 18),
                'timbral': (18, 306),  # MFCCs + spectral features
                'rhythmic': (306, 315),
                'harmonic': (315, 387),  # Chroma + Tonnetz
                'groove': (387, 410),
                'structural': (410, 420)
            }

        split_features = {}
        for category, (start, end) in category_ranges.items():
            if end <= features.shape[1]:
                split_features[category] = features[:, start:end]
            else:
                warnings.warn(f"Category {category} range exceeds feature dimensions")

        return split_features

    def get_scaling_info(self) -> Dict:
        """
        Get information about fitted scaling parameters.

        Returns
        -------
        dict
            Scaling parameters and statistics
        """
        if not self.is_fitted:
            return {'fitted': False}

        info = {
            'fitted': True,
            'method': self.scaling_method,
        }

        if self.scaling_method == 'zscore':
            info['mean_of_means'] = float(np.mean(self.mean_))
            info['mean_of_stds'] = float(np.mean(self.std_))

        elif self.scaling_method == 'minmax':
            info['global_min'] = float(np.min(self.min_))
            info['global_max'] = float(np.max(self.max_))
            info['target_range'] = self.feature_range

        return info

    def save(self, filepath: str):
        """
        Save preprocessor parameters to JSON.

        Parameters
        ----------
        filepath : str
            Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")

        data = {
            'scaling_method': self.scaling_method,
            'feature_range': self.feature_range,
            'is_fitted': self.is_fitted
        }

        if self.scaling_method == 'zscore':
            data['mean'] = self.mean_.tolist()
            data['std'] = self.std_.tolist()

        elif self.scaling_method == 'minmax':
            data['min'] = self.min_.tolist()
            data['max'] = self.max_.tolist()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FeaturePreprocessor':
        """
        Load preprocessor from JSON file.

        Parameters
        ----------
        filepath : str
            Path to saved preprocessor

        Returns
        -------
        FeaturePreprocessor
            Loaded preprocessor
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        preprocessor = cls(
            scaling_method=data['scaling_method'],
            feature_range=tuple(data['feature_range'])
        )

        preprocessor.is_fitted = data['is_fitted']

        if data['scaling_method'] == 'zscore':
            preprocessor.mean_ = np.array(data['mean'])
            preprocessor.std_ = np.array(data['std'])

        elif data['scaling_method'] == 'minmax':
            preprocessor.min_ = np.array(data['min'])
            preprocessor.max_ = np.array(data['max'])

        print(f"✓ Preprocessor loaded from {filepath}")
        return preprocessor


def auto_preprocess(
    features: np.ndarray,
    scaling_method: str = 'zscore',
    check_scaling: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, FeaturePreprocessor]:
    """
    Automatically preprocess features with sensible defaults.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    scaling_method : str
        Scaling method to apply
    check_scaling : bool
        Check if already scaled
    verbose : bool
        Print processing info

    Returns
    -------
    processed : np.ndarray
        Preprocessed features
    preprocessor : FeaturePreprocessor
        Fitted preprocessor
    """
    preprocessor = FeaturePreprocessor(scaling_method=scaling_method)

    if check_scaling and scaling_method != 'none':
        scaling_status = preprocessor.check_if_already_scaled(features)

        if verbose:
            print("=" * 60)
            print("Feature Scaling Check")
            print("=" * 60)
            print(f"Mean: {scaling_status['mean']:.4f}")
            print(f"Std:  {scaling_status['std']:.4f}")
            print(f"Min:  {scaling_status['min']:.4f}")
            print(f"Max:  {scaling_status['max']:.4f}")

            if scaling_status['appears_zscore']:
                print("✓ Features appear to be Z-score scaled already")
            elif scaling_status['appears_minmax']:
                print("✓ Features appear to be MinMax scaled already")
            else:
                print("⚠ Features do not appear to be scaled")
            print("=" * 60)

    # Apply preprocessing
    processed = preprocessor.fit_transform(features)

    if verbose:
        print(f"\n✓ Applied {scaling_method} scaling")
        print(f"  Output shape: {processed.shape}")
        print(f"  Output range: [{processed.min():.4f}, {processed.max():.4f}]")

    return processed, preprocessor


if __name__ == '__main__':
    # Test preprocessor
    print("Testing Feature Preprocessor...\n")

    # Create synthetic data
    np.random.seed(42)
    features = np.random.randn(100, 170) * 10 + 5  # Mean≈5, Std≈10

    print("Original features:")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std:  {features.std():.4f}")
    print(f"  Min:  {features.min():.4f}")
    print(f"  Max:  {features.max():.4f}")

    # Test Z-score scaling
    print("\n" + "=" * 60)
    print("Testing Z-score scaling...")
    print("=" * 60)

    preprocessor = FeaturePreprocessor(scaling_method='zscore')
    scaled = preprocessor.fit_transform(features)

    print(f"Scaled mean: {scaled.mean():.4f}")
    print(f"Scaled std:  {scaled.std():.4f}")

    # Test inverse transform
    recovered = preprocessor.inverse_transform(scaled)
    print(f"Recovery error: {np.max(np.abs(features - recovered)):.2e}")

    # Test MinMax scaling
    print("\n" + "=" * 60)
    print("Testing MinMax scaling...")
    print("=" * 60)

    preprocessor2 = FeaturePreprocessor(scaling_method='minmax')
    scaled2 = preprocessor2.fit_transform(features)

    print(f"Scaled min: {scaled2.min():.4f}")
    print(f"Scaled max: {scaled2.max():.4f}")

    # Test save/load
    print("\n" + "=" * 60)
    print("Testing save/load...")
    print("=" * 60)

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    preprocessor.save(temp_path)
    loaded = FeaturePreprocessor.load(temp_path)

    print(f"Parameters match: {np.allclose(preprocessor.mean_, loaded.mean_)}")

    # Cleanup
    Path(temp_path).unlink()
