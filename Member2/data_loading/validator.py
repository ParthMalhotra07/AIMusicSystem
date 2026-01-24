"""
Data Validation Module
Validates feature data quality and integrity from Member 1's output
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings


class FeatureValidator:
    """Validates feature matrices and detects data quality issues."""

    def __init__(self, expected_dim: int = 170):
        """
        Initialize validator.

        Parameters
        ----------
        expected_dim : int
            Expected feature vector dimension from Member 1
        """
        self.expected_dim = expected_dim
        self.validation_results = {}

    def validate_all(
        self,
        features: np.ndarray,
        file_paths: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Run all validation checks.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix (N x D)
        file_paths : list, optional
            List of file paths for error reporting
        verbose : bool
            Print validation results

        Returns
        -------
        is_valid : bool
            Whether data passes all checks
        results : dict
            Detailed validation results
        """
        results = {}

        # Check 1: Dimension validation
        results['dimensions'] = self._check_dimensions(features)

        # Check 2: Missing values
        results['missing_values'] = self._check_missing_values(features)

        # Check 3: Infinite values
        results['infinite_values'] = self._check_infinite_values(features)

        # Check 4: Feature ranges
        results['feature_ranges'] = self._check_feature_ranges(features)

        # Check 5: Variance (detect constant features)
        results['variance'] = self._check_variance(features)

        # Check 6: Outliers
        results['outliers'] = self._check_outliers(features)

        # Check 7: Data distribution
        results['distribution'] = self._check_distribution(features)

        # Overall validity
        is_valid = all([
            results['dimensions']['valid'],
            results['missing_values']['valid'],
            results['infinite_values']['valid'],
            results['variance']['valid']
        ])

        self.validation_results = results

        if verbose:
            self._print_results(results, is_valid)

        return is_valid, results

    def _check_dimensions(self, features: np.ndarray) -> Dict:
        """Check if feature dimensions match expected."""
        n_samples, n_features = features.shape
        valid = n_features == self.expected_dim

        return {
            'valid': valid,
            'n_samples': n_samples,
            'n_features': n_features,
            'expected_features': self.expected_dim,
            'message': f'Found {n_features} features, expected {self.expected_dim}'
        }

    def _check_missing_values(self, features: np.ndarray) -> Dict:
        """Check for NaN values."""
        nan_mask = np.isnan(features)
        n_nan = np.sum(nan_mask)
        nan_ratio = n_nan / features.size

        # Find which samples/features have NaN
        nan_rows = np.where(nan_mask.any(axis=1))[0]
        nan_cols = np.where(nan_mask.any(axis=0))[0]

        valid = n_nan == 0

        return {
            'valid': valid,
            'n_missing': int(n_nan),
            'missing_ratio': float(nan_ratio),
            'affected_samples': nan_rows.tolist()[:10],  # First 10
            'affected_features': nan_cols.tolist()[:10],
            'message': f'{n_nan} missing values ({nan_ratio*100:.2f}%)'
        }

    def _check_infinite_values(self, features: np.ndarray) -> Dict:
        """Check for infinite values."""
        inf_mask = np.isinf(features)
        n_inf = np.sum(inf_mask)
        inf_ratio = n_inf / features.size

        inf_rows = np.where(inf_mask.any(axis=1))[0]
        inf_cols = np.where(inf_mask.any(axis=0))[0]

        valid = n_inf == 0

        return {
            'valid': valid,
            'n_infinite': int(n_inf),
            'infinite_ratio': float(inf_ratio),
            'affected_samples': inf_rows.tolist()[:10],
            'affected_features': inf_cols.tolist()[:10],
            'message': f'{n_inf} infinite values ({inf_ratio*100:.2f}%)'
        }

    def _check_feature_ranges(self, features: np.ndarray) -> Dict:
        """Check feature value ranges."""
        mins = np.min(features, axis=0)
        maxs = np.max(features, axis=0)
        ranges = maxs - mins

        # Detect features with extreme ranges
        extreme_threshold = 1e6
        extreme_features = np.where(ranges > extreme_threshold)[0]

        return {
            'min_value': float(np.min(mins)),
            'max_value': float(np.max(maxs)),
            'mean_range': float(np.mean(ranges)),
            'extreme_features': extreme_features.tolist()[:10],
            'message': f'Value range: [{np.min(mins):.2f}, {np.max(maxs):.2f}]'
        }

    def _check_variance(self, features: np.ndarray) -> Dict:
        """Check for zero or near-zero variance features."""
        variances = np.var(features, axis=0)

        # Detect constant or near-constant features
        zero_var_threshold = 1e-10
        zero_var_features = np.where(variances < zero_var_threshold)[0]

        valid = len(zero_var_features) < features.shape[1] * 0.1  # Allow up to 10%

        return {
            'valid': valid,
            'n_zero_variance': len(zero_var_features),
            'zero_variance_features': zero_var_features.tolist()[:10],
            'mean_variance': float(np.mean(variances)),
            'message': f'{len(zero_var_features)} features have zero variance'
        }

    def _check_outliers(self, features: np.ndarray) -> Dict:
        """Detect outliers using z-score method."""
        # Calculate z-scores
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)

        # Avoid division by zero
        std[std == 0] = 1.0

        z_scores = np.abs((features - mean) / std)

        # Count extreme outliers (|z| > 5)
        outlier_threshold = 5.0
        outlier_mask = z_scores > outlier_threshold
        n_outliers = np.sum(outlier_mask)
        outlier_ratio = n_outliers / features.size

        outlier_rows = np.where(outlier_mask.any(axis=1))[0]

        return {
            'n_outliers': int(n_outliers),
            'outlier_ratio': float(outlier_ratio),
            'affected_samples': outlier_rows.tolist()[:10],
            'message': f'{n_outliers} extreme outliers detected ({outlier_ratio*100:.2f}%)'
        }

    def _check_distribution(self, features: np.ndarray) -> Dict:
        """Analyze feature distribution."""
        mean = np.mean(features)
        std = np.std(features)
        median = np.median(features)
        q25, q75 = np.percentile(features, [25, 75])

        return {
            'mean': float(mean),
            'std': float(std),
            'median': float(median),
            'q25': float(q25),
            'q75': float(q75),
            'message': f'Mean: {mean:.4f}, Std: {std:.4f}'
        }

    def _print_results(self, results: Dict, is_valid: bool):
        """Print validation results."""
        print("=" * 60)
        print("🔍 Feature Validation Results")
        print("=" * 60)

        # Dimensions
        dim_result = results['dimensions']
        status = "✓" if dim_result['valid'] else "✗"
        print(f"{status} Dimensions: {dim_result['n_samples']} samples × {dim_result['n_features']} features")

        # Missing values
        missing_result = results['missing_values']
        status = "✓" if missing_result['valid'] else "✗"
        print(f"{status} Missing values: {missing_result['message']}")

        # Infinite values
        inf_result = results['infinite_values']
        status = "✓" if inf_result['valid'] else "✗"
        print(f"{status} Infinite values: {inf_result['message']}")

        # Variance
        var_result = results['variance']
        status = "✓" if var_result['valid'] else "✗"
        print(f"{status} Variance: {var_result['message']}")

        # Outliers
        outlier_result = results['outliers']
        print(f"⚠ Outliers: {outlier_result['message']}")

        # Distribution
        dist_result = results['distribution']
        print(f"📊 Distribution: {dist_result['message']}")

        print("=" * 60)
        if is_valid:
            print("✅ Data validation PASSED")
        else:
            print("❌ Data validation FAILED - check errors above")
        print("=" * 60)

    def get_problematic_samples(self, features: np.ndarray) -> np.ndarray:
        """
        Get indices of problematic samples.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Indices of problematic samples
        """
        problematic = set()

        # Samples with NaN
        nan_mask = np.isnan(features)
        nan_samples = np.where(nan_mask.any(axis=1))[0]
        problematic.update(nan_samples)

        # Samples with inf
        inf_mask = np.isinf(features)
        inf_samples = np.where(inf_mask.any(axis=1))[0]
        problematic.update(inf_samples)

        return np.array(sorted(problematic))

    def remove_problematic_samples(
        self,
        features: np.ndarray,
        file_paths: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """
        Remove problematic samples from dataset.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        file_paths : list, optional
            List of file paths
        verbose : bool
            Print removal info

        Returns
        -------
        clean_features : np.ndarray
            Cleaned feature matrix
        clean_paths : list or None
            Cleaned file paths (if provided)
        """
        problematic_idx = self.get_problematic_samples(features)

        if len(problematic_idx) == 0:
            if verbose:
                print("✓ No problematic samples found")
            return features, file_paths

        # Remove problematic samples
        clean_mask = np.ones(len(features), dtype=bool)
        clean_mask[problematic_idx] = False

        clean_features = features[clean_mask]

        clean_paths = None
        if file_paths is not None:
            clean_paths = [p for i, p in enumerate(file_paths) if clean_mask[i]]

        if verbose:
            print(f"⚠ Removed {len(problematic_idx)} problematic samples")
            print(f"  Remaining: {len(clean_features)} samples")

        return clean_features, clean_paths


def quick_validate(features: np.ndarray, expected_dim: int = 170) -> bool:
    """
    Quick validation check (dimensions and missing values only).

    Parameters
    ----------
    features : np.ndarray
        Feature matrix
    expected_dim : int
        Expected feature dimension

    Returns
    -------
    bool
        Whether data is valid
    """
    # Check dimensions
    if features.ndim != 2:
        warnings.warn(f"Expected 2D array, got {features.ndim}D")
        return False

    if features.shape[1] != expected_dim:
        warnings.warn(f"Expected {expected_dim} features, got {features.shape[1]}")
        return False

    # Check for NaN/inf
    if np.any(np.isnan(features)):
        warnings.warn("Found NaN values")
        return False

    if np.any(np.isinf(features)):
        warnings.warn("Found infinite values")
        return False

    return True


if __name__ == '__main__':
    # Test validator with synthetic data
    print("Testing Feature Validator...\n")

    # Create test data
    np.random.seed(42)
    n_samples = 100
    n_features = 170

    # Good data
    good_features = np.random.randn(n_samples, n_features)

    validator = FeatureValidator(expected_dim=170)
    is_valid, results = validator.validate_all(good_features, verbose=True)

    print("\n" + "=" * 60)
    print("Testing with problematic data...\n")

    # Problematic data
    bad_features = good_features.copy()
    bad_features[5, 10] = np.nan
    bad_features[10, 20] = np.inf
    bad_features[:, 50] = 0  # Zero variance

    is_valid, results = validator.validate_all(bad_features, verbose=True)

    # Test cleanup
    print("\nTesting cleanup...")
    clean_features, _ = validator.remove_problematic_samples(bad_features, verbose=True)
    print(f"Clean shape: {clean_features.shape}")
