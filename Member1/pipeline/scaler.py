"""
Feature Scaling Module

Implements Min-Max and Z-score normalization for feature vectors.
Ensures that different features have comparable scales for AI models.

Why scaling matters:
- Tempo (20-200 BPM) would dominate over Spectral Centroid (0.0-1.0)
- AI models converge faster with normalized inputs
- Distance-based algorithms (clustering, kNN) require scaled features
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import numpy as np


class FeatureScaler(ABC):
    """Abstract base class for feature scalers."""
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        """Fit the scaler to the data."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the transformation."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save scaler parameters to file."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'FeatureScaler':
        """Load scaler from file."""
        pass


class MinMaxScaler(FeatureScaler):
    """
    Min-Max normalization to scale features to [0, 1] range.
    
    Formula: X_scaled = (X - X_min) / (X_max - X_min)
    
    Pros:
    - Bounded output [0, 1]
    - Preserves zero entries if original range includes 0
    
    Cons:
    - Sensitive to outliers
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize MinMaxScaler.
        
        Parameters
        ----------
        feature_range : tuple, optional
            Desired range of transformed data (default: (0, 1))
        """
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute min and max from training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
            
        Returns
        -------
        self
        """
        X = np.atleast_2d(X)
        
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.data_range_ = self.max_ - self.min_
        
        # Handle constant features (avoid division by zero)
        self.data_range_[self.data_range_ == 0] = 1.0
        
        # Compute scale factor
        range_min, range_max = self.feature_range
        self.scale_ = (range_max - range_min) / self.data_range_
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features to the target range.
        
        Parameters
        ----------
        X : np.ndarray
            Data to transform
            
        Returns
        -------
        np.ndarray
            Scaled data
        """
        if self.min_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        X = np.atleast_2d(X)
        
        range_min, range_max = self.feature_range
        X_scaled = (X - self.min_) * self.scale_ + range_min
        
        return X_scaled
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling transformation.
        
        Parameters
        ----------
        X : np.ndarray
            Scaled data
            
        Returns
        -------
        np.ndarray
            Original scale data
        """
        if self.min_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        X = np.atleast_2d(X)
        
        range_min, range_max = self.feature_range
        X_original = (X - range_min) / self.scale_ + self.min_
        
        return X_original
    
    def save(self, path: str) -> None:
        """Save scaler parameters to JSON file."""
        params = {
            'type': 'MinMaxScaler',
            'feature_range': self.feature_range,
            'min_': self.min_.tolist() if self.min_ is not None else None,
            'max_': self.max_.tolist() if self.max_ is not None else None,
            'scale_': self.scale_.tolist() if self.scale_ is not None else None,
            'data_range_': self.data_range_.tolist() if self.data_range_ is not None else None,
        }
        
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'MinMaxScaler':
        """Load scaler from JSON file."""
        with open(path, 'r') as f:
            params = json.load(f)
        
        scaler = cls(feature_range=tuple(params['feature_range']))
        scaler.min_ = np.array(params['min_']) if params['min_'] else None
        scaler.max_ = np.array(params['max_']) if params['max_'] else None
        scaler.scale_ = np.array(params['scale_']) if params['scale_'] else None
        scaler.data_range_ = np.array(params['data_range_']) if params['data_range_'] else None
        
        return scaler


class ZScoreScaler(FeatureScaler):
    """
    Z-score normalization (standardization).
    
    Formula: X_scaled = (X - mean) / std
    
    Pros:
    - Handles outliers better than MinMax
    - Centered at 0, useful for many ML algorithms
    
    Cons:
    - Output is unbounded
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Initialize ZScoreScaler.
        
        Parameters
        ----------
        with_mean : bool, optional
            Center data (default: True)
        with_std : bool, optional
            Scale to unit variance (default: True)
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'ZScoreScaler':
        """
        Compute mean and std from training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features)
            
        Returns
        -------
        self
        """
        X = np.atleast_2d(X)
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        
        if self.with_std:
            self.var_ = np.var(X, axis=0)
            self.std_ = np.sqrt(self.var_)
            # Handle constant features
            self.std_[self.std_ == 0] = 1.0
        else:
            self.std_ = np.ones(X.shape[1])
            self.var_ = np.ones(X.shape[1])
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features.
        
        Parameters
        ----------
        X : np.ndarray
            Data to transform
            
        Returns
        -------
        np.ndarray
            Standardized data
        """
        if self.mean_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        X = np.atleast_2d(X)
        
        X_scaled = (X - self.mean_) / self.std_
        
        return X_scaled
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the standardization.
        
        Parameters
        ----------
        X : np.ndarray
            Standardized data
            
        Returns
        -------
        np.ndarray
            Original scale data
        """
        if self.mean_ is None:
            raise ValueError("Scaler has not been fitted. Call fit() first.")
        
        X = np.atleast_2d(X)
        
        X_original = X * self.std_ + self.mean_
        
        return X_original
    
    def save(self, path: str) -> None:
        """Save scaler parameters to JSON file."""
        params = {
            'type': 'ZScoreScaler',
            'with_mean': self.with_mean,
            'with_std': self.with_std,
            'mean_': self.mean_.tolist() if self.mean_ is not None else None,
            'std_': self.std_.tolist() if self.std_ is not None else None,
            'var_': self.var_.tolist() if self.var_ is not None else None,
        }
        
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ZScoreScaler':
        """Load scaler from JSON file."""
        with open(path, 'r') as f:
            params = json.load(f)
        
        scaler = cls(
            with_mean=params['with_mean'],
            with_std=params['with_std']
        )
        scaler.mean_ = np.array(params['mean_']) if params['mean_'] else None
        scaler.std_ = np.array(params['std_']) if params['std_'] else None
        scaler.var_ = np.array(params['var_']) if params['var_'] else None
        
        return scaler


def create_scaler(method: str = 'zscore', **kwargs) -> FeatureScaler:
    """
    Factory function to create a scaler.
    
    Parameters
    ----------
    method : str
        Scaling method: 'minmax' or 'zscore'
    **kwargs
        Additional arguments for the scaler
        
    Returns
    -------
    FeatureScaler
        Configured scaler instance
    """
    if method == 'minmax':
        return MinMaxScaler(**kwargs)
    elif method == 'zscore':
        return ZScoreScaler(**kwargs)
    else:
        raise ValueError(f"Unknown scaling method: {method}")


def load_scaler(path: str) -> FeatureScaler:
    """
    Load a scaler from file.
    
    Parameters
    ----------
    path : str
        Path to scaler JSON file
        
    Returns
    -------
    FeatureScaler
        Loaded scaler
    """
    with open(path, 'r') as f:
        params = json.load(f)
    
    scaler_type = params['type']
    
    if scaler_type == 'MinMaxScaler':
        return MinMaxScaler.load(path)
    elif scaler_type == 'ZScoreScaler':
        return ZScoreScaler.load(path)
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
