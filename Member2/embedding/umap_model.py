"""
UMAP Embedding Model
Nonlinear dimensionality reduction using Uniform Manifold Approximation and Projection
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional
import json

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠ UMAP not available. Install with: pip install umap-learn")

from .base import BaseEmbeddingModel


class UMAPModel(BaseEmbeddingModel):
    """UMAP-based embedding model (nonlinear)."""

    def __init__(
        self,
        input_dim: int = 170,
        embedding_dim: int = 32,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
    ):
        """
        Initialize UMAP model.

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        embedding_dim : int
            Target embedding dimension
        n_neighbors : int
            Number of neighbors for manifold approximation
            - Smaller: focus on local structure
            - Larger: preserve global structure
        min_dist : float
            Minimum distance between points in embedding
            - Smaller: tighter clusters
            - Larger: more spread out
        metric : str
            Distance metric ('euclidean', 'cosine', 'manhattan', etc.)
        random_state : int
            Random seed
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")

        super().__init__(input_dim, embedding_dim, name="UMAP")

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

        # Initialize UMAP
        self.umap = umap.UMAP(
            n_components=embedding_dim,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=False
        )

    def fit(self, X: np.ndarray, **kwargs) -> 'UMAPModel':
        """
        Fit UMAP on data.

        Parameters
        ----------
        X : np.ndarray
            Training data (N x input_dim)

        Returns
        -------
        self
        """
        verbose = kwargs.get('verbose', True)

        if verbose:
            print(f"\n🔧 Fitting UMAP model")
            print(f"   Input dimension: {self.input_dim}")
            print(f"   Embedding dimension: {self.embedding_dim}")
            print(f"   N neighbors: {self.n_neighbors}")
            print(f"   Min distance: {self.min_dist}")
            print(f"   Metric: {self.metric}")
            print("=" * 60)

        # Fit UMAP
        self.umap.fit(X)
        self.is_fitted = True

        if verbose:
            print(f"✅ UMAP fitted successfully")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to UMAP embedding space.

        Parameters
        ----------
        X : np.ndarray
            Input data (N x input_dim)

        Returns
        -------
        embeddings : np.ndarray
            UMAP embeddings (N x embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        return self.umap.transform(X)

    def save(self, filepath: str):
        """Save UMAP model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save UMAP model
        joblib.dump(self.umap, str(filepath))

        # Save metadata
        meta_path = filepath.with_suffix('.json')
        metadata = {
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'metric': self.metric,
            'random_state': self.random_state
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ UMAP model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'UMAPModel':
        """Load UMAP model from disk."""
        filepath = Path(filepath)

        # Load metadata
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Create model
        model = cls(
            input_dim=metadata['input_dim'],
            embedding_dim=metadata['embedding_dim'],
            n_neighbors=metadata['n_neighbors'],
            min_dist=metadata['min_dist'],
            metric=metadata['metric'],
            random_state=metadata['random_state']
        )

        # Load UMAP
        model.umap = joblib.load(str(filepath))
        model.is_fitted = True

        print(f"✓ UMAP model loaded from {filepath}")
        return model

    def print_summary(self):
        """Print model summary."""
        print("=" * 60)
        print("📊 UMAP Model Summary")
        print("=" * 60)
        print(f"Input dimension: {self.input_dim}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"N neighbors: {self.n_neighbors}")
        print(f"Min distance: {self.min_dist}")
        print(f"Metric: {self.metric}")
        print(f"Fitted: {self.is_fitted}")
        print("=" * 60)


if __name__ == '__main__':
    if not UMAP_AVAILABLE:
        print("❌ UMAP not available. Install with: pip install umap-learn")
    else:
        # Test UMAP model
        print("Testing UMAP Model...\n")

        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(500, 170)
        X_test = np.random.randn(100, 170)

        # Create and fit model
        model = UMAPModel(
            input_dim=170,
            embedding_dim=32,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )

        print("Fitting UMAP...")
        model.fit(X_train, verbose=True)

        # Transform
        embeddings_train = model.transform(X_train[:10])
        embeddings_test = model.transform(X_test)

        print(f"\n✓ Train embeddings shape: {embeddings_train.shape}")
        print(f"✓ Test embeddings shape: {embeddings_test.shape}")

        # Print summary
        print()
        model.print_summary()

        print("\n✅ UMAP test complete")
