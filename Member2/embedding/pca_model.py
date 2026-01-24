"""
PCA Embedding Model
Linear dimensionality reduction baseline using Principal Component Analysis
"""

import numpy as np
from sklearn.decomposition import PCA
import joblib
from pathlib import Path
from typing import Optional
import json

from .base import BaseEmbeddingModel


class PCAModel(BaseEmbeddingModel):
    """PCA-based embedding model (linear baseline)."""

    def __init__(
        self,
        input_dim: int = 170,
        embedding_dim: int = 32,
        whiten: bool = True,
        random_state: int = 42
    ):
        """
        Initialize PCA model.

        Parameters
        ----------
        input_dim : int
            Input feature dimension
        embedding_dim : int
            Target embedding dimension (number of components)
        whiten : bool
            Whether to whiten the data (unit variance)
        random_state : int
            Random seed for reproducibility
        """
        super().__init__(input_dim, embedding_dim, name="PCA")

        self.whiten = whiten
        self.random_state = random_state

        # Initialize sklearn PCA
        self.pca = PCA(
            n_components=embedding_dim,
            whiten=whiten,
            random_state=random_state
        )

        # Statistics
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None

    def fit(self, X: np.ndarray, **kwargs) -> 'PCAModel':
        """
        Fit PCA on data.

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
            print(f"\n🔧 Fitting PCA model")
            print(f"   Input dimension: {self.input_dim}")
            print(f"   Embedding dimension: {self.embedding_dim}")
            print(f"   Whitening: {self.whiten}")
            print("=" * 60)

        # Fit PCA
        self.pca.fit(X)

        # Store statistics
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        self.is_fitted = True

        if verbose:
            print(f"✅ PCA fitted successfully")
            print(f"   Explained variance (first 5 components): {self.explained_variance_ratio_[:5]}")
            print(f"   Cumulative explained variance: {self.cumulative_variance_ratio_[-1]:.4f}")
            print(f"   Total variance retained: {self.cumulative_variance_ratio_[-1]*100:.2f}%")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to PCA embedding space.

        Parameters
        ----------
        X : np.ndarray
            Input data (N x input_dim)

        Returns
        -------
        embeddings : np.ndarray
            PCA embeddings (N x embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        return self.pca.transform(X)

    def inverse_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reconstruct original features from embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            PCA embeddings (N x embedding_dim)

        Returns
        -------
        reconstructed : np.ndarray
            Reconstructed features (N x input_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse_transform")

        return self.pca.inverse_transform(embeddings)

    def get_components(self) -> np.ndarray:
        """
        Get PCA components (principal axes).

        Returns
        -------
        components : np.ndarray
            Principal components (embedding_dim x input_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.pca.components_

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio for each component.

        Returns
        -------
        variance_ratios : np.ndarray
            Explained variance ratios
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.explained_variance_ratio_

    def save(self, filepath: str):
        """Save PCA model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save PCA model
        joblib.dump(self.pca, str(filepath))

        # Save metadata
        meta_path = filepath.with_suffix('.json')
        metadata = {
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'whiten': self.whiten,
            'random_state': self.random_state,
            'explained_variance_ratio': self.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': self.cumulative_variance_ratio_.tolist()
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ PCA model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'PCAModel':
        """Load PCA model from disk."""
        filepath = Path(filepath)

        # Load metadata
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Create model
        model = cls(
            input_dim=metadata['input_dim'],
            embedding_dim=metadata['embedding_dim'],
            whiten=metadata['whiten'],
            random_state=metadata['random_state']
        )

        # Load PCA
        model.pca = joblib.load(str(filepath))
        model.explained_variance_ratio_ = np.array(metadata['explained_variance_ratio'])
        model.cumulative_variance_ratio_ = np.array(metadata['cumulative_variance_ratio'])
        model.is_fitted = True

        print(f"✓ PCA model loaded from {filepath}")
        return model

    def plot_explained_variance(self, save_path: Optional[str] = None):
        """
        Plot explained variance by component.

        Parameters
        ----------
        save_path : str, optional
            Path to save plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Individual variance
        ax1.bar(range(1, len(self.explained_variance_ratio_) + 1),
                self.explained_variance_ratio_,
                alpha=0.7,
                color='steelblue')
        ax1.set_xlabel('Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('Variance per Component', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3, axis='y')

        # Cumulative variance
        ax2.plot(range(1, len(self.cumulative_variance_ratio_) + 1),
                 self.cumulative_variance_ratio_,
                 marker='o',
                 linewidth=2,
                 markersize=4,
                 color='darkred')
        ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")

        plt.show()

    def get_feature_importance(self, feature_names: Optional[list] = None, top_k: int = 10) -> dict:
        """
        Get most important features for each principal component.

        Parameters
        ----------
        feature_names : list, optional
            Names of original features
        top_k : int
            Number of top features to return per component

        Returns
        -------
        dict
            Component -> top features mapping
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.input_dim)]

        components = self.get_components()
        importance = {}

        for i, component in enumerate(components):
            # Get top k features by absolute weight
            top_indices = np.argsort(np.abs(component))[-top_k:][::-1]
            top_features = [(feature_names[idx], component[idx]) for idx in top_indices]
            importance[f"PC{i+1}"] = top_features

        return importance

    def print_summary(self):
        """Print model summary."""
        if not self.is_fitted:
            print("❌ Model not fitted yet")
            return

        print("=" * 60)
        print("📊 PCA Model Summary")
        print("=" * 60)
        print(f"Input dimension: {self.input_dim}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Whitening: {self.whiten}")
        print(f"\nExplained Variance:")
        print(f"  Total retained: {self.cumulative_variance_ratio_[-1]*100:.2f}%")
        print(f"  First 5 components: {self.explained_variance_ratio_[:5]}")
        print(f"  Last 5 components: {self.explained_variance_ratio_[-5:]}")
        print("=" * 60)


if __name__ == '__main__':
    # Test PCA model
    print("Testing PCA Model...\n")

    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(500, 170)
    X_test = np.random.randn(100, 170)

    # Create and fit model
    model = PCAModel(
        input_dim=170,
        embedding_dim=32,
        whiten=True
    )

    print("Fitting PCA...")
    model.fit(X_train, verbose=True)

    # Transform
    embeddings = model.transform(X_test)
    print(f"\n✓ Embeddings shape: {embeddings.shape}")

    # Inverse transform (reconstruction)
    reconstructed = model.inverse_transform(embeddings)
    mse = np.mean((X_test - reconstructed) ** 2)
    print(f"✓ Reconstruction MSE: {mse:.4f}")

    # Print summary
    print()
    model.print_summary()

    print("\n✅ PCA test complete")
