"""
Hierarchical Clustering Model
Agglomerative (bottom-up) hierarchical clustering
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import json
from pathlib import Path
from typing import Optional

from .base import BaseClusteringModel


class HierarchicalModel(BaseClusteringModel):
    """Hierarchical agglomerative clustering model."""

    def __init__(
        self,
        n_clusters: int = 5,
        linkage: str = 'ward',
        distance_threshold: Optional[float] = None
    ):
        """
        Initialize hierarchical clustering model.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters (if distance_threshold is None)
        linkage : str
            Linkage criterion:
            - 'ward': Minimize variance (default, best for most cases)
            - 'complete': Maximum distance
            - 'average': Average distance
            - 'single': Minimum distance
        distance_threshold : float, optional
            Distance threshold for automatic cluster detection
            If set, n_clusters is ignored
        """
        super().__init__(name="Hierarchical")

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold

        # Initialize sklearn model
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters if distance_threshold is None else None,
            linkage=linkage,
            distance_threshold=distance_threshold
        )

    def fit(self, X: np.ndarray, **kwargs) -> 'HierarchicalModel':
        """
        Fit hierarchical clustering on embeddings.

        Parameters
        ----------
        X : np.ndarray
            Embeddings (N x D)

        Returns
        -------
        self
        """
        verbose = kwargs.get('verbose', True)

        if verbose:
            print(f"\n🔧 Fitting Hierarchical clustering")
            if self.distance_threshold is None:
                print(f"   Number of clusters: {self.n_clusters}")
            else:
                print(f"   Distance threshold: {self.distance_threshold}")
            print(f"   Linkage: {self.linkage}")
            print(f"   Number of samples: {len(X)}")
            print("=" * 60)

        # Fit model
        self.model.fit(X)

        # Store results
        self.labels_ = self.model.labels_
        self.n_clusters_ = len(np.unique(self.labels_))
        self.is_fitted = True

        if verbose:
            print(f"✅ Hierarchical clustering fitted successfully")
            print(f"   Discovered {self.n_clusters_} clusters")
            print(f"   Cluster sizes: {np.bincount(self.labels_)}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new samples to nearest existing cluster.

        Note: Hierarchical clustering doesn't have a native predict method.
        We assign to the nearest cluster center.

        Parameters
        ----------
        X : np.ndarray
            New embeddings (M x D)

        Returns
        -------
        labels : np.ndarray
            Cluster assignments (M,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict")

        # For hierarchical clustering, we need the original training data
        # to compute cluster centers. This is a limitation.
        # In practice, store training data or use KNN-based assignment.

        raise NotImplementedError(
            "Hierarchical clustering doesn't support predict for new data. "
            "Use fit_predict or KNN-based assignment."
        )

    def get_dendrogram_data(self, X: np.ndarray) -> dict:
        """
        Compute linkage matrix for dendrogram plotting.

        Parameters
        ----------
        X : np.ndarray
            Embeddings (same as used in fit)

        Returns
        -------
        dict
            Linkage matrix and other dendrogram data
        """
        # Compute linkage matrix
        Z = linkage(X, method=self.linkage)

        return {
            'linkage_matrix': Z,
            'method': self.linkage,
            'n_samples': len(X)
        }

    def plot_dendrogram(
        self,
        X: np.ndarray,
        max_display: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Plot hierarchical dendrogram.

        Parameters
        ----------
        X : np.ndarray
            Embeddings (same as used in fit)
        max_display : int
            Maximum number of leaf nodes to display
        save_path : str, optional
            Path to save plot
        """
        import matplotlib.pyplot as plt

        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        print("\n🌳 Generating dendrogram...")

        # Compute linkage
        Z = linkage(X, method=self.linkage)

        # Plot
        plt.figure(figsize=(12, 6))
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=max_display,
            show_leaf_counts=True,
            leaf_font_size=10
        )

        plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage} linkage)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Dendrogram saved to {save_path}")

        plt.show()

    def save(self, filepath: str):
        """Save hierarchical model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save sklearn model
        joblib.dump(self.model, str(filepath))

        # Save metadata
        meta_path = filepath.with_suffix('.json')
        metadata = {
            'n_clusters': int(self.n_clusters_),
            'linkage': self.linkage,
            'distance_threshold': self.distance_threshold,
            'cluster_sizes': np.bincount(self.labels_).tolist()
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Hierarchical model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'HierarchicalModel':
        """Load hierarchical model."""
        filepath = Path(filepath)

        # Load metadata
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Create model
        model = cls(
            n_clusters=metadata['n_clusters'],
            linkage=metadata['linkage'],
            distance_threshold=metadata.get('distance_threshold')
        )

        # Load sklearn model
        model.model = joblib.load(str(filepath))
        model.labels_ = model.model.labels_
        model.n_clusters_ = metadata['n_clusters']
        model.is_fitted = True

        print(f"✓ Hierarchical model loaded from {filepath}")
        return model

    def print_summary(self):
        """Print model summary."""
        if not self.is_fitted:
            print("❌ Model not fitted yet")
            return

        print("=" * 60)
        print("📊 Hierarchical Clustering Summary")
        print("=" * 60)
        print(f"Number of clusters: {self.n_clusters_}")
        print(f"Linkage method: {self.linkage}")

        if self.distance_threshold is not None:
            print(f"Distance threshold: {self.distance_threshold}")

        print(f"\nCluster Sizes:")
        for i, size in enumerate(np.bincount(self.labels_)):
            print(f"  Cluster {i}: {size} samples")
        print("=" * 60)


if __name__ == '__main__':
    # Test hierarchical clustering
    print("Testing Hierarchical Clustering Model...\n")

    np.random.seed(42)

    # Create synthetic clusters
    cluster1 = np.random.randn(30, 10) + 5
    cluster2 = np.random.randn(25, 10) - 5
    cluster3 = np.random.randn(20, 10)

    X = np.vstack([cluster1, cluster2, cluster3])

    # Create and fit model
    model = HierarchicalModel(n_clusters=3, linkage='ward')
    model.fit(X, verbose=True)

    # Print summary
    print()
    model.print_summary()

    print("\n✓ Hierarchical clustering test complete")
