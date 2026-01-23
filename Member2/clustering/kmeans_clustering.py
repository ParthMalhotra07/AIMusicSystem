"""
K-Means Clustering Model
Partition-based clustering using K-means algorithm
"""

import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
import joblib
import json
from pathlib import Path
from typing import Optional

from .base import BaseClusteringModel


class KMeansModel(BaseClusteringModel):
    """K-means clustering model."""

    def __init__(
        self,
        n_clusters: int = 5,
        init: str = 'k-means++',
        n_init: int = 20,
        max_iter: int = 300,
        random_state: int = 42
    ):
        """
        Initialize K-means model.

        Parameters
        ----------
        n_clusters : int
            Number of clusters
        init : str
            Initialization method ('k-means++', 'random')
        n_init : int
            Number of initializations (best is kept)
        max_iter : int
            Maximum iterations
        random_state : int
            Random seed
        """
        super().__init__(name="KMeans")

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

        # Initialize sklearn K-means
        self.kmeans = SKLearnKMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )

        # Cluster centers
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray, **kwargs) -> 'KMeansModel':
        """
        Fit K-means on embeddings.

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
            print(f"\n🔧 Fitting K-Means clustering")
            print(f"   Number of clusters: {self.n_clusters}")
            print(f"   Initialization: {self.init}")
            print(f"   Number of samples: {len(X)}")
            print("=" * 60)

        # Fit K-means
        self.kmeans.fit(X)

        # Store results
        self.labels_ = self.kmeans.labels_
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.inertia_ = self.kmeans.inertia_
        self.n_clusters_ = self.n_clusters
        self.is_fitted = True

        if verbose:
            print(f"✅ K-Means fitted successfully")
            print(f"   Inertia (WCSS): {self.inertia_:.2f}")
            print(f"   Cluster sizes: {np.bincount(self.labels_)}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data.

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

        return self.kmeans.predict(X)

    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centroids.

        Returns
        -------
        centers : np.ndarray
            Cluster centers (n_clusters x D)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.cluster_centers_

    def get_distances_to_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Compute distance from each point to its assigned cluster center.

        Parameters
        ----------
        X : np.ndarray
            Embeddings (N x D)

        Returns
        -------
        distances : np.ndarray
            Distances (N,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        labels = self.predict(X)
        distances = np.zeros(len(X))

        for i, (x, label) in enumerate(zip(X, labels)):
            center = self.cluster_centers_[label]
            distances[i] = np.linalg.norm(x - center)

        return distances

    def save(self, filepath: str):
        """Save K-means model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save sklearn model
        joblib.dump(self.kmeans, str(filepath))

        # Save metadata
        meta_path = filepath.with_suffix('.json')
        metadata = {
            'n_clusters': self.n_clusters,
            'init': self.init,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'inertia': float(self.inertia_),
            'cluster_sizes': np.bincount(self.labels_).tolist()
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ K-Means model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'KMeansModel':
        """Load K-means model."""
        filepath = Path(filepath)

        # Load metadata
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Create model
        model = cls(
            n_clusters=metadata['n_clusters'],
            init=metadata['init'],
            n_init=metadata['n_init'],
            max_iter=metadata['max_iter'],
            random_state=metadata['random_state']
        )

        # Load sklearn model
        model.kmeans = joblib.load(str(filepath))
        model.labels_ = model.kmeans.labels_
        model.cluster_centers_ = model.kmeans.cluster_centers_
        model.inertia_ = model.kmeans.inertia_
        model.n_clusters_ = metadata['n_clusters']
        model.is_fitted = True

        print(f"✓ K-Means model loaded from {filepath}")
        return model

    def plot_elbow(
        self,
        X: np.ndarray,
        k_range: tuple = (2, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot elbow curve for finding optimal k.

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        k_range : tuple
            (min_k, max_k)
        save_path : str, optional
            Path to save plot
        """
        import matplotlib.pyplot as plt

        k_min, k_max = k_range
        k_values = range(k_min, min(k_max + 1, len(X)))
        inertias = []

        print(f"\n🔍 Computing elbow curve for k={k_min} to {k_max}...")

        for k in k_values:
            model = KMeansModel(n_clusters=k, random_state=self.random_state)
            model.fit(X, verbose=False)
            inertias.append(model.inertia_)
            print(f"  k={k}: Inertia={model.inertia_:.2f}")

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Elbow plot saved to {save_path}")

        plt.show()

    def print_summary(self):
        """Print model summary."""
        if not self.is_fitted:
            print("❌ Model not fitted yet")
            return

        print("=" * 60)
        print("📊 K-Means Model Summary")
        print("=" * 60)
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Initialization: {self.init}")
        print(f"Inertia (WCSS): {self.inertia_:.2f}")
        print(f"\nCluster Sizes:")
        for i, size in enumerate(np.bincount(self.labels_)):
            print(f"  Cluster {i}: {size} samples")
        print("=" * 60)


if __name__ == '__main__':
    # Test K-means
    print("Testing K-Means Model...\n")

    np.random.seed(42)

    # Create synthetic clusters
    cluster1 = np.random.randn(30, 10) + 5
    cluster2 = np.random.randn(25, 10) - 5
    cluster3 = np.random.randn(20, 10)

    X = np.vstack([cluster1, cluster2, cluster3])

    # Create and fit model
    model = KMeansModel(n_clusters=3, n_init=10)
    model.fit(X, verbose=True)

    # Print summary
    print()
    model.print_summary()

    # Test prediction
    new_point = np.random.randn(1, 10) + 5
    predicted_cluster = model.predict(new_point)
    print(f"\nTest prediction: {predicted_cluster}")

    print("\n✓ K-Means test complete")
