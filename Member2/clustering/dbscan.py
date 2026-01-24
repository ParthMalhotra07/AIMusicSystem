"""
DBSCAN Clustering Model
Density-Based Spatial Clustering of Applications with Noise
"""

import numpy as np
from sklearn.cluster import DBSCAN as SKLearnDBSCAN
import joblib
import json
from pathlib import Path
from typing import Optional

from .base import BaseClusteringModel


class DBSCANModel(BaseClusteringModel):
    """DBSCAN density-based clustering model."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ):
        """
        Initialize DBSCAN model.

        Parameters
        ----------
        eps : float
            Maximum distance between two samples to be considered neighbors
            - Smaller: More clusters, tighter
            - Larger: Fewer clusters, more inclusive
        min_samples : int
            Minimum number of samples in a neighborhood to form a core point
            - Smaller: More sensitive, more clusters
            - Larger: Less sensitive, fewer clusters
        metric : str
            Distance metric ('euclidean', 'cosine', 'manhattan', etc.)
        """
        super().__init__(name="DBSCAN")

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        # Initialize sklearn DBSCAN
        self.dbscan = SKLearnDBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )

        # Core sample indices
        self.core_sample_indices_ = None

    def fit(self, X: np.ndarray, **kwargs) -> 'DBSCANModel':
        """
        Fit DBSCAN on embeddings.

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
            print(f"\n🔧 Fitting DBSCAN clustering")
            print(f"   Eps (neighborhood radius): {self.eps}")
            print(f"   Min samples: {self.min_samples}")
            print(f"   Metric: {self.metric}")
            print(f"   Number of samples: {len(X)}")
            print("=" * 60)

        # Fit DBSCAN
        self.dbscan.fit(X)

        # Store results
        self.labels_ = self.dbscan.labels_
        self.core_sample_indices_ = self.dbscan.core_sample_indices_

        # Count clusters (excluding noise points labeled as -1)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

        self.is_fitted = True

        if verbose:
            n_noise = list(self.labels_).count(-1)
            print(f"✅ DBSCAN fitted successfully")
            print(f"   Discovered {self.n_clusters_} clusters")
            print(f"   Noise points: {n_noise}")

            if self.n_clusters_ > 0:
                cluster_sizes = np.bincount(self.labels_[self.labels_ >= 0])
                print(f"   Cluster sizes: {cluster_sizes}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data.

        Note: DBSCAN doesn't have native predict. We assign new points
        to the nearest core point's cluster, or mark as noise.

        Parameters
        ----------
        X : np.ndarray
            New embeddings (M x D)

        Returns
        -------
        labels : np.ndarray
            Cluster assignments (M,), -1 for noise
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict")

        # This is a simplified prediction
        # In practice, you'd need access to the original training data
        raise NotImplementedError(
            "DBSCAN doesn't support predict for new data by default. "
            "Use fit_predict or implement custom nearest-neighbor assignment."
        )

    def get_core_samples(self) -> np.ndarray:
        """
        Get indices of core samples.

        Core samples are points with at least min_samples neighbors
        within eps distance.

        Returns
        -------
        indices : np.ndarray
            Indices of core samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.core_sample_indices_

    def get_noise_samples(self) -> np.ndarray:
        """
        Get indices of noise samples.

        Noise samples are points that don't belong to any cluster.

        Returns
        -------
        indices : np.ndarray
            Indices of noise samples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return np.where(self.labels_ == -1)[0]

    def save(self, filepath: str):
        """Save DBSCAN model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save sklearn model
        joblib.dump(self.dbscan, str(filepath))

        # Save metadata
        n_noise = int(np.sum(self.labels_ == -1))
        meta_path = filepath.with_suffix('.json')
        metadata = {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'n_clusters': int(self.n_clusters_),
            'n_noise': n_noise,
        }

        if self.n_clusters_ > 0:
            metadata['cluster_sizes'] = np.bincount(self.labels_[self.labels_ >= 0]).tolist()

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ DBSCAN model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DBSCANModel':
        """Load DBSCAN model."""
        filepath = Path(filepath)

        # Load metadata
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Create model
        model = cls(
            eps=metadata['eps'],
            min_samples=metadata['min_samples'],
            metric=metadata['metric']
        )

        # Load sklearn model
        model.dbscan = joblib.load(str(filepath))
        model.labels_ = model.dbscan.labels_
        model.core_sample_indices_ = model.dbscan.core_sample_indices_
        model.n_clusters_ = metadata['n_clusters']
        model.is_fitted = True

        print(f"✓ DBSCAN model loaded from {filepath}")
        return model

    def print_summary(self):
        """Print model summary."""
        if not self.is_fitted:
            print("❌ Model not fitted yet")
            return

        n_noise = int(np.sum(self.labels_ == -1))

        print("=" * 60)
        print("📊 DBSCAN Clustering Summary")
        print("=" * 60)
        print(f"Eps (neighborhood radius): {self.eps}")
        print(f"Min samples: {self.min_samples}")
        print(f"Metric: {self.metric}")
        print(f"\nDiscovered {self.n_clusters_} clusters")
        print(f"Noise points: {n_noise}")

        if self.n_clusters_ > 0:
            print(f"\nCluster Sizes:")
            for i, size in enumerate(np.bincount(self.labels_[self.labels_ >= 0])):
                print(f"  Cluster {i}: {size} samples")

        print(f"\nCore samples: {len(self.core_sample_indices_)}")
        print("=" * 60)

    @staticmethod
    def estimate_eps(X: np.ndarray, k: int = 5) -> float:
        """
        Estimate good eps value using k-distance plot.

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        k : int
            k-nearest neighbors to consider

        Returns
        -------
        float
            Suggested eps value
        """
        from sklearn.neighbors import NearestNeighbors

        print(f"\n🔍 Estimating eps using {k}-nearest neighbors...")

        # Compute k-nearest neighbor distances
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(X)
        distances, _ = nbrs.kneighbors(X)

        # Take the distance to kth neighbor
        k_distances = distances[:, -1]
        k_distances = np.sort(k_distances)

        # Suggest eps as the "elbow" point (approximation)
        # Use 90th percentile as a heuristic
        suggested_eps = np.percentile(k_distances, 90)

        print(f"✓ Suggested eps: {suggested_eps:.4f}")
        print(f"  (Based on 90th percentile of {k}-distances)")

        return suggested_eps


if __name__ == '__main__':
    # Test DBSCAN
    print("Testing DBSCAN Model...\n")

    np.random.seed(42)

    # Create synthetic clusters with noise
    cluster1 = np.random.randn(30, 10) + 5
    cluster2 = np.random.randn(25, 10) - 5
    noise = np.random.randn(5, 10) * 10  # Scattered noise

    X = np.vstack([cluster1, cluster2, noise])

    # Estimate eps
    eps = DBSCANModel.estimate_eps(X, k=5)

    # Create and fit model
    model = DBSCANModel(eps=eps, min_samples=5)
    model.fit(X, verbose=True)

    # Print summary
    print()
    model.print_summary()

    print("\n✓ DBSCAN test complete")
