"""
Base Clustering Module
Abstract base class for all clustering algorithms
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


class BaseClusteringModel(ABC):
    """
    Abstract base class for clustering models.

    All clustering models must implement:
    - fit(X): Learn cluster assignments
    - predict(X): Assign new samples to clusters
    - get_labels(): Get cluster labels
    """

    def __init__(self, name: str = "BaseCluster"):
        """
        Initialize clustering model.

        Parameters
        ----------
        name : str
            Model name for logging
        """
        self.name = name
        self.is_fitted = False
        self.labels_ = None
        self.n_clusters_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> 'BaseClusteringModel':
        """
        Fit the clustering model on data.

        Parameters
        ----------
        X : np.ndarray
            Embedding data (N x D)
        **kwargs
            Additional arguments

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new data.

        Parameters
        ----------
        X : np.ndarray
            New embedding data (M x D)

        Returns
        -------
        labels : np.ndarray
            Cluster assignments (M,)
        """
        pass

    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit and predict in one step.

        Parameters
        ----------
        X : np.ndarray
            Embedding data (N x D)

        Returns
        -------
        labels : np.ndarray
            Cluster assignments (N,)
        """
        self.fit(X, **kwargs)
        return self.labels_

    def get_labels(self) -> np.ndarray:
        """Get cluster labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.labels_

    def get_n_clusters(self) -> int:
        """Get number of clusters."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.n_clusters_

    @abstractmethod
    def save(self, filepath: str):
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseClusteringModel':
        """Load model from disk."""
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'n_clusters': self.n_clusters_
        }

    def __repr__(self) -> str:
        """String representation."""
        params = self.get_params()
        param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
        return f"{self.__class__.__name__}({param_str})"


class ClusterAnalyzer:
    """
    Analyzes clustering results and computes statistics.
    """

    @staticmethod
    def get_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
        """
        Get size of each cluster.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        dict
            Cluster ID -> size
        """
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @staticmethod
    def get_cluster_centers(X: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute centroid of each cluster.

        Parameters
        ----------
        X : np.ndarray
            Embeddings (N x D)
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        dict
            Cluster ID -> centroid
        """
        centers = {}
        for label in np.unique(labels):
            mask = labels == label
            centers[int(label)] = np.mean(X[mask], axis=0)
        return centers

    @staticmethod
    def get_intra_cluster_distances(X: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """
        Compute average intra-cluster distance (cohesion).

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        dict
            Cluster ID -> average intra-cluster distance
        """
        from scipy.spatial.distance import pdist, squareform

        distances = {}
        for label in np.unique(labels):
            mask = labels == label
            cluster_points = X[mask]

            if len(cluster_points) > 1:
                # Compute pairwise distances within cluster
                dist_matrix = squareform(pdist(cluster_points, metric='euclidean'))
                # Average of upper triangle (excluding diagonal)
                distances[int(label)] = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
            else:
                distances[int(label)] = 0.0

        return distances

    @staticmethod
    def get_inter_cluster_distances(
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[tuple, float]:
        """
        Compute distances between cluster centroids (separation).

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        dict
            (cluster_i, cluster_j) -> distance
        """
        from scipy.spatial.distance import euclidean

        centers = ClusterAnalyzer.get_cluster_centers(X, labels)
        distances = {}

        cluster_ids = sorted(centers.keys())
        for i, id1 in enumerate(cluster_ids):
            for id2 in cluster_ids[i+1:]:
                dist = euclidean(centers[id1], centers[id2])
                distances[(id1, id2)] = float(dist)

        return distances

    @staticmethod
    def get_cluster_summary(
        X: np.ndarray,
        labels: np.ndarray,
        song_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Get comprehensive cluster statistics.

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        labels : np.ndarray
            Cluster assignments
        song_names : list, optional
            Song names

        Returns
        -------
        dict
            Comprehensive statistics
        """
        n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise (-1)

        summary = {
            'n_samples': len(labels),
            'n_clusters': n_clusters,
            'cluster_sizes': ClusterAnalyzer.get_cluster_sizes(labels),
            'intra_cluster_distances': ClusterAnalyzer.get_intra_cluster_distances(X, labels),
        }

        # Add song names per cluster if provided
        if song_names is not None:
            summary['cluster_songs'] = {}
            for label in np.unique(labels):
                mask = labels == label
                summary['cluster_songs'][int(label)] = [
                    song_names[i] for i in np.where(mask)[0]
                ]

        # Compute balance (how evenly distributed)
        sizes = list(summary['cluster_sizes'].values())
        if len(sizes) > 1:
            summary['balance_score'] = 1.0 - (np.std(sizes) / np.mean(sizes))
        else:
            summary['balance_score'] = 1.0

        return summary

    @staticmethod
    def print_summary(summary: Dict, verbose: bool = True):
        """
        Print cluster summary.

        Parameters
        ----------
        summary : dict
            Cluster statistics
        verbose : bool
            Print detailed info
        """
        print("=" * 60)
        print("📊 Cluster Analysis Summary")
        print("=" * 60)
        print(f"Total samples: {summary['n_samples']}")
        print(f"Number of clusters: {summary['n_clusters']}")
        print(f"Balance score: {summary['balance_score']:.4f}")

        print(f"\nCluster Sizes:")
        for cluster_id, size in summary['cluster_sizes'].items():
            pct = 100 * size / summary['n_samples']
            print(f"  Cluster {cluster_id}: {size} songs ({pct:.1f}%)")

        if verbose and 'intra_cluster_distances' in summary:
            print(f"\nIntra-cluster Cohesion (lower = tighter):")
            for cluster_id, dist in summary['intra_cluster_distances'].items():
                print(f"  Cluster {cluster_id}: {dist:.4f}")

        if verbose and 'cluster_songs' in summary:
            print(f"\nSongs per Cluster:")
            for cluster_id, songs in summary['cluster_songs'].items():
                print(f"\n  Cluster {cluster_id}:")
                for song in songs[:5]:  # Show first 5
                    print(f"    - {song}")
                if len(songs) > 5:
                    print(f"    ... and {len(songs) - 5} more")

        print("=" * 60)


if __name__ == '__main__':
    # Test analyzer with synthetic data
    print("Testing Cluster Analyzer...\n")

    np.random.seed(42)

    # Create synthetic clusters
    cluster1 = np.random.randn(20, 10) + np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(15, 10) + np.array([0, 0, 5, 5, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(10, 10) + np.array([0, 0, 0, 0, 5, 5, 0, 0, 0, 0])

    X = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0]*20 + [1]*15 + [2]*10)
    song_names = [f"song_{i:02d}" for i in range(len(X))]

    # Analyze
    analyzer = ClusterAnalyzer()
    summary = analyzer.get_cluster_summary(X, labels, song_names)
    analyzer.print_summary(summary, verbose=True)

    print("\n✓ Analyzer test complete")