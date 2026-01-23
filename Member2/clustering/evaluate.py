"""
Clustering Evaluation Module
Computes quality metrics for clustering results
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)


class ClusteringEvaluator:
    """
    Evaluates clustering quality using multiple metrics.
    """

    @staticmethod
    def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Silhouette Coefficient.

        Measures how similar samples are to their own cluster compared to other clusters.

        Range: [-1, 1]
        - 1: Perfect clustering (samples are far from neighboring clusters)
        - 0: Overlapping clusters
        - -1: Samples assigned to wrong clusters

        Parameters
        ----------
        X : np.ndarray
            Embeddings (N x D)
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        float
            Silhouette score
        """
        # Need at least 2 clusters
        n_clusters = len(np.unique(labels[labels >= 0]))
        if n_clusters < 2 or n_clusters >= len(X):
            return -1.0

        try:
            return silhouette_score(X, labels, metric='euclidean')
        except:
            return -1.0

    @staticmethod
    def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Davies-Bouldin Index.

        Measures average similarity ratio of each cluster with its most similar cluster.

        Range: [0, ∞)
        - Lower is better (0 is perfect)
        - Measures separation and compactness

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        float
            Davies-Bouldin index
        """
        n_clusters = len(np.unique(labels[labels >= 0]))
        if n_clusters < 2:
            return float('inf')

        try:
            return davies_bouldin_score(X, labels)
        except:
            return float('inf')

    @staticmethod
    def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Calinski-Harabasz Index (Variance Ratio Criterion).

        Ratio of between-cluster dispersion to within-cluster dispersion.

        Range: [0, ∞)
        - Higher is better
        - Measures how well-separated and compact clusters are

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        float
            Calinski-Harabasz score
        """
        n_clusters = len(np.unique(labels[labels >= 0]))
        if n_clusters < 2:
            return 0.0

        try:
            return calinski_harabasz_score(X, labels)
        except:
            return 0.0

    @staticmethod
    def inertia(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute inertia (within-cluster sum of squares).

        Sum of squared distances from each point to its cluster center.

        Range: [0, ∞)
        - Lower is better (0 = perfect)
        - Used for elbow method

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        float
            Inertia
        """
        total_inertia = 0.0

        for label in np.unique(labels):
            if label == -1:  # Skip noise points
                continue

            mask = labels == label
            cluster_points = X[mask]

            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                distances = np.sum((cluster_points - centroid) ** 2, axis=1)
                total_inertia += np.sum(distances)

        return total_inertia

    @staticmethod
    def cluster_balance(labels: np.ndarray) -> float:
        """
        Compute cluster balance score.

        Measures how evenly samples are distributed across clusters.

        Range: [0, 1]
        - 1: Perfect balance (all clusters same size)
        - 0: Completely unbalanced

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments

        Returns
        -------
        float
            Balance score
        """
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)

        if len(counts) < 2:
            return 1.0

        # Coefficient of variation (inverted)
        mean_size = np.mean(counts)
        std_size = np.std(counts)

        if mean_size == 0:
            return 0.0

        cv = std_size / mean_size
        balance = 1.0 / (1.0 + cv)

        return balance

    @staticmethod
    def evaluate_all(
        X: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Parameters
        ----------
        X : np.ndarray
            Embeddings (N x D)
        labels : np.ndarray
            Cluster assignments
        verbose : bool
            Print results

        Returns
        -------
        dict
            All metrics
        """
        metrics = {}

        # Silhouette (higher is better)
        metrics['silhouette_score'] = ClusteringEvaluator.silhouette(X, labels)

        # Davies-Bouldin (lower is better)
        metrics['davies_bouldin_index'] = ClusteringEvaluator.davies_bouldin(X, labels)

        # Calinski-Harabasz (higher is better)
        metrics['calinski_harabasz_score'] = ClusteringEvaluator.calinski_harabasz(X, labels)

        # Inertia (lower is better)
        metrics['inertia'] = ClusteringEvaluator.inertia(X, labels)

        # Balance (higher is better)
        metrics['balance_score'] = ClusteringEvaluator.cluster_balance(labels)

        # Number of clusters
        metrics['n_clusters'] = len(np.unique(labels[labels >= 0]))

        # Number of noise points (for DBSCAN)
        metrics['n_noise'] = int(np.sum(labels == -1))

        if verbose:
            ClusteringEvaluator.print_metrics(metrics)

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """
        Print evaluation metrics in a readable format.

        Parameters
        ----------
        metrics : dict
            Evaluation metrics
        """
        print("=" * 60)
        print("📊 Clustering Quality Metrics")
        print("=" * 60)

        print(f"\n🔢 Basic Info:")
        print(f"  Number of clusters: {metrics['n_clusters']}")
        if metrics['n_noise'] > 0:
            print(f"  Noise points: {metrics['n_noise']}")

        print(f"\n📈 Quality Scores (higher is better):")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"    (-1 to 1, optimal > 0.5)")
        print(f"  Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")
        print(f"    (higher = better separation)")

        print(f"\n📉 Quality Scores (lower is better):")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        print(f"    (< 1 is good)")
        print(f"  Inertia (WCSS): {metrics['inertia']:.2f}")
        print(f"    (within-cluster sum of squares)")

        print(f"\n⚖️  Cluster Balance:")
        print(f"  Balance Score: {metrics['balance_score']:.4f}")
        print(f"    (1 = perfect balance, 0 = unbalanced)")

        # Overall assessment
        print(f"\n✨ Overall Assessment:")
        if metrics['silhouette_score'] > 0.5:
            assessment = "Excellent clustering"
        elif metrics['silhouette_score'] > 0.3:
            assessment = "Good clustering"
        elif metrics['silhouette_score'] > 0.1:
            assessment = "Fair clustering"
        else:
            assessment = "Poor clustering (consider different k or method)"

        print(f"  {assessment}")

        print("=" * 60)

    @staticmethod
    def compare_results(
        X: np.ndarray,
        results: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple clustering results.

        Parameters
        ----------
        X : np.ndarray
            Embeddings
        results : dict
            name -> labels mapping

        Returns
        -------
        dict
            name -> metrics mapping
        """
        all_metrics = {}

        for name, labels in results.items():
            metrics = ClusteringEvaluator.evaluate_all(X, labels, verbose=False)
            all_metrics[name] = metrics

        return all_metrics

    @staticmethod
    def print_comparison(comparison: Dict[str, Dict[str, float]]):
        """
        Print comparison table.

        Parameters
        ----------
        comparison : dict
            Comparison results from compare_results()
        """
        print("\n" + "=" * 80)
        print("🔍 Clustering Method Comparison")
        print("=" * 80)

        # Header
        methods = list(comparison.keys())
        print(f"{'Metric':<30}", end="")
        for method in methods:
            print(f"{method:>15}", end="")
        print()
        print("-" * 80)

        # Metrics to compare
        metrics_to_show = [
            ('n_clusters', 'Number of Clusters', 'int'),
            ('silhouette_score', 'Silhouette Score ↑', 'float'),
            ('davies_bouldin_index', 'Davies-Bouldin ↓', 'float'),
            ('calinski_harabasz_score', 'Calinski-Harabasz ↑', 'float'),
            ('balance_score', 'Balance Score ↑', 'float'),
        ]

        for key, label, dtype in metrics_to_show:
            print(f"{label:<30}", end="")
            for method in methods:
                value = comparison[method].get(key, 0)
                if dtype == 'int':
                    print(f"{int(value):>15}", end="")
                else:
                    print(f"{value:>15.4f}", end="")
            print()

        print("=" * 80)

        # Recommend best
        best_method = max(
            methods,
            key=lambda m: comparison[m]['silhouette_score']
        )

        print(f"\n✅ Recommended: **{best_method}**")
        print(f"   (Highest silhouette score: {comparison[best_method]['silhouette_score']:.4f})")
        print("=" * 80)


def find_optimal_k(
    X: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    method: str = 'kmeans',
    verbose: bool = True
) -> int:
    """
    Find optimal number of clusters using elbow method.

    Parameters
    ----------
    X : np.ndarray
        Embeddings
    k_range : tuple
        (min_k, max_k)
    method : str
        Clustering method ('kmeans', 'hierarchical')
    verbose : bool
        Print progress

    Returns
    -------
    int
        Optimal k
    """
    from sklearn.cluster import KMeans

    k_min, k_max = k_range
    inertias = []
    silhouettes = []
    k_values = range(k_min, min(k_max + 1, len(X)))

    if verbose:
        print(f"🔍 Finding optimal k in range {k_range}...")

    for k in k_values:
        if method == 'kmeans':
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
        else:
            from sklearn.cluster import AgglomerativeClustering
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)

        inertia = ClusteringEvaluator.inertia(X, labels)
        silhouette = ClusteringEvaluator.silhouette(X, labels)

        inertias.append(inertia)
        silhouettes.append(silhouette)

        if verbose:
            print(f"  k={k}: Silhouette={silhouette:.4f}, Inertia={inertia:.2f}")

    # Find k with best silhouette
    best_idx = np.argmax(silhouettes)
    optimal_k = list(k_values)[best_idx]

    if verbose:
        print(f"\n✅ Optimal k: {optimal_k}")

    return optimal_k


if __name__ == '__main__':
    # Test evaluator
    print("Testing Clustering Evaluator...\n")

    np.random.seed(42)

    # Create synthetic clusters
    cluster1 = np.random.randn(20, 10) + 5
    cluster2 = np.random.randn(15, 10) - 5
    cluster3 = np.random.randn(10, 10)

    X = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0]*20 + [1]*15 + [2]*10)

    # Evaluate
    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate_all(X, labels, verbose=True)

    print("\n✓ Evaluator test complete")
