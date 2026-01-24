"""
Comparison Visualization Module
Compare different embedding and clustering methods side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns
from sklearn.manifold import TSNE


class ComparisonVisualizer:
    """
    Visualizes comparisons between different methods.
    """

    @staticmethod
    def compare_embedding_methods(
        embeddings_dict: Dict[str, np.ndarray],
        original_features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        projection_method: str = 'tsne',
        figsize: Tuple[int, int] = (18, 5),
        save_path: Optional[str] = None
    ):
        """
        Compare multiple embedding methods side-by-side in 2D.

        Parameters
        ----------
        embeddings_dict : dict
            method_name -> embeddings mapping
            e.g., {'PCA': pca_embeddings, 'Autoencoder': ae_embeddings}
        original_features : np.ndarray
            Original high-dimensional features for comparison
        labels : np.ndarray, optional
            Cluster labels for coloring
        projection_method : str
            Method to project to 2D ('tsne', 'pca')
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
        """
        n_methods = len(embeddings_dict) + 1  # +1 for original features
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)

        if n_methods == 1:
            axes = [axes]

        print(f"🔧 Comparing {len(embeddings_dict)} embedding methods using {projection_method.upper()}...")

        # Project original features
        print("   Projecting original features...")
        if projection_method == 'tsne':
            perplexity = min(30, len(original_features) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
            original_2d = tsne.fit_transform(original_features)
        elif projection_method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            original_2d = pca.fit_transform(original_features)
        else:
            raise ValueError(f"Unknown projection method: {projection_method}")

        # Plot original features
        ComparisonVisualizer._plot_single_projection(
            axes[0],
            original_2d,
            labels,
            title=f"Original Features ({original_features.shape[1]}D)",
            show_legend=(0 == 0)
        )

        # Plot each embedding method
        for idx, (method_name, embeddings) in enumerate(embeddings_dict.items(), start=1):
            print(f"   Projecting {method_name}...")

            if projection_method == 'tsne':
                perplexity = min(30, len(embeddings) - 1)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
                coords_2d = tsne.fit_transform(embeddings)
            elif projection_method == 'pca':
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                coords_2d = pca.fit_transform(embeddings)

            ComparisonVisualizer._plot_single_projection(
                axes[idx],
                coords_2d,
                labels,
                title=f"{method_name} ({embeddings.shape[1]}D)",
                show_legend=(idx == 0)
            )

        plt.suptitle(
            f"Embedding Method Comparison (2D projection via {projection_method.upper()})",
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Embedding comparison saved to {save_path}")

        plt.show()

    @staticmethod
    def _plot_single_projection(
        ax,
        coords_2d: np.ndarray,
        labels: Optional[np.ndarray],
        title: str,
        show_legend: bool = True
    ):
        """Helper to plot a single 2D projection."""
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                label_name = f"Cluster {label}" if label >= 0 else "Noise"

                ax.scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    c=[colors[i]],
                    label=label_name,
                    alpha=0.7,
                    s=80,
                    edgecolors='black',
                    linewidths=0.5
                )

            if show_legend:
                ax.legend(fontsize=9, loc='best')
        else:
            ax.scatter(
                coords_2d[:, 0],
                coords_2d[:, 1],
                c='steelblue',
                alpha=0.7,
                s=80,
                edgecolors='black',
                linewidths=0.5
            )

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=10)
        ax.set_ylabel('Component 2', fontsize=10)
        ax.grid(alpha=0.3)

    @staticmethod
    def compare_clustering_methods(
        embeddings: np.ndarray,
        labels_dict: Dict[str, np.ndarray],
        projection_method: str = 'tsne',
        figsize: Tuple[int, int] = (18, 5),
        save_path: Optional[str] = None
    ):
        """
        Compare multiple clustering methods on the same embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings to cluster (N x D)
        labels_dict : dict
            method_name -> labels mapping
            e.g., {'K-Means': kmeans_labels, 'DBSCAN': dbscan_labels}
        projection_method : str
            Method to project to 2D
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
        """
        n_methods = len(labels_dict)
        fig, axes = plt.subplots(1, n_methods, figsize=figsize)

        if n_methods == 1:
            axes = [axes]

        print(f"🔧 Comparing {n_methods} clustering methods...")

        # Project embeddings once
        print(f"   Projecting embeddings using {projection_method.upper()}...")
        if projection_method == 'tsne':
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
            coords_2d = tsne.fit_transform(embeddings)
        elif projection_method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown projection method: {projection_method}")

        # Plot each clustering method
        for idx, (method_name, labels) in enumerate(labels_dict.items()):
            n_clusters = len(set(labels[labels >= 0]))
            n_noise = int(np.sum(labels == -1))

            title = f"{method_name}\n({n_clusters} clusters"
            if n_noise > 0:
                title += f", {n_noise} noise)"
            else:
                title += ")"

            ComparisonVisualizer._plot_single_projection(
                axes[idx],
                coords_2d,
                labels,
                title=title,
                show_legend=(idx == 0)
            )

        plt.suptitle(
            "Clustering Method Comparison",
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Clustering comparison saved to {save_path}")

        plt.show()

    @staticmethod
    def compare_features_by_cluster(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ):
        """
        Compare feature distributions across clusters.

        Parameters
        ----------
        features : np.ndarray
            Original features (N x D)
        labels : np.ndarray
            Cluster assignments
        feature_names : list, optional
            Names of features
        top_k : int
            Number of top features to show
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
        """
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(features.shape[1])]

        unique_labels = sorted(set(labels[labels >= 0]))
        n_clusters = len(unique_labels)

        print(f"🔧 Comparing features across {n_clusters} clusters...")

        # Calculate mean feature values per cluster
        cluster_means = []
        for label in unique_labels:
            mask = labels == label
            cluster_mean = features[mask].mean(axis=0)
            cluster_means.append(cluster_mean)

        cluster_means = np.array(cluster_means)  # (n_clusters x n_features)

        # Find features with highest variance across clusters
        feature_variance = cluster_means.var(axis=0)
        top_features_idx = np.argsort(feature_variance)[-top_k:][::-1]

        # Plot heatmap
        plt.figure(figsize=figsize)

        data_to_plot = cluster_means[:, top_features_idx].T
        selected_names = [feature_names[i] for i in top_features_idx]

        sns.heatmap(
            data_to_plot,
            cmap='RdYlGn',
            center=0,
            xticklabels=[f"Cluster {l}" for l in unique_labels],
            yticklabels=selected_names,
            cbar_kws={'label': 'Mean Value (normalized)'},
            linewidths=0.5,
            linecolor='gray',
            annot=False
        )

        plt.title(
            f"Top {top_k} Discriminative Features Across Clusters",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Feature comparison saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_feature_importance(
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        method: str = 'variance',
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance for clustering.

        Parameters
        ----------
        features : np.ndarray
            Original features (N x D)
        labels : np.ndarray
            Cluster assignments
        feature_names : list, optional
            Names of features
        top_k : int
            Number of top features to show
        method : str
            Importance method: 'variance', 'anova'
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
        """
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(features.shape[1])]

        print(f"🔧 Computing feature importance using {method}...")

        if method == 'variance':
            # Calculate variance of cluster means
            unique_labels = sorted(set(labels[labels >= 0]))
            cluster_means = []

            for label in unique_labels:
                mask = labels == label
                cluster_mean = features[mask].mean(axis=0)
                cluster_means.append(cluster_mean)

            cluster_means = np.array(cluster_means)
            importance = cluster_means.var(axis=0)
            importance_name = "Variance Across Clusters"

        elif method == 'anova':
            # Calculate ANOVA F-statistic
            from sklearn.feature_selection import f_classif

            # Filter out noise points
            mask = labels >= 0
            importance, _ = f_classif(features[mask], labels[mask])
            importance_name = "ANOVA F-Statistic"

        else:
            raise ValueError(f"Unknown method: {method}")

        # Get top features
        top_idx = np.argsort(importance)[-top_k:]
        top_importance = importance[top_idx]
        top_names = [feature_names[i] for i in top_idx]

        # Plot
        plt.figure(figsize=figsize)

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importance)))

        bars = plt.barh(
            range(len(top_importance)),
            top_importance,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        plt.yticks(range(len(top_names)), top_names, fontsize=10)
        plt.xlabel(importance_name, fontsize=12)
        plt.title(
            f"Top {top_k} Most Important Features for Clustering",
            fontsize=14,
            fontweight='bold'
        )
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_metrics_comparison(
        metrics_dict: Dict[str, Dict[str, float]],
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        Compare metrics across different methods.

        Parameters
        ----------
        metrics_dict : dict
            method_name -> metrics_dict mapping
            e.g., {'K-Means': {'silhouette': 0.5, 'davies_bouldin': 1.2}, ...}
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
        """
        print("🔧 Comparing clustering metrics...")

        # Extract metrics
        methods = list(metrics_dict.keys())
        metric_names = list(next(iter(metrics_dict.values())).keys())

        # Create subplots
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        # Plot each metric
        for idx, metric_name in enumerate(metric_names):
            values = [metrics_dict[method].get(metric_name, 0) for method in methods]

            # Determine if higher is better
            higher_is_better = metric_name.lower() not in ['davies_bouldin', 'inertia']

            if higher_is_better:
                colors = ['green' if v == max(values) else 'steelblue' for v in values]
            else:
                colors = ['green' if v == min(values) else 'steelblue' for v in values]

            bars = axes[idx].bar(
                methods,
                values,
                color=colors,
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5
            )

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

            axes[idx].set_title(
                metric_name.replace('_', ' ').title(),
                fontsize=12,
                fontweight='bold'
            )
            axes[idx].set_ylabel('Score', fontsize=11)
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)

        plt.suptitle(
            "Clustering Metrics Comparison",
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Metrics comparison saved to {save_path}")

        plt.show()

    @staticmethod
    def create_comparison_report(
        embeddings_dict: Dict[str, Dict],
        clustering_dict: Dict[str, Dict],
        save_path: str = "comparison_report.txt"
    ):
        """
        Generate comprehensive comparison report.

        Parameters
        ----------
        embeddings_dict : dict
            method_name -> {'embeddings': array, 'metrics': dict}
        clustering_dict : dict
            method_name -> {'labels': array, 'metrics': dict}
        save_path : str
            Path to save report
        """
        report = []
        report.append("=" * 80)
        report.append("METHOD COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")

        # Embedding methods
        if embeddings_dict:
            report.append("EMBEDDING METHODS")
            report.append("-" * 80)
            report.append("")

            for method_name, data in embeddings_dict.items():
                embeddings = data['embeddings']
                metrics = data.get('metrics', {})

                report.append(f"{method_name}:")
                report.append(f"  Embedding dimensions: {embeddings.shape[1]}")
                report.append(f"  Number of samples: {embeddings.shape[0]}")

                if metrics:
                    report.append("  Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            report.append(f"    {key}: {value:.4f}")
                        else:
                            report.append(f"    {key}: {value}")

                report.append("")

        # Clustering methods
        if clustering_dict:
            report.append("CLUSTERING METHODS")
            report.append("-" * 80)
            report.append("")

            for method_name, data in clustering_dict.items():
                labels = data['labels']
                metrics = data.get('metrics', {})

                n_clusters = len(set(labels[labels >= 0]))
                n_noise = int(np.sum(labels == -1))

                report.append(f"{method_name}:")
                report.append(f"  Number of clusters: {n_clusters}")

                if n_noise > 0:
                    report.append(f"  Noise points: {n_noise}")

                if n_clusters > 0:
                    cluster_sizes = np.bincount(labels[labels >= 0])
                    report.append(f"  Cluster sizes: {cluster_sizes.tolist()}")

                if metrics:
                    report.append("  Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            report.append(f"    {key}: {value:.4f}")
                        else:
                            report.append(f"    {key}: {value}")

                report.append("")

        report.append("=" * 80)

        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"✓ Comparison report saved to {save_path}")


if __name__ == '__main__':
    # Test comparison visualizer
    print("Testing Comparison Visualizer...\n")

    np.random.seed(42)

    # Create synthetic data with 3 clusters
    cluster1 = np.random.randn(20, 50) + 5
    cluster2 = np.random.randn(15, 50) - 5
    cluster3 = np.random.randn(10, 50)

    original_features = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0]*20 + [1]*15 + [2]*10)

    # Create synthetic embeddings (PCA vs Random projection)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=14)
    pca_embeddings = pca.fit_transform(original_features)

    # Random projection as "autoencoder"
    ae_embeddings = np.random.randn(len(original_features), 14)

    embeddings_dict = {
        'PCA': pca_embeddings,
        'Autoencoder': ae_embeddings
    }

    # Test embedding comparison
    visualizer = ComparisonVisualizer()
    visualizer.compare_embedding_methods(
        embeddings_dict,
        original_features,
        labels=true_labels,
        projection_method='pca'
    )

    print("\n✓ Comparison visualizer test complete")
