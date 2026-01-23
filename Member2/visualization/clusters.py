"""
Cluster Visualization Module
Visualizes clustering results and cluster characteristics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import seaborn as sns


class ClusterVisualizer:
    """
    Visualizes clustering results.
    """

    @staticmethod
    def plot_cluster_distribution(
        labels: np.ndarray,
        title: str = "Cluster Size Distribution",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot cluster size distribution as bar chart.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            Save path
        """
        # Count cluster sizes
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

        # Create bar plot
        plt.figure(figsize=figsize)

        bars = plt.bar(
            unique_labels,
            counts,
            color='steelblue',
            alpha=0.7,
            edgecolor='black',
            linewidth=1.5
        )

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Songs', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(unique_labels)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Cluster distribution plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_cluster_pie(
        labels: np.ndarray,
        title: str = "Cluster Distribution",
        figsize: Tuple[int, int] = (8, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot cluster distribution as pie chart.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            Save path
        """
        # Count cluster sizes
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

        # Create pie chart
        plt.figure(figsize=figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        labels_text = [f"Cluster {label}" for label in unique_labels]

        wedges, texts, autotexts = plt.pie(
            counts,
            labels=labels_text,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 11}
        )

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Pie chart saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_cluster_heatmap(
        labels: np.ndarray,
        song_names: Optional[List[str]] = None,
        title: str = "Song-Cluster Assignment Heatmap",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of song-cluster assignments.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments
        song_names : list, optional
            Song names
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            Save path
        """
        if song_names is None:
            song_names = [f"Song {i}" for i in range(len(labels))]

        # Create binary matrix: songs × clusters
        unique_labels = sorted(set(labels[labels >= 0]))
        matrix = np.zeros((len(labels), len(unique_labels)))

        for i, label in enumerate(labels):
            if label >= 0:
                cluster_idx = unique_labels.index(label)
                matrix[i, cluster_idx] = 1

        # Create heatmap
        plt.figure(figsize=figsize)

        sns.heatmap(
            matrix.T,
            cmap='YlOrRd',
            cbar_kws={'label': 'Assignment'},
            xticklabels=song_names if len(song_names) <= 50 else False,
            yticklabels=[f"Cluster {l}" for l in unique_labels],
            linewidths=0.5,
            linecolor='gray'
        )

        plt.xlabel('Songs', fontsize=12)
        plt.ylabel('Clusters', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Heatmap saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_cluster_sizes_comparison(
        results: Dict[str, np.ndarray],
        title: str = "Cluster Sizes Across Methods",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Compare cluster sizes across different clustering methods.

        Parameters
        ----------
        results : dict
            method_name -> labels mapping
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            Save path
        """
        plt.figure(figsize=figsize)

        n_methods = len(results)
        x = np.arange(n_methods)
        width = 0.15

        # Get max number of clusters
        max_clusters = max([len(np.unique(labels[labels >= 0])) for labels in results.values()])

        colors = plt.cm.tab10(np.linspace(0, 1, max_clusters))

        # Plot bars for each cluster
        for cluster_id in range(max_clusters):
            sizes = []

            for method_name in results.keys():
                labels = results[method_name]
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)

                if cluster_id < len(unique_labels):
                    sizes.append(counts[cluster_id])
                else:
                    sizes.append(0)

            plt.bar(
                x + cluster_id * width,
                sizes,
                width,
                label=f'Cluster {cluster_id}',
                color=colors[cluster_id],
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )

        plt.xlabel('Clustering Method', fontsize=12)
        plt.ylabel('Cluster Size', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(x + width * (max_clusters - 1) / 2, list(results.keys()))
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Comparison plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_songs_per_cluster(
        labels: np.ndarray,
        song_names: List[str],
        max_songs_per_cluster: int = 10,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot songs grouped by cluster.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments
        song_names : list
            Song names
        max_songs_per_cluster : int
            Max songs to show per cluster
        figsize : tuple
            Figure size
        save_path : str, optional
            Save path
        """
        unique_labels = sorted(set(labels[labels >= 0]))
        n_clusters = len(unique_labels)

        fig, axes = plt.subplots(
            n_clusters,
            1,
            figsize=figsize,
            sharex=False
        )

        if n_clusters == 1:
            axes = [axes]

        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_songs = [song_names[j] for j in np.where(mask)[0]]

            # Take first N songs
            display_songs = cluster_songs[:max_songs_per_cluster]

            # Create bar
            y_pos = np.arange(len(display_songs))
            axes[i].barh(y_pos, [1] * len(display_songs), color='steelblue', alpha=0.7)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(display_songs, fontsize=9)
            axes[i].set_xlabel('')
            axes[i].set_title(f'Cluster {label} ({len(cluster_songs)} songs)', fontsize=12, fontweight='bold')
            axes[i].set_xlim([0, 1.5])
            axes[i].set_xticks([])
            axes[i].grid(axis='x', alpha=0.3)

            if len(cluster_songs) > max_songs_per_cluster:
                axes[i].text(
                    0.5, -0.5, f'... and {len(cluster_songs) - max_songs_per_cluster} more',
                    fontsize=9, style='italic', ha='center'
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Songs-per-cluster plot saved to {save_path}")

        plt.show()

    @staticmethod
    def create_cluster_report(
        labels: np.ndarray,
        song_names: List[str],
        metrics: Optional[Dict] = None,
        save_path: str = "cluster_report.txt"
    ):
        """
        Generate text report of clustering results.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments
        song_names : list
            Song names
        metrics : dict, optional
            Clustering metrics
        save_path : str
            Path to save report
        """
        report = []
        report.append("=" * 80)
        report.append("CLUSTERING REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall stats
        n_samples = len(labels)
        n_clusters = len(set(labels[labels >= 0]))
        n_noise = int(np.sum(labels == -1))

        report.append(f"Total songs: {n_samples}")
        report.append(f"Number of clusters: {n_clusters}")
        if n_noise > 0:
            report.append(f"Noise points: {n_noise}")
        report.append("")

        # Metrics
        if metrics:
            report.append("Quality Metrics:")
            report.append("-" * 40)
            for key, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")

        # Cluster details
        report.append("Cluster Details:")
        report.append("=" * 80)

        for label in sorted(set(labels[labels >= 0])):
            mask = labels == label
            cluster_songs = [song_names[i] for i in np.where(mask)[0]]

            report.append(f"\nCluster {label} ({len(cluster_songs)} songs):")
            report.append("-" * 40)
            for song in cluster_songs:
                report.append(f"  - {song}")

        report.append("\n" + "=" * 80)

        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"✓ Cluster report saved to {save_path}")


if __name__ == '__main__':
    # Test cluster visualizer
    print("Testing Cluster Visualizer...\n")

    np.random.seed(42)

    # Create synthetic data
    labels = np.array([0]*20 + [1]*15 + [2]*10)
    song_names = [f"song_{i:02d}" for i in range(len(labels))]

    visualizer = ClusterVisualizer()

    # Test distribution plot
    visualizer.plot_cluster_distribution(labels)

    print("\n✓ Cluster visualizer test complete")
