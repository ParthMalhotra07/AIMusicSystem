"""
Embedding Visualization Module
Projects high-dimensional embeddings to 2D/3D for visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, List, Tuple
from pathlib import Path

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EmbeddingVisualizer:
    """
    Visualizes high-dimensional embeddings in 2D/3D.
    """

    @staticmethod
    def project_to_2d(
        embeddings: np.ndarray,
        method: str = 'tsne',
        **kwargs
    ) -> np.ndarray:
        """
        Project embeddings to 2D.

        Parameters
        ----------
        embeddings : np.ndarray
            High-dimensional embeddings (N x D)
        method : str
            Projection method: 'tsne', 'umap', 'pca'
        **kwargs
            Additional arguments for the projection method

        Returns
        -------
        coords_2d : np.ndarray
            2D coordinates (N x 2)
        """
        if method == 'tsne':
            perplexity = kwargs.get('perplexity', min(30, len(embeddings) - 1))
            random_state = kwargs.get('random_state', 42)

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
                n_iter=1000
            )
            coords_2d = tsne.fit_transform(embeddings)

        elif method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not installed. Run: pip install umap-learn")

            n_neighbors = kwargs.get('n_neighbors', min(15, len(embeddings) - 1))
            min_dist = kwargs.get('min_dist', 0.1)
            random_state = kwargs.get('random_state', 42)

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state
            )
            coords_2d = reducer.fit_transform(embeddings)

        elif method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(embeddings)

        else:
            raise ValueError(f"Unknown method: {method}")

        return coords_2d

    @staticmethod
    def plot_embeddings_2d(
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        song_names: Optional[List[str]] = None,
        title: str = "Music Embedding Space",
        method: str = 'tsne',
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show_labels: bool = False,
        alpha: float = 0.7,
        s: int = 100
    ):
        """
        Plot embeddings in 2D with optional cluster coloring.

        Parameters
        ----------
        embeddings : np.ndarray
            High-dimensional embeddings (N x D)
        labels : np.ndarray, optional
            Cluster labels for coloring
        song_names : list, optional
            Song names for annotation
        title : str
            Plot title
        method : str
            Projection method ('tsne', 'umap', 'pca')
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save plot
        show_labels : bool
            Annotate points with song names
        alpha : float
            Point transparency
        s : int
            Point size
        """
        # Project to 2D
        print(f"🔧 Projecting embeddings to 2D using {method.upper()}...")
        coords_2d = EmbeddingVisualizer.project_to_2d(embeddings, method=method)

        # Create plot
        plt.figure(figsize=figsize)

        if labels is not None:
            # Color by cluster
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                label_name = f"Cluster {label}" if label >= 0 else "Noise"

                plt.scatter(
                    coords_2d[mask, 0],
                    coords_2d[mask, 1],
                    c=[colors[i]],
                    label=label_name,
                    alpha=alpha,
                    s=s,
                    edgecolors='black',
                    linewidths=0.5
                )
        else:
            # Single color
            plt.scatter(
                coords_2d[:, 0],
                coords_2d[:, 1],
                alpha=alpha,
                s=s,
                c='steelblue',
                edgecolors='black',
                linewidths=0.5
            )

        # Annotate with song names
        if show_labels and song_names:
            for i, name in enumerate(song_names):
                plt.annotate(
                    name,
                    (coords_2d[i, 0], coords_2d[i, 1]),
                    fontsize=8,
                    alpha=0.7
                )

        plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')

        if labels is not None:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_embeddings_3d(
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        song_names: Optional[List[str]] = None,
        title: str = "Music Embedding Space (3D)",
        method: str = 'tsne',
        save_path: Optional[str] = None
    ):
        """
        Plot embeddings in 3D (interactive).

        Parameters
        ----------
        embeddings : np.ndarray
            High-dimensional embeddings
        labels : np.ndarray, optional
            Cluster labels
        song_names : list, optional
            Song names
        title : str
            Plot title
        method : str
            Projection method
        save_path : str, optional
            Save path
        """
        from mpl_toolkits.mplot3d import Axes3D

        # Project to 3D
        print(f"🔧 Projecting embeddings to 3D using {method.upper()}...")

        if method == 'tsne':
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
            coords_3d = tsne.fit_transform(embeddings)

        elif method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not installed")

            n_neighbors = min(15, len(embeddings) - 1)
            reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
            coords_3d = reducer.fit_transform(embeddings)

        elif method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            coords_3d = pca.fit_transform(embeddings)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Create 3D plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                label_name = f"Cluster {label}" if label >= 0 else "Noise"

                ax.scatter(
                    coords_3d[mask, 0],
                    coords_3d[mask, 1],
                    coords_3d[mask, 2],
                    c=[colors[i]],
                    label=label_name,
                    s=50,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5
                )
        else:
            ax.scatter(
                coords_3d[:, 0],
                coords_3d[:, 1],
                coords_3d[:, 2],
                c='steelblue',
                s=50,
                alpha=0.7
            )

        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=11)
        ax.set_zlabel(f'{method.upper()} Dimension 3', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')

        if labels is not None:
            ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 3D plot saved to {save_path}")

        plt.show()

    @staticmethod
    def create_interactive_plot(
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        song_names: Optional[List[str]] = None,
        title: str = "Interactive Music Space",
        save_path: str = "embedding_interactive.html"
    ):
        """
        Create interactive plot using Plotly (HTML output).

        Parameters
        ----------
        embeddings : np.ndarray
            High-dimensional embeddings
        labels : np.ndarray, optional
            Cluster labels
        song_names : list, optional
            Song names for hover text
        title : str
            Plot title
        save_path : str
            Path to save HTML file
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("⚠ Plotly not installed. Install with: pip install plotly")
            return

        # Project to 2D
        coords_2d = EmbeddingVisualizer.project_to_2d(embeddings, method='tsne')

        # Prepare data
        if song_names is None:
            song_names = [f"Song {i}" for i in range(len(embeddings))]

        if labels is None:
            labels = np.zeros(len(embeddings))

        # Create figure
        fig = go.Figure()

        for label in np.unique(labels):
            mask = labels == label
            label_name = f"Cluster {label}" if label >= 0 else "Noise"

            fig.add_trace(go.Scatter(
                x=coords_2d[mask, 0],
                y=coords_2d[mask, 1],
                mode='markers',
                name=label_name,
                text=[song_names[i] for i in np.where(mask)[0]],
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white'))
            ))

        fig.update_layout(
            title=title,
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            hovermode='closest',
            width=1000,
            height=700,
            template='plotly_white'
        )

        # Save
        fig.write_html(save_path)
        print(f"✓ Interactive plot saved to {save_path}")
        print(f"  Open in browser to explore!")


if __name__ == '__main__':
    # Test visualizer
    print("Testing Embedding Visualizer...\n")

    np.random.seed(42)

    # Create synthetic embeddings with 3 clusters
    cluster1 = np.random.randn(20, 14) + 5
    cluster2 = np.random.randn(15, 14) - 5
    cluster3 = np.random.randn(10, 14)

    embeddings = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0]*20 + [1]*15 + [2]*10)
    song_names = [f"song_{i:02d}" for i in range(len(embeddings))]

    # Test 2D visualization
    visualizer = EmbeddingVisualizer()
    visualizer.plot_embeddings_2d(
        embeddings,
        labels=labels,
        song_names=song_names,
        title="Test Embedding Space",
        method='tsne'
    )

    print("\n✓ Visualizer test complete")
