"""
Member 2: Embedding & Clustering Engineer - Configuration
Centralized configuration for dimensionality reduction, clustering, and visualization
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Expected input from Member 1
    expected_feature_dim: int = 170  # Dimension of feature vectors from Member 1

    # Member 1 output paths
    member1_output_dir: str = "../Member1/output"
    default_feature_file: Optional[str] = None  # If None, will search for latest

    # Feature categories (from Member 1's schema)
    feature_categories: Tuple[str, ...] = (
        'time_domain', 'timbral', 'rhythmic', 'harmonic', 'groove', 'structural'
    )

    # Data validation
    allow_missing_values: bool = False
    max_missing_ratio: float = 0.05  # Max 5% missing values allowed

    # Preprocessing
    apply_scaling: bool = True  # Re-scale even if Member 1 scaled
    scaling_method: str = 'zscore'  # 'zscore' or 'minmax'


@dataclass
class EmbeddingConfig:
    """Configuration for embedding/dimensionality reduction."""

    # Embedding dimensions
    embedding_dim: int = 32  # Target embedding dimension (170 → 32)

    # Model type: 'autoencoder', 'pca', 'umap', or 'tsne'
    model_type: str = 'autoencoder'

    # Autoencoder architecture
    encoder_layers: Tuple[int, ...] = (128, 64, 32)  # 170 → 128 → 64 → 32
    decoder_layers: Tuple[int, ...] = (64, 128, 170)  # 32 → 64 → 128 → 170
    activation: str = 'relu'
    dropout_rate: float = 0.2

    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2

    # PCA parameters (if model_type='pca')
    pca_n_components: int = 32
    pca_whiten: bool = True

    # UMAP parameters (if model_type='umap')
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = 'cosine'

    # t-SNE parameters (if model_type='tsne')
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""

    # Clustering method: 'kmeans', 'agglomerative', 'dbscan', 'gmm'
    method: str = 'kmeans'

    # Number of clusters (for kmeans, agglomerative, gmm)
    n_clusters: int = 8
    n_clusters_range: Tuple[int, int] = (3, 15)  # For automatic selection

    # K-Means parameters
    kmeans_n_init: int = 20
    kmeans_max_iter: int = 300

    # DBSCAN parameters
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5

    # Agglomerative parameters
    agglomerative_linkage: str = 'ward'  # 'ward', 'complete', 'average'

    # Automatic cluster selection
    auto_select_clusters: bool = False  # Use silhouette score to find optimal k
    selection_metric: str = 'silhouette'  # 'silhouette', 'davies_bouldin', 'calinski_harabasz'


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    # 2D projection method: 'tsne', 'umap', 'pca'
    projection_method: str = 'tsne'

    # t-SNE for visualization
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000

    # UMAP for visualization
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Plot settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 150
    colormap: str = 'tab10'
    point_size: int = 50
    alpha: float = 0.7

    # Output
    save_plots: bool = True
    plot_format: str = 'png'  # 'png', 'pdf', 'svg'


@dataclass
class OutputConfig:
    """Configuration for output and export."""

    # Output directories
    output_dir: str = "./output"
    models_dir: str = "./models"
    plots_dir: str = "./plots"

    # Embedding output
    embeddings_filename: str = "embeddings.npy"
    cluster_labels_filename: str = "cluster_labels.npy"

    # Model checkpoints
    save_model: bool = True
    model_filename: str = "embedding_model.pth"

    # Metadata export
    export_cluster_info: bool = True
    cluster_info_filename: str = "cluster_info.json"

    # Create subdirectories
    create_timestamp_dirs: bool = True


@dataclass
class Member2Config:
    """Master configuration for Member 2."""

    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Project paths
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    def get_member1_path(self) -> str:
        """Get Member 1 directory path."""
        return os.path.normpath(os.path.join(self.project_root, '..', 'Member1'))

    def get_output_dir(self) -> str:
        """Get output directory path."""
        return os.path.join(self.project_root, self.output.output_dir)

    def get_models_dir(self) -> str:
        """Get models directory path."""
        return os.path.join(self.project_root, self.output.models_dir)

    def get_plots_dir(self) -> str:
        """Get plots directory path."""
        return os.path.join(self.project_root, self.output.plots_dir)

    def create_directories(self):
        """Create all necessary output directories."""
        os.makedirs(self.get_output_dir(), exist_ok=True)
        os.makedirs(self.get_models_dir(), exist_ok=True)
        os.makedirs(self.get_plots_dir(), exist_ok=True)


# Global default configuration
DEFAULT_CONFIG = Member2Config()


if __name__ == '__main__':
    # Test configuration
    config = Member2Config()
    print("=" * 60)
    print("Member 2 Configuration")
    print("=" * 60)
    print(f"Expected Feature Dimension: {config.data.expected_feature_dim}")
    print(f"Embedding Dimension: {config.embedding.embedding_dim}")
    print(f"Embedding Model: {config.embedding.model_type}")
    print(f"Clustering Method: {config.clustering.method}")
    print(f"Number of Clusters: {config.clustering.n_clusters}")
    print(f"Visualization Method: {config.visualization.projection_method}")
    print(f"Output Directory: {config.get_output_dir()}")