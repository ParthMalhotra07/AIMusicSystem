"""
Comprehensive Visualization Test Script
Tests all visualization modules with real pipeline data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
import argparse

from visualization import (
    EmbeddingVisualizer,
    ClusterVisualizer,
    ComparisonVisualizer,
    InteractiveDashboard
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test visualization modules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test all visualizations
    python test_visualization.py

    # Use real data
    python test_visualization.py --embeddings output/embeddings.npy --labels output/labels.npy

    # Generate dashboard only
    python test_visualization.py --dashboard_only
        """
    )

    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to embeddings .npy file'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Path to cluster labels .npy file'
    )
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='Path to original features .npy file'
    )
    parser.add_argument(
        '--dashboard_only',
        action='store_true',
        help='Only generate interactive dashboard'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output/visualization',
        help='Output directory for plots'
    )

    return parser.parse_args()


def load_or_generate_data(args):
    """Load real data or generate synthetic data."""
    if args.embeddings and Path(args.embeddings).exists():
        print("📂 Loading real data...")
        embeddings = np.load(args.embeddings)
        labels = np.load(args.labels) if args.labels else None
        features = np.load(args.features) if args.features else None

        song_names = [f"song_{i:02d}" for i in range(len(embeddings))]

        print(f"✓ Loaded {len(embeddings)} songs")
        print(f"  Embedding dimensions: {embeddings.shape[1]}")

        if labels is not None:
            n_clusters = len(set(labels[labels >= 0]))
            print(f"  Number of clusters: {n_clusters}")

        return embeddings, labels, features, song_names

    else:
        print("📦 Generating synthetic test data...")
        np.random.seed(42)

        # Create 3 clusters with different characteristics
        cluster1 = np.random.randn(20, 14) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(15, 14) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(10, 14) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        embeddings = np.vstack([cluster1, cluster2, cluster3])
        labels = np.array([0]*20 + [1]*15 + [2]*10)
        song_names = [f"song_{i:02d}" for i in range(len(embeddings))]

        # Generate high-dimensional features
        features = np.random.randn(len(embeddings), 50)
        for i in range(len(embeddings)):
            if labels[i] == 0:
                features[i, :10] += 3
            elif labels[i] == 1:
                features[i, 10:20] += 3
            else:
                features[i, 20:30] += 3

        print(f"✓ Generated {len(embeddings)} synthetic songs")
        print(f"  3 clusters with distinct characteristics")

        return embeddings, labels, features, song_names


def test_embedding_visualizer(embeddings, labels, song_names, output_dir):
    """Test embedding visualization."""
    print("\n" + "=" * 80)
    print("1️⃣  Testing Embedding Visualizer")
    print("=" * 80)

    visualizer = EmbeddingVisualizer()

    # Test 2D visualization with t-SNE
    print("\n📊 Creating 2D t-SNE visualization...")
    visualizer.plot_embeddings_2d(
        embeddings=embeddings,
        labels=labels,
        song_names=song_names,
        title="Music Embedding Space (t-SNE)",
        method='tsne',
        save_path=str(output_dir / "embeddings_tsne_2d.png")
    )

    # Test 2D visualization with PCA
    print("\n📊 Creating 2D PCA visualization...")
    visualizer.plot_embeddings_2d(
        embeddings=embeddings,
        labels=labels,
        song_names=song_names,
        title="Music Embedding Space (PCA)",
        method='pca',
        save_path=str(output_dir / "embeddings_pca_2d.png")
    )

    # Test interactive plot
    print("\n📊 Creating interactive HTML plot...")
    visualizer.create_interactive_plot(
        embeddings=embeddings,
        labels=labels,
        song_names=song_names,
        title="Interactive Music Space",
        save_path=str(output_dir / "embeddings_interactive.html")
    )

    print("\n✅ Embedding visualizer tests complete")


def test_cluster_visualizer(labels, song_names, output_dir):
    """Test cluster visualization."""
    print("\n" + "=" * 80)
    print("2️⃣  Testing Cluster Visualizer")
    print("=" * 80)

    visualizer = ClusterVisualizer()

    # Test distribution plot
    print("\n📊 Creating cluster distribution plot...")
    visualizer.plot_cluster_distribution(
        labels=labels,
        title="Cluster Size Distribution",
        save_path=str(output_dir / "cluster_distribution.png")
    )

    # Test pie chart
    print("\n📊 Creating cluster pie chart...")
    visualizer.plot_cluster_pie(
        labels=labels,
        title="Cluster Distribution",
        save_path=str(output_dir / "cluster_pie.png")
    )

    # Test songs per cluster
    print("\n📊 Creating songs-per-cluster plot...")
    visualizer.plot_songs_per_cluster(
        labels=labels,
        song_names=song_names,
        max_songs_per_cluster=10,
        save_path=str(output_dir / "songs_per_cluster.png")
    )

    # Test cluster report
    print("\n📄 Creating cluster report...")
    metrics = {
        'silhouette_score': 0.6523,
        'davies_bouldin_index': 0.8234,
        'n_clusters': len(set(labels[labels >= 0]))
    }
    visualizer.create_cluster_report(
        labels=labels,
        song_names=song_names,
        metrics=metrics,
        save_path=str(output_dir / "cluster_report.txt")
    )

    print("\n✅ Cluster visualizer tests complete")


def test_comparison_visualizer(embeddings, labels, features, output_dir):
    """Test comparison visualization."""
    print("\n" + "=" * 80)
    print("3️⃣  Testing Comparison Visualizer")
    print("=" * 80)

    visualizer = ComparisonVisualizer()

    # Create alternative embeddings for comparison
    print("\n🔧 Creating alternative embeddings...")
    from sklearn.decomposition import PCA

    pca_dim = min(14, embeddings.shape[0] - 1)
    pca = PCA(n_components=pca_dim)
    pca_embeddings = pca.fit_transform(embeddings)

    # Random projection as "autoencoder"
    ae_embeddings = embeddings + np.random.randn(*embeddings.shape) * 0.5

    embeddings_dict = {
        'Original': embeddings,
        'PCA': pca_embeddings,
        'Perturbed': ae_embeddings
    }

    # Test embedding method comparison
    print("\n📊 Comparing embedding methods...")
    visualizer.compare_embedding_methods(
        embeddings_dict=embeddings_dict,
        original_features=features,
        labels=labels,
        projection_method='pca',
        save_path=str(output_dir / "embedding_comparison.png")
    )

    # Create alternative clustering for comparison
    print("\n🔧 Creating alternative clusterings...")
    from sklearn.cluster import KMeans, DBSCAN

    kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(embeddings)
    dbscan_labels = DBSCAN(eps=2.0, min_samples=3).fit_predict(embeddings)

    labels_dict = {
        'Original': labels,
        'K-Means': kmeans_labels,
        'DBSCAN': dbscan_labels
    }

    # Test clustering method comparison
    print("\n📊 Comparing clustering methods...")
    visualizer.compare_clustering_methods(
        embeddings=embeddings,
        labels_dict=labels_dict,
        projection_method='pca',
        save_path=str(output_dir / "clustering_comparison.png")
    )

    # Test feature importance
    print("\n📊 Plotting feature importance...")
    feature_names = [f"Feature_{i:03d}" for i in range(features.shape[1])]
    visualizer.plot_feature_importance(
        features=features,
        labels=labels,
        feature_names=feature_names,
        top_k=20,
        method='variance',
        save_path=str(output_dir / "feature_importance.png")
    )

    # Test metrics comparison
    print("\n📊 Comparing metrics...")
    metrics_dict = {
        'K-Means': {'silhouette_score': 0.6523, 'davies_bouldin_index': 0.8234},
        'DBSCAN': {'silhouette_score': 0.7123, 'davies_bouldin_index': 0.7456},
        'Hierarchical': {'silhouette_score': 0.5987, 'davies_bouldin_index': 0.9012}
    }
    visualizer.plot_metrics_comparison(
        metrics_dict=metrics_dict,
        save_path=str(output_dir / "metrics_comparison.png")
    )

    print("\n✅ Comparison visualizer tests complete")


def test_dashboard(embeddings, labels, song_names, features, output_dir):
    """Test interactive dashboard generation."""
    print("\n" + "=" * 80)
    print("4️⃣  Testing Interactive Dashboard")
    print("=" * 80)

    # Project embeddings to 2D for dashboard
    from sklearn.manifold import TSNE
    print("\n🔧 Projecting embeddings to 2D...")
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Prepare metrics
    metrics = {
        'silhouette_score': 0.6523,
        'davies_bouldin_index': 0.8234,
        'n_clusters': len(set(labels[labels >= 0])),
        'n_samples': len(embeddings)
    }

    comparison_metrics = {
        'K-Means': {'silhouette_score': 0.6523, 'davies_bouldin_index': 0.8234},
        'DBSCAN': {'silhouette_score': 0.7123, 'davies_bouldin_index': 0.7456},
        'Hierarchical': {'silhouette_score': 0.5987, 'davies_bouldin_index': 0.9012}
    }

    feature_names = [f"Feature_{i:03d}" for i in range(features.shape[1])]

    # Generate dashboard
    print("\n🎨 Generating comprehensive dashboard...")
    InteractiveDashboard.generate_dashboard(
        embeddings=embeddings_2d,
        labels=labels,
        song_names=song_names,
        original_features=features,
        feature_names=feature_names,
        metrics=metrics,
        comparison_metrics=comparison_metrics,
        title="Music Clustering Dashboard",
        save_path=str(output_dir / "dashboard.html")
    )

    print("\n✅ Dashboard generation complete")


def main():
    """Main test pipeline."""
    args = parse_args()

    print("=" * 80)
    print("🎨 Member 2 Visualization Module Test")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Load or generate data
    embeddings, labels, features, song_names = load_or_generate_data(args)

    if args.dashboard_only:
        # Only generate dashboard
        test_dashboard(embeddings, labels, song_names, features, output_dir)
    else:
        # Run all tests
        test_embedding_visualizer(embeddings, labels, song_names, output_dir)
        test_cluster_visualizer(labels, song_names, output_dir)
        test_comparison_visualizer(embeddings, labels, features, output_dir)
        test_dashboard(embeddings, labels, song_names, features, output_dir)

    print("\n" + "=" * 80)
    print("✅ All Visualization Tests Complete!")
    print("=" * 80)
    print(f"\n📂 Results saved to: {output_dir}")
    print("\n🔍 Generated files:")
    for file in sorted(output_dir.iterdir()):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
