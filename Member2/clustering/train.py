"""
Clustering Training Script
Command-line interface for training clustering models on embeddings
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from clustering.kmeans_clustering import KMeansModel
from clustering.hierarchical import HierarchicalModel
from clustering.dbscan import DBSCANModel
from clustering.evaluate import ClusteringEvaluator, find_optimal_k
from clustering.base import ClusterAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train clustering models on music embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # K-means with 5 clusters
    python train.py --method kmeans --n_clusters 5 --embeddings embeddings.npy

    # Auto-find optimal k
    python train.py --method kmeans --auto_k --k_range 3 10

    # DBSCAN with auto eps estimation
    python train.py --method dbscan --auto_eps

    # Hierarchical clustering
    python train.py --method hierarchical --n_clusters 5 --linkage ward

    # Compare all methods
    python train.py --compare --embeddings embeddings.npy
        """
    )

    # Input/Output
    parser.add_argument(
        '--embeddings', '-e',
        type=str,
        required=True,
        help='Path to embeddings .npy file from Member 2'
    )
    parser.add_argument(
        '--song_mapping',
        type=str,
        default=None,
        help='Path to song_mapping.json (optional)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./output/clustering',
        help='Output directory'
    )

    # Method selection
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['kmeans', 'hierarchical', 'dbscan', 'all'],
        default='kmeans',
        help='Clustering method'
    )

    # K-means parameters
    parser.add_argument(
        '--n_clusters', '-k',
        type=int,
        default=5,
        help='Number of clusters (for kmeans, hierarchical)'
    )
    parser.add_argument(
        '--auto_k',
        action='store_true',
        help='Automatically find optimal k (kmeans only)'
    )
    parser.add_argument(
        '--k_range',
        nargs=2,
        type=int,
        default=[3, 10],
        help='Range for auto k search (min max)'
    )

    # Hierarchical parameters
    parser.add_argument(
        '--linkage',
        type=str,
        choices=['ward', 'complete', 'average', 'single'],
        default='ward',
        help='Linkage method for hierarchical clustering'
    )

    # DBSCAN parameters
    parser.add_argument(
        '--eps',
        type=float,
        default=0.5,
        help='Eps for DBSCAN (neighborhood radius)'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=5,
        help='Min samples for DBSCAN'
    )
    parser.add_argument(
        '--auto_eps',
        action='store_true',
        help='Auto-estimate eps for DBSCAN'
    )

    # Options
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all clustering methods'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots (elbow, dendrogram, etc.)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    return parser.parse_args()


def load_data(args):
    """Load embeddings and song mapping."""
    verbose = not args.quiet

    if verbose:
        print("\n📂 Loading embeddings...")

    # Load embeddings
    embeddings = np.load(args.embeddings)

    if verbose:
        print(f"✓ Loaded embeddings: {embeddings.shape}")

    # Load song mapping if available
    song_names = None
    if args.song_mapping and Path(args.song_mapping).exists():
        with open(args.song_mapping, 'r') as f:
            mapping = json.load(f)
            song_names = mapping.get('song_names', None)

        if verbose and song_names:
            print(f"✓ Loaded {len(song_names)} song names")

    return embeddings, song_names


def train_kmeans(X, args, song_names=None):
    """Train K-means model."""
    verbose = not args.quiet

    # Auto-find k if requested
    if args.auto_k:
        optimal_k = find_optimal_k(
            X,
            k_range=tuple(args.k_range),
            method='kmeans',
            verbose=verbose
        )
        n_clusters = optimal_k
    else:
        n_clusters = args.n_clusters

    # Train model
    model = KMeansModel(n_clusters=n_clusters, n_init=20)
    model.fit(X, verbose=verbose)

    # Evaluate
    if verbose:
        print("\n📊 Evaluating K-Means...")

    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate_all(X, model.labels_, verbose=verbose)

    # Cluster analysis
    if song_names:
        analyzer = ClusterAnalyzer()
        summary = analyzer.get_cluster_summary(X, model.labels_, song_names)
        analyzer.print_summary(summary, verbose=verbose)

    # Plot elbow if requested
    if args.plot:
        model.plot_elbow(X, k_range=tuple(args.k_range), save_path='output/kmeans_elbow.png')

    return model, metrics


def train_hierarchical(X, args, song_names=None):
    """Train hierarchical model."""
    verbose = not args.quiet

    # Train model
    model = HierarchicalModel(
        n_clusters=args.n_clusters,
        linkage=args.linkage
    )
    model.fit(X, verbose=verbose)

    # Evaluate
    if verbose:
        print("\n📊 Evaluating Hierarchical...")

    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate_all(X, model.labels_, verbose=verbose)

    # Cluster analysis
    if song_names:
        analyzer = ClusterAnalyzer()
        summary = analyzer.get_cluster_summary(X, model.labels_, song_names)
        analyzer.print_summary(summary, verbose=verbose)

    # Plot dendrogram if requested
    if args.plot:
        model.plot_dendrogram(X, save_path='output/hierarchical_dendrogram.png')

    return model, metrics


def train_dbscan(X, args, song_names=None):
    """Train DBSCAN model."""
    verbose = not args.quiet

    # Auto-estimate eps if requested
    if args.auto_eps:
        eps = DBSCANModel.estimate_eps(X, k=5)
    else:
        eps = args.eps

    # Train model
    model = DBSCANModel(eps=eps, min_samples=args.min_samples)
    model.fit(X, verbose=verbose)

    # Evaluate
    if verbose:
        print("\n📊 Evaluating DBSCAN...")

    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate_all(X, model.labels_, verbose=verbose)

    # Cluster analysis
    if song_names:
        analyzer = ClusterAnalyzer()
        summary = analyzer.get_cluster_summary(X, model.labels_, song_names)
        analyzer.print_summary(summary, verbose=verbose)

    return model, metrics


def compare_all_methods(X, args, song_names=None):
    """Compare all clustering methods."""
    verbose = not args.quiet

    if verbose:
        print("\n" + "=" * 80)
        print("🔬 Comparing All Clustering Methods")
        print("=" * 80)

    results = {}

    # K-means
    if verbose:
        print("\n1️⃣  Training K-Means...")

    kmeans = KMeansModel(n_clusters=args.n_clusters, n_init=20)
    kmeans.fit(X, verbose=False)
    results['K-Means'] = kmeans.labels_

    # Hierarchical
    if verbose:
        print("2️⃣  Training Hierarchical...")

    hierarchical = HierarchicalModel(n_clusters=args.n_clusters, linkage='ward')
    hierarchical.fit(X, verbose=False)
    results['Hierarchical'] = hierarchical.labels_

    # DBSCAN
    if verbose:
        print("3️⃣  Training DBSCAN...")

    eps = DBSCANModel.estimate_eps(X, k=5) if args.auto_eps else args.eps
    dbscan = DBSCANModel(eps=eps, min_samples=args.min_samples)
    dbscan.fit(X, verbose=False)
    results['DBSCAN'] = dbscan.labels_

    # Evaluate all
    evaluator = ClusteringEvaluator()
    comparison = evaluator.compare_results(X, results)
    evaluator.print_comparison(comparison)

    return results, comparison


def save_results(model, metrics, args):
    """Save clustering results."""
    verbose = not args.quiet

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"{args.method}_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    # Save model
    model_path = run_dir / f"{args.method}_model.pkl"
    model.save(str(model_path))

    # Save labels
    labels_path = run_dir / "cluster_labels.npy"
    np.save(labels_path, model.labels_)

    if verbose:
        print(f"\n💾 Results saved to {run_dir}")
        print(f"   - Model: {model_path.name}")
        print(f"   - Labels: {labels_path.name}")

    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"   - Metrics: {metrics_path.name}")

    return run_dir


def main():
    """Main training pipeline."""
    args = parse_args()

    print("=" * 80)
    print("🎵 Music Clustering Training")
    print("=" * 80)
    print(f"Method: {args.method}")
    if args.method in ['kmeans', 'hierarchical']:
        print(f"Number of clusters: {args.n_clusters}")
    print("=" * 80)

    # Load data
    X, song_names = load_data(args)

    # Train
    if args.compare:
        results, comparison = compare_all_methods(X, args, song_names)
    elif args.method == 'kmeans':
        model, metrics = train_kmeans(X, args, song_names)
        save_results(model, metrics, args)
    elif args.method == 'hierarchical':
        model, metrics = train_hierarchical(X, args, song_names)
        save_results(model, metrics, args)
    elif args.method == 'dbscan':
        model, metrics = train_dbscan(X, args, song_names)
        save_results(model, metrics, args)

    print("\n" + "=" * 80)
    print("✅ Clustering Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
