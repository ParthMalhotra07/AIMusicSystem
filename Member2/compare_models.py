"""
Model Comparison Script
Compares different embedding models side-by-side to help select the best one
"""

import sys
import os
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

from data_loading import load_features, find_latest_feature_file
from embedding import AutoencoderModel, PCAModel, UMAPModel, EmbeddingEvaluator


def compare_all_models(
    X,
    embedding_dim=32,
    epochs=50,
    verbose=True
):
    """
    Train and compare all three models.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (N x 170)
    embedding_dim : int
        Target embedding dimension
    epochs : int
        Training epochs for autoencoder
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results for each model
    """
    results = {}

    # Split data for autoencoder
    n_val = int(len(X) * 0.2)
    X_train = X[n_val:]
    X_val = X[:n_val]

    if verbose:
        print("=" * 60)
        print("🔬 Comparing Embedding Models")
        print("=" * 60)
        print(f"Data shape: {X.shape}")
        print(f"Embedding dimension: {embedding_dim}")
        print("=" * 60)

    # 1. PCA (fastest, linear baseline)
    if verbose:
        print("\n1️⃣  Training PCA...")
        print("-" * 60)

    start_time = time.time()
    pca = PCAModel(input_dim=170, embedding_dim=embedding_dim)
    pca.fit(X, verbose=verbose)
    pca_embeddings = pca.transform(X)
    pca_reconstructed = pca.inverse_transform(pca_embeddings)
    pca_time = time.time() - start_time

    evaluator = EmbeddingEvaluator()
    pca_metrics = evaluator.evaluate_all(X, pca_embeddings, pca_reconstructed)

    results['PCA'] = {
        'model': pca,
        'embeddings': pca_embeddings,
        'metrics': pca_metrics,
        'training_time': pca_time
    }

    if verbose:
        print(f"✓ PCA completed in {pca_time:.2f}s")

    # 2. Autoencoder (nonlinear, best for complex patterns)
    if verbose:
        print("\n2️⃣  Training Autoencoder...")
        print("-" * 60)

    start_time = time.time()
    autoencoder = AutoencoderModel(
        input_dim=170,
        embedding_dim=embedding_dim,
        encoder_layers=(128, 64),
        epochs=epochs,
        batch_size=32,
        early_stopping_patience=10,
        device='cpu'  # Use CPU for comparison
    )
    autoencoder.fit(X_train, X_val=X_val, verbose=verbose)
    ae_embeddings = autoencoder.transform(X)
    ae_reconstructed = autoencoder.reconstruct(X)
    ae_time = time.time() - start_time

    ae_metrics = evaluator.evaluate_all(X, ae_embeddings, ae_reconstructed)

    results['Autoencoder'] = {
        'model': autoencoder,
        'embeddings': ae_embeddings,
        'metrics': ae_metrics,
        'training_time': ae_time
    }

    if verbose:
        print(f"✓ Autoencoder completed in {ae_time:.2f}s")

    # 3. UMAP (nonlinear, good for visualization)
    try:
        if verbose:
            print("\n3️⃣  Training UMAP...")
            print("-" * 60)

        start_time = time.time()
        umap_model = UMAPModel(
            input_dim=170,
            embedding_dim=embedding_dim,
            n_neighbors=15,
            min_dist=0.1
        )
        umap_model.fit(X, verbose=verbose)
        umap_embeddings = umap_model.transform(X)
        umap_time = time.time() - start_time

        umap_metrics = evaluator.evaluate_all(X, umap_embeddings, None)

        results['UMAP'] = {
            'model': umap_model,
            'embeddings': umap_embeddings,
            'metrics': umap_metrics,
            'training_time': umap_time
        }

        if verbose:
            print(f"✓ UMAP completed in {umap_time:.2f}s")

    except ImportError:
        if verbose:
            print("⚠ UMAP not available (install with: pip install umap-learn)")
        results['UMAP'] = None

    return results


def print_comparison_table(results):
    """Print comparison table of all models."""
    print("\n" + "=" * 80)
    print("📊 Model Comparison Summary")
    print("=" * 80)

    # Header
    print(f"{'Metric':<40} {'PCA':>12} {'Autoencoder':>12} {'UMAP':>12}")
    print("-" * 80)

    # Training time
    print(f"{'Training Time (seconds)':<40} "
          f"{results['PCA']['training_time']:>12.2f} "
          f"{results['Autoencoder']['training_time']:>12.2f} "
          f"{results.get('UMAP', {}).get('training_time', 0):>12.2f}")

    # Metrics
    metric_names = [
        ('neighborhood_preservation_k5', 'Neighborhood Preservation (k=5)'),
        ('neighborhood_preservation_k10', 'Neighborhood Preservation (k=10)'),
        ('reconstruction_mse', 'Reconstruction MSE'),
        ('explained_variance_ratio', 'Explained Variance Ratio')
    ]

    for metric_key, metric_display in metric_names:
        pca_val = results['PCA']['metrics'].get(metric_key, '-')
        ae_val = results['Autoencoder']['metrics'].get(metric_key, '-')
        umap_val = results.get('UMAP', {}).get('metrics', {}).get(metric_key, '-')

        pca_str = f"{pca_val:.4f}" if isinstance(pca_val, float) else str(pca_val)
        ae_str = f"{ae_val:.4f}" if isinstance(ae_val, float) else str(ae_val)
        umap_str = f"{umap_val:.4f}" if isinstance(umap_val, float) else str(umap_val)

        print(f"{metric_display:<40} {pca_str:>12} {ae_str:>12} {umap_str:>12}")

    print("=" * 80)


def recommend_model(results):
    """
    Recommend best model based on metrics.

    Returns
    -------
    str
        Recommended model name
    """
    print("\n" + "=" * 80)
    print("🎯 Model Recommendation")
    print("=" * 80)

    scores = {}

    # Score PCA
    pca_metrics = results['PCA']['metrics']
    pca_score = (
        pca_metrics['neighborhood_preservation_k10'] * 0.4 +
        (1 - pca_metrics['reconstruction_mse'] / 10) * 0.3 +
        pca_metrics['explained_variance_ratio'] * 0.3
    )
    scores['PCA'] = max(0, pca_score)

    # Score Autoencoder
    ae_metrics = results['Autoencoder']['metrics']
    ae_score = (
        ae_metrics['neighborhood_preservation_k10'] * 0.4 +
        (1 - ae_metrics['reconstruction_mse'] / 10) * 0.4 +
        ae_metrics['explained_variance_ratio'] * 0.2
    )
    scores['Autoencoder'] = max(0, ae_score)

    # Score UMAP (if available)
    if results.get('UMAP'):
        umap_metrics = results['UMAP']['metrics']
        umap_score = (
            umap_metrics['neighborhood_preservation_k10'] * 0.5 +
            umap_metrics['explained_variance_ratio'] * 0.5
        )
        scores['UMAP'] = max(0, umap_score)

    # Recommendations
    print("\n📈 Scores (higher is better):")
    for model_name, score in scores.items():
        print(f"   {model_name}: {score:.4f}")

    best_model = max(scores, key=scores.get)

    print(f"\n✅ Recommended Model: **{best_model}**\n")

    # Detailed reasoning
    print("Reasoning:")
    print(f"  • PCA: Fast, linear baseline. Good for quick prototyping.")
    print(f"     - Training time: {results['PCA']['training_time']:.2f}s")
    print(f"     - Best for: Speed, interpretability")

    print(f"\n  • Autoencoder: Learns nonlinear patterns. Best quality.")
    print(f"     - Training time: {results['Autoencoder']['training_time']:.2f}s")
    print(f"     - Best for: Accuracy, reconstruction quality")

    if results.get('UMAP'):
        print(f"\n  • UMAP: Preserves local structure. Good for visualization.")
        print(f"     - Training time: {results['UMAP']['training_time']:.2f}s")
        print(f"     - Best for: 2D/3D visualization, local similarity")

    print("\n" + "=" * 80)

    return best_model


def plot_embeddings_comparison(results, save_path=None):
    """Plot 2D projections of embeddings from each model."""
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))

    if len(results) == 1:
        axes = [axes]

    for idx, (model_name, result) in enumerate(results.items()):
        if result is None:
            continue

        embeddings = result['embeddings']

        # Project to 2D using t-SNE (if not already 2D)
        if embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            coords_2d = tsne.fit_transform(embeddings)
        else:
            coords_2d = embeddings

        # Plot
        axes[idx].scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.6, s=30)
        axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Dimension 1')
        axes[idx].set_ylabel('Dimension 2')
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")

    plt.show()


def main():
    """Main comparison pipeline."""
    print("=" * 60)
    print("🔬 Embedding Model Comparison Tool")
    print("=" * 60)

    # Load data
    print("\n📂 Loading feature data...")
    latest_file = find_latest_feature_file('../Member1/output', format='parquet')

    if latest_file is None:
        print("❌ No feature files found in ../Member1/output/")
        print("   Please run Member 1 first to extract features.")
        sys.exit(1)

    dataset, _ = load_features(latest_file, preprocess=True, verbose=False)
    X = dataset.get_feature_matrix()

    print(f"✓ Loaded {len(X)} songs with {X.shape[1]} features\n")

    # Compare models
    results = compare_all_models(
        X,
        embedding_dim=32,
        epochs=50,
        verbose=True
    )

    # Print comparison
    print_comparison_table(results)

    # Recommend best model
    best_model = recommend_model(results)

    # Plot comparison
    print("\n📊 Generating comparison plot...")
    plot_embeddings_comparison(results, save_path='output/model_comparison.png')

    print("\n✅ Comparison complete!")
    print(f"   Recommended model: {best_model}")


if __name__ == '__main__':
    main()
