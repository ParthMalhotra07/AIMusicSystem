"""
Training Script for Embedding Models
Command-line interface for training and evaluating embedding models
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from data_loading import load_features, find_latest_feature_file
from config import Member2Config
from embedding.autoencoder import AutoencoderModel
from embedding.pca_model import PCAModel
from embedding.umap_model import UMAPModel
from embedding.base import EmbeddingEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train embedding models for music features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train autoencoder with default settings
    python train.py --model autoencoder --dim 32

    # Train PCA baseline
    python train.py --model pca --dim 32

    # Train UMAP with custom parameters
    python train.py --model umap --dim 32 --n_neighbors 20

    # Specify input file
    python train.py --model autoencoder --input ../Member1/output/features.parquet
        """
    )

    # Input/Output
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to feature file from Member 1 (auto-detect if not specified)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='./output',
        help='Output directory for embeddings and models'
    )

    # Model selection
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['autoencoder', 'pca', 'umap'],
        default='autoencoder',
        help='Embedding model type'
    )
    parser.add_argument(
        '--dim', '-d',
        type=int,
        default=32,
        help='Embedding dimension'
    )

    # Autoencoder parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs (autoencoder only)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (autoencoder only)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate (autoencoder only)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (autoencoder only)'
    )

    # UMAP parameters
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=15,
        help='Number of neighbors (UMAP only)'
    )
    parser.add_argument(
        '--min_dist',
        type=float,
        default=0.1,
        help='Minimum distance (UMAP only)'
    )

    # Data parameters
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--scaling',
        type=str,
        choices=['zscore', 'minmax', 'none'],
        default='zscore',
        help='Feature scaling method'
    )

    # Options
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate embedding quality'
    )
    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='Save training/evaluation plots'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    return parser.parse_args()


def load_data(args, verbose=True):
    """Load feature data from Member 1."""
    if args.input is None:
        # Auto-find latest file
        if verbose:
            print("🔍 Searching for Member 1's output...")

        latest_file = find_latest_feature_file(
            '../Member1/output',
            format='parquet'
        )

        if latest_file is None:
            print("❌ No feature files found. Please specify --input")
            sys.exit(1)

        input_file = latest_file
    else:
        input_file = args.input

    if verbose:
        print(f"📂 Loading features from: {input_file}\n")

    # Load features
    dataset, preprocessor = load_features(
        input_file,
        preprocess=(args.scaling != 'none'),
        scaling_method=args.scaling,
        validate=True,
        verbose=verbose
    )

    return dataset, preprocessor


def split_data(X, val_split, random_state=42):
    """Split data into train and validation sets."""
    n_samples = len(X)
    n_val = int(n_samples * val_split)

    # Shuffle indices
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    X_train = X[train_indices]
    X_val = X[val_indices]

    return X_train, X_val, train_indices, val_indices


def create_model(args):
    """Create embedding model based on arguments."""
    if args.model == 'autoencoder':
        model = AutoencoderModel(
            input_dim=170,
            embedding_dim=args.dim,
            encoder_layers=(128, 64),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            early_stopping_patience=args.patience
        )

    elif args.model == 'pca':
        model = PCAModel(
            input_dim=170,
            embedding_dim=args.dim,
            whiten=True
        )

    elif args.model == 'umap':
        model = UMAPModel(
            input_dim=170,
            embedding_dim=args.dim,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric='cosine'
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def train_model(model, X_train, X_val, args):
    """Train the embedding model."""
    verbose = not args.quiet

    if args.model == 'autoencoder':
        model.fit(X_train, X_val=X_val, verbose=verbose)
    else:
        model.fit(X_train, verbose=verbose)

    return model


def evaluate_model(model, X, reconstructed=None, verbose=True):
    """Evaluate embedding quality."""
    embeddings = model.transform(X)

    evaluator = EmbeddingEvaluator()
    metrics = evaluator.evaluate_all(X, embeddings, reconstructed)

    if verbose:
        evaluator.print_evaluation(metrics)

    return metrics, embeddings


def save_outputs(model, embeddings, dataset, metrics, args):
    """Save model, embeddings, and results."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"{args.model}_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    verbose = not args.quiet

    # Save model
    model_path = run_dir / f"{args.model}_model.pkl"
    model.save(str(model_path))

    # Save embeddings
    embeddings_path = run_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    if verbose:
        print(f"✓ Embeddings saved to {embeddings_path}")

    # Save song mapping
    song_mapping = {
        'file_paths': dataset.get_file_paths(),
        'song_names': dataset.get_song_names()
    }
    mapping_path = run_dir / "song_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump(song_mapping, f, indent=2)
    if verbose:
        print(f"✓ Song mapping saved to {mapping_path}")

    # Save metrics
    if metrics:
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"✓ Metrics saved to {metrics_path}")

    # Save training config
    config = {
        'model_type': args.model,
        'embedding_dim': args.dim,
        'input_file': args.input,
        'n_samples': len(embeddings),
        'scaling_method': args.scaling,
        'timestamp': timestamp
    }

    if args.model == 'autoencoder':
        config.update({
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        })

    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"✓ Config saved to {config_path}")

    # Save plots
    if args.save_plots:
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        if args.model == 'autoencoder' and hasattr(model, 'plot_training_history'):
            model.plot_training_history(save_path=str(plots_dir / "training_history.png"))

        if args.model == 'pca' and hasattr(model, 'plot_explained_variance'):
            model.plot_explained_variance(save_path=str(plots_dir / "explained_variance.png"))

    if verbose:
        print(f"\n✅ All outputs saved to {run_dir}")

    return run_dir


def main():
    """Main training pipeline."""
    args = parse_args()

    print("=" * 60)
    print("🎼 Embedding Model Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Embedding dimension: {args.dim}")
    print(f"Scaling: {args.scaling}")
    print("=" * 60)

    # Load data
    dataset, preprocessor = load_data(args, verbose=not args.quiet)
    X = dataset.get_feature_matrix()

    # Split data
    if args.val_split > 0 and args.model == 'autoencoder':
        X_train, X_val, train_idx, val_idx = split_data(X, args.val_split)
        if not args.quiet:
            print(f"\n📊 Data split:")
            print(f"   Training: {len(X_train)} samples")
            print(f"   Validation: {len(X_val)} samples")
    else:
        X_train = X
        X_val = None

    # Create model
    if not args.quiet:
        print(f"\n🔧 Creating {args.model} model...")

    model = create_model(args)

    # Train model
    if not args.quiet:
        print(f"\n🚀 Training {args.model} model...")

    model = train_model(model, X_train, X_val, args)

    # Get embeddings
    embeddings = model.transform(X)

    # Evaluate
    metrics = None
    if args.evaluate:
        if not args.quiet:
            print(f"\n📊 Evaluating embedding quality...")

        reconstructed = None
        if args.model == 'autoencoder':
            reconstructed = model.reconstruct(X)
        elif args.model == 'pca':
            reconstructed = model.inverse_transform(embeddings)

        metrics, embeddings = evaluate_model(
            model, X, reconstructed,
            verbose=not args.quiet
        )

    # Save outputs
    if not args.quiet:
        print(f"\n💾 Saving outputs...")

    run_dir = save_outputs(model, embeddings, dataset, metrics, args)

    # Summary
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Output directory: {run_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
