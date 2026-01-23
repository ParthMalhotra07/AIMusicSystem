"""
Base Embedding Model
Abstract base class for all embedding models (Autoencoder, PCA, UMAP, etc.)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for embedding models.

    All embedding models must implement:
    - fit(X): Train the model
    - transform(X): Get embeddings
    - save(path): Save model
    - load(path): Load model
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        name: str = "BaseModel"
    ):
        """
        Initialize embedding model.

        Parameters
        ----------
        input_dim : int
            Input feature dimension (e.g., 170)
        embedding_dim : int
            Target embedding dimension (e.g., 32)
        name : str
            Model name for logging
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> 'BaseEmbeddingModel':
        """
        Fit the embedding model on data.

        Parameters
        ----------
        X : np.ndarray
            Training data (N x input_dim)
        **kwargs
            Additional arguments (e.g., validation data, epochs)

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to embedding space.

        Parameters
        ----------
        X : np.ndarray
            Input data (N x input_dim)

        Returns
        -------
        embeddings : np.ndarray
            Embedded data (N x embedding_dim)
        """
        pass

    def fit_transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : np.ndarray
            Training data (N x input_dim)
        **kwargs
            Additional arguments for fit

        Returns
        -------
        embeddings : np.ndarray
            Embedded data (N x embedding_dim)
        """
        self.fit(X, **kwargs)
        return self.transform(X)

    @abstractmethod
    def save(self, filepath: str):
        """
        Save model to disk.

        Parameters
        ----------
        filepath : str
            Path to save model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseEmbeddingModel':
        """
        Load model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved model

        Returns
        -------
        model : BaseEmbeddingModel
            Loaded model
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns
        -------
        dict
            Model parameters
        """
        return {
            'name': self.name,
            'input_dim': self.input_dim,
            'embedding_dim': self.embedding_dim,
            'is_fitted': self.is_fitted
        }

    def __repr__(self) -> str:
        """String representation."""
        params = self.get_params()
        param_str = ', '.join([f'{k}={v}' for k, v in params.items()])
        return f"{self.__class__.__name__}({param_str})"


class EmbeddingEvaluator:
    """
    Evaluates quality of embeddings.
    """

    @staticmethod
    def reconstruction_error(
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> float:
        """
        Calculate mean squared error between original and reconstructed.

        Parameters
        ----------
        original : np.ndarray
            Original features
        reconstructed : np.ndarray
            Reconstructed features

        Returns
        -------
        float
            MSE reconstruction error
        """
        return np.mean((original - reconstructed) ** 2)

    @staticmethod
    def explained_variance(
        original: np.ndarray,
        embeddings: np.ndarray
    ) -> float:
        """
        Calculate how much variance is explained by embeddings.

        Parameters
        ----------
        original : np.ndarray
            Original features (N x D_original)
        embeddings : np.ndarray
            Embeddings (N x D_embed)

        Returns
        -------
        float
            Explained variance ratio
        """
        # Calculate variance in original space
        original_variance = np.var(original)

        # For dimensionality reduction, compare to variance in embedding space
        embedding_variance = np.var(embeddings)

        return embedding_variance / original_variance

    @staticmethod
    def neighborhood_preservation(
        original: np.ndarray,
        embeddings: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Calculate how well k-nearest neighbors are preserved.

        Parameters
        ----------
        original : np.ndarray
            Original features
        embeddings : np.ndarray
            Embeddings
        k : int
            Number of neighbors

        Returns
        -------
        float
            Neighborhood preservation score (0-1, higher is better)
        """
        from sklearn.neighbors import NearestNeighbors

        # Find k-nearest neighbors in original space
        nn_original = NearestNeighbors(n_neighbors=k+1)
        nn_original.fit(original)
        _, indices_original = nn_original.kneighbors(original)

        # Find k-nearest neighbors in embedding space
        nn_embedding = NearestNeighbors(n_neighbors=k+1)
        nn_embedding.fit(embeddings)
        _, indices_embedding = nn_embedding.kneighbors(embeddings)

        # Calculate overlap (excluding self)
        preserved = 0
        for i in range(len(original)):
            neighbors_orig = set(indices_original[i, 1:])  # Exclude self
            neighbors_embed = set(indices_embedding[i, 1:])
            overlap = len(neighbors_orig & neighbors_embed)
            preserved += overlap / k

        return preserved / len(original)

    @staticmethod
    def evaluate_all(
        original: np.ndarray,
        embeddings: np.ndarray,
        reconstructed: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Run all evaluation metrics.

        Parameters
        ----------
        original : np.ndarray
            Original features
        embeddings : np.ndarray
            Embeddings
        reconstructed : np.ndarray, optional
            Reconstructed features (if available, e.g., from autoencoder)

        Returns
        -------
        dict
            Dictionary of metric name -> score
        """
        metrics = {}

        # Neighborhood preservation
        metrics['neighborhood_preservation_k5'] = EmbeddingEvaluator.neighborhood_preservation(
            original, embeddings, k=5
        )
        metrics['neighborhood_preservation_k10'] = EmbeddingEvaluator.neighborhood_preservation(
            original, embeddings, k=10
        )

        # Reconstruction error (if available)
        if reconstructed is not None:
            metrics['reconstruction_mse'] = EmbeddingEvaluator.reconstruction_error(
                original, reconstructed
            )

        # Explained variance
        metrics['explained_variance_ratio'] = EmbeddingEvaluator.explained_variance(
            original, embeddings
        )

        return metrics

    @staticmethod
    def print_evaluation(metrics: Dict[str, float]):
        """
        Pretty-print evaluation metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary of metrics
        """
        print("=" * 60)
        print("📊 Embedding Quality Metrics")
        print("=" * 60)

        for name, value in metrics.items():
            # Format name
            display_name = name.replace('_', ' ').title()
            print(f"{display_name:.<50} {value:.4f}")

        print("=" * 60)


if __name__ == '__main__':
    # Test evaluator with synthetic data
    print("Testing Embedding Evaluator...\n")

    np.random.seed(42)

    # Create synthetic data
    original = np.random.randn(100, 170)
    embeddings = np.random.randn(100, 32)
    reconstructed = original + np.random.randn(100, 170) * 0.1  # Add noise

    # Evaluate
    evaluator = EmbeddingEvaluator()
    metrics = evaluator.evaluate_all(original, embeddings, reconstructed)

    # Print results
    evaluator.print_evaluation(metrics)

    print("\n✓ Evaluator test complete")
