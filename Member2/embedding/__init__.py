"""
Embedding Module for Member 2
Dimensionality reduction models for learning compact music representations
"""

from .base import BaseEmbeddingModel, EmbeddingEvaluator
from .autoencoder import AutoencoderModel, AutoencoderNetwork
from .pca_model import PCAModel
from .umap_model import UMAPModel

__all__ = [
    # Base classes
    'BaseEmbeddingModel',
    'EmbeddingEvaluator',

    # Models
    'AutoencoderModel',
    'AutoencoderNetwork',
    'PCAModel',
    'UMAPModel',
]

__version__ = '1.0.0'
__author__ = 'Member 2 - Embedding & Clustering Engineer'
