"""
Clustering Module for Member 2
Clusters music embeddings into similar groups without labels
"""

from .base import BaseClusteringModel, ClusterAnalyzer
from .evaluate import ClusteringEvaluator, find_optimal_k
from .kmeans_clustering import KMeansModel
from .hierarchical import HierarchicalModel
from .dbscan import DBSCANModel

__all__ = [
    # Base classes
    'BaseClusteringModel',
    'ClusterAnalyzer',

    # Evaluation
    'ClusteringEvaluator',
    'find_optimal_k',

    # Clustering models
    'KMeansModel',
    'HierarchicalModel',
    'DBSCANModel',
]

__version__ = '1.0.0'
__author__ = 'Member 2 - Embedding & Clustering Engineer'
