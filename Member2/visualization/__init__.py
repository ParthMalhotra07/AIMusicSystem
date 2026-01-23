"""
Visualization Module for Member 2
Provides comprehensive visualization tools for embeddings and clustering results
"""

from .embeddings import EmbeddingVisualizer
from .clusters import ClusterVisualizer
from .comparison import ComparisonVisualizer
from .dashboard import InteractiveDashboard

__all__ = [
    'EmbeddingVisualizer',
    'ClusterVisualizer',
    'ComparisonVisualizer',
    'InteractiveDashboard'
]

__version__ = '1.0.0'
