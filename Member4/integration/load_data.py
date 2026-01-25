"""
Data Loading Module for AI Music Recommendation System.

Provides robust data loaders with automatic mock data generation fallback.
Handles song features (CSV), embeddings (NPY), and song IDs (NPY).
"""

import os
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for mock data generation
MOCK_NUM_SONGS = 200
MOCK_EMBEDDING_DIM = 64
MOCK_NUM_MFCC = 13
MOCK_NUM_CHROMA = 12


def _generate_mock_song_ids(n: int = MOCK_NUM_SONGS) -> np.ndarray:
    """Generate mock song IDs."""
    return np.array([f"song_{i:04d}" for i in range(n)])


def _generate_mock_embeddings(
    n: int = MOCK_NUM_SONGS, 
    dim: int = MOCK_EMBEDDING_DIM
) -> np.ndarray:
    """Generate mock normalized embeddings."""
    np.random.seed(42)  # Reproducibility
    embeddings = np.random.randn(n, dim).astype(np.float32)
    # Normalize to unit vectors for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


def _generate_mock_features(n: int = MOCK_NUM_SONGS) -> pd.DataFrame:
    """Generate mock song features DataFrame with realistic audio properties."""
    np.random.seed(42)
    
    data = {
        "song_id": [f"song_{i:04d}" for i in range(n)],
        "tempo": np.random.uniform(60, 180, n),  # BPM
        "rms": np.random.uniform(0.01, 0.5, n),  # Energy
        "spectral_centroid": np.random.uniform(500, 5000, n),  # Hz
    }
    
    # Add MFCC features (mfcc_mean_1 to mfcc_mean_13)
    for i in range(1, MOCK_NUM_MFCC + 1):
        data[f"mfcc_mean_{i}"] = np.random.uniform(-20, 20, n)
    
    # Add Chroma features (chroma_mean_1 to chroma_mean_12)
    for i in range(1, MOCK_NUM_CHROMA + 1):
        data[f"chroma_mean_{i}"] = np.random.uniform(0, 1, n)
    
    # Add some additional mock metadata for display purposes
    data["duration_sec"] = np.random.uniform(120, 360, n)
    
    return pd.DataFrame(data)


def load_features(path: str = "data/song_features.csv") -> Tuple[pd.DataFrame, bool]:
    """
    Load song features from CSV file.
    
    Args:
        path: Path to the features CSV file.
        
    Returns:
        Tuple of (DataFrame with features, bool indicating if mock data was used)
    """
    is_mock = False
    
    try:
        if os.path.exists(path):
            logger.info(f"Loading features from {path}")
            df = pd.read_csv(path)
            
            # Validate required columns
            required_cols = ["song_id", "tempo", "spectral_centroid"]
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                logger.warning(f"Missing required columns: {missing}. Using mock data.")
                df = _generate_mock_features()
                is_mock = True
            else:
                logger.info(f"Loaded {len(df)} songs from {path}")
                
                # Check for RMS column (may have different names)
                if "rms" not in df.columns and "rms_mean" in df.columns:
                    df["rms"] = df["rms_mean"]
                elif "rms" not in df.columns:
                    df["rms"] = np.random.uniform(0.01, 0.5, len(df))
                    logger.warning("RMS column not found, generated random values")
        else:
            logger.warning(f"Features file not found at {path}. Generating mock data.")
            df = _generate_mock_features()
            is_mock = True
            
    except Exception as e:
        logger.error(f"Error loading features: {e}. Generating mock data.")
        df = _generate_mock_features()
        is_mock = True
    
    return df, is_mock


def load_embeddings(
    emb_path: str = "data/song_embeddings.npy",
    ids_path: str = "data/song_ids.npy"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], bool]:
    """
    Load embeddings and song IDs from NPY files.
    
    Args:
        emb_path: Path to embeddings NPY file.
        ids_path: Path to song IDs NPY file.
        
    Returns:
        Tuple of (embeddings array, song_ids array, id_to_idx mapping, is_mock flag)
    """
    is_mock = False
    
    try:
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            logger.info(f"Loading embeddings from {emb_path}")
            embeddings = np.load(emb_path)
            
            logger.info(f"Loading song IDs from {ids_path}")
            song_ids = np.load(ids_path, allow_pickle=True)
            
            # Validate alignment
            if len(embeddings) != len(song_ids):
                logger.warning(
                    f"Mismatch: {len(embeddings)} embeddings vs {len(song_ids)} IDs. "
                    "Using mock data."
                )
                embeddings = _generate_mock_embeddings()
                song_ids = _generate_mock_song_ids()
                is_mock = True
            else:
                logger.info(f"Loaded {len(embeddings)} embeddings with dim={embeddings.shape[1]}")
        else:
            missing = []
            if not os.path.exists(emb_path):
                missing.append(emb_path)
            if not os.path.exists(ids_path):
                missing.append(ids_path)
            logger.warning(f"Missing files: {missing}. Generating mock data.")
            embeddings = _generate_mock_embeddings()
            song_ids = _generate_mock_song_ids()
            is_mock = True
            
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}. Generating mock data.")
        embeddings = _generate_mock_embeddings()
        song_ids = _generate_mock_song_ids()
        is_mock = True
    
    # Build id_to_idx mapping
    id_to_idx = {str(sid): idx for idx, sid in enumerate(song_ids)}
    
    return embeddings, song_ids, id_to_idx, is_mock


def get_mfcc_columns(df: pd.DataFrame) -> list:
    """Get MFCC column names from DataFrame."""
    # Try different naming conventions
    mfcc_cols = [col for col in df.columns if col.startswith("mfcc_mean_")]
    if not mfcc_cols:
        mfcc_cols = [col for col in df.columns if col.startswith("mfcc_")]
    return sorted(mfcc_cols, key=lambda x: int(x.split("_")[-1]))


def get_chroma_columns(df: pd.DataFrame) -> list:
    """Get Chroma column names from DataFrame."""
    chroma_cols = [col for col in df.columns if col.startswith("chroma_mean_")]
    if not chroma_cols:
        chroma_cols = [col for col in df.columns if col.startswith("chroma_")]
    return sorted(chroma_cols, key=lambda x: int(x.split("_")[-1]))


def has_chroma_features(df: pd.DataFrame) -> bool:
    """Check if DataFrame has chroma features."""
    return len(get_chroma_columns(df)) > 0


def has_rms_feature(df: pd.DataFrame) -> bool:
    """Check if DataFrame has RMS energy feature."""
    return "rms" in df.columns or "rms_mean" in df.columns


def get_song_features(song_id: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """Get features for a specific song."""
    row = df[df["song_id"] == song_id]
    if len(row) == 0:
        return None
    return row.iloc[0]


def load_from_database() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, int], bool]:
    """
    Load data from the SQLite database (includes user-uploaded songs).
    
    Returns:
        Tuple of (features_df, embeddings, song_ids, id_to_idx, is_empty)
    """
    try:
        from .database import get_database
        
        db = get_database()
        
        # Get all songs from database
        songs = db.get_all_songs()
        
        if not songs:
            logger.info("Database is empty, will use file-based loading")
            return None, None, None, None, True
        
        # Get embeddings (already handles dimension alignment)
        embeddings, song_ids = db.get_all_embeddings()
        
        # Get full feature vectors for feature-based similarity
        feature_matrix, _ = db.get_all_features()
        
        if len(embeddings) == 0:
            return None, None, None, None, None, True
        
        # Get songs with MFCC/chroma features for explainability
        songs_with_features = db.get_all_songs_with_features()
        
        # Build features DataFrame with MFCC and chroma columns
        features_df = pd.DataFrame(songs_with_features)
        
        # Ensure required columns exist
        for col in ['song_id', 'tempo', 'spectral_centroid', 'rms', 'duration_sec']:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Build id_to_idx mapping
        id_to_idx = {str(sid): idx for idx, sid in enumerate(song_ids)}
        
        logger.info(f"Loaded {len(songs)} songs from database")
        return features_df, embeddings, feature_matrix, song_ids, id_to_idx, False
        
    except Exception as e:
        logger.warning(f"Failed to load from database: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, True


# Streamlit caching helpers
def load_all_data(
    features_path: str = None,
    embeddings_path: str = None,
    ids_path: str = None,
    use_database: bool = True
) -> Dict:
    """
    Load all data with a single function call (suitable for Streamlit caching).
    
    Automatically resolves paths relative to the Member4 directory.
    First tries to load from database (includes user-uploaded songs),
    then falls back to file-based loading.
    
    Parameters:
        features_path: Path to features CSV (optional)
        embeddings_path: Path to embeddings NPY (optional)
        ids_path: Path to song IDs NPY (optional)
        use_database: Whether to try loading from database first
    
    Returns:
        Dictionary containing all loaded data and metadata.
    """
    from pathlib import Path
    
    # Get the Member4 root directory
    member4_root = Path(__file__).parent.parent
    data_dir = member4_root / "data"
    
    # Try database first (includes user uploads)
    if use_database:
        features_df, embeddings, feature_matrix, song_ids, id_to_idx, is_empty = load_from_database()
        
        if not is_empty and embeddings is not None and len(embeddings) > 0:
            logger.info("Using database for data loading")
            return {
                "features_df": features_df,
                "embeddings": embeddings,
                "feature_matrix": feature_matrix,
                "song_ids": song_ids,
                "id_to_idx": id_to_idx,
                "is_mock": False,
                "num_songs": len(song_ids),
                "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                "has_chroma": False,  # Database doesn't store individual chroma
                "has_rms": "rms" in features_df.columns,
                "source": "database"
            }
    
    # Fall back to file-based loading
    logger.info("Using file-based data loading")
    
    # Use default paths relative to Member4/data if not specified
    if features_path is None:
        features_path = str(data_dir / "song_features.csv")
    if embeddings_path is None:
        embeddings_path = str(data_dir / "song_embeddings.npy")
    if ids_path is None:
        ids_path = str(data_dir / "song_ids.npy")
    
    logger.info(f"Loading data from: {data_dir}")
    
    features_df, features_mock = load_features(features_path)
    embeddings, song_ids, id_to_idx, embeddings_mock = load_embeddings(embeddings_path, ids_path)
    
    # Ensure alignment between features and embeddings
    features_song_ids = set(features_df["song_id"].values)
    embedding_song_ids = set(song_ids)
    
    common_ids = features_song_ids & embedding_song_ids
    
    if len(common_ids) < len(embedding_song_ids):
        logger.warning(
            f"Only {len(common_ids)} songs have both features and embeddings. "
            "Some songs may be missing from recommendations."
        )
    
    return {
        "features_df": features_df,
        "embeddings": embeddings,
        "song_ids": song_ids,
        "id_to_idx": id_to_idx,
        "is_mock": features_mock or embeddings_mock,
        "num_songs": len(song_ids),
        "embedding_dim": embeddings.shape[1],
        "has_chroma": has_chroma_features(features_df),
        "has_rms": has_rms_feature(features_df),
        "source": "files"
    }


if __name__ == "__main__":
    # Test the module
    print("Testing load_data module...")
    data = load_all_data()
    print(f"Loaded {data['num_songs']} songs")
    print(f"Embedding dimension: {data['embedding_dim']}")
    print(f"Using mock data: {data['is_mock']}")
    print(f"Has chroma features: {data['has_chroma']}")
    print(f"Has RMS feature: {data['has_rms']}")
    print("\nFeatures DataFrame columns:")
    print(data["features_df"].columns.tolist())
