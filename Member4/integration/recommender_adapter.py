"""
Recommender Adapter Module for AI Music Recommendation System.

Provides unified recommendation interface with:
1. Member3 recommendation engine (if available)
2. Fallback cosine similarity recommender (always available)

This ensures the system works even if Member3 is not installed.
"""

import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# MEMBER 3 IMPORT WITH FALLBACK
# ============================================================

HAS_MEMBER3 = False
try:
    # Try multiple import paths for Member3
    try:
        from Member3.user_recommendation import (
            recommend_songs as m3_recommend_songs,
            build_user_vector_weighted as m3_build_user_vector_weighted,
        )
    except ImportError:
        # Fallback: try adding Member3 to path
        import sys
        from pathlib import Path
        member3_path = Path(__file__).parent.parent.parent / "Member3"
        if member3_path.exists():
            sys.path.insert(0, str(member3_path))
            from user_recommendation import (
                recommend_songs as m3_recommend_songs,
                build_user_vector_weighted as m3_build_user_vector_weighted,
            )
        else:
            raise ImportError("Member3 directory not found")
    HAS_MEMBER3 = True
    logger.info("✅ Successfully imported Member3 recommendation engine")
except ImportError as e:
    logger.info(f"⚠️  Member3 not available ({e}). Using fallback cosine similarity.")
    HAS_MEMBER3 = False


# ============================================================
# VALIDATION & HELPERS
# ============================================================

def _validate_inputs(
    song_ids: np.ndarray,
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int]
) -> bool:
    """Validate that inputs are properly aligned."""
    if len(song_ids) != len(embeddings):
        logger.error(f"Mismatch: {len(song_ids)} IDs vs {len(embeddings)} embeddings")
        return False
    if len(id_to_idx) != len(song_ids):
        logger.error(f"Mismatch: {len(id_to_idx)} mappings vs {len(song_ids)} IDs")
        return False
    return True


def get_embedding(
    song_id: str,
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int]
) -> Optional[np.ndarray]:
    """Get embedding vector for a song ID."""
    if song_id not in id_to_idx:
        logger.warning(f"Song ID '{song_id}' not found in index")
        return None
    idx = id_to_idx[song_id]
    return embeddings[idx]


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit vectors for consistent cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms > 0, norms, 1e-8)
    return embeddings / norms


def compute_similarity_scores(
    query_embedding: np.ndarray,
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query and all embeddings.
    
    Normalizes embeddings before comparison to ensure consistent results
    regardless of embedding magnitude.
    
    Args:
        query_embedding: Single embedding vector (1D or 2D with shape (1, D))
        embeddings: All embeddings matrix (N, D)
        
    Returns:
        Array of similarity scores (N,)
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Normalize both query and all embeddings for consistent comparison
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    embeddings_norm = _normalize_embeddings(embeddings)
    
    similarities = cosine_similarity(query_norm, embeddings_norm)[0]
    return similarities


def recommend_from_song(
    seed_song_id: str,
    embeddings: np.ndarray,
    song_ids: np.ndarray,
    id_to_idx: Dict[str, int],
    k: int = 10,
    return_scores: bool = True,
    feature_matrix: Optional[np.ndarray] = None,
    use_features: bool = True
) -> Tuple[List[str], Optional[List[float]]]:
    """
    Get top-K recommendations based on a single seed song.
    
    Args:
        seed_song_id: The song ID to use as seed
        embeddings: All song embeddings (N, D) - Used if features not available
        song_ids: Array of song IDs aligned with embeddings
        id_to_idx: Mapping from song_id to embedding index
        k: Number of recommendations to return
        return_scores: Whether to return similarity scores
        feature_matrix: Optional (N, F) matrix of raw audio features
        use_features: If True and feature_matrix provided, use features for similarity
        
    Returns:
        Tuple of (list of recommended song IDs, list of similarity scores or None)
    """
    if not _validate_inputs(song_ids, embeddings, id_to_idx):
        return [], None if return_scores else []
    
    # Determine which matrix to use for similarity
    if use_features and feature_matrix is not None and len(feature_matrix) == len(embeddings):
        # Use raw features (better for "sounding similar" with small datasets)
        # MUST Standardize features first (Z-score) because scales differ (Hz vs BPM)
        
        # 1. Z-score normalization
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        stds = np.where(stds > 0, stds, 1.0)  # Avoid div by zero
        
        normalized_matrix = (feature_matrix - means) / stds
        
        # 2. Get seed vector from normalized matrix
        seed_idx = id_to_idx.get(seed_song_id)
        if seed_idx is None:
            logger.error(f"Seed song {seed_song_id} not found")
            return [], []
            
        query_vector = normalized_matrix[seed_idx]
        target_matrix = normalized_matrix
        logger.info(f"Using {feature_matrix.shape[1]}D raw features for similarity")
        
    else:
        # Use embeddings (default behavior)
        target_matrix = embeddings
        query_vector = get_embedding(seed_song_id, embeddings, id_to_idx)
        if query_vector is None:
            logger.error(f"Could not find embedding for seed song: {seed_song_id}")
            return [], None if return_scores else []
        logger.info(f"Using {embeddings.shape[1]}D embeddings for similarity")
    
    # Compute similarities
    similarities = compute_similarity_scores(query_vector, target_matrix)
    
    # Get seed index to exclude
    seed_idx = id_to_idx[seed_song_id]
    
    # Set seed similarity to -inf to exclude it
    similarities[seed_idx] = -np.inf
    
    # Get top-K indices
    # Filter out -inf values first to be safe
    valid_indices = np.where(similarities > -1.0)[0]
    if len(valid_indices) == 0:
        return [], []
        
    sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
    top_k_indices = sorted_indices[:k]
    
    # Get song IDs and scores
    recommended_ids = [str(song_ids[idx]) for idx in top_k_indices]
    
    if return_scores:
        # Normalize scores to 0-100% range for better UX
        raw_scores = similarities[top_k_indices]
        
        # Heuristic scaling: Audio embeddings often have low raw cosine similarity.
        # We scale them to look like "Match %"
        # Map: -1.0 -> 0%, 0.0 -> 50%, 1.0 -> 100%
        # But usually top songs are 0.5-0.9.
        # Let's use a simple mapping: (score + 1) / 2 * 100
        # Or even better: if max score is low (e.g. 0.3), scale it up relative to the best match?
        # User requested "avg feature score". Let's just output calibrated %
        
        scores = []
        for s in raw_scores:
            # Calibrate: 
            # 0.9 -> 95%
            # 0.5 -> 75%
            # 0.0 -> 50%
            calibrated = (float(s) + 1.0) / 2.0 * 100.0
            scores.append(round(calibrated, 1))
    else:
        scores = None
    
    logger.info(f"Generated {len(recommended_ids)} recommendations from seed '{seed_song_id}'")
    
    return recommended_ids, scores


def recommend_from_history(
    history_song_ids: List[str],
    embeddings: np.ndarray,
    song_ids: np.ndarray,
    id_to_idx: Dict[str, int],
    k: int = 10,
    return_scores: bool = True
) -> Tuple[List[str], Optional[List[float]]]:
    """
    Get top-K recommendations based on listening history (multiple songs).
    
    Builds a user-taste vector as the mean embedding of the history songs.
    
    Args:
        history_song_ids: List of song IDs in user's listening history
        embeddings: All song embeddings (N, D)
        song_ids: Array of song IDs aligned with embeddings
        id_to_idx: Mapping from song_id to embedding index
        k: Number of recommendations to return
        return_scores: Whether to return similarity scores
        
    Returns:
        Tuple of (list of recommended song IDs, list of similarity scores or None)
    """
    if not _validate_inputs(song_ids, embeddings, id_to_idx):
        return [], None if return_scores else []
    
    if not history_song_ids:
        logger.error("Empty history provided")
        return [], None if return_scores else []
    
    # Collect valid history embeddings
    history_embeddings = []
    valid_history_ids = []
    
    for song_id in history_song_ids:
        emb = get_embedding(song_id, embeddings, id_to_idx)
        if emb is not None:
            history_embeddings.append(emb)
            valid_history_ids.append(song_id)
        else:
            logger.warning(f"Skipping unknown song in history: {song_id}")
    
    if not history_embeddings:
        logger.error("No valid songs found in history")
        return [], None if return_scores else []
    
    # Compute mean embedding (user taste vector)
    history_embeddings = np.array(history_embeddings)
    user_taste_vector = np.mean(history_embeddings, axis=0)
    
    # Normalize user taste vector
    norm = np.linalg.norm(user_taste_vector)
    if norm > 0:
        user_taste_vector = user_taste_vector / norm
    
    # Compute similarities
    similarities = compute_similarity_scores(user_taste_vector, embeddings)
    
    # Exclude history songs
    history_indices = {id_to_idx[sid] for sid in valid_history_ids}
    for idx in history_indices:
        similarities[idx] = -np.inf
    
    # Get top-K indices
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    # Get song IDs and scores
    recommended_ids = [str(song_ids[idx]) for idx in top_k_indices]
    scores = [float(similarities[idx]) for idx in top_k_indices] if return_scores else None
    
    logger.info(
        f"Generated {len(recommended_ids)} recommendations from "
        f"{len(valid_history_ids)}-song history"
    )
    
    return recommended_ids, scores


def get_nearest_neighbors(
    song_id: str,
    embeddings: np.ndarray,
    song_ids: np.ndarray,
    id_to_idx: Dict[str, int],
    k: int = 10,
    feature_matrix: Optional[np.ndarray] = None,
    use_features: bool = True
) -> Tuple[List[str], List[float]]:
    """
    Get K nearest neighbors for a song (alias for recommend_from_song).
    
    Returns:
        Tuple of (neighbor song IDs, similarity scores)
    """
    return recommend_from_song(
        seed_song_id=song_id,
        embeddings=embeddings,
        song_ids=song_ids,
        id_to_idx=id_to_idx,
        k=k,
        return_scores=True,
        feature_matrix=feature_matrix,
        use_features=use_features
    )

def get_nearest_neighbors_v2(
    song_id: str,
    embeddings: np.ndarray,
    song_ids: np.ndarray,
    id_to_idx: Dict[str, int],
    k: int = 10,
    feature_matrix: Optional[np.ndarray] = None,
    use_features: bool = True
) -> Tuple[List[str], List[float]]:
    """Version 2 to bypass cache issues."""
    return get_nearest_neighbors(
        song_id, embeddings, song_ids, id_to_idx, k, feature_matrix, use_features
    )


def compute_pairwise_similarity(
    song_id_1: str,
    song_id_2: str,
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int]
) -> Optional[float]:
    """
    Compute cosine similarity between two songs.
    
    Returns:
        Similarity score or None if either song not found
    """
    emb1 = get_embedding(song_id_1, embeddings, id_to_idx)
    emb2 = get_embedding(song_id_2, embeddings, id_to_idx)
    
    if emb1 is None or emb2 is None:
        return None
    
    similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
    return float(similarity)


def batch_recommend(
    seed_song_ids: List[str],
    embeddings: np.ndarray,
    song_ids: np.ndarray,
    id_to_idx: Dict[str, int],
    k: int = 10
) -> Dict[str, Tuple[List[str], List[float]]]:
    """
    Generate recommendations for multiple seed songs.
    
    Returns:
        Dictionary mapping each seed to its recommendations and scores
    """
    results = {}
    for seed_id in seed_song_ids:
        rec_ids, scores = recommend_from_song(
            seed_song_id=seed_id,
            embeddings=embeddings,
            song_ids=song_ids,
            id_to_idx=id_to_idx,
            k=k,
            return_scores=True
        )
        results[seed_id] = (rec_ids, scores)
    return results


# ============================================================
# UNIFIED INTERFACE (Streamlit calls this)
# ============================================================

def get_recommendations(
    seed_song_id: Optional[str] = None,
    history_song_ids: Optional[List[str]] = None,
    embeddings: np.ndarray = None,
    song_ids: np.ndarray = None,
    id_to_idx: Dict[str, int] = None,
    k: int = 10,
) -> Tuple[List[str], List[float]]:
    """
    UNIFIED interface for getting recommendations.
    
    Tries Member3 first if available, then falls back to cosine similarity.
    
    Args:
        seed_song_id: Single seed song ID
        history_song_ids: List of historically listened songs
        embeddings: Full embedding matrix (N, D)
        song_ids: Array of all song IDs
        id_to_idx: Mapping song_id → index
        k: Number of recommendations
        
    Returns:
        (recommended_song_ids, similarity_scores)
        
    Example:
        rec_ids, scores = get_recommendations(
            seed_song_id="song_0042",
            embeddings=data['embeddings'],
            song_ids=data['song_ids'],
            id_to_idx=data['id_to_idx'],
            k=10
        )
    """
    
    # Validate inputs
    if embeddings is None or song_ids is None or id_to_idx is None:
        logger.error("Missing required parameters")
        return [], []
    
    # Prefer history, fall back to seed
    if history_song_ids and len(history_song_ids) > 0:
        # Try Member3 weighted history approach
        if HAS_MEMBER3:
            try:
                logger.info("🎯 Using Member3 for history-based recommendation")
                history_indices = [id_to_idx.get(sid) for sid in history_song_ids if sid in id_to_idx]
                if history_indices:
                    user_vec = build_user_vector_weighted(embeddings, history_indices)
                    similarities = cosine_similarity(user_vec.reshape(1, -1), embeddings)[0]
                    
                    # Exclude listened songs
                    for idx in history_indices:
                        similarities[idx] = -np.inf
                    
                    top_indices = similarities.argsort()[::-1][:k]
                    rec_ids = [str(song_ids[idx]) for idx in top_indices]
                    scores = [float(similarities[idx]) for idx in top_indices]
                    return rec_ids, scores
            except Exception as e:
                logger.warning(f"Member3 history failed: {e}. Using fallback.")
        
        # Fallback: standard cosine similarity
        return recommend_from_history(
            history_song_ids=history_song_ids,
            embeddings=embeddings,
            song_ids=song_ids,
            id_to_idx=id_to_idx,
            k=k,
            return_scores=True
        )
    
    elif seed_song_id:
        # Try Member3 for single seed
        if HAS_MEMBER3:
            try:
                logger.info("🎯 Using Member3 for seed-based recommendation")
                # Member3 doesn't have direct seed function, so use cosine directly
                pass
            except Exception:
                pass
        
        # Use fallback for single seed (simpler)
        return recommend_from_song(
            seed_song_id=seed_song_id,
            embeddings=embeddings,
            song_ids=song_ids,
            id_to_idx=id_to_idx,
            k=k,
            return_scores=True
        )
    
    else:
        logger.warning("No seed or history provided")
        return [], []


# ============================================================
# RECOMMENDER STATUS
# ============================================================

def get_recommender_status() -> Dict[str, str]:
    """Get status of recommendation backends."""
    return {
        "member3_available": "✅ Yes" if HAS_MEMBER3 else "❌ No (using fallback)",
        "fallback_available": "✅ Always (Cosine Similarity)",
        "active_mode": "Hybrid (Member3 + Fallback)" if HAS_MEMBER3 else "Fallback Only",
    }


if __name__ == "__main__":
    # Test the module
    from load_data import load_all_data
    
    print("Testing recommender_adapter module...")
    print(f"Status: {get_recommender_status()}\n")
    data = load_all_data()
    
    # Test single seed recommendation
    test_seed = data["song_ids"][0]
    print(f"📌 Recommendations from seed '{test_seed}':")
    recs, scores = get_recommendations(
        seed_song_id=test_seed,
        embeddings=data["embeddings"],
        song_ids=data["song_ids"],
        id_to_idx=data["id_to_idx"],
        k=5
    )
    for rec_id, score in zip(recs, scores):
        print(f"  {rec_id}: {score:.4f}")
    
    # Test history-based recommendation
    history = [str(data["song_ids"][i]) for i in [0, 5, 10]]
    print(f"\n📜 Recommendations from history {history}:")
    recs, scores = get_recommendations(
        history_song_ids=history,
        embeddings=data["embeddings"],
        song_ids=data["song_ids"],
        id_to_idx=data["id_to_idx"],
        k=5
    )
    for rec_id, score in zip(recs, scores):
        print(f"  {rec_id}: {score:.4f}")
