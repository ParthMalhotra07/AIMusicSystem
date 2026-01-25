"""
Explainability Engine for AI Music Recommendation System.

Transforms vector similarity into human-readable musical reasoning.
Provides component-wise similarity scores and natural language explanations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Similarity thresholds
THRESHOLD_VERY_SIMILAR = 0.85
THRESHOLD_MODERATELY_SIMILAR = 0.65
THRESHOLD_DIFFERENT = 0.4

# Feature-specific parameters
TEMPO_DECAY = 15.0  # BPM difference for decay
BRIGHTNESS_DECAY = 2000.0  # Spectral centroid difference for decay
ENERGY_DECAY = 0.1  # RMS difference for decay


def _get_mfcc_columns(df: pd.DataFrame) -> List[str]:
    """Get MFCC column names from DataFrame."""
    mfcc_cols = [col for col in df.columns if col.startswith("mfcc_mean_")]
    if not mfcc_cols:
        mfcc_cols = [col for col in df.columns if col.startswith("mfcc_")]
    return sorted(mfcc_cols, key=lambda x: int(x.split("_")[-1]))


def _get_chroma_columns(df: pd.DataFrame) -> List[str]:
    """Get Chroma column names from DataFrame."""
    chroma_cols = [col for col in df.columns if col.startswith("chroma_mean_")]
    if not chroma_cols:
        chroma_cols = [col for col in df.columns if col.startswith("chroma_")]
    return sorted(chroma_cols, key=lambda x: int(x.split("_")[-1]))


def _get_song_row(song_id: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """Get feature row for a song."""
    row = df[df["song_id"] == song_id]
    if len(row) == 0:
        return None
    return row.iloc[0]


def _compute_tempo_similarity(tempo1: float, tempo2: float) -> float:
    """Compute tempo similarity using exponential decay."""
    # Handle NaN values
    if np.isnan(tempo1) or np.isnan(tempo2):
        return 0.5  # Neutral if missing data
    diff = abs(tempo1 - tempo2)
    return float(np.exp(-diff / TEMPO_DECAY))


def _compute_timbre_similarity(mfcc1: np.ndarray, mfcc2: np.ndarray) -> Tuple[float, float]:
    """
    Compute timbre similarity from MFCC vectors.
    
    Returns:
        Tuple of (similarity score, L2 distance)
    """
    # Handle None or NaN values
    if mfcc1 is None or mfcc2 is None:
        return 0.5, 0.0
    if np.any(np.isnan(mfcc1)) or np.any(np.isnan(mfcc2)):
        return 0.5, 0.0
    mfcc_dist = float(np.linalg.norm(mfcc1 - mfcc2))
    similarity = float(np.exp(-mfcc_dist / 10.0))  # Normalized decay
    return similarity, mfcc_dist


def _compute_brightness_similarity(centroid1: float, centroid2: float) -> float:
    """Compute brightness similarity from spectral centroids."""
    # Handle NaN values
    if np.isnan(centroid1) or np.isnan(centroid2):
        return 0.5  # Neutral if missing data
    diff = abs(centroid1 - centroid2)
    return float(np.exp(-diff / BRIGHTNESS_DECAY))


def _compute_harmony_similarity(chroma1: np.ndarray, chroma2: np.ndarray) -> float:
    """Compute harmony similarity from chroma features using cosine similarity."""
    if chroma1 is None or chroma2 is None:
        return 0.5  # Neutral if no chroma data
    if len(chroma1) == 0 or len(chroma2) == 0:
        return 0.5  # Neutral if no chroma data
    
    # Handle NaN values
    if np.any(np.isnan(chroma1)) or np.any(np.isnan(chroma2)):
        return 0.5  # Neutral if NaN present
    
    chroma1 = chroma1.reshape(1, -1)
    chroma2 = chroma2.reshape(1, -1)
    return float(cosine_similarity(chroma1, chroma2)[0, 0])


def _compute_energy_similarity(rms1: float, rms2: float) -> float:
    """Compute energy similarity from RMS values."""
    # Handle NaN values
    if np.isnan(rms1) or np.isnan(rms2):
        return 0.5  # Neutral if missing data
    diff = abs(rms1 - rms2)
    return float(np.exp(-diff / ENERGY_DECAY))


def _compute_embedding_similarity(
    song_id_1: str,
    song_id_2: str,
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int]
) -> Optional[float]:
    """Compute cosine similarity between embeddings."""
    if song_id_1 not in id_to_idx or song_id_2 not in id_to_idx:
        return None
    
    idx1 = id_to_idx[song_id_1]
    idx2 = id_to_idx[song_id_2]
    
    emb1 = embeddings[idx1].reshape(1, -1)
    emb2 = embeddings[idx2].reshape(1, -1)
    
    return float(cosine_similarity(emb1, emb2)[0, 0])


def _similarity_level(score: float) -> str:
    """Convert similarity score to human-readable level."""
    if score >= THRESHOLD_VERY_SIMILAR:
        return "very similar"
    elif score >= THRESHOLD_MODERATELY_SIMILAR:
        return "moderately similar"
    elif score >= THRESHOLD_DIFFERENT:
        return "somewhat different"
    else:
        return "quite different"


def _generate_tempo_reason(tempo1: float, tempo2: float, score: float) -> str:
    """Generate explanation text for tempo similarity."""
    diff = abs(tempo1 - tempo2)
    
    if score >= THRESHOLD_VERY_SIMILAR:
        return f"Tempo is highly aligned ({tempo1:.0f} vs {tempo2:.0f} BPM, diff: {diff:.0f} BPM)."
    elif score >= THRESHOLD_MODERATELY_SIMILAR:
        return f"Tempo is reasonably close ({tempo1:.0f} vs {tempo2:.0f} BPM, diff: {diff:.0f} BPM)."
    else:
        return f"Tempo differs significantly ({tempo1:.0f} vs {tempo2:.0f} BPM, diff: {diff:.0f} BPM)."


def _generate_timbre_reason(mfcc_dist: float, score: float) -> str:
    """Generate explanation text for timbre similarity."""
    if score >= THRESHOLD_VERY_SIMILAR:
        return f"Timbre profile is very similar (MFCC distance: {mfcc_dist:.2f}), suggesting comparable instrumentation and vocal texture."
    elif score >= THRESHOLD_MODERATELY_SIMILAR:
        return f"Timbre profile shows moderate similarity (MFCC distance: {mfcc_dist:.2f}), indicating related sonic character."
    else:
        return f"Timbre profiles differ (MFCC distance: {mfcc_dist:.2f}), suggesting different instrumentation or vocal styles."


def _generate_brightness_reason(
    centroid1: float, 
    centroid2: float, 
    score: float
) -> str:
    """Generate explanation text for brightness similarity."""
    diff = abs(centroid1 - centroid2)
    
    if score >= THRESHOLD_VERY_SIMILAR:
        return f"Brightness matches closely (spectral centroid diff: {diff:.0f} Hz), indicating similar tonal color."
    elif score >= THRESHOLD_MODERATELY_SIMILAR:
        return f"Brightness is moderately similar (spectral centroid diff: {diff:.0f} Hz)."
    else:
        return f"Brightness differs (spectral centroid diff: {diff:.0f} Hz), suggesting different tonal character."


def _generate_harmony_reason(chroma_sim: float) -> str:
    """Generate explanation text for harmony similarity."""
    if chroma_sim >= THRESHOLD_VERY_SIMILAR:
        return f"Harmony patterns align well (chroma similarity: {chroma_sim:.2f}), indicating similar chord progressions."
    elif chroma_sim >= THRESHOLD_MODERATELY_SIMILAR:
        return f"Harmony shows moderate alignment (chroma similarity: {chroma_sim:.2f})."
    else:
        return f"Harmony patterns differ (chroma similarity: {chroma_sim:.2f}), suggesting different chord structures."


def _generate_energy_reason(rms1: float, rms2: float, score: float) -> str:
    """Generate explanation text for energy similarity."""
    if score >= THRESHOLD_VERY_SIMILAR:
        return f"Energy levels match (RMS: {rms1:.3f} vs {rms2:.3f}), indicating similar loudness/intensity."
    elif score >= THRESHOLD_MODERATELY_SIMILAR:
        return f"Energy levels are moderately similar (RMS: {rms1:.3f} vs {rms2:.3f})."
    else:
        return f"Energy levels differ (RMS: {rms1:.3f} vs {rms2:.3f}), suggesting different loudness profiles."


def explain_pair(
    seed_id: str,
    rec_id: str,
    features_df: pd.DataFrame,
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int]
) -> Dict:
    """
    Generate comprehensive explanation for why two songs are similar.
    
    Args:
        seed_id: The seed song ID
        rec_id: The recommended song ID
        features_df: DataFrame containing song features
        embeddings: Embedding matrix
        id_to_idx: Mapping from song_id to embedding index
        
    Returns:
        Dictionary containing:
        - cosine_similarity: Overall embedding similarity
        - component_scores: Individual feature similarities
        - feature_deltas: Raw feature differences
        - reasons: Human-readable explanation bullets
    """
    result = {
        "seed_id": seed_id,
        "rec_id": rec_id,
        "cosine_similarity": None,
        "component_scores": {},
        "feature_deltas": {},
        "reasons": [],
        "overall_assessment": "",
    }
    
    # Get song feature rows
    seed_row = _get_song_row(seed_id, features_df)
    rec_row = _get_song_row(rec_id, features_df)
    
    if seed_row is None:
        result["reasons"].append(f"⚠️ Features not found for seed song: {seed_id}")
        return result
    
    if rec_row is None:
        result["reasons"].append(f"⚠️ Features not found for recommended song: {rec_id}")
        return result
    
    # Compute embedding cosine similarity
    cos_sim = _compute_embedding_similarity(seed_id, rec_id, embeddings, id_to_idx)
    result["cosine_similarity"] = cos_sim
    
    # ===== TEMPO ANALYSIS =====
    if "tempo" in seed_row and "tempo" in rec_row:
        tempo1 = float(seed_row["tempo"])
        tempo2 = float(rec_row["tempo"])
        tempo_score = _compute_tempo_similarity(tempo1, tempo2)
        
        result["component_scores"]["tempo_score"] = tempo_score
        result["feature_deltas"]["tempo_diff"] = abs(tempo1 - tempo2)
        result["reasons"].append(_generate_tempo_reason(tempo1, tempo2, tempo_score))
    
    # ===== TIMBRE ANALYSIS (MFCC) =====
    mfcc_cols = _get_mfcc_columns(features_df)
    if mfcc_cols:
        mfcc1 = seed_row[mfcc_cols].values.astype(float)
        mfcc2 = rec_row[mfcc_cols].values.astype(float)
        timbre_score, mfcc_dist = _compute_timbre_similarity(mfcc1, mfcc2)
        
        result["component_scores"]["timbre_score"] = timbre_score
        result["feature_deltas"]["mfcc_dist"] = mfcc_dist
        result["reasons"].append(_generate_timbre_reason(mfcc_dist, timbre_score))
    
    # ===== BRIGHTNESS ANALYSIS (SPECTRAL CENTROID) =====
    if "spectral_centroid" in seed_row and "spectral_centroid" in rec_row:
        centroid1 = float(seed_row["spectral_centroid"])
        centroid2 = float(rec_row["spectral_centroid"])
        brightness_score = _compute_brightness_similarity(centroid1, centroid2)
        
        result["component_scores"]["brightness_score"] = brightness_score
        result["feature_deltas"]["centroid_diff"] = abs(centroid1 - centroid2)
        result["reasons"].append(
            _generate_brightness_reason(centroid1, centroid2, brightness_score)
        )
    
    # ===== HARMONY ANALYSIS (CHROMA) =====
    chroma_cols = _get_chroma_columns(features_df)
    if chroma_cols:
        chroma1 = seed_row[chroma_cols].values.astype(float)
        chroma2 = rec_row[chroma_cols].values.astype(float)
        harmony_score = _compute_harmony_similarity(chroma1, chroma2)
        
        result["component_scores"]["harmony_score"] = harmony_score
        result["feature_deltas"]["chroma_sim"] = harmony_score
        result["reasons"].append(_generate_harmony_reason(harmony_score))
    
    # ===== ENERGY ANALYSIS (RMS) =====
    rms_col = "rms" if "rms" in seed_row else ("rms_mean" if "rms_mean" in seed_row else None)
    if rms_col:
        rms1 = float(seed_row[rms_col])
        rms2 = float(rec_row[rms_col])
        energy_score = _compute_energy_similarity(rms1, rms2)
        
        result["component_scores"]["energy_score"] = energy_score
        result["feature_deltas"]["rms_diff"] = abs(rms1 - rms2)
        result["reasons"].append(_generate_energy_reason(rms1, rms2, energy_score))
    
    # ===== OVERALL ASSESSMENT =====
    if result["component_scores"]:
        avg_score = np.mean(list(result["component_scores"].values()))
        level = _similarity_level(avg_score)
        result["overall_assessment"] = (
            f"Overall, these tracks are {level} "
            f"(avg component score: {avg_score:.2f}, embedding similarity: {cos_sim:.2f})."
        )
    
    return result


def explain_recommendations(
    seed_id: str,
    rec_ids: List[str],
    features_df: pd.DataFrame,
    embeddings: np.ndarray,
    id_to_idx: Dict[str, int]
) -> List[Dict]:
    """
    Generate explanations for a list of recommendations.
    
    Returns:
        List of explanation dictionaries, one per recommendation
    """
    explanations = []
    for rec_id in rec_ids:
        explanation = explain_pair(
            seed_id=seed_id,
            rec_id=rec_id,
            features_df=features_df,
            embeddings=embeddings,
            id_to_idx=id_to_idx
        )
        explanations.append(explanation)
    return explanations


def get_similarity_summary(explanation: Dict) -> str:
    """Generate a one-line summary of the similarity."""
    cos_sim = explanation.get("cosine_similarity", 0) or 0
    scores = explanation.get("component_scores", {})
    
    if not scores:
        return f"Embedding similarity: {cos_sim:.2f}"
    
    avg_component = np.mean(list(scores.values()))
    level = _similarity_level(avg_component)
    
    return f"{level.capitalize()} (embedding: {cos_sim:.2f}, features: {avg_component:.2f})"


def format_explanation_for_display(explanation: Dict) -> str:
    """Format explanation as a multi-line string for display."""
    lines = []
    
    # Header
    lines.append(f"🎵 Seed: {explanation['seed_id']}")
    lines.append(f"🎶 Recommendation: {explanation['rec_id']}")
    lines.append("")
    
    # Cosine similarity
    cos_sim = explanation.get("cosine_similarity")
    if cos_sim is not None:
        lines.append(f"📊 Embedding Similarity: {cos_sim:.4f}")
    
    # Component scores
    scores = explanation.get("component_scores", {})
    if scores:
        lines.append("")
        lines.append("Component Scores:")
        for name, score in scores.items():
            display_name = name.replace("_", " ").title()
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"  {display_name}: {bar} {score:.2f}")
    
    # Reasons
    reasons = explanation.get("reasons", [])
    if reasons:
        lines.append("")
        lines.append("Musical Analysis:")
        for reason in reasons:
            lines.append(f"  • {reason}")
    
    # Overall assessment
    assessment = explanation.get("overall_assessment", "")
    if assessment:
        lines.append("")
        lines.append(f"📋 {assessment}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the module
    import sys
    sys.path.insert(0, str(__file__).replace("explainability/explain.py", "integration"))
    from load_data import load_all_data
    from recommender_adapter import recommend_from_song
    
    print("Testing explain module...")
    data = load_all_data()
    
    # Get a seed and recommendations
    seed_id = str(data["song_ids"][0])
    rec_ids, scores = recommend_from_song(
        seed_id,
        data["embeddings"],
        data["song_ids"],
        data["id_to_idx"],
        k=3
    )
    
    print(f"\nExplanation for seed '{seed_id}' and recommendation '{rec_ids[0]}':\n")
    explanation = explain_pair(
        seed_id,
        rec_ids[0],
        data["features_df"],
        data["embeddings"],
        data["id_to_idx"]
    )
    
    print(format_explanation_for_display(explanation))
