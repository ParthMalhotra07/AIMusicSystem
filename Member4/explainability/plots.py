"""
Visualization Module for AI Music Recommendation System.

Provides clean matplotlib-based visualizations for:
- Radar charts of component similarity scores
- PCA embedding space maps
- Feature comparison tables

CYBERPUNK THEME: Neon colors on dark backgrounds
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CYBERPUNK Color palette - Neon colors
COLORS = {
    "seed": "#ff00ff",        # Neon Magenta
    "recommended": "#ff8800",  # Neon Orange
    "other": "#444444",        # Dark Gray
    "highlight": "#00ff88",    # Neon Green
    "primary": "#00ffff",      # Neon Cyan
    "secondary": "#ff00ff",    # Neon Pink
    "background": "#0a0a0a",   # Near Black
    "grid": "#333333",         # Dark Grid
}

# Set dark theme globally
plt.style.use('dark_background')


def plot_radar(
    component_scores: Dict[str, float],
    title: str = "⚡ SIMILARITY COMPONENT ANALYSIS",
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Create a CYBERPUNK radar chart showing component similarity scores.
    
    Args:
        component_scores: Dictionary of component names to scores (0-1)
        title: Chart title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not component_scores:
        # Return empty figure if no scores
        fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["background"])
        ax.set_facecolor(COLORS["background"])
        ax.text(0.5, 0.5, "No component scores available", 
                ha='center', va='center', fontsize=12, color=COLORS["primary"])
        ax.set_axis_off()
        return fig
    
    # Prepare data
    categories = list(component_scores.keys())
    values = list(component_scores.values())
    
    # Clean up category names for display - UPPERCASE for cyberpunk
    display_names = [cat.replace("_score", "").replace("_", " ").upper() 
                     for cat in categories]
    
    # Number of variables
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop
    values = values + [values[0]]
    angles = angles + [angles[0]]
    display_names = display_names + [display_names[0]]
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True),
                           facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    
    # Plot with neon glow effect
    ax.plot(angles, values, linewidth=8, color=COLORS["primary"], alpha=0.2)  # Glow
    ax.plot(angles, values, 'o-', linewidth=3, color=COLORS["primary"])
    ax.fill(angles, values, alpha=0.3, color=COLORS["primary"])
    
    # Set the labels with neon color
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_names[:-1], size=11, color=COLORS["highlight"], fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8, color='#888888')
    
    # Add title with neon effect
    ax.set_title(title, size=16, fontweight='bold', pad=20, color=COLORS["secondary"])
    
    # Styled gridlines
    ax.grid(True, linestyle='--', alpha=0.4, color=COLORS["grid"])
    ax.spines['polar'].set_color(COLORS["primary"])
    ax.spines['polar'].set_linewidth(2)
    
    plt.tight_layout()
    return fig


def plot_embedding_map(
    embeddings: np.ndarray,
    song_ids: np.ndarray,
    seed_id: str,
    rec_ids: List[str],
    id_to_idx: Dict[str, int],
    title: str = "🗺️ MUSIC EMBEDDING SPACE (PCA)",
    figsize: Tuple[int, int] = (12, 10),
    show_labels: bool = False
) -> plt.Figure:
    """
    Create a CYBERPUNK 2D PCA visualization of the embedding space.
    
    Args:
        embeddings: Full embedding matrix (N, D)
        song_ids: Array of song IDs
        seed_id: The seed song ID to highlight
        rec_ids: List of recommended song IDs to highlight
        id_to_idx: Mapping from song_id to index
        title: Chart title
        figsize: Figure size
        show_labels: Whether to show song ID labels
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    explained_var = sum(pca.explained_variance_ratio_) * 100
    
    # Get indices for seed and recommendations
    seed_idx = id_to_idx.get(seed_id)
    rec_indices = [id_to_idx.get(rid) for rid in rec_ids if rid in id_to_idx]
    
    # Create masks
    all_indices = set(range(len(song_ids)))
    highlighted_indices = {seed_idx} | set(rec_indices) if seed_idx is not None else set(rec_indices)
    other_indices = all_indices - highlighted_indices
    
    # Plot other points (background) - dim dots
    if other_indices:
        other_idx_list = list(other_indices)
        ax.scatter(
            embeddings_2d[other_idx_list, 0],
            embeddings_2d[other_idx_list, 1],
            c=COLORS["other"],
            s=30,
            alpha=0.5,
            label="Other songs"
        )
    
    # Plot recommended points with neon glow
    if rec_indices:
        # Glow effect
        ax.scatter(
            embeddings_2d[rec_indices, 0],
            embeddings_2d[rec_indices, 1],
            c=COLORS["recommended"],
            s=400,
            alpha=0.2,
        )
        # Main point
        ax.scatter(
            embeddings_2d[rec_indices, 0],
            embeddings_2d[rec_indices, 1],
            c=COLORS["recommended"],
            s=150,
            alpha=0.9,
            marker='s',
            edgecolors='white',
            linewidths=1.5,
            label="Recommended"
        )
        
        if show_labels:
            for idx, rid in zip(rec_indices, rec_ids):
                ax.annotate(
                    rid,
                    (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    fontsize=8,
                    alpha=0.8,
                    color=COLORS["highlight"]
                )
    
    # Plot seed point (on top) with neon glow
    if seed_idx is not None:
        # Glow effect
        ax.scatter(
            embeddings_2d[seed_idx, 0],
            embeddings_2d[seed_idx, 1],
            c=COLORS["seed"],
            s=800,
            alpha=0.2,
        )
        # Main point
        ax.scatter(
            embeddings_2d[seed_idx, 0],
            embeddings_2d[seed_idx, 1],
            c=COLORS["seed"],
            s=400,
            alpha=1.0,
            marker='*',
            edgecolors='white',
            linewidths=2,
            label=f"Seed: {seed_id}"
        )
    
    # Draw neon lines from seed to recommendations
    if seed_idx is not None and rec_indices:
        for rec_idx in rec_indices:
            ax.plot(
                [embeddings_2d[seed_idx, 0], embeddings_2d[rec_idx, 0]],
                [embeddings_2d[seed_idx, 1], embeddings_2d[rec_idx, 1]],
                color=COLORS["primary"],
                linestyle='--',
                alpha=0.5,
                linewidth=1.5
            )
    
    # Labels and title with neon colors
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", 
                  fontsize=11, color=COLORS["primary"])
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", 
                  fontsize=11, color=COLORS["primary"])
    ax.set_title(f"{title}\n(Total explained variance: {explained_var:.1f}%)", 
                 fontsize=14, fontweight='bold', color=COLORS["secondary"])
    
    # Legend with dark background
    legend = ax.legend(loc='upper right', fontsize=10, facecolor=COLORS["background"],
                       edgecolor=COLORS["primary"], labelcolor='white')
    
    # Styled grid
    ax.grid(True, linestyle='--', alpha=0.3, color=COLORS["grid"])
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    
    plt.tight_layout()
    return fig


def plot_feature_comparison(
    seed_id: str,
    rec_id: str,
    features_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create a bar chart comparing features between seed and recommended song.
    
    Args:
        seed_id: Seed song ID
        rec_id: Recommended song ID
        features_df: DataFrame with song features
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get song rows
    seed_row = features_df[features_df["song_id"] == seed_id]
    rec_row = features_df[features_df["song_id"] == rec_id]
    
    if len(seed_row) == 0 or len(rec_row) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "Feature data not available", 
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()
        return fig
    
    seed_row = seed_row.iloc[0]
    rec_row = rec_row.iloc[0]
    
    # Plot 1: Main features comparison
    main_features = ["tempo", "spectral_centroid", "rms"]
    main_features = [f for f in main_features if f in seed_row.index]
    
    if main_features:
        x = np.arange(len(main_features))
        width = 0.35
        
        seed_vals = [float(seed_row[f]) for f in main_features]
        rec_vals = [float(rec_row[f]) for f in main_features]
        
        # Normalize for display
        max_vals = [max(abs(s), abs(r)) for s, r in zip(seed_vals, rec_vals)]
        seed_norm = [s/m if m > 0 else 0 for s, m in zip(seed_vals, max_vals)]
        rec_norm = [r/m if m > 0 else 0 for r, m in zip(rec_vals, max_vals)]
        
        bars1 = axes[0].bar(x - width/2, seed_norm, width, 
                           label=f'Seed: {seed_id}', color=COLORS["seed"])
        bars2 = axes[0].bar(x + width/2, rec_norm, width,
                           label=f'Rec: {rec_id}', color=COLORS["recommended"])
        
        axes[0].set_xlabel('Feature')
        axes[0].set_ylabel('Normalized Value')
        axes[0].set_title('Main Audio Features Comparison')
        axes[0].set_xticks(x)
        display_names = [f.replace("_", " ").title() for f in main_features]
        axes[0].set_xticklabels(display_names)
        axes[0].legend()
        axes[0].grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Add value annotations
        for bar, val in zip(bars1, seed_vals):
            axes[0].annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, rec_vals):
            axes[0].annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "Main features not available", 
                     ha='center', va='center', fontsize=12)
        axes[0].set_axis_off()
    
    # Plot 2: MFCC comparison
    mfcc_cols = [col for col in features_df.columns if 'mfcc' in col.lower()]
    mfcc_cols = sorted(mfcc_cols)[:6]  # First 6 MFCCs
    
    if mfcc_cols:
        x = np.arange(len(mfcc_cols))
        width = 0.35
        
        seed_mfcc = [float(seed_row[c]) for c in mfcc_cols]
        rec_mfcc = [float(rec_row[c]) for c in mfcc_cols]
        
        axes[1].bar(x - width/2, seed_mfcc, width, 
                   label=f'Seed', color=COLORS["seed"])
        axes[1].bar(x + width/2, rec_mfcc, width,
                   label=f'Rec', color=COLORS["recommended"])
        
        axes[1].set_xlabel('MFCC Coefficient')
        axes[1].set_ylabel('Value')
        axes[1].set_title('MFCC Comparison (Timbre)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'MFCC {i+1}' for i in range(len(mfcc_cols))])
        axes[1].legend()
        axes[1].grid(True, axis='y', linestyle='--', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "MFCC features not available", 
                     ha='center', va='center', fontsize=12)
        axes[1].set_axis_off()
    
    plt.tight_layout()
    return fig


def plot_feature_table(
    seed_id: str,
    rec_id: str,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a formatted comparison table of features.
    
    Args:
        seed_id: Seed song ID
        rec_id: Recommended song ID
        features_df: DataFrame with song features
        
    Returns:
        DataFrame with comparison table
    """
    # Get song rows
    seed_row = features_df[features_df["song_id"] == seed_id]
    rec_row = features_df[features_df["song_id"] == rec_id]
    
    if len(seed_row) == 0 or len(rec_row) == 0:
        return pd.DataFrame({"Error": ["Feature data not available"]})
    
    seed_row = seed_row.iloc[0]
    rec_row = rec_row.iloc[0]
    
    # Select key features for comparison
    feature_groups = {
        "Rhythm": ["tempo"],
        "Energy": ["rms"],
        "Brightness": ["spectral_centroid"],
        "Duration": ["duration_sec"],
    }
    
    # Build comparison data
    comparison_data = []
    
    for group, features in feature_groups.items():
        for feat in features:
            if feat in seed_row.index and feat in rec_row.index:
                seed_val = float(seed_row[feat])
                rec_val = float(rec_row[feat])
                diff = rec_val - seed_val
                diff_pct = (diff / seed_val * 100) if seed_val != 0 else 0
                
                comparison_data.append({
                    "Category": group,
                    "Feature": feat.replace("_", " ").title(),
                    f"Seed ({seed_id})": f"{seed_val:.2f}",
                    f"Rec ({rec_id})": f"{rec_val:.2f}",
                    "Difference": f"{diff:+.2f}",
                    "Diff %": f"{diff_pct:+.1f}%"
                })
    
    # Add MFCC summary
    mfcc_cols = [col for col in features_df.columns if 'mfcc' in col.lower()]
    if mfcc_cols:
        seed_mfcc = seed_row[mfcc_cols].values.astype(float)
        rec_mfcc = rec_row[mfcc_cols].values.astype(float)
        mfcc_dist = float(np.linalg.norm(seed_mfcc - rec_mfcc))
        
        comparison_data.append({
            "Category": "Timbre",
            "Feature": "MFCC Distance (L2)",
            f"Seed ({seed_id})": "-",
            f"Rec ({rec_id})": "-",
            "Difference": f"{mfcc_dist:.2f}",
            "Diff %": "-"
        })
    
    # Add Chroma summary
    chroma_cols = [col for col in features_df.columns if 'chroma' in col.lower()]
    if chroma_cols:
        seed_chroma = seed_row[chroma_cols].values.astype(float).reshape(1, -1)
        rec_chroma = rec_row[chroma_cols].values.astype(float).reshape(1, -1)
        from sklearn.metrics.pairwise import cosine_similarity
        chroma_sim = float(cosine_similarity(seed_chroma, rec_chroma)[0, 0])
        
        comparison_data.append({
            "Category": "Harmony",
            "Feature": "Chroma Similarity",
            f"Seed ({seed_id})": "-",
            f"Rec ({rec_id})": "-",
            "Difference": f"{chroma_sim:.3f}",
            "Diff %": "-"
        })
    
    return pd.DataFrame(comparison_data)


def plot_similarity_distribution(
    similarities: List[float],
    highlight_scores: Optional[List[float]] = None,
    title: str = "Similarity Score Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot histogram of similarity scores.
    
    Args:
        similarities: List of all similarity scores
        highlight_scores: Scores to highlight (e.g., recommendations)
        title: Chart title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(similarities, bins=50, color=COLORS["other"], alpha=0.7, 
            edgecolor='black', linewidth=0.5, label="All pairs")
    
    # Highlight specific scores
    if highlight_scores:
        for i, score in enumerate(highlight_scores):
            ax.axvline(x=score, color=COLORS["recommended"], 
                      linestyle='--', linewidth=2, alpha=0.8,
                      label=f"Rec {i+1}: {score:.3f}" if i < 3 else None)
    
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test the module
    import sys
    sys.path.insert(0, str(__file__).replace("explainability/plots.py", "integration"))
    from load_data import load_all_data
    from recommender_adapter import recommend_from_song
    
    print("Testing plots module...")
    data = load_all_data()
    
    # Get seed and recommendations
    seed_id = str(data["song_ids"][0])
    rec_ids, scores = recommend_from_song(
        seed_id,
        data["embeddings"],
        data["song_ids"],
        data["id_to_idx"],
        k=5
    )
    
    # Test radar chart
    print("Testing radar chart...")
    component_scores = {
        "tempo_score": 0.85,
        "timbre_score": 0.72,
        "brightness_score": 0.91,
        "harmony_score": 0.68,
        "energy_score": 0.79
    }
    fig_radar = plot_radar(component_scores)
    plt.savefig("test_radar.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved test_radar.png")
    
    # Test embedding map
    print("Testing embedding map...")
    fig_map = plot_embedding_map(
        data["embeddings"],
        data["song_ids"],
        seed_id,
        rec_ids,
        data["id_to_idx"]
    )
    plt.savefig("test_embedding_map.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved test_embedding_map.png")
    
    # Test feature table
    print("Testing feature table...")
    table = plot_feature_table(seed_id, rec_ids[0], data["features_df"])
    print(table.to_string())
    
    print("\nAll plot tests completed!")
