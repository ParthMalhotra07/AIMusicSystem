"""
Explainability Page - THE WINNING PAGE

Understand WHY songs are recommended with human-readable musical reasoning.
This is the core differentiator of the system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from integration.load_data import load_all_data
from integration.recommender_adapter import recommend_from_song
from explainability.explain import explain_pair, get_similarity_summary
from explainability.plots import plot_radar, plot_embedding_map, plot_feature_table, plot_feature_comparison
from styles import CYBERPUNK_CSS, MUSIC_VISUALIZER, SIDEBAR_TOGGLE_BUTTON

# Page config
st.set_page_config(
    page_title="📊 Explainability - Why This Song?",
    page_icon="📊",
    layout="wide"
)


def get_data():
    """Load data from database (always fresh)."""
    return load_all_data(use_database=True)


def main():
    # Apply cyberpunk theme
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)
    st.markdown(SIDEBAR_TOGGLE_BUTTON, unsafe_allow_html=True)
    
    # Animated Header
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1 style='
            font-family: Orbitron, sans-serif;
            font-size: 3rem;
            background: linear-gradient(90deg, #00ffff, #ff00ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0,255,255,0.5);
        '>🧠 EXPLAINABILITY ENGINE</h1>
        <p style='color: #00ffff; font-size: 1.2rem; letter-spacing: 3px;'>
            UNDERSTAND WHY WITH AI REASONING
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='
        background: rgba(0,255,136,0.05);
        border-left: 3px solid #00ff88;
        padding: 15px 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    '>
        <p style='color: #ccc; margin: 0;'>
            ⚡ <b style='color: #00ff88;'>This is the winning feature.</b> Every recommendation is explainable.
            See exactly WHY two songs are similar — broken down by musical components.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = get_data()
    
    if data["is_mock"]:
        st.markdown("""
        <div style='background: rgba(255,165,0,0.1); border: 1px solid orange; 
                    border-radius: 10px; padding: 10px; text-align: center; margin-bottom: 20px;'>
            <span style='color: orange;'>⚠️ DEMO MODE</span> — Using auto-generated mock data
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # INPUTS - Select Seed + Recommendation
    # ============================================
    
    st.subheader("🎵 Select Songs to Compare")
    
    song_ids_list = [str(sid) for sid in data["song_ids"]]
    
    # Check if we have recommendations from the Recommender page
    has_session_recs = (
        "last_seed" in st.session_state and 
        "last_recommendations" in st.session_state
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Seed song selection
        default_seed_idx = 0
        if has_session_recs:
            try:
                default_seed_idx = song_ids_list.index(st.session_state["last_seed"])
            except ValueError:
                default_seed_idx = 0
        
        seed_song = st.selectbox(
            "**Seed Song** (the original):",
            options=song_ids_list,
            index=default_seed_idx,
            help="The song you started with."
        )
    
    with col2:
        # Recommendation selection
        # Get recommendations for the seed
        rec_ids, rec_scores = recommend_from_song(
            seed_song_id=seed_song,
            embeddings=data["embeddings"],
            song_ids=data["song_ids"],
            id_to_idx=data["id_to_idx"],
            k=10
        )
        
        # Use session state if available and matching
        if has_session_recs and st.session_state["last_seed"] == seed_song:
            rec_options = st.session_state["last_recommendations"]
        else:
            rec_options = rec_ids
        
        rec_song = st.selectbox(
            "**Recommended Song** (to explain):",
            options=rec_options,
            index=0,
            help="The recommendation to analyze."
        )
    
    # Generate button
    if st.button("🔍 Generate Full Explanation", type="primary", use_container_width=True):
        with st.spinner("Analyzing musical similarity..."):
            explanation = explain_pair(
                seed_id=seed_song,
                rec_id=rec_song,
                features_df=data["features_df"],
                embeddings=data["embeddings"],
                id_to_idx=data["id_to_idx"]
            )
        
        st.session_state["current_explanation"] = explanation
    
    st.markdown("---")
    
    # Display explanation if available
    if "current_explanation" in st.session_state:
        explanation = st.session_state["current_explanation"]
        
        # Only show if it matches current selection
        if explanation["seed_id"] == seed_song and explanation["rec_id"] == rec_song:
            display_explanation(explanation, data)
        else:
            st.info("👆 Click 'Generate Full Explanation' to analyze the selected pair.")
    else:
        st.info("👆 Select songs above and click 'Generate Full Explanation' to see why they're similar.")
    
    # Sidebar info
    st.sidebar.header("📖 How to Read Explanations")
    st.sidebar.markdown("""
    **Component Scores (0 to 1):**
    
    - **Tempo**: BPM alignment
    - **Timbre**: Instrument/vocal texture (MFCC)
    - **Brightness**: Tonal color (spectral centroid)
    - **Harmony**: Chord patterns (chroma)
    - **Energy**: Loudness (RMS)
    
    **Similarity Levels:**
    - 🟢 ≥ 0.85: Very similar
    - 🟡 ≥ 0.65: Moderately similar  
    - 🟠 ≥ 0.40: Somewhat different
    - 🔴 < 0.40: Quite different
    """)


def display_explanation(explanation: dict, data: dict):
    """Display the full explanation with all 4 components."""
    
    st.markdown(f"## Comparing: **{explanation['seed_id']}** → **{explanation['rec_id']}**")
    
    # Overall similarity metrics
    cos_sim = explanation.get("cosine_similarity")
    if cos_sim is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔢 Embedding Similarity",
                f"{cos_sim:.4f}",
                help="Cosine similarity in the learned embedding space"
            )
        
        with col2:
            component_scores = explanation.get("component_scores", {})
            if component_scores:
                avg_score = sum(component_scores.values()) / len(component_scores)
                st.metric(
                    "📊 Avg Feature Score",
                    f"{avg_score:.3f}",
                    help="Average of all feature-based similarity scores"
                )
        
        with col3:
            summary = get_similarity_summary(explanation)
            level = summary.split("(")[0].strip()
            st.metric("📋 Assessment", level)
    
    st.markdown("---")
    
    # ============================================
    # ALL 4 OUTPUTS - (A) Bullets, (B) Table, (C) Radar, (D) Map
    # ============================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 (A) Explanation Bullets",
        "📋 (B) Feature Comparison Table", 
        "🎯 (C) Radar Chart",
        "🗺️ (D) Embedding Map"
    ])
    
    with tab1:
        display_text_explanation(explanation)
    
    with tab2:
        display_feature_table_section(explanation, data)
    
    with tab3:
        display_radar_chart(explanation)
    
    with tab4:
        display_embedding_map(explanation, data)


def display_text_explanation(explanation: dict):
    """(A) Display explanation bullets - human-readable reasoning."""
    st.markdown("### 📝 Human-Readable Explanation")
    st.markdown("*Why are these songs similar? Here's the musical reasoning:*")
    
    # Explanation bullets - THE KEY OUTPUT
    reasons = explanation.get("reasons", [])
    if reasons:
        st.markdown("---")
        for reason in reasons:
            # Add checkmark emoji for positive matches
            if "similar" in reason.lower() or "match" in reason.lower() or "align" in reason.lower() or "close" in reason.lower():
                st.markdown(f"✅ {reason}")
            else:
                st.markdown(f"ℹ️ {reason}")
    
    # Overall assessment
    assessment = explanation.get("overall_assessment", "")
    if assessment:
        st.markdown("---")
        st.success(f"**📋 Summary:** {assessment}")
    
    # Component scores table
    component_scores = explanation.get("component_scores", {})
    if component_scores:
        st.markdown("---")
        st.markdown("#### Component Similarity Scores")
        
        scores_data = []
        for name, score in component_scores.items():
            display_name = name.replace("_score", "").replace("_", " ").title()
            
            # Determine level with emoji
            if score >= 0.85:
                level = "🟢 Very Similar"
            elif score >= 0.65:
                level = "🟡 Moderately Similar"
            elif score >= 0.40:
                level = "🟠 Somewhat Different"
            else:
                level = "🔴 Quite Different"
            
            scores_data.append({
                "Component": display_name,
                "Score": f"{score:.3f}",
                "Level": level
            })
        
        scores_df = pd.DataFrame(scores_data)
        st.dataframe(scores_df, use_container_width=True, hide_index=True)


def display_feature_table_section(explanation: dict, data: dict):
    """(B) Display feature comparison table."""
    st.markdown("### 📋 Feature Comparison Table")
    st.markdown("*Side-by-side comparison of audio features:*")
    
    seed_id = explanation["seed_id"]
    rec_id = explanation["rec_id"]
    
    # Feature comparison table
    comparison_table = plot_feature_table(seed_id, rec_id, data["features_df"])
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    
    # Also show feature deltas
    feature_deltas = explanation.get("feature_deltas", {})
    if feature_deltas:
        st.markdown("#### Raw Feature Differences")
        
        deltas_data = []
        for name, delta in feature_deltas.items():
            display_name = name.replace("_", " ").title()
            deltas_data.append({
                "Feature": display_name,
                "Difference": f"{delta:.3f}" if abs(delta) < 1000 else f"{delta:.0f}"
            })
        
        deltas_df = pd.DataFrame(deltas_data)
        st.dataframe(deltas_df, use_container_width=True, hide_index=True)


def display_radar_chart(explanation: dict):
    """(C) Display the radar chart visualization."""
    st.markdown("### 🎯 Similarity Radar Chart")
    st.markdown("*Visual comparison of similarity across 5 musical dimensions:*")
    
    component_scores = explanation.get("component_scores", {})
    
    if component_scores:
        fig = plot_radar(
            component_scores=component_scores,
            title=f"Similarity: {explanation['seed_id']} vs {explanation['rec_id']}"
        )
        st.pyplot(fig)
        
        st.markdown("""
        **Reading the Radar:**
        - Each axis = one musical dimension
        - Values closer to **1.0** (outer edge) = **more similar**
        - Values closer to **0.0** (center) = **less similar**
        
        *A well-matched song pair will have a large, filled radar shape.*
        """)
    else:
        st.warning("No component scores available for radar chart.")


def display_embedding_map(explanation: dict, data: dict):
    """(D) Display the embedding space map highlighting the pair."""
    st.markdown("### 🗺️ Embedding Map — Position in Music Space")
    st.markdown("*See where these two songs sit relative to each other:*")
    
    seed_id = explanation["seed_id"]
    rec_id = explanation["rec_id"]
    
    # Get other recommendations for context
    rec_ids, _ = recommend_from_song(
        seed_song_id=seed_id,
        embeddings=data["embeddings"],
        song_ids=data["song_ids"],
        id_to_idx=data["id_to_idx"],
        k=5
    )
    
    # Make sure the explained rec is included
    if rec_id not in rec_ids:
        rec_ids = [rec_id] + rec_ids[:4]
    
    fig = plot_embedding_map(
        embeddings=data["embeddings"],
        song_ids=data["song_ids"],
        seed_id=seed_id,
        rec_ids=rec_ids,
        id_to_idx=data["id_to_idx"],
        title=f"Music Space: {seed_id} → {rec_id}",
        show_labels=False
    )
    
    st.pyplot(fig)
    
    st.markdown("""
    **What you're seeing:**
    - ⭐ **Star** = Seed song (your starting point)
    - 🟧 **Orange squares** = Recommended songs (including the one being explained)
    - ⚫ **Gray dots** = All other songs
    - **Dashed lines** = Connections from seed to recommendations
    
    *Songs that are close in this space share similar audio characteristics!*
    """)


if __name__ == "__main__":
    main()
