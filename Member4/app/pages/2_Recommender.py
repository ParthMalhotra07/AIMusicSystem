"""
Recommender Page - Get Audio-Based Music Recommendations

Supports single seed track and multi-song listening history modes.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from integration.load_data import load_all_data
from integration.recommender_adapter import recommend_from_song, recommend_from_history
from styles import CYBERPUNK_CSS, MUSIC_VISUALIZER

# Page config
st.set_page_config(
    page_title="🎯 Recommender - AI Music",
    page_icon="🎯",
    layout="wide"
)


@st.cache_data
def get_data():
    """Load and cache all data."""
    return load_all_data()


def check_audio_exists(song_id: str, audio_dir: str = "data/audio_samples") -> str:
    """Check if audio file exists for a song."""
    extensions = [".wav", ".mp3", ".ogg", ".flac"]
    for ext in extensions:
        path = os.path.join(project_root, audio_dir, f"{song_id}{ext}")
        if os.path.exists(path):
            return path
    return None


def main():
    # Apply cyberpunk theme
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)
    
    # Animated Header
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1 style='
            font-family: Orbitron, sans-serif;
            font-size: 3rem;
            background: linear-gradient(90deg, #ff00ff, #00ffff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(255,0,255,0.5);
        '>🎯 MUSIC RECOMMENDER</h1>
        <p style='color: #ff00ff; font-size: 1.2rem; letter-spacing: 3px;'>
            AI-POWERED AUDIO SIMILARITY ENGINE
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = get_data()
    
    if data["is_mock"]:
        st.markdown("""
        <div style='background: rgba(255,165,0,0.1); border: 1px solid orange; 
                    border-radius: 10px; padding: 10px; text-align: center; margin-bottom: 20px;'>
            <span style='color: orange;'>⚠️ DEMO MODE</span> — Using mock demonstration data
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendation mode selection
    mode = st.radio(
        "**Recommendation Mode:**",
        options=["Single Seed Track", "Listening History"],
        horizontal=True,
        help="Choose how to generate recommendations."
    )
    
    st.markdown("---")
    
    # Song list for selection
    song_ids_list = [str(sid) for sid in data["song_ids"]]
    
    # Number of recommendations
    k_recs = st.sidebar.slider(
        "Number of recommendations (K):",
        min_value=3,
        max_value=25,
        value=10,
        help="How many songs to recommend."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    Recommendations are based on **cosine similarity** 
    in the learned audio embedding space.
    
    - **Single Seed**: Find songs similar to one track
    - **Listening History**: Find songs similar to your taste profile
    """)
    
    if mode == "Single Seed Track":
        # Single seed mode
        st.subheader("🎵 Single Seed Recommendation")
        st.markdown("Select a song to find similar tracks.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            seed_song = st.selectbox(
                "Choose your seed song:",
                options=song_ids_list,
                index=0,
                help="The song to base recommendations on."
            )
        
        with col2:
            # Show seed song info
            seed_row = data["features_df"][data["features_df"]["song_id"] == seed_song]
            if len(seed_row) > 0:
                seed_row = seed_row.iloc[0]
                tempo = seed_row.get("tempo", "N/A")
                st.metric("Seed Tempo", f"{tempo:.0f} BPM" if tempo != "N/A" else "N/A")
        
        # Generate recommendations
        if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Finding similar songs..."):
                rec_ids, rec_scores = recommend_from_song(
                    seed_song_id=seed_song,
                    embeddings=data["embeddings"],
                    song_ids=data["song_ids"],
                    id_to_idx=data["id_to_idx"],
                    k=k_recs
                )
            
            st.success(f"Found {len(rec_ids)} recommendations based on **{seed_song}**!")
            
            # Store in session state for explainability page
            st.session_state["last_seed"] = seed_song
            st.session_state["last_recommendations"] = rec_ids
            st.session_state["last_scores"] = rec_scores
            
            # Display recommendations
            display_recommendations(rec_ids, rec_scores, data)
    
    else:
        # Listening history mode
        st.subheader("📚 Listening History Recommendation")
        st.markdown("Select multiple songs to build your taste profile.")
        
        # Multi-select for history
        history_songs = st.multiselect(
            "Build your listening history:",
            options=song_ids_list,
            default=song_ids_list[:3] if len(song_ids_list) >= 3 else song_ids_list[:1],
            help="Select songs you've enjoyed. Recommendations will match your combined taste."
        )
        
        if history_songs:
            st.markdown(f"**Selected {len(history_songs)} songs** for taste profile.")
            
            # Show selected songs in a compact view
            with st.expander("View selected songs"):
                for song_id in history_songs:
                    song_row = data["features_df"][data["features_df"]["song_id"] == song_id]
                    if len(song_row) > 0:
                        tempo = song_row.iloc[0].get("tempo", "N/A")
                        tempo_str = f"{tempo:.0f} BPM" if tempo != "N/A" else ""
                        st.write(f"• **{song_id}** {tempo_str}")
            
            # Generate recommendations
            if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
                with st.spinner("Analyzing your taste profile..."):
                    rec_ids, rec_scores = recommend_from_history(
                        history_song_ids=history_songs,
                        embeddings=data["embeddings"],
                        song_ids=data["song_ids"],
                        id_to_idx=data["id_to_idx"],
                        k=k_recs
                    )
                
                st.success(
                    f"Found {len(rec_ids)} recommendations based on "
                    f"your {len(history_songs)}-song listening history!"
                )
                
                # Store in session state
                st.session_state["last_seed"] = history_songs[0]  # Use first as reference
                st.session_state["last_history"] = history_songs
                st.session_state["last_recommendations"] = rec_ids
                st.session_state["last_scores"] = rec_scores
                
                # Display recommendations
                display_recommendations(rec_ids, rec_scores, data)
        else:
            st.warning("Please select at least one song to build your listening history.")


def display_recommendations(rec_ids, rec_scores, data):
    """Display recommendations in a clean format."""
    st.markdown("---")
    st.subheader("🎶 Your Recommendations")
    
    # Create columns for cards
    cols_per_row = 2
    
    for i in range(0, len(rec_ids), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(rec_ids):
                break
            
            rec_id = rec_ids[idx]
            score = rec_scores[idx] if rec_scores else 0
            
            with col:
                with st.container():
                    # Card header
                    st.markdown(f"### #{idx + 1} {rec_id}")
                    
                    # Similarity score bar - clamp to [0, 1] range for progress bar
                    # Cosine similarity can be negative, so we normalize it
                    score_display = max(0.0, min(1.0, score))  # Clamp to valid range
                    score_pct = int(score_display * 100)
                    st.progress(score_display, text=f"Similarity: {score:.3f}")
                    
                    # Song features
                    song_row = data["features_df"][data["features_df"]["song_id"] == rec_id]
                    if len(song_row) > 0:
                        song_row = song_row.iloc[0]
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            tempo = song_row.get("tempo", "N/A")
                            st.write(f"🥁 **Tempo:** {tempo:.0f} BPM" if tempo != "N/A" else "🥁 Tempo: N/A")
                        
                        with col_b:
                            centroid = song_row.get("spectral_centroid", "N/A")
                            st.write(f"✨ **Brightness:** {centroid:.0f}" if centroid != "N/A" else "✨ Brightness: N/A")
                    
                    # Check for audio playback
                    audio_path = check_audio_exists(rec_id)
                    if audio_path:
                        st.audio(audio_path, format="audio/wav")
                    
                    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🔗 Next Steps")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Explore Explanations:**
        
        Go to the **Explainability** page to understand 
        why these songs were recommended.
        """)
    
    with col2:
        st.markdown("""
        **Refine Results:**
        
        Adjust K or try different seed songs to explore 
        more of the music space.
        """)


if __name__ == "__main__":
    main()
