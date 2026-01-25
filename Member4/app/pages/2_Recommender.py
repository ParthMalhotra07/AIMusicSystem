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
from integration.database import get_database
from styles import CYBERPUNK_CSS, MUSIC_VISUALIZER, SIDEBAR_TOGGLE_BUTTON, ICON_FIX_SCRIPT

# Page config
st.set_page_config(
    page_title="🎯 Recommender - AI Music",
    page_icon="🎯",
    layout="wide"
)


def get_data():
    """Load data from database (always fresh)."""
    return load_all_data(use_database=True)


def get_audio_path(song_id: str) -> str:
    """Get audio file path for a song if it exists."""
    try:
        db = get_database()
        song_info = db.get_song(song_id)
        
        # 1. Check database path
        if song_info and song_info.get('filepath'):
            filepath = song_info['filepath']
            if os.path.exists(filepath):
                return filepath
        
        # 2. Check standard locations
        extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        directories = [
            db.audio_dir,  # data/audio_uploads
            project_root / "data" / "audio_samples",
            Path("data/audio_uploads"),
            Path("data/audio_samples")
        ]
        
        for directory in directories:
            if not isinstance(directory, Path):
                directory = Path(directory)
            
            if not directory.exists():
                continue
                
            for ext in extensions:
                path = directory / f"{song_id}{ext}"
                if path.exists():
                    print(f"🎵 Found audio at: {path}")  # Debug print
                    return str(path)
                    
    except Exception as e:
        print(f"Error finding audio for {song_id}: {e}")
    return None


def render_audio_player(song_id: str, label: str = None):
    """Render an audio player for a song if audio file exists."""
    audio_path = get_audio_path(song_id)
    
    if label:
        st.markdown(f"<p style='color: #00ffff; margin-bottom: 5px;'>{label}</p>", unsafe_allow_html=True)
    
    if audio_path:
        try:
            # Debug: show path being used
            # st.caption(f"📂 {audio_path}")
            
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            if len(audio_bytes) == 0:
                st.error("Audio file is empty")
                return False
            
            # Determine mime type from extension
            ext = os.path.splitext(audio_path)[1].lower()
            if ext == '.wav':
                mime = 'audio/wav'
            elif ext == '.ogg':
                mime = 'audio/ogg'
            elif ext == '.flac':
                mime = 'audio/flac'
            elif ext == '.m4a':
                mime = 'audio/mp4'
            else:
                mime = 'audio/mpeg'  # Changed from audio/mp3 to audio/mpeg (more compatible)
                
            st.audio(audio_bytes, format=mime)
            return True
        except FileNotFoundError:
            st.error(f"File not found: {audio_path}")
            return False
        except PermissionError:
            st.error(f"Permission denied: {audio_path}")
            return False
        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")
            return False
            
    st.warning(f"Audio not found for {song_id}")
    return False


def check_audio_exists(song_id: str, audio_dir: str = None) -> str:
    """Check if audio file exists for a song."""
    return get_audio_path(song_id)


def main():
    # Apply cyberpunk theme
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)
    st.markdown(SIDEBAR_TOGGLE_BUTTON, unsafe_allow_html=True)
    st.markdown(ICON_FIX_SCRIPT, unsafe_allow_html=True)
    
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
    
    # Single seed mode (no mode selection - just simple interface)
    st.subheader("🎵 Single Seed Recommendation")
    st.markdown("Select a song to find similar tracks.")
    
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
    Recommendations are based on **audio feature similarity**.
    
    Songs with similar tempo, timbre, and energy will be ranked higher.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Search filter for songs
        st.markdown("<p style='color: #00ffff; margin-bottom: 5px;'>🔍 Filter songs:</p>", unsafe_allow_html=True)
        seed_search = st.text_input("Filter", placeholder="Search...", label_visibility="collapsed", key="seed_search")
        
        # Filter song list based on search
        if seed_search:
            filtered_seeds = [s for s in song_ids_list if seed_search.lower() in s.lower()]
            if not filtered_seeds:
                filtered_seeds = song_ids_list
        else:
            filtered_seeds = song_ids_list
        
        seed_song = st.selectbox(
            "Choose your seed song:",
            options=filtered_seeds,
            index=0,
            help="The song to base recommendations on."
        )
        # Audio player for seed song
        render_audio_player(seed_song, "🎧 Preview Seed Song")
    
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
                k=k_recs,
                feature_matrix=data.get("feature_matrix"),
                use_features=True  # Use raw audio features for better similarity
            )
        
        st.success(f"Found {len(rec_ids)} recommendations based on **{seed_song}**!")
        
        # Store in session state for explainability page
        st.session_state["last_seed"] = seed_song
        st.session_state["last_recommendations"] = rec_ids
        st.session_state["last_scores"] = rec_scores
        
        # Display recommendations
        display_recommendations(rec_ids, rec_scores, data)


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
                    
                    # Score is now 0-100 from adapter
                    score_val = float(score)
                    score_display = max(0.0, min(100.0, score_val)) / 100.0
                    
                    st.progress(score_display)
                    st.caption(f"**Avg Feature Score:** {score_val:.1f}%")
                    
                    # --- MATCH TAGS (EXPLAINABILITY) ---
                    tags = []
                    seed_id = st.session_state.get("last_seed")
                    if seed_id:
                        seed_row = data["features_df"][data["features_df"]["song_id"] == seed_id]
                        rec_row = data["features_df"][data["features_df"]["song_id"] == rec_id]
                        
                        if not seed_row.empty and not rec_row.empty:
                            s_tempo = seed_row.iloc[0].get('tempo', 0)
                            r_tempo = rec_row.iloc[0].get('tempo', 0)
                            s_rms = seed_row.iloc[0].get('rms', 0)
                            r_rms = rec_row.iloc[0].get('rms', 0)
                            
                            # Tempo Check
                            if abs(s_tempo - r_tempo) < 5:
                                tags.append("🥁 Rhythm Twin")
                            elif abs(s_tempo - r_tempo) < 15:
                                tags.append("🎵 Similar Vibe")
                                
                            # Energy Check
                            if abs(s_rms - r_rms) < 0.05:
                                tags.append("⚡ Energy Match")
                            elif r_rms > 0.2:
                                tags.append("🔥 High Intensity")
                    
                    if tags:
                        tag_html = " ".join([f"<span style='background:rgba(255,0,255,0.2); color:#ff00ff; padding:2px 6px; border-radius:4px; font-size:0.8em; margin-right:4px;'>{t}</span>" for t in tags[:2]])
                        st.markdown(f"<div style='margin-bottom:8px;'>{tag_html}</div>", unsafe_allow_html=True)

                    # Song details
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
                    
                    # Audio player from database
                    render_audio_player(rec_id, "▶️ Play")
                    
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
