"""
Discover Page - Explore the Music Embedding Space

Interactive PCA visualization of the audio embedding space.
Proves that songs cluster naturally based on audio features.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np
import os

from integration.load_data import load_all_data
from integration.recommender_adapter import recommend_from_song
from integration.database import get_database
from explainability.plots import plot_embedding_map
from styles import CYBERPUNK_CSS, MUSIC_VISUALIZER, SIDEBAR_TOGGLE_BUTTON, ICON_FIX_SCRIPT

# Page config
st.set_page_config(
    page_title="🔍 Discover - Music Space",
    page_icon="🔍",
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
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
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
                mime = 'audio/mpeg'  # More compatible than audio/mp3
                
            st.audio(audio_bytes, format=mime)
            return True
        except Exception as e:
            st.error(f"Playback error: {e}")
            return False
            
    st.warning(f"Audio not found for {song_id}")
    return False


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
            background: linear-gradient(90deg, #00ff88, #00ffff, #ff00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0,255,255,0.5);
            animation: glow 2s ease-in-out infinite alternate;
        '>🔍 DISCOVER: THE MUSIC MATRIX</h1>
        <p style='
            color: #00ffff;
            font-size: 1.2rem;
            letter-spacing: 3px;
        '>NAVIGATE THE AUDIO EMBEDDING SPACE</p>
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
            ⚡ <b style='color: #00ff88;'>Songs cluster naturally</b> based on audio similarity — no tags needed.
            Select a song below to see its position in the embedding space.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = get_data()
    
    if data["is_mock"]:
        st.markdown("""
        <div style='
            background: rgba(255,165,0,0.1);
            border: 1px solid orange;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            margin-bottom: 20px;
        '>
            <span style='color: orange;'>⚠️ DEMO MODE</span> — Using auto-generated mock data
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar controls with cyberpunk styling
    st.sidebar.markdown("""
    <h2 style='
        color: #00ffff;
        font-family: Orbitron;
        text-align: center;
        border-bottom: 2px solid #00ffff;
        padding-bottom: 10px;
    '>🎛️ CONTROLS</h2>
    """, unsafe_allow_html=True)
    
    # Song selection
    song_ids_list = [str(sid) for sid in data["song_ids"]]
    
    # Add search filter for easier song finding
    st.sidebar.markdown("<p style='color: #00ffff; margin: 15px 0 5px 0;'>🔍 Filter songs:</p>", unsafe_allow_html=True)
    song_search = st.sidebar.text_input("Filter", placeholder="Search...", label_visibility="collapsed", key="discover_search")
    
    # Filter song list based on search
    if song_search:
        filtered_songs = [s for s in song_ids_list if song_search.lower() in s.lower()]
        if not filtered_songs:
            st.sidebar.warning("No songs match your filter")
            filtered_songs = song_ids_list
    else:
        filtered_songs = song_ids_list
    
    selected_song = st.sidebar.selectbox(
        "Select a song to explore:",
        options=filtered_songs,
        index=0,
        help="Choose a song to see its position in the embedding space and nearest neighbors."
    )
    
    # Number of neighbors
    k_neighbors = st.sidebar.slider(
        "Number of neighbors (K):",
        min_value=3,
        max_value=20,
        value=10,
        help="How many nearest neighbors to highlight."
    )
    
    # Show labels option
    show_labels = st.sidebar.checkbox(
        "Show song labels",
        value=False,
        help="Display song IDs on the plot (can be cluttered with many songs)."
    )
    
    st.sidebar.markdown("---")
    
    # Get nearest neighbors using feature-based similarity (same as Recommender)
    neighbor_ids, neighbor_scores = recommend_from_song(
        seed_song_id=selected_song,
        embeddings=data["embeddings"],
        song_ids=data["song_ids"],
        id_to_idx=data["id_to_idx"],
        k=k_neighbors,
        return_scores=True,
        feature_matrix=data.get("feature_matrix"),
        use_features=True
    )
    
    # ============================================
    # (A) EMBEDDING MAP - 2D Scatter Plot
    # ============================================
    
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.markdown("""
        <div style='
            background: rgba(0,255,255,0.05);
            border: 1px solid #00ffff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        '>
            <h3 style='color: #00ffff; font-family: Orbitron; margin: 0;'>
                🗺️ (A) EMBEDDING MAP — 2D VISUALIZATION
            </h3>
            <p style='color: #aaa; margin: 5px 0 0 0;'>Each dot = a song. Songs close together sound similar.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display the embedding map
        fig = plot_embedding_map(
            embeddings=data["embeddings"],
            song_ids=data["song_ids"],
            seed_id=selected_song,
            rec_ids=neighbor_ids,
            id_to_idx=data["id_to_idx"],
            title=f"Music Embedding Space — Selected: {selected_song}",
            show_labels=show_labels
        )
        
        st.pyplot(fig)
        
        st.markdown("""
        <div style='
            background: rgba(255,0,255,0.05);
            border-radius: 10px;
            padding: 15px;
        '>
            <p style='color: #ff00ff; margin-bottom: 10px;'><b>LEGEND:</b></p>
            <p style='color: #ccc; margin: 5px 0;'>⭐ <b style='color: #ff00ff;'>Star</b> = Selected song</p>
            <p style='color: #ccc; margin: 5px 0;'>🟧 <b style='color: orange;'>Orange squares</b> = Nearest neighbors</p>
            <p style='color: #ccc; margin: 5px 0;'>⚫ <b style='color: #666;'>Gray dots</b> = All other songs</p>
            <p style='color: #00ff88; font-size: 0.9rem; margin-top: 15px;'>
                <i>Notice how similar songs cluster together — this is the power of audio-based embeddings!</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # (B) SIMILAR SONGS LIST - Top K Nearest
    # ============================================
    
    with col_side:
        st.markdown(f"""
        <div style='
            background: rgba(255,0,255,0.05);
            border: 1px solid #ff00ff;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        '>
            <h3 style='color: #ff00ff; font-family: Orbitron; margin: 0;'>
                🎯 (B) TOP {k_neighbors} SIMILAR
            </h3>
            <p style='color: #aaa; margin: 5px 0 0 0;'>Ranked by cosine similarity</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display neighbors as a styled list
        if neighbor_ids and neighbor_scores:
            for i, (nid, score) in enumerate(zip(neighbor_ids, neighbor_scores), 1):
                # Get song features for display
                song_row = data["features_df"][data["features_df"]["song_id"] == nid]
                
                # Neon-styled similarity card
                neon_color = "#00ff88" if score > 0.8 else "#00ffff" if score > 0.6 else "#ff00ff"
                
                st.markdown(f"""
                <div style='
                    background: rgba(0,0,0,0.3);
                    border-left: 3px solid {neon_color};
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                '>
                    <span style='color: {neon_color}; font-weight: bold;'>#{i}</span>
                    <span style='color: #fff;'> {nid}</span>
                    <div style='
                        background: rgba(0,255,255,0.2);
                        border-radius: 5px;
                        margin-top: 5px;
                        overflow: hidden;
                    '>
                        <div style='
                            width: {score*100}%;
                            background: linear-gradient(90deg, #00ffff, {neon_color});
                            height: 8px;
                        '></div>
                    </div>
                    <span style='color: #aaa; font-size: 0.8rem;'>Similarity: {score:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Add audio player for each similar song
                render_audio_player(nid)
        else:
            st.warning("No neighbors found.")
    
    # Additional insights
    st.markdown("---")
    st.subheader("📊 Selected Song Details")
    
    # Get selected song features
    selected_row = data["features_df"][data["features_df"]["song_id"] == selected_song]
    
    if len(selected_row) > 0:
        selected_row = selected_row.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tempo = selected_row.get("tempo", "N/A")
            if tempo != "N/A":
                st.metric("🥁 Tempo", f"{tempo:.1f} BPM")
            else:
                st.metric("🥁 Tempo", "N/A")
        
        with col2:
            centroid = selected_row.get("spectral_centroid", "N/A")
            if centroid != "N/A":
                st.metric("✨ Brightness", f"{centroid:.0f} Hz")
            else:
                st.metric("✨ Brightness", "N/A")
        
        with col3:
            rms = selected_row.get("rms", selected_row.get("rms_mean", "N/A"))
            if rms != "N/A":
                st.metric("⚡ Energy (RMS)", f"{rms:.3f}")
            else:
                st.metric("⚡ Energy", "N/A")
        
        with col4:
            duration = selected_row.get("duration_sec", "N/A")
            if duration != "N/A":
                mins = int(duration // 60)
                secs = int(duration % 60)
                st.metric("⏱️ Duration", f"{mins}:{secs:02d}")
            else:
                st.metric("⏱️ Duration", "N/A")
        
        # Audio player for selected song
        st.markdown("""
        <div style='
            background: rgba(0,255,136,0.05);
            border: 1px solid #00ff88;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
        '>
            <h4 style='color: #00ff88; margin: 0 0 10px 0;'>🎵 Play Selected Song</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if not render_audio_player(selected_song):
            st.info("Audio file not available for this song. Upload songs to enable playback.")
    
    # Statistics about embedding space
    st.markdown("---")
    with st.expander("📈 Embedding Space Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Embedding Properties:**")
            st.write(f"- Number of songs: {data['num_songs']}")
            st.write(f"- Embedding dimension: {data['embedding_dim']}")
            
            # Compute some statistics
            embeddings = data["embeddings"]
            norms = np.linalg.norm(embeddings, axis=1)
            st.write(f"- Avg embedding norm: {norms.mean():.3f}")
            st.write(f"- Norm std dev: {norms.std():.3f}")
        
        with col2:
            st.markdown("**Similarity Distribution:**")
            
            # Sample pairwise similarities
            sample_size = min(100, len(embeddings))
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_emb = embeddings[indices]
            
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(sample_emb)
            
            # Get upper triangle (excluding diagonal)
            upper_tri = sim_matrix[np.triu_indices(sample_size, k=1)]
            
            st.write(f"- Mean similarity: {upper_tri.mean():.3f}")
            st.write(f"- Std similarity: {upper_tri.std():.3f}")
            st.write(f"- Min similarity: {upper_tri.min():.3f}")
            st.write(f"- Max similarity: {upper_tri.max():.3f}")


if __name__ == "__main__":
    main()
