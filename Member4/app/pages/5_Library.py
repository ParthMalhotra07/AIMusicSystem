"""
Library Page - Browse and Play Your Saved Songs

View all uploaded songs and play them directly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from integration.database import get_database
from styles import CYBERPUNK_CSS, SIDEBAR_TOGGLE_BUTTON

# Page config
st.set_page_config(
    page_title="📚 Library - Your Music",
    page_icon="📚",
    layout="wide"
)


def get_audio_path(song_id: str) -> str:
    """Get audio file path for a song if it exists."""
    try:
        db = get_database()
        song_info = db.get_song(song_id)
        if song_info and song_info.get('filepath'):
            filepath = song_info['filepath']
            if os.path.exists(filepath):
                return filepath
    except Exception:
        pass
    return None


def render_audio_player(song_id: str):
    """Render an audio player for a song if audio file exists."""
    audio_path = get_audio_path(song_id)
    if audio_path:
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format='audio/mp3')
        return True
    return False


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
            background: linear-gradient(90deg, #ff00ff, #00ffff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(255,0,255,0.5);
        '>📚 YOUR MUSIC LIBRARY</h1>
        <p style='
            color: #ff00ff;
            font-size: 1.2rem;
            letter-spacing: 3px;
        '>BROWSE AND PLAY YOUR UPLOADED SONGS</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all songs from database
    db = get_database()
    all_songs = db.get_all_songs()
    
    # Filter to only uploaded songs with audio files
    playable_songs = [s for s in all_songs if s.get('filepath') and os.path.exists(str(s.get('filepath', '')))]
    
    if not playable_songs:
        st.markdown("""
        <div style='
            background: rgba(255,165,0,0.1);
            border: 1px solid orange;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            margin: 50px 0;
        '>
            <h3 style='color: orange;'>🎵 No Songs in Library</h3>
            <p style='color: #ccc;'>Upload some songs on the <b>Upload</b> page to start building your library!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Stats
    st.markdown(f"""
    <div style='
        background: rgba(0,255,255,0.05);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    '>
        <span style='color: #00ffff; font-size: 1.2rem;'>
            🎶 <b>{len(playable_songs)}</b> songs in your library
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Search filter
    search = st.text_input("🔍 Search songs", placeholder="Type to filter...")
    
    # Filter songs
    if search:
        filtered_songs = [s for s in playable_songs if search.lower() in s['song_id'].lower() or search.lower() in s.get('filename', '').lower()]
    else:
        filtered_songs = playable_songs
    
    if not filtered_songs:
        st.warning("No songs match your search.")
        return
    
    st.markdown("---")
    
    # Display songs in a grid
    cols_per_row = 2
    
    for i in range(0, len(filtered_songs), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(filtered_songs):
                break
            
            song = filtered_songs[idx]
            song_id = song['song_id']
            
            with col:
                # Song card
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, rgba(0,0,0,0.5), rgba(30,30,50,0.5));
                    border: 1px solid #ff00ff;
                    border-radius: 15px;
                    padding: 20px;
                    margin-bottom: 15px;
                '>
                    <h3 style='
                        color: #00ffff;
                        font-family: Orbitron;
                        margin: 0 0 10px 0;
                        font-size: 1.1rem;
                    '>🎵 {song_id}</h3>
                    <p style='color: #888; font-size: 0.9rem; margin: 5px 0;'>
                        📁 {song.get('filename', 'Unknown')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Song info in columns
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    duration = song.get('duration_sec')
                    if duration:
                        mins = int(duration // 60)
                        secs = int(duration % 60)
                        st.markdown(f"<span style='color: #00ff88;'>⏱️ {mins}:{secs:02d}</span>", unsafe_allow_html=True)
                    
                    tempo = song.get('tempo')
                    if tempo:
                        st.markdown(f"<span style='color: #00ffff;'>🥁 {tempo:.0f} BPM</span>", unsafe_allow_html=True)
                
                with info_col2:
                    key = song.get('key_name')
                    if key:
                        st.markdown(f"<span style='color: #ff00ff;'>🎹 Key: {key}</span>", unsafe_allow_html=True)
                    
                    upload_date = song.get('upload_date', '')
                    if upload_date:
                        date_str = upload_date.split('T')[0] if 'T' in upload_date else upload_date[:10]
                        st.markdown(f"<span style='color: #888;'>📅 {date_str}</span>", unsafe_allow_html=True)
                
                # Audio player
                st.markdown("<br>", unsafe_allow_html=True)
                render_audio_player(song_id)
                
                st.markdown("---")
    
    # Delete song section
    st.markdown("---")
    with st.expander("🗑️ Manage Library"):
        st.warning("⚠️ Deleting a song removes it from the database and deletes the audio file permanently.")
        
        song_to_delete = st.selectbox(
            "Select song to delete:",
            options=[s['song_id'] for s in playable_songs],
            key="delete_song_select"
        )
        
        if st.button("🗑️ Delete Song", type="secondary"):
            try:
                db.delete_song(song_to_delete)
                st.success(f"Deleted {song_to_delete}")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting song: {e}")


if __name__ == "__main__":
    main()
