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
from styles import CYBERPUNK_CSS, SIDEBAR_TOGGLE_BUTTON, ICON_FIX_SCRIPT

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


def render_audio_player(song_id: str):
    """Render an audio player for a song if audio file exists."""
    audio_path = get_audio_path(song_id)
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
    
    # Search filter with styled container
    st.markdown("""
    <div style='
        background: rgba(0,255,255,0.05);
        border: 1px solid rgba(0,255,255,0.3);
        border-radius: 10px;
        padding: 10px 15px;
        margin-bottom: 15px;
    '>
        <p style='color: #00ffff; margin: 0 0 5px 0; font-size: 0.9rem;'>🔍 Search your library</p>
    </div>
    """, unsafe_allow_html=True)
    search = st.text_input("🔍 Search songs", placeholder="Search...", label_visibility="collapsed")
    
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
    
    # Manage library section
    st.markdown("---")
    with st.expander("⚙️ Manage Library"):
        tab1, tab2 = st.tabs(["✏️ Rename Song", "🗑️ Delete Song"])
        
        with tab1:
            st.markdown("Rename a song to give it a custom display name.")
            
            song_to_rename = st.selectbox(
                "Select song to rename:",
                options=[s['song_id'] for s in playable_songs],
                key="rename_song_select"
            )
            
            # Get current name
            current_song = next((s for s in playable_songs if s['song_id'] == song_to_rename), None)
            current_name = current_song.get('filename', song_to_rename) if current_song else song_to_rename
            
            new_name = st.text_input(
                "New display name:",
                value=current_name,
                key="new_song_name"
            )
            
            if st.button("💾 Save Name", type="primary"):
                if new_name and new_name != current_name:
                    try:
                        db.update_song_name(song_to_rename, new_name)
                        st.success(f"Renamed to: {new_name}")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error renaming: {e}")
                else:
                    st.warning("Please enter a different name.")
        
        with tab2:
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
