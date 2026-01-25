"""
Upload Page - Add New Songs to the System

End-to-end pipeline:
1. Upload MP3/WAV file
2. Extract audio features (Member1)
3. Generate embeddings (Member2)
4. Store in database
5. Make available for recommendations
"""

import sys
import os
from pathlib import Path
import tempfile
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np

# Import integration modules
from integration.database import get_database, SongDatabase
from integration.audio_pipeline import get_pipeline, check_pipeline_status, HAS_MEMBER1, HAS_MEMBER2
from styles import CYBERPUNK_CSS, SIDEBAR_TOGGLE_BUTTON

# Page config
st.set_page_config(
    page_title="🎵 Upload Songs - AI Music",
    page_icon="📤",
    layout="wide"
)


def show_pipeline_status():
    """Display pipeline component status."""
    status = check_pipeline_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if status['member1_available']:
            st.success("✅ Feature Extractor")
        else:
            st.error("❌ Feature Extractor")
    
    with col2:
        if status['member2_available']:
            st.success("✅ Embedding Model")
        else:
            st.error("❌ Embedding Model")
    
    with col3:
        st.success("✅ Database")
    
    with col4:
        if status.get('fast_mode_ready', False):
            st.success("⚡ Fast Mode")
        elif all(status.values()):
            st.success("✅ Pipeline Ready")
        else:
            st.warning("⚠️ Limited Mode")


def process_uploaded_file(uploaded_file, db: SongDatabase) -> dict:
    """
    Process an uploaded audio file through the fast pipeline.
    
    Uses optimized fast mode (~3-5 seconds vs 15-30 seconds).
    
    Returns dict with status and details.
    """
    result = {
        'success': False,
        'song_id': None,
        'message': '',
        'details': {},
        'processing_time': 0
    }
    
    import time
    start_time = time.time()
    
    try:
        # Get pipeline (uses fast mode by default)
        pipeline = get_pipeline()
        
        # Get existing features for PCA fallback
        existing_features, _ = db.get_all_features()
        if len(existing_features) == 0:
            existing_features = None
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            # Process through fast pipeline
            processed = pipeline.process_audio(tmp_path, existing_features)
            
            # Read audio data for storage
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            # Add to database
            song_id = db.add_song(
                filename=uploaded_file.name,
                features=processed['features'],
                embedding=processed['embedding'],
                metadata=processed['metadata'],
                audio_data=audio_data
            )
            
            processing_time = time.time() - start_time
            
            result['success'] = True
            result['song_id'] = song_id
            result['message'] = f"Processed in {processing_time:.1f} seconds!"
            result['processing_time'] = processing_time
            result['details'] = {
                'filename': uploaded_file.name,
                'duration': processed['metadata'].get('duration_seconds', 0),
                'tempo': processed['metadata'].get('tempo_bpm', 0),
                'key': processed['metadata'].get('key', 'Unknown'),
                'feature_dim': len(processed['features']),
                'embedding_dim': len(processed['embedding'])
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        result['message'] = f"Error processing file: {str(e)}"
        import traceback
        result['details']['error'] = traceback.format_exc()
    
    return result


def show_database_stats(db: SongDatabase):
    """Display database statistics."""
    songs = db.get_all_songs()
    
    st.markdown("### 📊 Database Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Songs", len(songs))
    
    with col2:
        user_uploaded = sum(1 for s in songs if s.get('is_user_uploaded', False))
        st.metric("User Uploaded", user_uploaded)
    
    with col3:
        imported = len(songs) - user_uploaded
        st.metric("Pre-loaded", imported)


def show_song_library(db: SongDatabase):
    """Display the song library."""
    songs = db.get_all_songs()
    
    if not songs:
        st.info("📭 No songs in database. Upload some songs to get started!")
        return
    
    st.markdown("### 🎵 Song Library")
    
    # Create a nice table view
    for i, song in enumerate(songs[:20]):  # Show first 20
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            
            with col1:
                icon = "📤" if song.get('is_user_uploaded') else "📀"
                st.markdown(f"{icon} **{song['filename']}**")
            
            with col2:
                if song.get('tempo'):
                    st.markdown(f"🎵 {song['tempo']:.1f} BPM")
                else:
                    st.markdown("🎵 --")
            
            with col3:
                if song.get('duration_sec'):
                    mins = int(song['duration_sec'] // 60)
                    secs = int(song['duration_sec'] % 60)
                    st.markdown(f"⏱️ {mins}:{secs:02d}")
                else:
                    st.markdown("⏱️ --")
            
            with col4:
                st.markdown(f"🆔 `{song['song_id'][:15]}...`")
            
            with col5:
                if st.button("🗑️", key=f"delete_{song['song_id']}", help="Delete song"):
                    db.delete_song(song['song_id'])
                    st.rerun()
            
            st.markdown("---")
    
    if len(songs) > 20:
        st.info(f"Showing 20 of {len(songs)} songs")


def main():
    # Apply cyberpunk theme
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)
    st.markdown(SIDEBAR_TOGGLE_BUTTON, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1 style='
            font-family: Orbitron, sans-serif;
            font-size: 3rem;
            background: linear-gradient(90deg, #00ff88, #00ffff, #ff00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        '>📤 UPLOAD SONGS</h1>
        <p style='color: #00ffff; font-size: 1.2rem; letter-spacing: 3px;'>
            ADD YOUR MUSIC TO THE AI RECOMMENDATION ENGINE
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize database
    db = get_database()
    
    # Show pipeline status
    st.markdown("### ⚡ System Status")
    show_pipeline_status()
    
    st.markdown("---")
    
    # Upload section
    st.markdown("### 📁 Upload New Song")
    
    st.markdown("""
    <div style='background: rgba(0,255,255,0.1); border: 1px solid #00ffff; 
                border-radius: 10px; padding: 15px; margin-bottom: 20px;'>
        <p style='color: #00ffff; margin: 0;'>
            <strong>⚡ FAST MODE:</strong> Processes in ~3-5 seconds (analyzes first 30s)<br>
            <strong>Supported formats:</strong> MP3, WAV, FLAC, OGG, M4A<br>
            <strong>Features extracted:</strong> MFCCs, spectral, tempo, chroma, energy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'flac', 'ogg', 'm4a'],
        help="Upload an audio file to add to the recommendation system"
    )
    
    if uploaded_file is not None:
        st.markdown(f"**Selected file:** {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            process_btn = st.button("⚡ Process & Add", type="primary", use_container_width=True)
        
        if process_btn:
            # Simple spinner - processing is fast now
            with st.spinner("⚡ Processing audio (fast mode ~3-5s)..."):
                # Process the file
                result = process_uploaded_file(uploaded_file, db)
            # Show result
            if result['success']:
                st.success(f"✅ {result['message']}")
                
                # Show details
                details = result['details']
                st.markdown("#### 📋 Song Details")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⏱️ Process Time", f"{result.get('processing_time', 0):.1f}s")
                with col2:
                    st.metric("Duration", f"{details.get('duration', 0):.1f}s")
                with col3:
                    st.metric("Tempo", f"{details.get('tempo', 0):.1f} BPM")
                with col4:
                    st.metric("Features", details.get('feature_dim', 0))
                
                st.info(f"🆔 Song ID: `{result['song_id']}`")
                
                # Success message - song is now available everywhere
                st.markdown("""
                <div style='background: rgba(0,255,136,0.2); border: 1px solid #00ff88; 
                            border-radius: 10px; padding: 15px; margin: 15px 0;'>
                    <p style='color: #00ff88; margin: 0; font-weight: bold;'>
                        ✨ Song saved to database!
                    </p>
                    <p style='color: #ccc; margin: 5px 0 0 0;'>
                        Your song is now available on <strong>Discover</strong> and <strong>Recommender</strong> pages.
                        It will persist even after you close the app!
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Refresh button
                if st.button("🔄 Upload Another"):
                    st.rerun()
            else:
                st.error(f"❌ {result['message']}")
                if 'error' in result.get('details', {}):
                    with st.expander("Show error details"):
                        st.code(result['details']['error'])
    
    st.markdown("---")
    
    # Database stats
    show_database_stats(db)
    
    st.markdown("---")
    
    # Song library
    show_song_library(db)
    
    # Sidebar info
    st.sidebar.markdown("### ⚡ Fast Mode Info")
    st.sidebar.markdown("""
    **Processing optimizations:**
    
    - 🎵 Only first **30 seconds** analyzed
    - 📊 **60 key features** extracted
    - ⚡ **~3-5 seconds** per song
    
    **Features extracted:**
    - MFCCs (timbre)
    - Spectral features
    - Tempo & rhythm
    - Chroma (harmony)
    - Energy patterns
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 How it works")
    st.sidebar.markdown("""
    1. 🎵 **Upload** - Select audio file
    2. ⚡ **Extract** - Fast feature analysis
    3. 🧠 **Embed** - Generate embeddings  
    4. 💾 **Store** - Save to database
    5. 🎯 **Ready** - Recommend!
    """)
    
    st.sidebar.markdown("---")
    
    # Database management
    st.sidebar.markdown("### 🛠️ Database Tools")
    
    if st.sidebar.button("📥 Import Existing Data"):
        main_data_dir = project_root.parent / "data"
        local_data_dir = project_root / "data"
        
        # Try main data folder first
        if (main_data_dir / "song_features.npy").exists():
            data_dir = main_data_dir
        elif (local_data_dir / "song_features.npy").exists():
            data_dir = local_data_dir
        else:
            data_dir = None
        
        if data_dir:
            with st.spinner("Importing..."):
                db.import_from_numpy(
                    str(data_dir / "song_features.npy"),
                    str(data_dir / "song_embeddings.npy"),
                    str(data_dir / "song_ids.npy"),
                    str(data_dir / "song_features.csv")
                )
            st.sidebar.success("✅ Data imported!")
            st.rerun()
        else:
            st.sidebar.error("No existing data found")
    
    if st.sidebar.button("📤 Export to Files"):
        output_dir = project_root / "data" / "exported"
        db.export_to_numpy(str(output_dir))
        st.sidebar.success(f"✅ Exported!")


if __name__ == "__main__":
    main()
