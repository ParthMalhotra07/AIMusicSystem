"""
Main Streamlit Application - AI Music Recommendation System

Audio-First (Tag-Free) AI Music Discovery Dashboard
CYBERPUNK EDITION 🎵
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="🎵 Audio-First Music Recommendation",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import cyberpunk styles
from styles import CYBERPUNK_CSS, MUSIC_VISUALIZER, HEADPHONES_ANIMATION, get_cyber_card, get_neon_header, SIDEBAR_TOGGLE_BUTTON, ICON_FIX_SCRIPT

# Import data loaders
from integration.load_data import load_all_data

# Apply cyberpunk theme
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

# Add hamburger menu button for sidebar toggle
st.markdown(SIDEBAR_TOGGLE_BUTTON, unsafe_allow_html=True)

# Inject JavaScript to fix keyboard icon text rendering
st.markdown(ICON_FIX_SCRIPT, unsafe_allow_html=True)


def get_data():
    """Load data from database (always fresh)."""
    return load_all_data(use_database=True)


def main():
    """Main application entry point."""
    
    # Sidebar with cyberpunk style
    st.sidebar.markdown(HEADPHONES_ANIMATION, unsafe_allow_html=True)
    st.sidebar.title("🎵 AUDIO-FIRST AI")
    st.sidebar.markdown(MUSIC_VISUALIZER, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ⚡ How It Works
    
    This system **understands music from audio** 
    (numbers), not tags or genres.
    
    1. 🎵 Audio → Extract features
    2. 🧠 Features → Learn embeddings  
    3. 📊 Embeddings → Find similar songs
    4. 💡 Explain → Musical reasoning
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚀 Quick Links")
    st.sidebar.page_link("pages/1_Discover.py", label="Discover Music Space", icon="🔍")
    st.sidebar.page_link("pages/2_Recommender.py", label="Get Recommendations", icon="🎯")
    st.sidebar.page_link("pages/3_Explainability.py", label="See Explanations", icon="📊")
    st.sidebar.page_link("pages/4_Upload.py", label="Upload New Songs", icon="📤")
    
    # Load data
    with st.spinner("⚡ Initializing neural pathways..."):
        data = get_data()
    
    # ============================================
    # HERO SECTION - Cyberpunk Style
    # ============================================
    
    # Animated header
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <div style='font-size: 60px; margin-bottom: 10px;'>🎧</div>
        <h1 style='
            font-family: Orbitron, sans-serif;
            font-size: 3.5rem;
            background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 3s linear infinite;
            margin: 0;
        '>AUDIO-FIRST AI</h1>
        <h2 style='
            font-family: Orbitron, sans-serif;
            color: #00ffff;
            font-weight: 300;
            letter-spacing: 10px;
            text-shadow: 0 0 20px #00ffff;
        '>MUSIC RECOMMENDATION</h2>
        <p style='color: #ff00ff; font-size: 1.2rem; letter-spacing: 5px;'>[ TAG-FREE • EXPLAINABLE • INTELLIGENT ]</p>
    </div>
    <style>
        @keyframes shine {
            to { background-position: 200% center; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Animated music visualizer
    st.markdown(MUSIC_VISUALIZER, unsafe_allow_html=True)
    
    # One-liner value proposition
    st.markdown("""
    <div style='
        text-align: center; 
        padding: 30px; 
        background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1)); 
        border: 1px solid rgba(0,255,255,0.3);
        border-radius: 15px; 
        margin: 30px 0;
        position: relative;
        overflow: hidden;
    '>
        <p style='font-size: 1.5rem; color: #e0e0e0; margin: 0; font-family: Rajdhani, sans-serif;'>
            🎵 We analyze <span style="color: #00ffff; font-weight: bold;">sound patterns</span> — 
            tempo, energy, brightness, harmony<br>
            <span style='color: #ff00ff;'>No tags. No genres. Pure audio intelligence.</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ============================================
    # ACTION BUTTONS - Cyberpunk Neon Cards
    # ============================================
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='
            text-align: center; 
            padding: 30px; 
            border: 2px solid #00ff88; 
            border-radius: 15px; 
            height: 220px;
            background: rgba(0,255,136,0.05);
        '>
            <div style='font-size: 50px; margin-bottom: 10px;'>🔍</div>
            <h3 style='color: #00ff88; font-family: Orbitron;'>EXPLORE</h3>
            <p style='color: #aaa;'>Navigate the music embedding space</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 ENTER THE MATRIX", use_container_width=True, type="secondary"):
            st.switch_page("pages/1_Discover.py")
    
    with col2:
        st.markdown("""
        <div style='
            text-align: center; 
            padding: 30px; 
            border: 2px solid #ff00ff; 
            border-radius: 15px; 
            height: 220px;
            background: rgba(255,0,255,0.05);
        '>
            <div style='font-size: 50px; margin-bottom: 10px;'>🎯</div>
            <h3 style='color: #ff00ff; font-family: Orbitron;'>RECOMMEND</h3>
            <p style='color: #aaa;'>Get AI-powered song suggestions</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 GET RECOMMENDATIONS", use_container_width=True, type="primary"):
            st.switch_page("pages/2_Recommender.py")
    
    with col3:
        st.markdown("""
        <div style='
            text-align: center; 
            padding: 30px; 
            border: 2px solid #00ffff; 
            border-radius: 15px; 
            height: 220px;
            background: rgba(0,255,255,0.05);
        '>
            <div style='font-size: 50px; margin-bottom: 10px;'>🧠</div>
            <h3 style='color: #00ffff; font-family: Orbitron;'>EXPLAIN</h3>
            <p style='color: #aaa;'>Understand WHY with AI reasoning</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧠 SEE THE LOGIC", use_container_width=True, type="secondary"):
            st.switch_page("pages/3_Explainability.py")
    
    st.markdown("---")
    
    # ============================================
    # KEY DIFFERENTIATORS - Cyberpunk Cards
    # ============================================
    
    st.markdown("""
    <h2 style='text-align: center; color: #00ffff; font-family: Orbitron; margin: 40px 0;'>
        ⚡ WHAT MAKES THIS DIFFERENT ⚡
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(0,0,0,0.5));
            border-left: 3px solid #00ffff;
            padding: 20px;
            border-radius: 10px;
        '>
            <h3 style='color: #00ffff;'>🎵 AUDIO-FIRST</h3>
            <p style='color: #ccc;'>
                We analyze the <b>actual sound</b> — tempo, timbre, 
                brightness, harmony, energy — not metadata.
            </p>
            <p style='color: #00ff88; font-size: 0.9rem;'>No genre labels. No artist bias.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(255,0,255,0.1), rgba(0,0,0,0.5));
            border-left: 3px solid #ff00ff;
            padding: 20px;
            border-radius: 10px;
        '>
            <h3 style='color: #ff00ff;'>🧠 EXPLAINABLE AI</h3>
            <p style='color: #ccc;'>
                Every recommendation comes with <b>human-readable 
                reasoning</b>: why two songs sound similar.
            </p>
            <p style='color: #00ff88; font-size: 0.9rem;'>Not a black box.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,0,0,0.5));
            border-left: 3px solid #00ff88;
            padding: 20px;
            border-radius: 10px;
        '>
            <h3 style='color: #00ff88;'>📊 RESEARCH-GRADE</h3>
            <p style='color: #ccc;'>
                Built on acoustic feature extraction and 
                learned embeddings. Visualize the music space.
            </p>
            <p style='color: #00ff88; font-size: 0.9rem;'>Songs cluster naturally.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================
    # DATA STATUS - Cyberpunk Style
    # ============================================
    
    if data["is_mock"]:
        st.markdown("""
        <div style='
            background: rgba(255,165,0,0.1);
            border: 1px solid orange;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        '>
            <span style='color: orange;'>⚠️ DEMO MODE</span> | 
            Running with auto-generated mock data (200 songs) | 
            <span style='color: #00ff88;'>System fully operational</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='
            background: rgba(0,255,136,0.1);
            border: 1px solid #00ff88;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        '>
            <span style='color: #00ff88;'>✅ LIVE DATA</span> | 
            Connected to your music dataset
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Dataset stats with neon glow
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎵 SONGS", data["num_songs"])
    with col2:
        st.metric("📐 DIMENSIONS", data["embedding_dim"])
    with col3:
        st.metric("🎨 HARMONY", "ACTIVE" if data["has_chroma"] else "OFF")
    with col4:
        st.metric("⚡ ENERGY", "ACTIVE" if data["has_rms"] else "OFF")
    
    st.markdown("---")
    
    # ============================================
    # HOW IT WORKS - Technical Overview
    # ============================================
    
    with st.expander("🔧 TECHNICAL SPECS", expanded=False):
        st.markdown("""
        ### The Neural Pipeline
        
        ```
        🎵 Raw Audio Signal
            ↓
        📊 Feature Extraction (tempo, MFCC, spectral centroid, chroma, RMS)
            ↓
        🧠 Embedding Learning (songs → vectors in latent space)
            ↓
        🔍 Similarity Search (cosine similarity)
            ↓
        💡 Explainability Engine (component-wise analysis)
            ↓
        🎯 Recommendations + Explanations
        ```
        
        ### Similarity Components
        
        | Component | What It Measures | Formula |
        |-----------|-----------------|---------|
        | **Tempo** | Rhythmic alignment | `exp(-|BPM₁ - BPM₂| / 15)` |
        | **Timbre** | Instrument/vocal texture | `exp(-L2(MFCC₁, MFCC₂) / 10)` |
        | **Brightness** | Tonal color | `exp(-|centroid₁ - centroid₂| / 2000)` |
        | **Harmony** | Chord patterns | `cosine(chroma₁, chroma₂)` |
        | **Energy** | Loudness profile | `exp(-|RMS₁ - RMS₂| / 0.1)` |
        """)
    
    # ============================================
    # FOOTER - Cyberpunk Style
    # ============================================
    
    st.markdown("---")
    st.markdown(MUSIC_VISUALIZER, unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p style='
            font-family: Orbitron, sans-serif;
            color: #00ffff;
            font-size: 1.2rem;
            text-shadow: 0 0 10px #00ffff;
        '>🎧 AUDIO-FIRST AI MUSIC SYSTEM</p>
        <p style='color: #666; letter-spacing: 3px;'>
            TAG-FREE • EXPLAINABLE • INTELLIGENT
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
