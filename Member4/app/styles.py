"""
Cyberpunk Theme Styles for AI Music Recommendation System

Neon colors, glowing effects, and animated elements.
"""

CYBERPUNK_CSS = """
<style>
/* ============================================
   CYBERPUNK THEME - GLOBAL STYLES
   ============================================ */

/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

/* Root variables */
:root {
    --neon-pink: #ff00ff;
    --neon-cyan: #00ffff;
    --neon-blue: #00d4ff;
    --neon-purple: #bf00ff;
    --neon-green: #00ff88;
    --neon-yellow: #ffff00;
    --neon-orange: #ff6600;
    --dark-bg: #0a0a0f;
    --darker-bg: #050508;
    --card-bg: rgba(15, 15, 25, 0.8);
    --glass-bg: rgba(255, 255, 255, 0.05);
}

/* Main container dark background */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
    background-attachment: fixed;
}

/* Add animated grid background */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: gridMove 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes gridMove {
    0% { transform: translate(0, 0); }
    100% { transform: translate(50px, 50px); }
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(0, 255, 255, 0.2);
}

[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 2px;
    height: 100%;
    background: linear-gradient(180deg, transparent, var(--neon-cyan), transparent);
    animation: sidebarGlow 3s ease-in-out infinite;
}

@keyframes sidebarGlow {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
}

/* ============================================
   TYPOGRAPHY
   ============================================ */

h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
    animation: textGlow 2s ease-in-out infinite alternate;
}

@keyframes textGlow {
    from { filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5)); }
    to { filter: drop-shadow(0 0 20px rgba(255, 0, 255, 0.5)); }
}

p, span, label, .stMarkdown p {
    font-family: 'Rajdhani', sans-serif !important;
    color: #e0e0e0 !important;
    word-break: break-word;
    overflow-wrap: break-word;
    max-width: 100%;
}

/* ============================================
   BUTTONS - NEON GLOW
   ============================================ */

.stButton > button {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.1), rgba(255, 0, 255, 0.1)) !important;
    border: 2px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    border-radius: 5px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
    font-size: 0.9rem !important;
    white-space: normal !important;
    word-break: break-word !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.3), rgba(255, 0, 255, 0.3)) !important;
    box-shadow: 
        0 0 20px var(--neon-cyan),
        0 0 40px var(--neon-pink),
        inset 0 0 20px rgba(0, 255, 255, 0.1) !important;
    transform: translateY(-2px) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--neon-pink), var(--neon-purple)) !important;
    border: none !important;
    color: white !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 
        0 0 30px var(--neon-pink),
        0 0 60px var(--neon-purple) !important;
}

/* ============================================
   METRICS - CYBERPUNK CARDS
   ============================================ */

[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    border-radius: 10px !important;
    padding: 12px 8px !important;
    position: relative;
    overflow: hidden;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink), var(--neon-cyan));
    background-size: 200% 100%;
    animation: borderGlow 2s linear infinite;
}

@keyframes borderGlow {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 10px var(--neon-cyan);
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    margin: 5px 0 !important;
    word-break: break-word;
    text-align: center;
    line-height: 1.2;
}

[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 0.8rem !important;
    word-break: break-word;
    text-align: center;
    white-space: normal !important;
    line-height: 1.3;
}

/* ============================================
   SELECT BOXES & INPUTS
   ============================================ */

.stSelectbox > div > div {
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    border-radius: 5px !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3) !important;
}

.stMultiSelect > div > div {
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
}

/* ============================================
   PROGRESS BARS - NEON
   ============================================ */

.stProgress > div > div {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink)) !important;
    box-shadow: 0 0 20px var(--neon-cyan);
}

/* ============================================
   DATAFRAMES & TABLES
   ============================================ */

.stDataFrame {
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

[data-testid="stDataFrameContainer"] {
    background: var(--card-bg) !important;
}

/* ============================================
   TABS - CYBERPUNK STYLE
   ============================================ */

.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 5px;
    gap: 5px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    color: #888 !important;
    border-radius: 5px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--neon-cyan) !important;
    background: rgba(0, 255, 255, 0.1);
}

.stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    background: rgba(0, 255, 255, 0.2) !important;
    border: 1px solid var(--neon-cyan) !important;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
}

/* ============================================
   EXPANDERS
   ============================================ */

.streamlit-expanderHeader {
    font-family: 'Orbitron', sans-serif !important;
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
    border-radius: 5px !important;
    color: var(--neon-cyan) !important;
}

/* ============================================
   INFO/SUCCESS/WARNING BOXES
   ============================================ */

.stAlert {
    border-radius: 10px !important;
    border-left: 4px solid var(--neon-cyan) !important;
    background: var(--card-bg) !important;
}

/* ============================================
   SLIDER
   ============================================ */

.stSlider > div > div > div {
    background: var(--neon-cyan) !important;
}

/* ============================================
   ANIMATED MUSIC VISUALIZER (decorative)
   ============================================ */

.music-visualizer {
    display: flex;
    align-items: flex-end;
    justify-content: center;
    height: 60px;
    gap: 4px;
    margin: 20px 0;
}

.music-bar {
    width: 8px;
    background: linear-gradient(180deg, var(--neon-cyan), var(--neon-pink));
    border-radius: 4px;
    animation: musicBounce 0.5s ease-in-out infinite;
    box-shadow: 0 0 10px var(--neon-cyan);
}

.music-bar:nth-child(1) { animation-delay: 0s; height: 20px; }
.music-bar:nth-child(2) { animation-delay: 0.1s; height: 35px; }
.music-bar:nth-child(3) { animation-delay: 0.2s; height: 50px; }
.music-bar:nth-child(4) { animation-delay: 0.3s; height: 30px; }
.music-bar:nth-child(5) { animation-delay: 0.4s; height: 45px; }
.music-bar:nth-child(6) { animation-delay: 0.2s; height: 25px; }
.music-bar:nth-child(7) { animation-delay: 0.1s; height: 40px; }
.music-bar:nth-child(8) { animation-delay: 0s; height: 55px; }
.music-bar:nth-child(9) { animation-delay: 0.3s; height: 30px; }

@keyframes musicBounce {
    0%, 100% { transform: scaleY(0.5); opacity: 0.7; }
    50% { transform: scaleY(1); opacity: 1; }
}

/* ============================================
   HEADPHONES ANIMATION
   ============================================ */

.headphones-container {
    text-align: center;
    font-size: 80px;
    animation: headphonesBounce 2s ease-in-out infinite;
    text-shadow: 
        0 0 20px var(--neon-cyan),
        0 0 40px var(--neon-pink);
}

@keyframes headphonesBounce {
    0%, 100% { transform: translateY(0) rotate(-5deg); }
    50% { transform: translateY(-10px) rotate(5deg); }
}

/* ============================================
   GLOWING CARD CONTAINER
   ============================================ */

.cyber-card {
    background: var(--card-bg);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 15px;
    padding: 25px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.cyber-card:hover {
    border-color: var(--neon-cyan);
    box-shadow: 
        0 0 20px rgba(0, 255, 255, 0.3),
        inset 0 0 30px rgba(0, 255, 255, 0.05);
}

.cyber-card::after {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(
        45deg,
        transparent 40%,
        rgba(0, 255, 255, 0.1) 50%,
        transparent 60%
    );
    animation: cardShine 3s ease-in-out infinite;
    pointer-events: none;
}

@keyframes cardShine {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

/* ============================================
   PULSING DOT INDICATOR
   ============================================ */

.pulse-dot {
    width: 12px;
    height: 12px;
    background: var(--neon-green);
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.5s ease-in-out infinite;
    box-shadow: 0 0 20px var(--neon-green);
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.7; }
}

/* ============================================
   NEON TEXT CLASSES
   ============================================ */

.neon-text-cyan {
    color: var(--neon-cyan) !important;
    text-shadow: 0 0 10px var(--neon-cyan), 0 0 20px var(--neon-cyan);
}

.neon-text-pink {
    color: var(--neon-pink) !important;
    text-shadow: 0 0 10px var(--neon-pink), 0 0 20px var(--neon-pink);
}

.neon-text-green {
    color: var(--neon-green) !important;
    text-shadow: 0 0 10px var(--neon-green), 0 0 20px var(--neon-green);
}

/* ============================================
   FLOATING PARTICLES (decorative)
   ============================================ */

.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    overflow: hidden;
    z-index: -1;
}

.particle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: var(--neon-cyan);
    border-radius: 50%;
    animation: float 15s infinite;
    opacity: 0.5;
}

@keyframes float {
    0%, 100% {
        transform: translateY(100vh) translateX(0);
        opacity: 0;
    }
    10% { opacity: 0.5; }
    90% { opacity: 0.5; }
    100% {
        transform: translateY(-100vh) translateX(100px);
        opacity: 0;
    }
}

/* ============================================
   SONG CARD STYLING
   ============================================ */

.song-card {
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.05), rgba(255, 0, 255, 0.05));
    border: 1px solid rgba(0, 255, 255, 0.2);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: all 0.3s ease;
}

.song-card:hover {
    transform: translateX(10px);
    border-color: var(--neon-pink);
    box-shadow: -5px 0 20px rgba(255, 0, 255, 0.3);
}

/* ============================================
   RECOMMENDATION SCORE BAR
   ============================================ */

.score-bar-container {
    background: rgba(0, 0, 0, 0.5);
    border-radius: 10px;
    overflow: hidden;
    height: 24px;
    position: relative;
}

.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink));
    border-radius: 10px;
    transition: width 1s ease;
    box-shadow: 0 0 20px var(--neon-cyan);
}

.score-bar-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    color: white;
    text-shadow: 0 0 10px black;
}

/* ============================================
   COLUMNS & CONTAINERS - IMPROVED SPACING
   ============================================ */

[data-testid="column"] {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: stretch;
    gap: 8px;
    min-height: 140px;
}

.block-container {
    padding: 1rem 1rem !important;
    max-width: 100%;
}

/* Subheader text wrapping */
[data-testid="stHeading"] {
    word-break: break-word;
    overflow-wrap: break-word;
    max-width: 100%;
}

h1, h2, h3, h4, h5, h6 {
    word-break: break-word;
    overflow-wrap: break-word;
    max-width: 100%;
}

/* Improved markdown rendering */
.stMarkdown {
    word-break: break-word;
    overflow-wrap: break-word;
}

</style>
"""

# HTML components
MUSIC_VISUALIZER = """
<div class="music-visualizer">
    <div class="music-bar"></div>
    <div class="music-bar"></div>
    <div class="music-bar"></div>
    <div class="music-bar"></div>
    <div class="music-bar"></div>
    <div class="music-bar"></div>
    <div class="music-bar"></div>
    <div class="music-bar"></div>
    <div class="music-bar"></div>
</div>
"""

HEADPHONES_ANIMATION = """
<div class="headphones-container">
    🎧
</div>
"""

PULSE_DOT = """<span class="pulse-dot"></span>"""

def get_cyber_card(content: str, title: str = "") -> str:
    """Wrap content in a cyberpunk-styled card."""
    title_html = f'<h4 style="color: #00ffff; margin-bottom: 15px;">{title}</h4>' if title else ""
    return f"""
    <div class="cyber-card">
        {title_html}
        {content}
    </div>
    """

def get_neon_header(text: str, level: int = 1) -> str:
    """Create a neon glowing header."""
    return f"""
    <h{level} style="
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: textGlow 2s ease-in-out infinite alternate;
    ">{text}</h{level}>
    """

def get_score_bar(score: float, label: str = "") -> str:
    """Create an animated score bar."""
    width = int(score * 100)
    return f"""
    <div style="margin: 10px 0;">
        <div style="color: #888; font-size: 12px; margin-bottom: 5px;">{label}</div>
        <div class="score-bar-container">
            <div class="score-bar-fill" style="width: {width}%;"></div>
            <div class="score-bar-text">{score:.3f}</div>
        </div>
    </div>
    """
