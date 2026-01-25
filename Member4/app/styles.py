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

/* Import Material Icons - fixes the text rendering issue for sidebar/expander icons */
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded');

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
   TEXT INPUT - COMPREHENSIVE STYLING
   ============================================ */

/* Text input container */
.stTextInput > div > div {
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    border-radius: 8px !important;
    padding: 2px !important;
}

.stTextInput > div > div:focus-within {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.4) !important;
}

/* The actual input element */
.stTextInput input {
    background: transparent !important;
    color: #e0e0e0 !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    padding: 10px 15px !important;
    border: none !important;
    width: 100% !important;
    min-width: 200px !important;
}

/* Placeholder text styling - crucial fix for truncation */
.stTextInput input::placeholder {
    color: rgba(0, 255, 255, 0.5) !important;
    font-style: italic !important;
    opacity: 1 !important;
    text-overflow: ellipsis !important;
}

.stTextInput input::-webkit-input-placeholder {
    color: rgba(0, 255, 255, 0.5) !important;
    font-style: italic !important;
}

.stTextInput input::-moz-placeholder {
    color: rgba(0, 255, 255, 0.5) !important;
    font-style: italic !important;
}

.stTextInput input:-ms-input-placeholder {
    color: rgba(0, 255, 255, 0.5) !important;
    font-style: italic !important;
}

/* Input label styling */
.stTextInput > label {
    color: var(--neon-cyan) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px !important;
}

/* Number input styling */
.stNumberInput > div > div {
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    border-radius: 8px !important;
}

.stNumberInput input {
    background: transparent !important;
    color: #e0e0e0 !important;
}

/* Text area styling */
.stTextArea > div > div {
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    border-radius: 8px !important;
}

.stTextArea textarea {
    background: transparent !important;
    color: #e0e0e0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

.stTextArea textarea::placeholder {
    color: rgba(0, 255, 255, 0.5) !important;
    font-style: italic !important;
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
   EXPANDERS - IMPROVED STYLING
   ============================================ */

.streamlit-expanderHeader {
    font-family: 'Orbitron', sans-serif !important;
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    color: var(--neon-cyan) !important;
    padding: 12px 15px !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.2) !important;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(0, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    background: var(--card-bg) !important;
    overflow: hidden;
}

[data-testid="stExpander"] > div {
    background: transparent !important;
}

/* ============================================
   MATERIAL ICONS/SYMBOLS FIX
   Completely hides text like 'keyboard_arrow_down' and
   replaces with unicode symbols
   ============================================ */

/* Material Symbols proper styling */
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-weight: normal;
    font-style: normal;
    font-size: 24px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}

/* CRITICAL FIX: Hide ALL keyboard_* text in the entire app */
/* Target any span containing material icon text */
span[class*="material"]:not(:empty) {
    font-size: 0 !important;
    color: transparent !important;
}

span[class*="material"]::before {
    font-size: 20px !important;
    color: var(--neon-cyan) !important;
    font-family: sans-serif !important;
}

/* Expander arrow icons - hide text, show symbols */
[data-testid="stExpander"] [class*="icon"],
[data-testid="stExpander"] summary span,
.streamlit-expanderHeader span {
    font-size: 0 !important;
    color: transparent !important;
    width: 24px !important;
    height: 24px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Expander collapsed state - down arrow */
[data-testid="stExpander"]:not([open]) [class*="icon"]::before,
[data-testid="stExpander"]:not([open]) summary span::before,
.streamlit-expanderHeader span::before {
    content: "▼" !important;
    font-size: 12px !important;
    color: var(--neon-cyan) !important;
}

/* Expander expanded state - up arrow */
[data-testid="stExpander"][open] [class*="icon"]::before,
[data-testid="stExpander"][open] summary span::before {
    content: "▲" !important;
    font-size: 12px !important;
    color: var(--neon-cyan) !important;
}

/* Sidebar collapse/expand buttons */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="baseButton-header"],
button[kind="header"] {
    font-size: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 46px !important;
    height: 42px !important;
    background: linear-gradient(135deg, #1a1a2e, #0f0f1a) !important;
    border: 2px solid var(--neon-cyan) !important;
    border-radius: 8px !important;
}

/* Hide SVG and all child spans with text */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="baseButton-header"],
button[kind="header"] {
    display: none !important;
}

[data-testid="collapsedControl"] *,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="baseButton-header"] span,
button[kind="header"] span {
    display: none !important;
}

/* Show hamburger menu for collapsed sidebar */  
[data-testid="collapsedControl"]::before {
    content: "☰" !important;
    font-size: 24px !important;
    font-family: sans-serif !important;
    color: var(--neon-cyan) !important;
    display: flex !important;
}

/* Show X for close sidebar button in sidebar */
[data-testid="stSidebarCollapseButton"]::before,
[data-testid="baseButton-header"]::before,
button[kind="header"]::before {
    content: "✕" !important;
    font-size: 18px !important;
    font-family: sans-serif !important;
    color: var(--neon-cyan) !important;
    display: flex !important;
}

/* Select box dropdown arrows */
[data-testid="stSelectbox"] [class*="icon"],
.stSelectbox svg,
[baseweb="select"] [class*="icon"] {
    font-size: 0 !important;
}

/* General fallback: Any element containing "keyboard" text make it invisible */
*:not(script):not(style) {
    text-indent: inherit;
}

/* Selectbox arrow icon replacement */
[baseweb="select"] [class*="icon"]::before {
    content: "▾" !important;
    font-size: 16px !important;
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

/* ============================================
   FILE UPLOADER - CYBERPUNK STYLE
   ============================================ */

[data-testid="stFileUploader"] {
    background: var(--card-bg) !important;
    border: 2px dashed rgba(0, 255, 255, 0.4) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.2) !important;
}

[data-testid="stFileUploader"] section {
    padding: 15px !important;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, var(--neon-cyan), var(--neon-purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', sans-serif !important;
    padding: 10px 20px !important;
}

/* ============================================
   AUDIO PLAYER - CYBERPUNK STYLE
   ============================================ */

audio {
    width: 100% !important;
    border-radius: 10px !important;
    filter: drop-shadow(0 0 5px rgba(0, 255, 255, 0.3)) !important;
}

audio::-webkit-media-controls-panel {
    background: linear-gradient(135deg, #1a1a2e, #0f0f1a) !important;
}

/* ============================================
   RADIO BUTTONS - CYBERPUNK STYLE
   ============================================ */

.stRadio > label {
    color: var(--neon-cyan) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 500 !important;
}

.stRadio > div {
    background: var(--card-bg) !important;
    border-radius: 10px !important;
    padding: 10px 15px !important;
}

.stRadio [role="radiogroup"] {
    gap: 15px !important;
}

/* ============================================
   SPINNER - CYBERPUNK STYLE
   ============================================ */

.stSpinner > div {
    border-color: var(--neon-cyan) transparent transparent transparent !important;
}

/* ============================================
   CHECKBOX - CYBERPUNK STYLE
   ============================================ */

.stCheckbox label {
    color: #e0e0e0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

.stCheckbox [data-testid="stCheckbox"] {
    background: var(--card-bg) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
    border-radius: 5px !important;
    padding: 8px 12px !important;
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


# Hamburger menu button to toggle sidebar
SIDEBAR_TOGGLE_BUTTON = """
<style>
/* ============================================
   SIDEBAR TOGGLE BUTTON STYLING
   Works for both expanded and collapsed states
   ============================================ */

/* Style the collapsed state toggle (appears when sidebar is hidden) */
[data-testid="collapsedControl"] {
    background: linear-gradient(135deg, #1a1a2e, #0f0f1a) !important;
    border: 2px solid #00ffff !important;
    border-radius: 8px !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.3) !important;
    top: 14px !important;
    left: 14px !important;
    width: 46px !important;
    height: 42px !important;
    padding: 0 !important;
    transition: all 0.3s ease !important;
    overflow: hidden !important;
}

[data-testid="collapsedControl"]:hover {
    background: linear-gradient(135deg, #2a2a4e, #1f1f3a) !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.6) !important;
    transform: scale(1.05) !important;
    border-color: #ff00ff !important;
}

/* Hide the SVG icon and any text */
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] span {
    display: none !important;
}

/* Create hamburger icon using pseudo-element */
[data-testid="collapsedControl"]::before {
    content: "≡";
    font-size: 28px !important;
    color: #00ffff !important;
    text-shadow: 0 0 10px #00ffff !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    line-height: 38px !important;
}

[data-testid="collapsedControl"]:hover::before {
    color: #ff00ff !important;
    text-shadow: 0 0 15px #ff00ff !important;
}

/* Style the close/collapse button inside the sidebar (X button) */
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebar"] [data-testid="baseButton-header"] {
    background: linear-gradient(135deg, #1a1a2e, #0f0f1a) !important;
    border: 2px solid #00ffff !important;
    border-radius: 8px !important;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.3) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]:hover,
[data-testid="stSidebar"] button[kind="header"]:hover,
[data-testid="stSidebar"] [data-testid="baseButton-header"]:hover {
    background: linear-gradient(135deg, #2a2a4e, #1f1f3a) !important;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.6) !important;
    border-color: #ff00ff !important;
    transform: scale(1.05) !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebar"] button[kind="header"] svg,
[data-testid="stSidebar"] [data-testid="baseButton-header"] svg {
    stroke: #00ffff !important;
    transition: all 0.3s ease !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]:hover svg,
[data-testid="stSidebar"] button[kind="header"]:hover svg,
[data-testid="stSidebar"] [data-testid="baseButton-header"]:hover svg {
    stroke: #ff00ff !important;
}
</style>
"""

# JavaScript injection to hide keyboard text and fix icon rendering
ICON_FIX_SCRIPT = """
<script>
// Function to hide keyboard text and replace with icons
function fixKeyboardIcons() {
    // Find all elements that might contain keyboard text
    const allElements = document.querySelectorAll('*');
    
    allElements.forEach(el => {
        // Skip script and style elements
        if (el.tagName === 'SCRIPT' || el.tagName === 'STYLE') return;
        
        // Check direct text content
        if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
            const text = el.textContent.trim();
            
            // Replace keyboard icon text with symbols
            if (text.includes('keyboard_double_arrow_right')) {
                el.textContent = '»';
                el.style.fontSize = '24px';
                el.style.color = '#00ffff';
            } else if (text.includes('keyboard_double_arrow_left')) {
                el.textContent = '«';
                el.style.fontSize = '24px';
                el.style.color = '#00ffff';
            } else if (text.includes('keyboard_arrow_down')) {
                el.textContent = '▼';
                el.style.fontSize = '14px';
                el.style.color = '#00ffff';
            } else if (text.includes('keyboard_arrow_up')) {
                el.textContent = '▲';
                el.style.fontSize = '14px';
                el.style.color = '#00ffff';
            } else if (text.includes('keyboard_arrow_right')) {
                el.textContent = '›';
                el.style.fontSize = '20px';
                el.style.color = '#00ffff';
            } else if (text.includes('keyboard_arrow_left')) {
                el.textContent = '‹';
                el.style.fontSize = '20px';
                el.style.color = '#00ffff';
            } else if (text.includes('close')) {
                el.textContent = '✕';
                el.style.fontSize = '18px';
                el.style.color = '#00ffff';
            } else if (text.includes('expand_more')) {
                el.textContent = '▼';
                el.style.fontSize = '14px';
                el.style.color = '#00ffff';
            } else if (text.includes('expand_less')) {
                el.textContent = '▲';
                el.style.fontSize = '14px';
                el.style.color = '#00ffff';
            }
        }
    });
    
    // Also target the collapsed control button specifically
    const collapsedControl = document.querySelector('[data-testid="collapsedControl"]');
    if (collapsedControl) {
        // Find any text nodes and replace
        const walker = document.createTreeWalker(
            collapsedControl,
            NodeFilter.SHOW_TEXT,
            null,
            false
        );
        
        let node;
        while (node = walker.nextNode()) {
            if (node.textContent.includes('keyboard')) {
                node.textContent = '';
            }
        }
    }
}

// Run on page load and periodically to catch dynamic content
document.addEventListener('DOMContentLoaded', fixKeyboardIcons);
setTimeout(fixKeyboardIcons, 500);
setTimeout(fixKeyboardIcons, 1000);
setTimeout(fixKeyboardIcons, 2000);
setTimeout(fixKeyboardIcons, 3000);

// Also observe for DOM changes
const observer = new MutationObserver(function(mutations) {
    fixKeyboardIcons();
});

// Start observing once DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
</script>
"""
