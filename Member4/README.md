# 🎵 Audio-First AI Music Recommendation System

> **Tag-Free Music Discovery** - Recommendations powered purely by audio analysis, not metadata.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Concept

Traditional music recommendation systems rely heavily on metadata: genres, tags, artist names, and user listening patterns. This approach has limitations:

- **Cold start problem**: New songs without tags can't be recommended
- **Genre bias**: Songs are confined to rigid categories
- **Missing cross-genre discoveries**: A jazz track with electronic elements won't be recommended to EDM fans

**Our solution**: Recommend music based purely on **audio-derived features**. We analyze the actual sound—tempo, timbre, brightness, harmony, and energy—and learn embeddings that capture musical similarity at a deeper level.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                         │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ Discover │  │ Recommender  │  │    Explainability       │   │
│  │  (PCA)   │  │ (Seed/Hist)  │  │  (Why recommended?)     │   │
│  └──────────┘  └──────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                            │
│  ┌─────────────────┐         ┌────────────────────────────┐    │
│  │   load_data.py  │         │  recommender_adapter.py    │    │
│  │  - CSV loader   │         │  - Cosine similarity       │    │
│  │  - NPY loader   │         │  - Single seed recs        │    │
│  │  - Mock fallback│         │  - History-based recs      │    │
│  └─────────────────┘         └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXPLAINABILITY ENGINE                         │
│  ┌─────────────────┐         ┌────────────────────────────┐    │
│  │   explain.py    │         │       plots.py             │    │
│  │  - Component    │         │  - Radar charts            │    │
│  │    scores       │         │  - PCA embedding maps      │    │
│  │  - Feature      │         │  - Feature comparison      │    │
│  │    deltas       │         │    visualizations          │    │
│  │  - Human-       │         │                            │    │
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit
```bash
streamlit run app/streamlit_app.py
```

Open your browser to `http://localhost:8501` — you'll see the cyberpunk-themed dashboard!

### 3. Prepare Your Data
- Place your song features CSV at: `data/song_features.csv`
- Place embeddings NPY at: `data/song_embeddings.npy`
- Place song IDs NPY at: `data/song_ids.npy`

If you don't have data files, the system generates mock data automatically.

## 📊 Data Format

### `song_features.csv`
Required columns:
- `song_id` (string)
- `tempo` (float)
- `spectral_centroid` (float)
- `rms` (float) - Energy/loudness

Optional:
- `mfcc_mean_1` through `mfcc_mean_13` - Timbre (MFCC coefficients)
- `chroma_mean_1` through `chroma_mean_12` - Harmony (Chroma features)
- `duration_sec` - Track length

### `song_embeddings.npy`
- Shape: `(num_songs, embedding_dim)` - typically 64-512 dimensions
- dtype: `float32`

### `song_ids.npy`
- Shape: `(num_songs,)`
- dtype: object (strings)

## 📱 Dashboard Pages

### 🔍 Discover
- 2D PCA visualization of the embedding space
- Explore song neighborhoods
- See which songs cluster together
- **Proves:** Audio similarity creates natural song clusters

### 🎯 Recommender  
- **Single Seed Mode**: Find songs similar to one track
- **Listening History Mode**: Build a taste profile from multiple songs
- Cosine similarity ranking
- Configurable K (number of recommendations)
- Audio playback (if files exist)

### 📊 Explainability (THE WINNING FEATURE)
- **4 Analysis Views:**
  1. **Explanation Bullets** - Human-readable musical reasoning
  2. **Feature Comparison Table** - Side-by-side feature analysis
  3. **Radar Chart** - Visual component similarity scores
  4. **Embedding Map** - Spatial positioning in music space

- **Component Scores (0-1):**
  - **Tempo** - BPM alignment
  - **Timbre** - MFCC-based texture similarity
  - **Brightness** - Spectral centroid color
  - **Harmony** - Chroma/chord alignment
  - **Energy** - RMS loudness match

## 🎨 Cyberpunk Theme

The UI features a **neon cyberpunk design**:
- Neon cyan (#00ffff), magenta (#ff00ff), green (#00ff88) accents
- Dark backgrounds with glowing animations
- Animated music visualizer bars
- Orbitron font for tech feel
- Smooth transitions and hover effects

## 🧠 How the Recommendation Engine Works

1. **Audio Feature Extraction**
   - Tempo (BPM)
   - MFCCs (timbre)
   - Spectral Centroid (brightness)
   - Chroma (harmony)
   - RMS (energy)

2. **Embedding Learning**
   - Convert audio features → dense vectors (embeddings)
   - Capture semantic music similarity
   - Normalized for cosine similarity

3. **Similarity Search**
   - Cosine similarity between embeddings
   - Fast nearest-neighbor retrieval
   - Supports single-seed or history-based modes

4. **Explainability**
   - Break down similarity into 5 components
   - Compute component scores independently
   - Generate human-readable explanations
   - Visual radar charts and spatial maps

## 💡 Key Features

✅ **Audio-First** - No genre tags needed  
✅ **Explainable** - Understand *why* songs are recommended  
✅ **Fast** - Cosine similarity in milliseconds  
✅ **Flexible** - Works with any audio feature set  
✅ **Visual** - PCA maps, radar charts, heatmaps  
✅ **Robust** - Auto-generates mock data if files missing  
✅ **Production-Ready** - Handles edge cases gracefully  

## 🛠️ Architecture Files

```
app/
  ├── streamlit_app.py          # Main dashboard entry
  ├── styles.py                 # Cyberpunk CSS + HTML helpers
  └── pages/
      ├── 1_Discover.py         # Embedding space explorer
      ├── 2_Recommender.py      # Recommendation engine UI
      └── 3_Explainability.py   # Explanation visualization

integration/
  ├── load_data.py              # CSV/NPY loaders + mock fallback
  └── recommender_adapter.py    # Cosine similarity engine

explainability/
  ├── explain.py                # Component score calculator
  └── plots.py                  # Matplotlib visualizations

data/
  ├── song_features.csv         # Audio features
  ├── song_embeddings.npy       # Learned embeddings
  ├── song_ids.npy              # Song ID array
  └── audio_samples/            # Optional: audio files for playback
```

## 🔄 Data Flow

```
User Interaction (Streamlit Pages)
           ↓
load_data.py (features + embeddings)
           ↓
recommender_adapter.py (similarity compute)
           ↓
explain.py (component analysis)
           ↓
plots.py (visualization)
           ↓
Streamlit Display (pages + CSS styling)
```

## 📝 Example Usage

### Programmatic Recommendations
```python
from integration.load_data import load_all_data
from integration.recommender_adapter import recommend_from_song

data = load_all_data()

# Get recommendations
rec_ids, scores = recommend_from_song(
    seed_song_id="song_0042",
    embeddings=data["embeddings"],
    song_ids=data["song_ids"],
    id_to_idx=data["id_to_idx"],
    k=10
)

print(f"Top recommendations: {rec_ids}")
print(f"Scores: {scores}")
```

### Programmatic Explanations
```python
from explainability.explain import explain_pair

explanation = explain_pair(
    seed_id="song_0042",
    rec_id="song_0107",
    features_df=data["features_df"],
    embeddings=data["embeddings"],
    id_to_idx=data["id_to_idx"]
)

print(explanation["reasons"])
print(f"Component scores: {explanation['component_scores']}")
```

## 🎯 Use Cases

1. **Music Discovery** - Help users find new songs they'll love
2. **Playlist Generation** - Auto-generate playlists based on taste
3. **Music Research** - Analyze audio similarity without metadata
4. **Cold-Start Problem** - Recommend new songs with no user history
5. **Cross-Genre Discovery** - Surface unexpected but similar tracks

## 📚 References

- MFCC (Mel-Frequency Cepstral Coefficients): Timbre representation
- Spectral Centroid: Brightness/tonal color
- Chroma Features: Harmonic content
- Cosine Similarity: Distance metric for embeddings
- PCA: Dimensionality reduction for visualization

## 🐛 Troubleshooting

**"No data found"** → Create `data/` directory and add CSV/NPY files, or system will auto-generate mock data

**"Audio files not found"** → Place `.wav` or `.mp3` files in `data/audio_samples/` with naming convention: `{song_id}.wav`

**"Embeddings dimension mismatch"** → Ensure all songs have embeddings of same dimension

**Performance slow** → Reduce number of songs or increase K sampling threshold

## 📄 License

MIT License - see LICENSE file for details

---

**Built with ❤️ for music lovers and AI researchers**
