# 🎵 AI Music Recommendation System

**DUHacks 5 - Audio-First Music Discovery Platform**

An intelligent music recommendation system that uses **audio feature analysis** and **deep embeddings** to discover similar songs, without relying on tags, genres, or metadata.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Team Contributions](#team-contributions)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Install Dependencies

```bash
cd /workspaces/AIMusicSystem
pip install -r requirements.txt
```

### Run the Dashboard

```bash
cd Member4
streamlit run app/streamlit_app.py
```

The dashboard will open at `http://localhost:8501` with a cyberpunk-themed interface featuring 3 interactive pages:

1. **📊 Discover** - Explore songs in embedding space using PCA
2. **🎯 Recommender** - Get personalized recommendations
3. **💡 Explainability** - Understand why songs are recommended

---

## ✨ Features

### 🎧 Audio Feature Extraction (Member1)
- Extracts 100+ audio features from raw music files
- **Temporal:** MFCCs, Chroma features
- **Rhythmic:** Tempo, Beat, Onset strength
- **Harmonic:** Spectral centroid, Spectral rolloff, Zero-crossing rate
- **Timbral:** Spectral contrast, Tonnetz
- **Groove:** Groove-based tempo curves
- Output: `data/song_features.csv`

### 🧠 Deep Embeddings & Clustering (Member2)
- Converts audio features → low-dimensional embeddings (64-512D)
- **Embedding models:** PCA, UMAP, Autoencoder
- **Clustering:** K-Means, Hierarchical, DBSCAN
- Output: `data/song_embeddings.npy`, trained models

### 🎯 Smart Recommendations (Member3)
- **History-based:** Recommends songs similar to your listening history
- **Seed-based:** Finds similar songs given one seed
- **Weighted recency:** Recent songs influence recommendations more
- **Cold-start fallback:** Works even for new users
- Dual implementation: Member3 + Cosine similarity fallback

### 🎨 Interactive Dashboard (Member4)
- **Cyberpunk UI:** Neon colors, dark mode, smooth animations
- **3 Discovery Pages:**
  - Visualize songs in 2D embedding space
  - Get top-K recommendations with similarity scores
  - Understand recommendations with component analysis
- **Explainability:** 5-component breakdown (tempo, timbre, brightness, harmony, energy)

---

## 🏗️ Architecture

### Data Flow

```
Raw Audio → Audio Features → Embeddings → Recommendations → UI + Explainability
(Member1)    (Member1)       (Member2)     (Member3)        (Member4)
```

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│              🎵 Streamlit Dashboard (Member4)            │
│         Cyberpunk UI with 3 interactive pages            │
├─────────────────────────────────────────────────────────┤
│              Integration Layer (Central APIs)            │
│  • load_all_data()         → unified data dict           │
│  • get_recommendations()   → Member3 + fallback          │
│  • explain_pair()          → explainability              │
├──────┬──────────────┬──────────────┬──────────────────┤
│      │              │              │                   │
│   Member1       Member2        Member3            Member4
│   Audio         Embeddings      Recs              Explainability
└─────────────────────────────────────────────────────────┘
```

---

## 👥 Team Contributions

### Member1: Audio Feature Engineering
- **Deliverable:** `audio_features/` module
- **Output:** `data/song_features.csv` (100+ audio features)
- **Key Files:** `main.py`, `pipeline/`, `preprocessing/`, `features/`

### Member2: Embeddings & Clustering  
- **Deliverable:** `embeddings/` module
- **Output:** `data/song_embeddings.npy`, trained models
- **Key Files:** `embedding/`, `clustering/`, `visualization/`

### Member3: Recommendation Engine
- **Deliverable:** `recommendation/` module  
- **Output:** Recommendation functions
- **Key File:** `user_recommendation.py` (64 lines, 6 functions)

### Member4: UI & Integration (Main Entry Point)
- **Deliverable:** `Member4/` with app, integration, explainability
- **Output:** Streamlit dashboard
- **Key Files:** `app/`, `integration/`, `explainability/`

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- 4GB RAM (8GB recommended)
- 2GB disk space

### Installation Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
cd Member4
streamlit run app/streamlit_app.py
```

---

## 💻 Usage

### Run Streamlit Dashboard (Recommended)

```bash
cd Member4
streamlit run app/streamlit_app.py
```

Visit: `http://localhost:8501`

### Use Python API

```python
from integration.load_data import load_all_data
from integration.recommender_adapter import get_recommendations

# Load data
data = load_all_data()

# Get recommendations
rec_ids, scores = get_recommendations(
    seed_song_id=str(data['song_ids'][0]),
    embeddings=data['embeddings'],
    song_ids=data['song_ids'],
    id_to_idx=data['id_to_idx'],
    k=10
)
```

### Command-Line Tools

**Extract Audio Features (Member1):**
```bash
cd audio_features
python3 main.py --input_dir /path/to/music --output song_features.csv
```

**Train Embeddings (Member2):**
```bash
cd embeddings
python3 embedding/train.py --input ../data/song_features.csv --method umap
```

---

## 📁 Project Structure

```
/workspaces/AIMusicSystem/
├── 📱 Member4/              # Main entry point (UI + Integration)
│   ├── app/                 # Streamlit dashboard
│   ├── integration/         # Unified APIs
│   ├── explainability/      # Explanation components
│   └── data/
├── 🎚️ Member1/            # Audio feature extraction
├── 🧠 Member2/             # Embeddings & clustering
├── 🎯 Member3/             # Recommendation engine
├── 📊 data/               # Shared data files
├── requirements.txt        # Merged dependencies
├── README.md              # This file
├── IMPLEMENTATION_GUIDE.md # Step-by-step setup
└── MERGE_STRUCTURE.md      # Architecture details
```

---

## 🔌 API Reference

### `load_all_data()`
Returns unified data dictionary with embeddings, songs, features

### `get_recommendations(seed_song_id=None, history_song_ids=None, ...)`
Returns list of recommended song IDs with similarity scores

### `analyze_components(song_idx, data)`
Returns 5-component analysis for explainability

---

## 🐛 Troubleshooting

**Issue:** "ModuleNotFoundError: No module named 'recommendation'"  
**Fix:** `touch Member3/__init__.py`

**Issue:** "FileNotFoundError: data/song_embeddings.npy"  
**Fix:** Mock data auto-generated; or run `cd Member2 && python3 embedding/train.py`

**Issue:** Streamlit stuck loading  
**Fix:** Run with debug: `streamlit run app/streamlit_app.py --logger.level=debug`

---

## 📝 For More Information

- **Setup Guide:** See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
- **Architecture:** See [MERGE_STRUCTURE.md](MERGE_STRUCTURE.md)
- **Member Docs:** See `Member{1,2,3,4}/README.md`

---

**Status:** ✅ Production Ready | **DUHacks 5** | **2024**
