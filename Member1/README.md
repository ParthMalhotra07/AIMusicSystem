# 🎵 Audio Feature Engineering Module

> **Transform raw audio into mathematical "fingerprints" for AI-powered music recommendation.**

This module eliminates metadata dependency by analyzing raw audio signals mathematically—enabling music recommendations based purely on how songs *sound*, not how they're tagged.

---

## 🎯 Overview

| Feature | Description |
|---------|-------------|
| **~170D Vector** | Fixed-length feature vector regardless of song duration |
| **3 Feature Categories** | Timbral (texture), Rhythmic (tempo), Harmonic (pitch) |
| **Multiple Exports** | CSV, NumPy (.npy), Parquet formats |
| **Batch Processing** | Process entire directories with progress tracking |

---

## 🚀 Quick Start

### Installation

```bash
cd Member1
pip install -r requirements.txt
```

### Single File Processing

```bash
python main.py --input song.mp3 --output features.csv
```

### Batch Processing

```bash
python main.py --input_dir ./music --output_dir ./output --format parquet
```

---

## 📊 Feature Vector Breakdown

### Timbral Features (Sound Texture)
| Feature | Dims | Description |
|---------|------|-------------|
| MFCCs | 13 × 4 | Spectral envelope "fingerprint" |
| Spectral Centroid | 1 × 4 | Brightness of sound |
| Spectral Contrast | 7 × 4 | Peak-valley differences |
| Zero-Crossing Rate | 1 × 4 | Noisiness indicator |

### Rhythmic Features (Energy)
| Feature | Dims | Description |
|---------|------|-------------|
| Tempo | 1 | BPM estimation |
| Onset Strength | 1 × 4 | Attack patterns |
| Beat Strength | 1 × 4 | Rhythm intensity |

### Harmonic Features (Pitch)
| Feature | Dims | Description |
|---------|------|-------------|
| Chroma | 12 × 4 | Pitch class distribution |
| Tonnetz | 6 × 4 | Harmonic relationships |

*Each time-varying feature has 4 statistics: mean, std, skew, kurtosis*

---

## 🔧 Python API

```python
from pipeline.extractor import AudioFeatureExtractor
from pipeline.scaler import ZScoreScaler

# Initialize
extractor = AudioFeatureExtractor()

# Extract features with metadata
features, metadata = extractor.process_file('song.mp3', return_metadata=True)

print(f"Tempo: {metadata['tempo_bpm']:.1f} BPM")
print(f"Key: {metadata['key']}")
print(f"Features: {len(features)} dimensions")

# Batch processing
features_matrix = extractor.process_batch(['song1.mp3', 'song2.mp3'])

# Scale features for AI
scaler = ZScoreScaler()
features_scaled = scaler.fit_transform(features_matrix)
```

---

## 📁 Project Structure

```
Member1/
├── main.py              # CLI entry point
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── preprocessing/       # Audio loading & spectral transforms
├── features/           # Feature extractors (timbral, rhythmic, harmonic)
├── pipeline/           # Main extraction pipeline & scalers
├── export/             # CSV, NPY, Parquet exporters
└── tests/              # Unit tests (43 tests, all passing)
```

---

## 🧪 Running Tests

```bash
python3 -m pytest tests/ -v
```

---

## 📤 Output Formats

| Format | Use Case |
|--------|----------|
| **CSV** | Human-readable, spreadsheet import |
| **NPY** | Fast NumPy loading |
| **Parquet** | Columnar queries, compression, Spark/Dask |

---

## 🔗 Integration with Other Modules

- **Member 2 (Clustering)**: Use exported features for unsupervised clustering
- **Member 3 (UI/Recs)**: Query for similar songs using feature vectors

Features are **independent** of downstream models—change the AI, keep the same ground truth.

---

## 📋 CLI Reference

```
python main.py [OPTIONS]

Options:
  --input, -i      Single audio file path
  --input_dir, -d  Directory of audio files
  --output, -o     Output file path
  --output_dir     Output directory (default: ./output)
  --format, -f     Output format: csv, npy, parquet (default: parquet)
  --sample_rate    Target sample rate (default: 22050)
  --scale          Scaling: none, minmax, zscore (default: zscore)
  --recursive, -r  Search subdirectories
  --quiet, -q      Suppress progress output
```

---

## 📖 Mathematics Behind the Features

### Why These Features?

1. **MFCCs** capture the *shape* of the spectral envelope—what makes a piano different from a guitar
2. **Chroma** captures *which notes* are present, regardless of octave—perfect for harmonic analysis
3. **Tempo** captures the *speed* and *danceability* of the track
4. **Statistical aggregation** (mean, std, skew, kurtosis) summarizes how features change over time

### Fixed-Length Output

The key innovation is **global pooling**: by computing statistics over all time frames, a 3-minute song and a 10-minute song both produce a vector of the **same dimension**.

---

*Built for the AI Music Recommendation System — Member 1: Audio Signal & Feature Engineering*
