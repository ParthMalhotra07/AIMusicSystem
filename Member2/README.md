# Member 2 Outputs for Member 3 (Recommendation Engine)

**Generated:** 2026-01-24 18:30:03

This directory contains all the processed data and trained models from Member 2's Embedding & Clustering pipeline, ready for use by Member 3's Recommendation Engine.

---

## 📊 Pipeline Summary

- **Input:** 15 songs with 336D audio features from Member 1
- **Embedding Model:** Autoencoder (336D → 14D)
  - Neighborhood Preservation Score: 0.7067
- **Clustering Model:** K-Means with 5 clusters
  - Silhouette Score: 0.0787
  - Davies-Bouldin Index: 1.4274

---

## 📁 Files Overview

### 1. **song_embeddings.npy** (968B)
- **Shape:** (15, 14)
- **Description:** Low-dimensional embeddings for all songs
- **Usage:** Use for computing song similarity via cosine distance or Euclidean distance
- **Format:** NumPy array, dtype=float32

```python
import numpy as np
embeddings = np.load('song_embeddings.npy')
# Shape: (15 songs, 14 dimensions)
```

### 2. **cluster_labels.npy** (188B)
- **Shape:** (15,)
- **Description:** Cluster assignment for each song (0-4)
- **Usage:** Filter songs by cluster for collaborative filtering
- **Format:** NumPy array, dtype=int

```python
labels = np.load('cluster_labels.npy')
# Array: [1, 1, 1, 4, 3, 1, 2, 3, 0, 2, 3, 4, 4, 0, 1]
```

### 3. **original_features.npy** (20KB)
- **Shape:** (15, 336)
- **Description:** Original high-dimensional audio features from Member 1
- **Usage:** For explainability - explain why songs are similar based on audio features
- **Format:** NumPy array, dtype=float32

### 4. **song_metadata.json** (783B)
- **Description:** Song names and file paths
- **Usage:** Map between song indices and actual song names
- **Fields:**
  - `song_names`: List of song names (e.g., "song_01", "song_02", ...)
  - `file_paths`: Original audio file paths
  - `n_songs`: Total number of songs (15)

```json
{
  "song_names": ["song_08", "song_09", "song_01", ...],
  "file_paths": ["../test_audio/song_08.wav", ...],
  "n_songs": 15
}
```

### 5. **cluster_info.json** (509B)
- **Description:** Detailed clustering information
- **Usage:** Understand cluster composition for cluster-based recommendations
- **Fields:**
  - `n_clusters`: Number of clusters (5)
  - `cluster_sizes`: Songs per cluster
  - `clustering_method`: "K-Means"
  - `songs_per_cluster`: Which songs belong to which cluster

```json
{
  "n_clusters": 5,
  "cluster_sizes": {"0": 2, "1": 5, "2": 2, "3": 3, "4": 3},
  "songs_per_cluster": {
    "0": ["song_09", "song_14"],
    "1": ["song_01", "song_15", "song_02", "song_03", "song_06"],
    ...
  }
}
```

### 6. **embedding_model.pkl** (415KB)
- **Description:** Trained Autoencoder model
- **Usage:** Generate embeddings for NEW songs
- **Format:** Pickled PyTorch model

```python
import torch
from Member2.embedding import AutoencoderModel

model = AutoencoderModel.load('embedding_model.pkl')
new_song_features = np.array([...])  # 336D features from Member 1
new_embedding = model.transform(new_song_features)
```

### 7. **clustering_model.pkl** (1.1KB)
- **Description:** Trained K-Means clustering model
- **Usage:** Assign NEW songs to existing clusters
- **Format:** Pickled scikit-learn model

```python
from Member2.clustering import KMeansModel

kmeans = KMeansModel.load('clustering_model.pkl')
new_cluster = kmeans.predict(new_embedding)
```

### 8. **preprocessor.json** (14KB)
- **Description:** Feature scaling parameters (mean, std)
- **Usage:** Preprocess NEW songs before embedding
- **Format:** JSON with scaling statistics

```python
from Member2.data_loading import FeaturePreprocessor

preprocessor = FeaturePreprocessor.load('preprocessor.json')
scaled_features = preprocessor.transform(raw_features)
```

### 9. **pipeline_config.json** (434B)
- **Description:** Complete pipeline configuration and metadata
- **Usage:** Reference for understanding how data was processed
- **Fields:** timestamp, model types, dimensions, scores, etc.

---

## 🎯 Recommended Workflow for Member 3

### Step 1: Load All Data

```python
import numpy as np
import json

# Load embeddings and labels
embeddings = np.load('song_embeddings.npy')
labels = np.load('cluster_labels.npy')
features = np.load('original_features.npy')

# Load metadata
with open('song_metadata.json') as f:
    metadata = json.load(f)
    song_names = metadata['song_names']

with open('cluster_info.json') as f:
    cluster_info = json.load(f)
```

### Step 2: Build Recommendation System

**Option A: Embedding-based (Content-Based Filtering)**
```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute similarity matrix
similarity_matrix = cosine_similarity(embeddings)

def recommend_similar_songs(song_idx, top_k=5):
    """Find K most similar songs."""
    similarities = similarity_matrix[song_idx]
    similar_idx = np.argsort(similarities)[::-1][1:top_k+1]
    return [(song_names[i], similarities[i]) for i in similar_idx]

# Example: Recommend songs similar to song_01 (index 2)
recommendations = recommend_similar_songs(2, top_k=3)
```

**Option B: Cluster-based (Collaborative Filtering)**
```python
def recommend_from_cluster(song_idx, top_k=3):
    """Recommend songs from the same cluster."""
    cluster_id = labels[song_idx]
    cluster_songs = np.where(labels == cluster_id)[0]
    cluster_songs = cluster_songs[cluster_songs != song_idx]

    # Rank by embedding similarity within cluster
    similarities = similarity_matrix[song_idx, cluster_songs]
    top_idx = cluster_songs[np.argsort(similarities)[::-1][:top_k]]

    return [song_names[i] for i in top_idx]
```

**Option C: Hybrid Approach**
Combine both methods with weighted scores.

### Step 3: Add Explainability

```python
# Load feature names from Member 1
# Explain WHY two songs are similar based on audio features

def explain_similarity(song_a_idx, song_b_idx, top_features=5):
    """Explain similarity using original features."""
    feat_a = features[song_a_idx]
    feat_b = features[song_b_idx]

    # Find features with smallest difference
    diff = np.abs(feat_a - feat_b)
    similar_features = np.argsort(diff)[:top_features]

    return similar_features  # Map to feature names from Member 1
```

### Step 4: Handle New Songs

```python
from Member2.data_loading import FeaturePreprocessor
from Member2.embedding import AutoencoderModel
from Member2.clustering import KMeansModel

# Load models
preprocessor = FeaturePreprocessor.load('preprocessor.json')
embedding_model = AutoencoderModel.load('embedding_model.pkl')
clustering_model = KMeansModel.load('clustering_model.pkl')

def process_new_song(new_features):
    """Process a new song through the pipeline."""
    # 1. Preprocess
    scaled = preprocessor.transform(new_features)

    # 2. Generate embedding
    embedding = embedding_model.transform(scaled)

    # 3. Assign cluster
    cluster = clustering_model.predict(embedding)

    return embedding, cluster

# Now use embedding for similarity-based recommendations
```

---

## 📈 Cluster Analysis

### Cluster Distribution:
- **Cluster 0:** 2 songs (13%)
- **Cluster 1:** 5 songs (33%) - Largest cluster
- **Cluster 2:** 2 songs (13%)
- **Cluster 3:** 3 songs (20%)
- **Cluster 4:** 3 songs (20%)

### Cluster Composition:
- **Cluster 0:** song_09, song_14
- **Cluster 1:** song_01, song_15, song_02, song_03, song_06
- **Cluster 2:** song_07, song_10
- **Cluster 3:** song_08, song_11, song_05
- **Cluster 4:** song_13, song_12, song_04

---

## ⚠️ Important Notes

1. **Dimensionality:** Embeddings are 14D (reduced from 336D) for efficient similarity computation
2. **Preprocessing:** All features were Z-score normalized - apply same preprocessing to new songs
3. **Model Format:** Models are saved in pickle format - ensure Member2 modules are importable
4. **Small Dataset:** Current data has only 15 songs - clustering quality will improve with more data
5. **Silhouette Score:** 0.0787 is relatively low due to small dataset size - expect better clustering with 100+ songs

---

## 🚀 Next Steps for Member 3

1. **Build Recommendation Engine:**
   - Implement similarity-based recommendations
   - Implement cluster-based recommendations
   - Create hybrid scoring system

2. **Add Explainability:**
   - Map embeddings back to audio features
   - Explain why songs are recommended
   - Show which audio properties (tempo, energy, etc.) are similar

3. **Create API/Interface:**
   - REST API for recommendations
   - Handle new song ingestion
   - Return ranked recommendations with scores

4. **Optimize for Scale:**
   - Use approximate nearest neighbors (FAISS, Annoy) for large datasets
   - Cache similarity matrices
   - Implement batch processing for new songs

---

## 📞 Questions?

If you have questions about the data format or pipeline, refer to:
- Member 2 code: `/Member2/`
- Pipeline script: `/Member2/run_complete_pipeline.py`
- Visualization dashboard: `../pipeline_results/visualizations/dashboard.html`

**Happy Recommending! 🎵**
