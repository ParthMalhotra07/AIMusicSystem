# Embedding Model Selection Guide

## Overview

You have **3 embedding models** to choose from:

| Model | Type | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **PCA** | Linear | ⚡ Very Fast | ⭐⭐ Good | Quick baseline, interpretability |
| **Autoencoder** | Nonlinear | 🐢 Slower | ⭐⭐⭐⭐ Excellent | Best quality, complex patterns |
| **UMAP** | Nonlinear | 🚀 Fast | ⭐⭐⭐ Very Good | Visualization, local structure |

---

## Quick Start

### Option 1: Compare All Models Automatically

```bash
cd Member2
python compare_models.py
```

This will:
- Train all 3 models
- Evaluate their quality
- Recommend the best one
- Generate comparison plots

### Option 2: Train a Specific Model

```bash
# Train autoencoder (recommended)
python embedding/train.py --model autoencoder --dim 32 --epochs 100

# Train PCA (fast baseline)
python embedding/train.py --model pca --dim 32

# Train UMAP
python embedding/train.py --model umap --dim 32 --n_neighbors 15
```

---

## How to Select the Best Model

### 1. **Decision Tree**

```
Start here
    │
    ├─ Need FAST results? → Use **PCA**
    │
    ├─ Want BEST quality? → Use **Autoencoder**
    │
    ├─ Need 2D visualization? → Use **UMAP** or **t-SNE**
    │
    └─ Not sure? → Run **compare_models.py**
```

### 2. **Evaluation Metrics**

The comparison script evaluates models on:

| Metric | What it Measures | Higher is Better? |
|--------|------------------|-------------------|
| **Neighborhood Preservation** | Are similar songs still neighbors in embedding space? | ✅ Yes |
| **Reconstruction MSE** | Can we recover original features? | ❌ No (lower is better) |
| **Explained Variance** | How much information is retained? | ✅ Yes |
| **Training Time** | How long does it take? | ❌ No (faster is better) |

### 3. **When to Use Each Model**

#### **Use PCA when:**
- ✅ You need results quickly (< 5 seconds)
- ✅ You want interpretable principal components
- ✅ You're doing initial exploration
- ✅ Linear relationships are sufficient
- ✅ You have limited computational resources

**Example:**
```python
from embedding import PCAModel

model = PCAModel(input_dim=170, embedding_dim=32)
model.fit(X)
embeddings = model.transform(X)

# Analyze which features are most important
model.plot_explained_variance()
```

#### **Use Autoencoder when:**
- ✅ You want the **best quality** embeddings
- ✅ You have time for training (~2-10 minutes)
- ✅ Your data has **nonlinear patterns**
- ✅ You need good reconstruction quality
- ✅ You want to learn complex musical relationships

**Example:**
```python
from embedding import AutoencoderModel

model = AutoencoderModel(
    input_dim=170,
    embedding_dim=32,
    encoder_layers=(128, 64),
    epochs=100,
    batch_size=32
)

model.fit(X_train, X_val=X_val)
embeddings = model.transform(X)

# View training progress
model.plot_training_history()
```

#### **Use UMAP when:**
- ✅ You need **2D/3D visualization**
- ✅ You want to preserve **local structure** (nearby songs)
- ✅ You need something faster than autoencoder but better than PCA
- ✅ You're creating interactive plots for demos

**Example:**
```python
from embedding import UMAPModel

model = UMAPModel(
    input_dim=170,
    embedding_dim=2,  # 2D for visualization
    n_neighbors=15,
    min_dist=0.1
)

model.fit(X)
embeddings_2d = model.transform(X)

# Now you can plot directly
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
```

---

## Detailed Training Examples

### Example 1: Train Autoencoder with Custom Parameters

```bash
python embedding/train.py \
    --model autoencoder \
    --dim 32 \
    --epochs 150 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --patience 15 \
    --evaluate \
    --save_plots
```

**Parameters explained:**
- `--dim 32`: Compress 170D → 32D
- `--epochs 150`: Train for up to 150 epochs
- `--batch_size 64`: Use larger batches (faster, but needs more memory)
- `--learning_rate 0.0005`: Lower learning rate for more stable training
- `--patience 15`: Stop if no improvement for 15 epochs
- `--evaluate`: Calculate quality metrics
- `--save_plots`: Save training curves

### Example 2: Train Multiple Dimensions

Try different embedding dimensions to find the best:

```bash
# 16D (more compression)
python embedding/train.py --model autoencoder --dim 16

# 32D (balanced)
python embedding/train.py --model autoencoder --dim 32

# 64D (less compression, more info retained)
python embedding/train.py --model autoencoder --dim 64
```

**Rule of thumb:**
- **16D**: Very compact, good for large datasets, some info loss
- **32D**: Sweet spot for most use cases ✅
- **64D**: High quality, but more expensive for clustering

### Example 3: Python API Usage

```python
from data_loading import load_features
from embedding import AutoencoderModel, PCAModel
from embedding.base import EmbeddingEvaluator

# Load data
dataset, _ = load_features('../Member1/output/features.parquet', preprocess=True)
X = dataset.get_feature_matrix()

# Split data
n_val = int(len(X) * 0.2)
X_train, X_val = X[n_val:], X[:n_val]

# Train autoencoder
model = AutoencoderModel(input_dim=170, embedding_dim=32, epochs=100)
model.fit(X_train, X_val=X_val, verbose=True)

# Get embeddings
embeddings = model.transform(X)

# Evaluate quality
evaluator = EmbeddingEvaluator()
reconstructed = model.reconstruct(X)
metrics = evaluator.evaluate_all(X, embeddings, reconstructed)
evaluator.print_evaluation(metrics)

# Save model
model.save('models/my_autoencoder.pkl')

# Later: Load model
loaded_model = AutoencoderModel.load('models/my_autoencoder.pkl')
new_embeddings = loaded_model.transform(new_data)
```

---

## Understanding the Metrics

### 1. **Neighborhood Preservation (0-1, higher is better)**

Measures if k-nearest neighbors in original space remain neighbors in embedding space.

- **0.8-1.0**: Excellent - similar songs stay together ✅
- **0.6-0.8**: Good - reasonable preservation
- **< 0.6**: Poor - losing similarity information ❌

### 2. **Reconstruction MSE (lower is better)**

How well can we reconstruct the original 170D features from embeddings?

- **< 0.1**: Excellent reconstruction ✅
- **0.1-0.5**: Good - most info retained
- **> 0.5**: Poor - too much information lost ❌

### 3. **Explained Variance Ratio (0-1, higher is better)**

What fraction of the original variance is captured?

- **> 0.9**: Excellent - retains 90%+ of information ✅
- **0.7-0.9**: Good - reasonable compression
- **< 0.7**: High compression - more info loss

---

## Common Issues & Solutions

### Issue 1: Autoencoder Not Converging

**Symptoms:**
- Training loss stays high
- Validation loss oscillates

**Solutions:**
```bash
# Lower learning rate
python embedding/train.py --model autoencoder --learning_rate 0.0001

# Increase epochs
python embedding/train.py --model autoencoder --epochs 200

# Add more capacity
# Edit config.py to use larger encoder_layers: (256, 128, 64)
```

### Issue 2: PCA Explains Too Little Variance

**Symptoms:**
- Cumulative variance < 0.7 with 32 components

**Solutions:**
```bash
# Use more components
python embedding/train.py --model pca --dim 64

# Or switch to autoencoder for nonlinear patterns
python embedding/train.py --model autoencoder --dim 32
```

### Issue 3: Training Takes Too Long

**Solutions:**
```bash
# Use PCA instead
python embedding/train.py --model pca --dim 32

# Or reduce epochs/batch size
python embedding/train.py --model autoencoder --epochs 50 --batch_size 64

# Or use smaller data subset for testing
```

### Issue 4: Out of Memory (GPU/RAM)

**Solutions:**
```bash
# Reduce batch size
python embedding/train.py --model autoencoder --batch_size 16

# Force CPU usage (edit code to set device='cpu')
```

---

## Recommended Workflow for Hackathon

### Phase 1: Quick Baseline (5 minutes)
```bash
# Get quick results with PCA
python embedding/train.py --model pca --dim 32 --evaluate
```

### Phase 2: Best Quality (15 minutes)
```bash
# Train autoencoder for best results
python embedding/train.py --model autoencoder --dim 32 --epochs 100 --evaluate --save_plots
```

### Phase 3: Visualization (10 minutes)
```bash
# Create 2D embeddings for demo plots
python embedding/train.py --model umap --dim 2 --evaluate
```

### Phase 4: Compare & Select (5 minutes)
```bash
# Compare all models and get recommendation
python compare_models.py
```

---

## Output Files

After training, you'll get:

```
output/
└── autoencoder_20240115_120000/
    ├── autoencoder_model.pkl      # Trained model weights
    ├── autoencoder_model.json     # Model metadata
    ├── embeddings.npy             # Embedding vectors (N x 32)
    ├── song_mapping.json          # Song names and paths
    ├── metrics.json               # Evaluation metrics
    ├── config.json                # Training configuration
    └── plots/
        └── training_history.png   # Loss curves
```

**Use these files for:**
- `embeddings.npy` → Input for clustering (Member 3)
- `song_mapping.json` → Map cluster IDs back to songs
- `autoencoder_model.pkl` → Embed new songs later

---

## Next Steps

After selecting and training your embedding model:

1. ✅ **Save the embeddings**: `embeddings.npy`
2. ✅ **Save the model**: For inference on new songs
3. 🚧 **Move to Component 3**: Clustering
4. 🚧 **Move to Component 4**: Visualization

---

## Judge-Facing Explanation

> "We compared three approaches: PCA (linear baseline), Autoencoder (deep learning), and UMAP (manifold learning). The [CHOSEN MODEL] achieved the best neighborhood preservation score of [X.XX], meaning similar-sounding songs remain neighbors in our learned 32-dimensional embedding space. This proves we can compress 170 acoustic features into a compact representation without losing musical similarity information."

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Compare all models | `python compare_models.py` |
| Train autoencoder | `python embedding/train.py --model autoencoder --dim 32` |
| Train PCA | `python embedding/train.py --model pca --dim 32` |
| Train UMAP | `python embedding/train.py --model umap --dim 32` |
| Test autoencoder | `python embedding/autoencoder.py` |
| Test PCA | `python embedding/pca_model.py` |
| Test UMAP | `python embedding/umap_model.py` |

---

**For help:** Run any script with `--help`
```bash
python embedding/train.py --help
```
