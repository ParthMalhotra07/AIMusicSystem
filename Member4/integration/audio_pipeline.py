"""
End-to-End Audio Processing Pipeline

Integrates Member1 (Audio → Features) and Member2 (Features → Embeddings)
for real-time processing of uploaded audio files.

HIGH QUALITY VERSION:
- Analyzes FULL song (not just first 30 seconds)
- 512-dimensional embeddings for better discrimination
- More audio features for accuracy
- Higher quality similarity scores
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging
from typing import Optional, Tuple, Dict
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Member1 and Member2 to path
MEMBER4_ROOT = Path(__file__).parent.parent
AIMUSIICSYSTEM_ROOT = MEMBER4_ROOT.parent
MEMBER1_PATH = AIMUSIICSYSTEM_ROOT / "Member1"
MEMBER2_PATH = AIMUSIICSYSTEM_ROOT / "Member2"

sys.path.insert(0, str(MEMBER1_PATH))
sys.path.insert(0, str(MEMBER2_PATH))

# Import Member1 components (Audio Feature Extraction)
try:
    from pipeline.extractor import AudioFeatureExtractor
    from config import Config, AudioConfig, FeatureConfig, PipelineConfig
    HAS_MEMBER1 = True
    logger.info("✅ Member1 (Audio Feature Extractor) loaded successfully")
except ImportError as e:
    HAS_MEMBER1 = False
    logger.warning(f"⚠️ Member1 not available: {e}")

# Import Member2 components (Embedding Generation)
try:
    from embedding.autoencoder import AutoencoderModel
    from embedding.pca_model import PCAModel
    from data_loading.preprocessor import FeaturePreprocessor
    HAS_MEMBER2 = True
    logger.info("✅ Member2 (Embedding Models) loaded successfully")
except ImportError as e:
    HAS_MEMBER2 = False
    logger.warning(f"⚠️ Member2 not available: {e}")


def create_fast_config() -> 'Config':
    """
    Create an optimized config for fast audio processing.
    
    Optimizations:
    - Lower sample rate (16000 Hz vs 22050 Hz) = 27% fewer samples
    - Larger hop length (1024 vs 512) = 50% fewer frames
    - Fewer MFCCs (13 vs 20) = faster computation
    - Only mean/std statistics (vs mean/std/skew/kurtosis)
    """
    audio_config = AudioConfig(
        sample_rate=16000,      # Lower sample rate for speed
        mono=True,
        n_fft=2048,
        hop_length=1024,        # Larger hop = fewer frames
        win_length=2048,
        n_mels=64,              # Fewer mel bands
        fmin=50.0,
        fmax=7500.0
    )
    
    feature_config = FeatureConfig(
        n_mfcc=13,              # Standard 13 MFCCs (faster)
        n_chroma=12,
        n_bands=6,
        statistics=('mean', 'std')  # Only 2 stats instead of 4
    )
    
    pipeline_config = PipelineConfig(
        n_jobs=1,               # Single thread for simplicity
        batch_size=1
    )
    
    return Config(
        audio=audio_config,
        features=feature_config,
        pipeline=pipeline_config
    )


class FastAudioFeatureExtractor:
    """
    Simplified, fast feature extractor for real-time processing.
    
    Only extracts essential features:
    - MFCCs (13 coefficients × 2 stats = 26 features)
    - Spectral centroid, rolloff, bandwidth (3 × 2 = 6 features)
    - Tempo (1 feature)
    - RMS energy (2 features)
    - Chroma (12 × 2 = 24 features)
    
    Total: ~60 features (vs 336 in full mode)
    Processing time: ~2-3 seconds (vs 15-30 seconds)
    """
    
    def __init__(self, max_duration: float = None, sample_rate: int = 22050):
        """
        Initialize feature extractor.
        
        Parameters
        ----------
        max_duration : float or None
            Maximum audio duration to process (seconds). None = full song.
        sample_rate : int
            Target sample rate (22050 for better quality)
        """
        self.max_duration = max_duration  # None = full song
        self.sample_rate = sample_rate
        self.n_mfcc = 13
        self.hop_length = 1024
        self.n_fft = 2048
        
    def process_file(self, audio_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract features from audio file (fast mode).
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
            
        Returns
        -------
        features : np.ndarray
            Feature vector
        metadata : dict
            Audio metadata
        """
        import librosa
        
        # Load full song (or max_duration if specified)
        y, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
            duration=self.max_duration  # None = full song
        )
        
        features = []
        
        # 1. MFCCs (essential timbral features)
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        features.extend([mfccs.mean(axis=1), mfccs.std(axis=1)])
        
        # 2. Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        features.append(np.array([spec_cent.mean(), spec_cent.std()]))
        features.append(np.array([spec_rolloff.mean(), spec_rolloff.std()]))
        features.append(np.array([spec_bw.mean(), spec_bw.std()]))
        
        # 3. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        features.append(np.array([rms.mean(), rms.std()]))
        
        # 4. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        # Handle both scalar and array tempo returns
        tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        features.append(np.array([tempo_val]))
        
        # 5. Chroma (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        features.extend([chroma.mean(axis=1), chroma.std(axis=1)])
        
        # 6. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        features.append(np.array([zcr.mean(), zcr.std()]))
        
        # Concatenate all features
        feature_vector = np.concatenate([f.flatten() for f in features]).astype(np.float32)
        
        # Metadata
        metadata = {
            'duration_seconds': len(y) / sr,
            'tempo_bpm': tempo_val,
            'sample_rate': sr,
            'spectral_centroid': float(spec_cent.mean()),
            'rms': float(rms.mean()),
            'file_path': audio_path,
            'file_name': os.path.basename(audio_path),
            'key': 'Unknown',  # Skip key detection for speed
            'key_confidence': 0.0
        }
        
        return feature_vector, metadata


class AudioProcessingPipeline:
    """
    End-to-end pipeline for processing audio files.
    
    Pipeline stages:
    1. Load audio file
    2. Extract audio features (Member1 for Complete Mode, Fast Extractor for Fast Mode)
    3. Scale/preprocess features (Member2 preprocessor)
    4. Generate embeddings (Member2 autoencoder for Complete, custom for Fast)
    
    FAST MODE: ~3-5 seconds per song, 61 features → 512D custom embedding
    COMPLETE MODE: ~15-60 seconds per song, 336D Member1 features → 14D Member2 autoencoder embedding
    """
    
    def __init__(
        self,
        embedding_model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        embedding_dim: int = 512,
        use_pca_fallback: bool = True,
        fast_mode: bool = True,
        max_duration: float = None,
        use_member_pipeline: bool = False
    ):
        """
        Initialize the audio processing pipeline.
        
        Parameters
        ----------
        embedding_model_path : str, optional
            Path to trained embedding model (.pkl or .pt)
        preprocessor_path : str, optional
            Path to feature preprocessor (.json)
        embedding_dim : int
            Target embedding dimension (512 for Fast Mode, 14 for Member2 autoencoder)
        use_pca_fallback : bool
            Use PCA if trained model not available
        fast_mode : bool
            Use fast feature extraction
        max_duration : float or None
            Max audio duration (None = analyze full song)
        use_member_pipeline : bool
            If True, use Member1→Member2 full pipeline (Complete Mode)
        """
        self.use_pca_fallback = use_pca_fallback
        self.fast_mode = fast_mode
        self.max_duration = max_duration
        self.use_member_pipeline = use_member_pipeline
        
        # Set embedding dimension based on mode
        if use_member_pipeline and HAS_MEMBER1 and HAS_MEMBER2:
            self.embedding_dim = 14  # Member2 autoencoder output
            self.feature_dim = 336   # Member1 output
        else:
            self.embedding_dim = embedding_dim or 512
            self.feature_dim = 60  # Fast mode
        
        # Initialize feature extractors
        self.member1_extractor = None
        self.fast_extractor = None
        self.member2_model = None
        self.member2_preprocessor = None
        
        # COMPLETE MODE: Use Member1 + Member2
        if use_member_pipeline and HAS_MEMBER1 and HAS_MEMBER2:
            try:
                # Initialize Member1 full feature extractor
                self.member1_extractor = AudioFeatureExtractor()
                self.feature_dim = self.member1_extractor.feature_dim
                logger.info(f"🎵 Member1 AudioFeatureExtractor initialized ({self.feature_dim}D features)")
                
                # Load Member2 autoencoder model
                if embedding_model_path is None:
                    embedding_model_path = str(MEMBER2_PATH / "embedding_model.pkl")
                
                if os.path.exists(embedding_model_path):
                    self.member2_model = AutoencoderModel.load(embedding_model_path)
                    self.embedding_dim = 14  # Autoencoder output dimension
                    logger.info(f"🔮 Member2 Autoencoder loaded ({self.embedding_dim}D embeddings)")
                else:
                    logger.warning(f"⚠️ Member2 model not found at {embedding_model_path}")
                
                # Load Member2 preprocessor
                if preprocessor_path is None:
                    preprocessor_path = str(MEMBER2_PATH / "preprocessor.json")
                
                if os.path.exists(preprocessor_path):
                    self.member2_preprocessor = FeaturePreprocessor.load(preprocessor_path)
                    logger.info("📊 Member2 preprocessor loaded")
                else:
                    logger.warning(f"⚠️ Preprocessor not found at {preprocessor_path}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Member pipeline: {e}. Falling back to Fast Mode.")
                self.use_member_pipeline = False
                self._init_fast_mode()
        else:
            # FAST MODE: Use simplified extractor
            self._init_fast_mode()
    
    def _init_fast_mode(self):
        """Initialize fast mode components."""
        # Ensure embedding dimension is sufficient for fast mode features
        # Fast mode logic writes 61+ features into the embedding, so it needs space
        if self.embedding_dim < 64:
            self.embedding_dim = 512
            
        self.fast_extractor = FastAudioFeatureExtractor(
            max_duration=self.max_duration,
            sample_rate=22050
        )
        self.feature_dim = 60
        duration_str = f"{self.max_duration}s" if self.max_duration else "full song"
        logger.info(f"⚡ Fast feature extractor initialized ({duration_str}, ~60 features)")

    def _load_embedding_model(self, model_path: Optional[str]):
        """Load or create embedding model (for fast mode only)."""
        # Member pipeline handles its own model loading in __init__
        if self.use_member_pipeline:
            return
            
        if model_path and os.path.exists(model_path):
            try:
                if model_path.endswith('.pt') or model_path.endswith('.pth'):
                    self.embedding_model = AutoencoderModel.load(model_path)
                    logger.info(f"Autoencoder model loaded from {model_path}")
                elif model_path.endswith('.pkl'):
                    import joblib
                    self.embedding_model = joblib.load(model_path)
                    logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        if self.embedding_model is None and self.use_pca_fallback and HAS_MEMBER2:
            logger.info("Using PCA fallback for embeddings")
    
    def extract_features(self, audio_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract audio features from a file.
        
        Complete Mode: Uses Member1 AudioFeatureExtractor (336D features)
        Fast Mode: Uses FastAudioFeatureExtractor (~61D features)
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
            
        Returns
        -------
        features : np.ndarray
            Feature vector
        metadata : dict
            Audio metadata (tempo, key, duration, etc.)
        """
        # COMPLETE MODE: Use Member1 full feature extractor
        if self.use_member_pipeline and self.member1_extractor is not None:
            try:
                logger.info(f"🎵 Using Member1 AudioFeatureExtractor for {audio_path}")
                features, metadata = self.member1_extractor.process_file(
                    audio_path, 
                    return_metadata=True
                )
                logger.info(f"   → Member1 extracted {len(features)}D features")
                return features, metadata
            except Exception as e:
                logger.error(f"Member1 feature extraction failed: {e}")
                raise
        
        # FAST MODE: Use fast extractor
        if self.fast_extractor is not None:
            try:
                return self.fast_extractor.process_file(audio_path)
            except Exception as e:
                logger.error(f"Fast feature extraction failed: {e}")
                raise
        
        raise RuntimeError("No feature extractor available. Install librosa.")
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess/scale features for embedding model.
        
        Complete Mode: Uses Member2 preprocessor
        Fast Mode: Uses simple z-score normalization
        
        Parameters
        ----------
        features : np.ndarray
            Raw feature vector(s)
            
        Returns
        -------
        scaled_features : np.ndarray
            Preprocessed features
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # COMPLETE MODE: Use Member2 preprocessor
        if self.use_member_pipeline and self.member2_preprocessor is not None:
            try:
                scaled = self.member2_preprocessor.transform(features)
                logger.info(f"📊 Member2 preprocessor applied")
                return scaled
            except Exception as e:
                logger.warning(f"Member2 preprocessing failed: {e}, using z-score")
        
        # FAST MODE: Simple z-score normalization
        return (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-8)
    
    def generate_embedding(
        self, 
        features: np.ndarray,
        existing_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate embedding from features.
        
        Complete Mode: Uses Member2 Autoencoder (336D → 14D)
        Fast Mode: Creates 512D embedding from feature groups
        
        Parameters
        ----------
        features : np.ndarray
            Feature vector(s) to embed
        existing_features : np.ndarray, optional
            Existing feature matrix (for fitting PCA if needed)
            
        Returns
        -------
        embedding : np.ndarray
            Low-dimensional embedding (normalized for cosine similarity)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # COMPLETE MODE: Use Member2 Autoencoder
        if self.use_member_pipeline and self.member2_model is not None:
            try:
                # Adapt dimensions if needed
                model_input_dim = self.member2_model.input_dim
                current_dim = features.shape[1]
                
                features_adapted = features
                if current_dim != model_input_dim:
                    logger.warning(f"Feature dim mismatch: {current_dim} vs {model_input_dim}. Adapting...")
                    if current_dim > model_input_dim:
                        # Truncate
                        features_adapted = features[:, :model_input_dim]
                    else:
                        # Pad with zeros
                        padding = np.zeros((features.shape[0], model_input_dim - current_dim), dtype=features.dtype)
                        features_adapted = np.hstack([features, padding])
                
                logger.info(f"🔮 Using Member2 Autoencoder ({features_adapted.shape[1]}D → {self.embedding_dim}D)")
                embedding = self.member2_model.transform(features_adapted)
                
                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                embedding = embedding / (norm + 1e-8)
                
                logger.info(f"   → Member2 generated {embedding.shape[1]}D embedding")
                return embedding.astype(np.float32)
            except Exception as e:
                logger.error(f"Member2 embedding failed: {e}")
                # Fall through to PCA fallback
        
        # FAST MODE: Generate 512D embedding from feature groups
        if not self.use_member_pipeline:
            # Features layout (~61 features):
            # [0:13] MFCC means, [13:26] MFCC stds
            # [26:28] spectral centroid (mean, std)
            # [28:30] spectral rolloff (mean, std)
            # [30:32] spectral bandwidth (mean, std)
            # [32:34] RMS energy (mean, std)
            # [34:35] tempo
            # [35:47] chroma means, [47:59] chroma stds
            # [59:61] ZCR (mean, std)
            
            embeddings = []
            for feat in features:
                # Create 512-dimensional embedding
                embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                
                # Normalize input features to prevent scale issues
                feat_normalized = feat / (np.abs(feat).max() + 1e-8)
                
                idx = 0
                
                # === Section 1: Raw Features (first 61 dims) ===
                raw_len = min(len(feat_normalized), 61)
                embedding[idx:idx+raw_len] = feat_normalized[:raw_len]
                idx += 64  # Reserve 64 dims for raw features
                
                # === Section 2: MFCC Details (64 dims) ===
                if len(feat) > 26:
                    mfcc_mean = feat[0:13]
                    mfcc_std = feat[13:26]
                    
                    # MFCC means normalized (13 dims)
                    embedding[idx:idx+13] = mfcc_mean / (np.abs(mfcc_mean).max() + 1e-8)
                    idx += 13
                    
                    # MFCC stds normalized (13 dims)
                    embedding[idx:idx+13] = mfcc_std / (np.abs(mfcc_std).max() + 1e-8)
                    idx += 13
                    
                    # MFCC deltas (differences between adjacent coefficients) (12 dims)
                    mfcc_deltas = np.diff(mfcc_mean)
                    embedding[idx:idx+12] = mfcc_deltas / (np.abs(mfcc_deltas).max() + 1e-8)
                    idx += 12
                    
                    # MFCC products (pairwise for first 6) (15 dims)
                    pair_idx = 0
                    for i in range(6):
                        for j in range(i+1, 6):
                            if idx + pair_idx < self.embedding_dim:
                                embedding[idx + pair_idx] = mfcc_mean[i] * mfcc_mean[j] / 100.0
                                pair_idx += 1
                    idx += 15
                    
                    # Skip to next section
                    idx = 128
                
                # === Section 3: Spectral Features Expanded (64 dims) ===
                if len(feat) > 34:
                    centroid = feat[26]
                    rolloff = feat[28]
                    bandwidth = feat[30]
                    rms = feat[32]
                    tempo = feat[34] if len(feat) > 34 else 120.0
                    
                    # Normalized spectral features (5 dims)
                    embedding[idx] = centroid / 5000.0
                    embedding[idx+1] = rolloff / 10000.0
                    embedding[idx+2] = bandwidth / 3000.0
                    embedding[idx+3] = rms * 5.0
                    embedding[idx+4] = (tempo - 100.0) / 60.0
                    idx += 5
                    
                    # Spectral ratios (important for timbre discrimination) (10 dims)
                    embedding[idx] = centroid / (rolloff + 1e-8)
                    embedding[idx+1] = bandwidth / (centroid + 1e-8)
                    embedding[idx+2] = centroid / (bandwidth + 1e-8)
                    embedding[idx+3] = rms * centroid / 1000.0
                    embedding[idx+4] = tempo * rms
                    embedding[idx+5] = np.log1p(centroid) / 10.0
                    embedding[idx+6] = np.log1p(rolloff) / 10.0
                    embedding[idx+7] = np.log1p(bandwidth) / 10.0
                    embedding[idx+8] = np.sqrt(centroid) / 100.0
                    embedding[idx+9] = np.sqrt(rolloff) / 100.0
                    idx += 10
                    
                    # Skip to next section
                    idx = 192
                
                # === Section 4: Chroma Features Expanded (128 dims) ===
                if len(feat) > 59:
                    chroma_mean = feat[35:47]
                    chroma_std = feat[47:59]
                    
                    # Chroma means (12 dims)
                    embedding[idx:idx+12] = chroma_mean
                    idx += 12
                    
                    # Chroma stds (12 dims)
                    embedding[idx:idx+12] = chroma_std
                    idx += 12
                    
                    # Chroma deltas (11 dims)
                    chroma_deltas = np.diff(chroma_mean)
                    embedding[idx:idx+11] = chroma_deltas
                    idx += 11
                    
                    # Chroma cross-correlations (major key relationships) (12 dims)
                    # Perfect 5th intervals (7 semitones)
                    for i in range(12):
                        fifth = (i + 7) % 12
                        embedding[idx + i] = chroma_mean[i] * chroma_mean[fifth]
                    idx += 12
                    
                    # Major 3rd intervals (4 semitones) (12 dims)
                    for i in range(12):
                        third = (i + 4) % 12
                        embedding[idx + i] = chroma_mean[i] * chroma_mean[third]
                    idx += 12
                    
                    # Key strength indicators (12 dims)
                    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
                    for i in range(12):
                        rotated = np.roll(major_template, i)
                        embedding[idx + i] = np.dot(chroma_mean, rotated)
                    idx += 12
                    
                    # Skip to next section
                    idx = 320
                
                # === Section 5: Statistical Features (64 dims) ===
                # Overall feature statistics
                embedding[idx] = np.mean(feat_normalized)
                embedding[idx+1] = np.std(feat_normalized)
                embedding[idx+2] = np.min(feat_normalized)
                embedding[idx+3] = np.max(feat_normalized)
                embedding[idx+4] = np.median(feat_normalized)
                idx += 5
                
                # Feature group statistics
                if len(feat) > 26:
                    embedding[idx] = np.mean(feat[0:13])  # MFCC mean avg
                    embedding[idx+1] = np.std(feat[0:13])  # MFCC mean std
                    embedding[idx+2] = np.mean(feat[13:26])  # MFCC std avg
                    idx += 3
                
                if len(feat) > 47:
                    embedding[idx] = np.mean(feat[35:47])  # Chroma mean avg
                    embedding[idx+1] = np.std(feat[35:47])  # Chroma mean std
                    idx += 2
                
                # Skip to next section
                idx = 384
                
                # === Section 6: Polynomial Feature Expansion (128 dims) ===
                # Use key features to generate polynomial combinations
                if len(feat) > 34:
                    key_feats = np.array([
                        feat[0] / 50.0,   # MFCC1
                        feat[1] / 20.0,   # MFCC2
                        feat[2] / 20.0,   # MFCC3
                        feat[26] / 5000.0,  # Centroid
                        feat[32] * 5.0,     # RMS
                        feat[34] / 120.0    # Tempo (normalized around 120 BPM)
                    ])
                    
                    # Single features (6 dims)
                    for i, kf in enumerate(key_feats):
                        if idx + i < self.embedding_dim:
                            embedding[idx + i] = kf
                    idx += 6
                    
                    # Squared features (6 dims)
                    for i, kf in enumerate(key_feats):
                        if idx + i < self.embedding_dim:
                            embedding[idx + i] = kf ** 2
                    idx += 6
                    
                    # Pairwise products (15 dims)
                    pair_idx = 0
                    for i in range(len(key_feats)):
                        for j in range(i+1, len(key_feats)):
                            if idx + pair_idx < self.embedding_dim:
                                embedding[idx + pair_idx] = key_feats[i] * key_feats[j]
                                pair_idx += 1
                    idx += 15
                    
                    # Sinusoidal transforms for periodicity (24 dims)
                    for i, kf in enumerate(key_feats):
                        if idx + i*4 + 3 < self.embedding_dim:
                            embedding[idx + i*4] = np.sin(kf * np.pi)
                            embedding[idx + i*4 + 1] = np.cos(kf * np.pi)
                            embedding[idx + i*4 + 2] = np.sin(kf * 2 * np.pi)
                            embedding[idx + i*4 + 3] = np.cos(kf * 2 * np.pi)
                    idx += 24
                
                # Fill remaining with random projections of features
                remaining = self.embedding_dim - idx
                if remaining > 0 and len(feat_normalized) > 0:
                    np.random.seed(42)  # Reproducible projections
                    projection_matrix = np.random.randn(len(feat_normalized), remaining) / np.sqrt(len(feat_normalized))
                    embedding[idx:] = feat_normalized @ projection_matrix
                
                # L2 Normalize to unit vector for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings.append(embedding)
            
            return np.array(embeddings, dtype=np.float32)
        
        # If we have a trained model, use it (for full mode)
        if self.embedding_model is not None and hasattr(self.embedding_model, 'transform'):
            try:
                return self.embedding_model.transform(features)
            except Exception as e:
                logger.warning(f"Model transform failed: {e}, falling back to PCA")
        
        # PCA fallback (only for full mode with matching dimensions)
        if HAS_MEMBER2 and existing_features is not None:
            if features.shape[1] == existing_features.shape[1]:
                pca = PCAModel(
                    input_dim=features.shape[1],
                    embedding_dim=min(self.embedding_dim, features.shape[1], existing_features.shape[0])
                )
                # Fit on existing data + new data
                all_features = np.vstack([existing_features, features])
                pca.fit(all_features, verbose=False)
                return pca.transform(features)
            else:
                logger.warning(f"Feature dimension mismatch: {features.shape[1]} vs {existing_features.shape[1]}")
        
        # Last resort: truncate/project to embedding_dim
        logger.warning("No embedding model available, using simple projection")
        return features[:, :self.embedding_dim]
    
    def process_audio(
        self,
        audio_path: str,
        existing_features: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Full pipeline: audio file → features → embedding.
        
        Parameters
        ----------
        audio_path : str
            Path to audio file
        existing_features : np.ndarray, optional
            Existing features for PCA fitting
            
        Returns
        -------
        result : dict
            Contains 'features', 'embedding', 'metadata'
        """
        logger.info(f"Processing audio: {audio_path}")
        
        # Step 1: Extract features
        features, metadata = self.extract_features(audio_path)
        logger.info(f"  → Extracted {len(features)} features")
        
        # Step 2: Preprocess
        scaled_features = self.preprocess_features(features)
        logger.info(f"  → Preprocessed features")
        
        # Step 3: Generate embedding
        embedding = self.generate_embedding(scaled_features, existing_features)
        logger.info(f"  → Generated embedding (dim={embedding.shape[-1]})")
        
        return {
            'features': features,
            'scaled_features': scaled_features.flatten(),
            'embedding': embedding.flatten(),
            'metadata': metadata
        }
    
    def process_uploaded_file(
        self,
        uploaded_file,
        existing_features: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Process an uploaded file (e.g., from Streamlit).
        
        Parameters
        ----------
        uploaded_file : UploadedFile
            Streamlit uploaded file object
        existing_features : np.ndarray, optional
            Existing features for PCA fitting
            
        Returns
        -------
        result : dict
            Processing results
        """
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        try:
            result = self.process_audio(tmp_path, existing_features)
            result['original_filename'] = uploaded_file.name
            return result
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# Singleton instance for efficiency
_pipeline_instance = None


def get_pipeline(
    embedding_model_path: Optional[str] = None,
    preprocessor_path: Optional[str] = None
) -> AudioProcessingPipeline:
    """
    Get or create the audio processing pipeline singleton.
    
    Parameters
    ----------
    embedding_model_path : str, optional
        Path to embedding model
    preprocessor_path : str, optional
        Path to preprocessor
        
    Returns
    -------
    pipeline : AudioProcessingPipeline
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        # Try to find model files
        if embedding_model_path is None:
            model_path = MEMBER2_PATH / "embedding_model.pkl"
            if model_path.exists():
                embedding_model_path = str(model_path)
        
        if preprocessor_path is None:
            prep_path = MEMBER2_PATH / "preprocessor.json"
            if prep_path.exists():
                preprocessor_path = str(prep_path)
        
        _pipeline_instance = AudioProcessingPipeline(
            embedding_model_path=embedding_model_path,
            preprocessor_path=preprocessor_path,
            fast_mode=True,  # Use feature extraction mode
            max_duration=None  # Analyze FULL song for best quality
        )
    
    return _pipeline_instance


def check_pipeline_status() -> Dict[str, bool]:
    """Check the status of pipeline components."""
    # Fast mode only needs librosa, not full Member1
    try:
        import librosa
        fast_mode_ready = True
    except ImportError:
        fast_mode_ready = False
    
    return {
        'member1_available': HAS_MEMBER1,
        'member2_available': HAS_MEMBER2,
        'feature_extractor_ready': fast_mode_ready or HAS_MEMBER1,
        'embedding_model_ready': HAS_MEMBER2 or fast_mode_ready,
        'fast_mode_ready': fast_mode_ready,
    }


if __name__ == "__main__":
    # Test the pipeline
    print("=" * 60)
    print("Audio Processing Pipeline Test (FAST MODE)")
    print("=" * 60)
    
    status = check_pipeline_status()
    print("\nPipeline Status:")
    for key, value in status.items():
        status_icon = "✅" if value else "❌"
        print(f"  {status_icon} {key}: {value}")
    
    if status['fast_mode_ready']:
        print("\n⚡ Fast mode is ready! Processing time: ~3-5 seconds per song")
    
    if HAS_MEMBER1 and HAS_MEMBER2:
        print("✅ Full pipeline is also available for detailed analysis")
    else:
        print("⚠️ Full pipeline missing some components (fast mode still works)")
