"""
End-to-End Audio Processing Pipeline

Integrates Member1 (Audio → Features) and Member2 (Features → Embeddings)
for real-time processing of uploaded audio files.

OPTIMIZED VERSION - Uses fast mode for quick processing:
- Only processes first 30 seconds of audio
- Lower sample rate (16000 Hz)
- Reduced feature set for speed
- Simplified statistics
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
    
    def __init__(self, max_duration: float = 30.0, sample_rate: int = 16000):
        """
        Initialize fast extractor.
        
        Parameters
        ----------
        max_duration : float
            Maximum audio duration to process (seconds)
        sample_rate : int
            Target sample rate
        """
        self.max_duration = max_duration
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
        
        # Load only first N seconds
        y, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
            duration=self.max_duration
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
    2. Extract audio features (Member1 or Fast Mode)
    3. Scale/preprocess features
    4. Generate embeddings (Member2 or PCA)
    
    FAST MODE (default): ~3-5 seconds per song
    FULL MODE: ~15-30 seconds per song (more accurate)
    """
    
    def __init__(
        self,
        embedding_model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        embedding_dim: int = 14,
        use_pca_fallback: bool = True,
        fast_mode: bool = True,
        max_duration: float = 30.0
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
            Target embedding dimension
        use_pca_fallback : bool
            Use PCA if trained model not available
        fast_mode : bool
            Use fast feature extraction (recommended for real-time)
        max_duration : float
            Max audio duration to process in seconds (fast mode only)
        """
        self.embedding_dim = embedding_dim
        self.use_pca_fallback = use_pca_fallback
        self.fast_mode = fast_mode
        self.max_duration = max_duration
        
        # Initialize feature extractor
        self.feature_extractor = None
        self.fast_extractor = None
        
        if fast_mode:
            # Use fast extractor (always available if librosa is installed)
            self.fast_extractor = FastAudioFeatureExtractor(
                max_duration=max_duration,
                sample_rate=16000
            )
            self.feature_dim = 60  # Approximate fast mode dimension
            logger.info(f"⚡ Fast feature extractor initialized (max {max_duration}s, ~60 features)")
        elif HAS_MEMBER1:
            try:
                fast_config = create_fast_config()
                self.feature_extractor = AudioFeatureExtractor(fast_config)
                self.feature_dim = self.feature_extractor.feature_dim
                logger.info(f"Feature extractor initialized (output dim: {self.feature_dim})")
            except Exception as e:
                logger.error(f"Failed to initialize feature extractor: {e}")
                # Fallback to fast mode
                self.fast_extractor = FastAudioFeatureExtractor(max_duration=max_duration)
                self.feature_dim = 60
        
        # Initialize preprocessor (for scaling features)
        self.preprocessor = None
        if preprocessor_path and os.path.exists(preprocessor_path):
            try:
                self.preprocessor = FeaturePreprocessor.load(preprocessor_path)
                logger.info(f"Preprocessor loaded from {preprocessor_path}")
            except Exception as e:
                logger.warning(f"Failed to load preprocessor: {e}")
        
        # Initialize embedding model
        self.embedding_model = None
        self._load_embedding_model(embedding_model_path)
    
    def _load_embedding_model(self, model_path: Optional[str]):
        """Load or create embedding model."""
        if model_path and os.path.exists(model_path):
            try:
                # Try loading autoencoder
                if model_path.endswith('.pt') or model_path.endswith('.pth'):
                    self.embedding_model = AutoencoderModel.load(model_path)
                    logger.info(f"Autoencoder model loaded from {model_path}")
                elif model_path.endswith('.pkl'):
                    import joblib
                    self.embedding_model = joblib.load(model_path)
                    logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        # Use PCA fallback if no model loaded
        if self.embedding_model is None and self.use_pca_fallback and HAS_MEMBER2:
            logger.info("Using PCA fallback for embeddings")
            # Will fit PCA on existing data when first used
            self.embedding_model = None  # Will be created dynamically
    
    def extract_features(self, audio_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Extract audio features from a file.
        
        Uses fast extractor by default for real-time processing.
        
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
        # Use fast extractor if available (preferred for speed)
        if self.fast_extractor is not None:
            try:
                return self.fast_extractor.process_file(audio_path)
            except Exception as e:
                logger.error(f"Fast feature extraction failed: {e}")
                raise
        
        # Fall back to full Member1 extractor
        if self.feature_extractor is not None:
            try:
                features, metadata = self.feature_extractor.process_file(
                    audio_path, 
                    return_metadata=True
                )
                return features, metadata
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                raise
        
        raise RuntimeError("No feature extractor available. Install librosa.")
    
    def preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """
        Preprocess/scale features for embedding model.
        
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
        
        # In fast mode, always use simple normalization (preprocessor expects 336 features)
        if self.fast_mode:
            # Simple z-score normalization for fast mode
            return (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-8)
        
        if self.preprocessor is not None:
            return self.preprocessor.transform(features)
        
        # Simple z-score normalization as fallback
        return (features - features.mean()) / (features.std() + 1e-8)
    
    def generate_embedding(
        self, 
        features: np.ndarray,
        existing_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate embedding from features.
        
        For fast mode, creates a meaningful embedding by:
        1. Grouping features by type (MFCC, spectral, chroma, etc.)
        2. Computing weighted averages within each group
        3. Normalizing to unit vector for cosine similarity
        
        This ensures different songs with similar audio characteristics
        will have similar embeddings.
        
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
        
        # In fast mode, create a meaningful embedding from feature groups
        if self.fast_mode:
            # Fast mode features layout (61 features):
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
                embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                
                # Group 1: Timbre (MFCCs) - dims 0-3
                if len(feat) > 13:
                    mfcc_mean = feat[0:13]
                    # Use first 4 MFCCs (most important for timbre)
                    embedding[0:4] = mfcc_mean[0:4] / (np.abs(mfcc_mean[0:4]).max() + 1e-8)
                
                # Group 2: Brightness/Spectral - dims 4-6
                if len(feat) > 32:
                    embedding[4] = feat[26] / 5000.0  # spectral centroid (normalize ~0-5000 Hz)
                    embedding[5] = feat[28] / 10000.0  # spectral rolloff
                    embedding[6] = feat[30] / 3000.0  # spectral bandwidth
                
                # Group 3: Energy - dims 7-8
                if len(feat) > 34:
                    embedding[7] = feat[32] * 5.0  # RMS energy (scale up, usually 0-0.2)
                    embedding[8] = feat[33] * 5.0  # RMS std
                
                # Group 4: Tempo - dim 9
                if len(feat) > 35:
                    embedding[9] = (feat[34] - 100.0) / 60.0  # Normalize tempo (60-160 BPM range)
                
                # Group 5: Harmony (Chroma) - dims 10-12
                if len(feat) > 47:
                    chroma_mean = feat[35:47]
                    # Summarize 12 chroma bins into 3 features
                    embedding[10] = chroma_mean[0:4].mean()  # C, C#, D, D#
                    embedding[11] = chroma_mean[4:8].mean()  # E, F, F#, G
                    embedding[12] = chroma_mean[8:12].mean()  # G#, A, A#, B
                
                # Group 6: Rhythm/Texture - dim 13
                if len(feat) > 60:
                    embedding[13] = feat[59] * 10.0  # ZCR (scale up)
                
                # Normalize to unit vector for cosine similarity
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
            fast_mode=True,  # Use fast mode by default
            max_duration=30.0  # Only process first 30 seconds
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
