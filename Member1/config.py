# Audio Feature Engineering - Configuration
# All audio processing parameters are centralized here

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AudioConfig:
    """Configuration for audio loading and preprocessing."""
    
    # Sampling parameters
    sample_rate: int = 22050  # Standard for music analysis
    mono: bool = True  # Convert to mono for consistency
    
    # STFT parameters
    n_fft: int = 2048  # FFT window size
    hop_length: int = 512  # Hop between frames
    win_length: int = 2048  # Window length
    
    # Mel spectrogram parameters
    n_mels: int = 128  # Number of Mel bands
    fmin: float = 20.0  # Minimum frequency
    fmax: float = 8000.0  # Maximum frequency (Nyquist/2 approx)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    
    # MFCC parameters
    n_mfcc: int = 20  # Number of MFCCs to extract (expanded for richer representation)
    
    # Chroma parameters
    n_chroma: int = 12  # 12-bin pitch class
    
    # Spectral contrast parameters
    n_bands: int = 6  # Number of frequency bands
    
    # Statistics to compute per feature
    statistics: Tuple[str, ...] = ('mean', 'std', 'skew', 'kurtosis')


@dataclass
class PipelineConfig:
    """Configuration for the extraction pipeline."""
    
    # Processing
    n_jobs: int = -1  # Parallel jobs (-1 = all cores)
    batch_size: int = 32  # Files per batch
    
    # Output vector configuration
    target_vector_size: int = 170  # Approximate target dimension
    
    # Supported audio formats
    supported_formats: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')


@dataclass
class ScalerConfig:
    """Configuration for feature scaling."""
    
    method: str = 'zscore'  # 'minmax' or 'zscore'
    feature_range: Tuple[float, float] = (0.0, 1.0)  # For minmax scaling


@dataclass
class ExportConfig:
    """Configuration for data export."""
    
    # Default export format
    default_format: str = 'parquet'  # 'csv', 'npy', 'parquet'
    
    # CSV options
    csv_float_precision: int = 6
    
    # Parquet options
    parquet_compression: str = 'snappy'


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Paths
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    
    def get_output_dir(self) -> str:
        """Get default output directory."""
        return os.path.join(self.project_root, 'output')
    
    def get_sample_audio_dir(self) -> str:
        """Get sample audio directory."""
        return os.path.join(self.project_root, 'sample_audio')


# Global default configuration
DEFAULT_CONFIG = Config()


# Feature vector schema (for documentation and validation)
FEATURE_SCHEMA = {
    'time_domain': {
        'short_time_energy': {'dims': 1, 'stats': 4, 'total': 4},
        'rms_energy': {'dims': 1, 'stats': 4, 'total': 4},
        'zero_crossing_rate': {'dims': 1, 'stats': 4, 'total': 4},
        'inter_beat_interval': {'dims': 1, 'stats': 4, 'total': 4},
        'dynamic_range': {'dims': 1, 'stats': 1, 'total': 1},
        'onset_rate': {'dims': 1, 'stats': 1, 'total': 1},
    },
    'timbral': {
        'mfcc': {'dims': 20, 'stats': 4, 'total': 80},
        'mfcc_delta': {'dims': 20, 'stats': 4, 'total': 80},
        'mfcc_delta2': {'dims': 20, 'stats': 4, 'total': 80},
        'spectral_centroid': {'dims': 1, 'stats': 4, 'total': 4},
        'spectral_rolloff': {'dims': 1, 'stats': 4, 'total': 4},
        'spectral_contrast': {'dims': 7, 'stats': 4, 'total': 28},
        'spectral_bandwidth': {'dims': 1, 'stats': 4, 'total': 4},
        'spectral_flatness': {'dims': 1, 'stats': 4, 'total': 4},
        'spectral_flux': {'dims': 1, 'stats': 4, 'total': 4},
    },
    'rhythmic': {
        'tempo': {'dims': 1, 'stats': 1, 'total': 1},
        'beat_strength': {'dims': 1, 'stats': 4, 'total': 4},
        'onset_strength': {'dims': 1, 'stats': 4, 'total': 4},
    },
    'harmonic': {
        'chroma': {'dims': 12, 'stats': 4, 'total': 48},
        'tonnetz': {'dims': 6, 'stats': 4, 'total': 24},
    },
    'groove': {
        'fundamental_frequency': {'dims': 1, 'stats': 4, 'total': 4},
        'harmonic_energy_ratio': {'dims': 1, 'stats': 1, 'total': 1},
        'rhythmic_complexity': {'dims': 1, 'stats': 1, 'total': 1},
        'syncopation_index': {'dims': 1, 'stats': 1, 'total': 1},
        'perceived_loudness': {'dims': 1, 'stats': 1, 'total': 1},
        'crest_factor': {'dims': 1, 'stats': 1, 'total': 1},
        'attack_time': {'dims': 1, 'stats': 4, 'total': 4},
        'decay_time': {'dims': 1, 'stats': 4, 'total': 4},
        'beat_strength_dist': {'dims': 1, 'stats': 4, 'total': 4},
    },
    'structural': {
        'section_count': {'dims': 1, 'stats': 1, 'total': 1},
        'section_duration': {'dims': 1, 'stats': 4, 'total': 4},
        'repetition_score': {'dims': 1, 'stats': 1, 'total': 1},
        'novelty_curve': {'dims': 1, 'stats': 4, 'total': 4},
    }
}


def get_total_feature_dims() -> int:
    """Calculate total feature vector dimensions from schema."""
    total = 0
    for category in FEATURE_SCHEMA.values():
        for feature in category.values():
            total += feature['total']
    return total


if __name__ == '__main__':
    config = Config()
    print(f"Sample Rate: {config.audio.sample_rate}")
    print(f"Total Feature Dimensions: {get_total_feature_dims()}")
