"""
Audio Feature Extractor Pipeline

This is the main orchestration module that combines all feature
extraction steps into a single, easy-to-use pipeline.

Usage:
    extractor = AudioFeatureExtractor()
    features = extractor.process_file('song.mp3')
    batch_features = extractor.process_batch(['song1.mp3', 'song2.mp3'])
"""

import os
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# Import configuration
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DEFAULT_CONFIG, FEATURE_SCHEMA, get_total_feature_dims

# Import preprocessing
from preprocessing.audio_loader import load_audio, validate_audio, normalize_amplitude
from preprocessing.spectral import compute_mel_spectrogram

# Import feature extractors
from features.timbral import extract_all_timbral_features
from features.rhythmic import extract_all_rhythmic_features, extract_tempo
from features.harmonic import extract_all_harmonic_features, estimate_key
from features.time_domain import extract_all_time_domain_features
from features.groove import extract_all_groove_features
from features.structural import extract_all_structural_features
from features.statistics import compute_statistics, aggregate_features


class AudioFeatureExtractor:
    """
    Main pipeline for extracting audio features.
    
    This class orchestrates the entire feature extraction process:
    1. Load and validate audio
    2. Extract timbral features (MFCCs, spectral)
    3. Extract rhythmic features (tempo, beats)
    4. Extract harmonic features (chroma, tonnetz)
    5. Aggregate into fixed-length vector
    
    Attributes
    ----------
    config : Config
        Configuration object with all processing parameters
    feature_names : list
        Names of all features in the output vector
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the feature extractor.
        
        Parameters
        ----------
        config : Config, optional
            Configuration object (default: DEFAULT_CONFIG)
        """
        self.config = config or DEFAULT_CONFIG
        self._feature_names = None
    
    @property
    def feature_names(self) -> List[str]:
        """Get the names of all features in the output vector."""
        if self._feature_names is None:
            self._feature_names = self._generate_feature_names()
        return self._feature_names
    
    @property
    def feature_dim(self) -> int:
        """Get the total dimension of the feature vector."""
        return get_total_feature_dims()
    
    def _generate_feature_names(self) -> List[str]:
        """Generate names for all features in the output vector."""
        names = []
        stats = self.config.features.statistics
        n_mfcc = self.config.features.n_mfcc
        
        # Time-domain features
        for feature in ['short_time_energy', 'rms_energy', 'zero_crossing_rate']:
            for stat in stats:
                names.append(f'{feature}_{stat}')
        
        for stat in stats:
            names.append(f'inter_beat_interval_{stat}')
        names.append('dynamic_range')
        names.append('onset_rate')
        
        # Timbral features - MFCCs
        for prefix in ['mfcc', 'mfcc_delta', 'mfcc_delta2']:
            for i in range(n_mfcc):
                for stat in stats:
                    names.append(f'{prefix}_{i}_{stat}')
        
        # Other spectral features
        for feature in ['spectral_centroid', 'spectral_rolloff', 
                        'spectral_bandwidth', 'spectral_flatness', 'spectral_flux']:
            for stat in stats:
                names.append(f'{feature}_{stat}')
        
        # Spectral contrast (7 bands)
        for i in range(7):
            for stat in stats:
                names.append(f'spectral_contrast_{i}_{stat}')
        
        # Rhythmic features
        names.append('tempo')
        
        for feature in ['onset_strength', 'beat_strength']:
            for stat in stats:
                names.append(f'{feature}_{stat}')
        
        # Harmonic features
        for i in range(self.config.features.n_chroma):
            for stat in stats:
                names.append(f'chroma_{i}_{stat}')
        
        for i in range(6):  # Tonnetz has 6 dimensions
            for stat in stats:
                names.append(f'tonnetz_{i}_{stat}')
        
        # Groove features
        for stat in stats:
            names.append(f'fundamental_frequency_{stat}')
        
        for feature in ['harmonic_energy_ratio', 'rhythmic_complexity', 
                        'syncopation_index', 'perceived_loudness', 'crest_factor']:
            names.append(feature)
        
        for feature in ['attack_time', 'decay_time', 'beat_strength_dist']:
            for stat in stats:
                names.append(f'{feature}_{stat}')
        
        # Structural features
        names.append('section_count')
        for stat in stats:
            names.append(f'section_duration_{stat}')
        names.append('repetition_score')
        for stat in stats:
            names.append(f'novelty_curve_{stat}')
        
        return names
    
    def process_file(
        self,
        audio_path: str,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Extract features from a single audio file.
        
        Parameters
        ----------
        audio_path : str
            Path to the audio file
        return_metadata : bool, optional
            Whether to return metadata alongside features (default: False)
            
        Returns
        -------
        features : np.ndarray
            Fixed-length feature vector
        metadata : dict, optional
            Metadata including file info, tempo, key, etc.
        """
        # Load audio
        y, sr = load_audio(
            audio_path,
            sr=self.config.audio.sample_rate,
            mono=self.config.audio.mono
        )
        
        # Validate
        validation = validate_audio(y, sr)
        if not validation['is_valid']:
            print(f"Warning: Audio validation issues for {audio_path}:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        # Normalize amplitude
        y = normalize_amplitude(y, method='peak')
        
        # Extract all features
        features_dict = self._extract_all_features(y, sr)
        
        # Aggregate to fixed-length vector
        feature_vector = self._aggregate_to_vector(features_dict)
        
        if return_metadata:
            metadata = self._extract_metadata(audio_path, y, sr, features_dict)
            return feature_vector, metadata
        
        return feature_vector
    
    def _extract_all_features(
        self,
        y: np.ndarray,
        sr: int
    ) -> Dict[str, np.ndarray]:
        """Extract all feature categories from audio."""
        n_fft = self.config.audio.n_fft
        hop_length = self.config.audio.hop_length
        n_mfcc = self.config.features.n_mfcc
        n_mels = self.config.audio.n_mels
        
        features = {}
        
        # Time-domain features (STE, IBI, Dynamic Range, Onset Rate)
        time_domain = extract_all_time_domain_features(y, sr, n_fft, hop_length)
        features.update(time_domain)
        
        # Timbral features (MFCCs + Delta + Delta2, Spectral)
        timbral = extract_all_timbral_features(
            y, sr, n_fft, hop_length, n_mfcc, n_mels
        )
        features.update(timbral)
        
        # Rhythmic features (Tempo, Beat, Onset)
        rhythmic = extract_all_rhythmic_features(y, sr, hop_length)
        features.update(rhythmic)
        
        # Harmonic features (Chroma, Tonnetz)
        harmonic = extract_all_harmonic_features(y, sr, hop_length, n_fft)
        features.update(harmonic)
        
        # Groove features (F0, Syncopation, LUFS, Attack/Decay)
        groove = extract_all_groove_features(y, sr, n_fft, hop_length)
        features.update(groove)
        
        # Structural features (Sections, Repetition, Novelty)
        structural = extract_all_structural_features(y, sr, hop_length)
        features.update(structural)
        
        return features
    
    def _aggregate_to_vector(
        self,
        features_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Aggregate features into a fixed-length vector."""
        stats = self.config.features.statistics
        aggregated_parts = []
        
        # Process each feature category
        for name, feature in features_dict.items():
            if feature is None:
                continue
            
            # Ensure 2D
            if np.isscalar(feature):
                aggregated_parts.append(np.array([feature]))
                continue
            
            if feature.ndim == 1:
                feature = feature.reshape(1, -1)
            
            # Special case: single value features (tempo)
            if feature.shape[1] == 1:
                aggregated_parts.append(feature.flatten())
            else:
                # Compute statistics over time
                agg = compute_statistics(feature, stats)
                aggregated_parts.append(agg)
        
        return np.concatenate(aggregated_parts).astype(np.float32)
    
    def _extract_metadata(
        self,
        audio_path: str,
        y: np.ndarray,
        sr: int,
        features_dict: Dict
    ) -> Dict:
        """Extract metadata about the audio file."""
        # Estimate key
        key_info = estimate_key(y, sr)
        
        return {
            'file_path': audio_path,
            'file_name': os.path.basename(audio_path),
            'duration_seconds': len(y) / sr,
            'sample_rate': sr,
            'tempo_bpm': float(features_dict.get('tempo', [[120.0]])[0][0]),
            'key': key_info['key'],
            'key_confidence': key_info['correlation'],
        }
    
    def process_batch(
        self,
        audio_paths: List[str],
        n_jobs: int = -1,
        show_progress: bool = True,
        return_metadata: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[Dict]]]:
        """
        Process multiple audio files in batch.
        
        Parameters
        ----------
        audio_paths : list
            List of paths to audio files
        n_jobs : int, optional
            Number of parallel jobs (-1 = all cores)
        show_progress : bool, optional
            Show progress bar (default: True)
        return_metadata : bool, optional
            Return metadata for each file (default: False)
            
        Returns
        -------
        features : np.ndarray
            Feature matrix of shape (n_files, n_features)
        metadata : list, optional
            List of metadata dicts for each file
        """
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        
        n_files = len(audio_paths)
        features_list = []
        metadata_list = []
        failed_files = []
        
        # Process files
        iterator = tqdm(audio_paths, desc="Extracting features") if show_progress else audio_paths
        
        for path in iterator:
            try:
                if return_metadata:
                    feat, meta = self.process_file(path, return_metadata=True)
                    features_list.append(feat)
                    metadata_list.append(meta)
                else:
                    feat = self.process_file(path)
                    features_list.append(feat)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                failed_files.append(path)
                # Add zeros for failed files
                features_list.append(np.zeros(self.feature_dim, dtype=np.float32))
                if return_metadata:
                    metadata_list.append({'file_path': path, 'error': str(e)})
        
        # Stack into matrix
        features_matrix = np.stack(features_list, axis=0)
        
        if failed_files:
            print(f"Failed to process {len(failed_files)} files")
        
        if return_metadata:
            return features_matrix, metadata_list
        
        return features_matrix
    
    def get_feature_vector_info(self) -> Dict:
        """Get information about the feature vector structure."""
        return {
            'total_dimensions': self.feature_dim,
            'feature_names': self.feature_names,
            'schema': FEATURE_SCHEMA,
            'statistics_used': self.config.features.statistics,
        }


def extract_features_from_file(audio_path: str, config: Optional[Config] = None) -> np.ndarray:
    """
    Convenience function to extract features from a single file.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file
    config : Config, optional
        Configuration object
        
    Returns
    -------
    np.ndarray
        Feature vector
    """
    extractor = AudioFeatureExtractor(config)
    return extractor.process_file(audio_path)


def extract_features_from_directory(
    directory: str,
    output_path: Optional[str] = None,
    config: Optional[Config] = None,
    recursive: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features from all audio files in a directory.
    
    Parameters
    ----------
    directory : str
        Path to directory containing audio files
    output_path : str, optional
        Path to save features (auto-determined format from extension)
    config : Config, optional
        Configuration object
    recursive : bool, optional
        Search subdirectories (default: True)
        
    Returns
    -------
    features : np.ndarray
        Feature matrix
    file_paths : list
        List of processed file paths
    """
    config = config or DEFAULT_CONFIG
    
    # Find all audio files
    audio_files = []
    supported = config.pipeline.supported_formats
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f)[1].lower() in supported:
                    audio_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in supported:
                audio_files.append(os.path.join(directory, f))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {directory}")
    
    print(f"Found {len(audio_files)} audio files")
    
    # Extract features
    extractor = AudioFeatureExtractor(config)
    features = extractor.process_batch(audio_files)
    
    # Save if output path provided
    if output_path:
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.npy':
            np.save(output_path, features)
        elif ext == '.csv':
            import pandas as pd
            df = pd.DataFrame(features, columns=extractor.feature_names)
            df['file_path'] = audio_files
            df.to_csv(output_path, index=False)
        elif ext == '.parquet':
            import pandas as pd
            df = pd.DataFrame(features, columns=extractor.feature_names)
            df['file_path'] = audio_files
            df.to_parquet(output_path)
        
        print(f"Saved features to {output_path}")
    
    return features, audio_files
