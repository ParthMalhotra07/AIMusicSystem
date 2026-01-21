"""
Tests for Feature Extraction Pipeline
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.timbral import (
    extract_mfccs,
    extract_spectral_centroid,
    extract_spectral_rolloff,
    extract_spectral_contrast,
    extract_zero_crossing_rate,
)
from features.rhythmic import (
    extract_tempo,
    extract_beat_frames,
    extract_onset_strength,
)
from features.harmonic import (
    extract_chroma,
    extract_tonnetz,
)
from features.statistics import (
    compute_statistics,
    aggregate_features,
    global_pooling,
)


# Create synthetic audio for testing
@pytest.fixture
def synthetic_audio():
    """Generate synthetic audio for testing."""
    sr = 22050
    duration = 5.0
    t = np.arange(int(sr * duration)) / sr
    
    # Create a simple tone with some harmonics
    y = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4
    y += 0.25 * np.sin(2 * np.pi * 880 * t)  # A5
    y += 0.1 * np.sin(2 * np.pi * 1320 * t)  # E6
    
    # Add some noise
    y += 0.05 * np.random.randn(len(y))
    
    return y.astype(np.float32), sr


class TestTimbralFeatures:
    """Tests for timbral feature extraction."""
    
    def test_mfcc_shape(self, synthetic_audio):
        """Test MFCC extraction returns correct shape."""
        y, sr = synthetic_audio
        n_mfcc = 13
        
        mfccs = extract_mfccs(y, sr, n_mfcc=n_mfcc)
        
        assert mfccs.shape[0] == n_mfcc
        assert mfccs.shape[1] > 0  # Should have frames
    
    def test_mfcc_values(self, synthetic_audio):
        """Test MFCC values are reasonable."""
        y, sr = synthetic_audio
        
        mfccs = extract_mfccs(y, sr)
        
        assert not np.any(np.isnan(mfccs))
        assert not np.any(np.isinf(mfccs))
    
    def test_spectral_centroid_shape(self, synthetic_audio):
        """Test spectral centroid returns correct shape."""
        y, sr = synthetic_audio
        
        centroid = extract_spectral_centroid(y, sr)
        
        assert centroid.shape[0] == 1  # Single feature
        assert centroid.shape[1] > 0
    
    def test_spectral_rolloff_shape(self, synthetic_audio):
        """Test spectral rolloff returns correct shape."""
        y, sr = synthetic_audio
        
        rolloff = extract_spectral_rolloff(y, sr)
        
        assert rolloff.shape[0] == 1
        assert rolloff.shape[1] > 0
    
    def test_spectral_contrast_shape(self, synthetic_audio):
        """Test spectral contrast returns correct shape."""
        y, sr = synthetic_audio
        n_bands = 6
        
        contrast = extract_spectral_contrast(y, sr, n_bands=n_bands)
        
        # Returns n_bands + 1 (includes mean)
        assert contrast.shape[0] == n_bands + 1
    
    def test_zcr_shape(self, synthetic_audio):
        """Test zero-crossing rate returns correct shape."""
        y, sr = synthetic_audio
        
        zcr = extract_zero_crossing_rate(y)
        
        assert zcr.shape[0] == 1


class TestRhythmicFeatures:
    """Tests for rhythmic feature extraction."""
    
    def test_tempo_extraction(self, synthetic_audio):
        """Test tempo extraction returns reasonable value."""
        y, sr = synthetic_audio
        
        tempo = extract_tempo(y, sr)
        
        assert isinstance(tempo, float)
        assert 20 <= tempo <= 300  # Reasonable BPM range
    
    def test_beat_frames_returns_tuple(self, synthetic_audio):
        """Test beat tracking returns tempo and frames."""
        y, sr = synthetic_audio
        
        tempo, beats = extract_beat_frames(y, sr)
        
        assert isinstance(tempo, float)
        assert isinstance(beats, np.ndarray)
    
    def test_onset_strength_shape(self, synthetic_audio):
        """Test onset strength envelope shape."""
        y, sr = synthetic_audio
        
        onset = extract_onset_strength(y, sr)
        
        assert onset.ndim == 1
        assert len(onset) > 0


class TestHarmonicFeatures:
    """Tests for harmonic feature extraction."""
    
    def test_chroma_shape(self, synthetic_audio):
        """Test chroma extraction returns 12 bins."""
        y, sr = synthetic_audio
        
        chroma = extract_chroma(y, sr)
        
        assert chroma.shape[0] == 12  # 12 pitch classes
        assert chroma.shape[1] > 0
    
    def test_chroma_values(self, synthetic_audio):
        """Test chroma values are in valid range."""
        y, sr = synthetic_audio
        
        chroma = extract_chroma(y, sr)
        
        assert not np.any(np.isnan(chroma))
        # Chroma features are typically non-negative
    
    def test_tonnetz_shape(self, synthetic_audio):
        """Test tonnetz extraction returns 6 dimensions."""
        y, sr = synthetic_audio
        
        tonnetz = extract_tonnetz(y, sr)
        
        assert tonnetz.shape[0] == 6  # 6D tonal space
        assert tonnetz.shape[1] > 0


class TestStatistics:
    """Tests for statistical aggregation."""
    
    def test_compute_statistics_shape(self):
        """Test statistics computation returns correct shape."""
        n_features = 5
        n_frames = 100
        feature_matrix = np.random.randn(n_features, n_frames)
        
        stats = compute_statistics(feature_matrix)
        
        # Default: 4 statistics per feature
        assert len(stats) == n_features * 4
    
    def test_compute_statistics_values(self):
        """Test statistics values are correct."""
        # Known data
        feature = np.array([[1, 2, 3, 4, 5]])
        
        stats = compute_statistics(feature, ('mean', 'std'))
        
        assert stats[0] == pytest.approx(3.0, rel=0.01)  # mean
        assert stats[1] == pytest.approx(np.std([1,2,3,4,5]), rel=0.01)  # std
    
    def test_global_pooling_mean_std(self):
        """Test mean_std pooling."""
        n_features = 10
        n_frames = 50
        feature_matrix = np.random.randn(n_features, n_frames)
        
        pooled = global_pooling(feature_matrix, method='mean_std')
        
        # mean_std returns 2 * n_features
        assert len(pooled) == n_features * 2
    
    def test_aggregate_features(self):
        """Test feature dictionary aggregation."""
        features_dict = {
            'feature_a': np.random.randn(3, 50),
            'feature_b': np.random.randn(5, 50),
        }
        
        aggregated = aggregate_features(features_dict)
        
        assert len(aggregated) > 0
        assert not np.any(np.isnan(aggregated))


class TestFixedLengthOutput:
    """Tests to ensure fixed-length output regardless of input duration."""
    
    def test_different_durations_same_output(self):
        """Test that different length audio produces same dimension output."""
        sr = 22050
        
        # Short audio (2 seconds)
        y_short = np.random.randn(sr * 2).astype(np.float32)
        
        # Long audio (10 seconds)
        y_long = np.random.randn(sr * 10).astype(np.float32)
        
        # Extract MFCCs
        mfcc_short = extract_mfccs(y_short, sr)
        mfcc_long = extract_mfccs(y_long, sr)
        
        # Feature dimension should be same
        assert mfcc_short.shape[0] == mfcc_long.shape[0]
        
        # Apply statistics to get fixed length
        stats_short = compute_statistics(mfcc_short)
        stats_long = compute_statistics(mfcc_long)
        
        # Output dimension should be identical
        assert len(stats_short) == len(stats_long)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
