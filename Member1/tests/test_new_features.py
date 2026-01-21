"""
Tests for Time Domain, Groove, and Structural Feature Extraction
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.time_domain import (
    extract_short_time_energy,
    extract_rms_energy,
    extract_inter_beat_interval,
    extract_dynamic_range,
    extract_onset_rate,
    extract_zero_crossing_rate,
    extract_all_time_domain_features,
)
from features.groove import (
    extract_fundamental_frequency,
    extract_harmonic_energy_ratio,
    extract_rhythmic_complexity,
    extract_syncopation_index,
    extract_perceived_loudness,
    extract_crest_factor,
    extract_all_groove_features,
)
from features.structural import (
    extract_novelty_curve,
    detect_sections,
    extract_repetition_score,
    extract_section_count,
    extract_all_structural_features,
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
    
    # Add some rhythmic impulses
    for i in range(0, len(y), sr // 2):  # Every 0.5 seconds (120 BPM)
        end = min(i + 500, len(y))
        y[i:end] += 0.3 * np.exp(-np.arange(end - i) / 50)
    
    # Add some noise
    y += 0.05 * np.random.randn(len(y))
    
    return y.astype(np.float32), sr


class TestTimeDomainFeatures:
    """Tests for time-domain feature extraction."""
    
    def test_short_time_energy_shape(self, synthetic_audio):
        """Test STE returns correct shape."""
        y, sr = synthetic_audio
        ste = extract_short_time_energy(y)
        
        assert ste.shape[0] == 1
        assert ste.shape[1] > 0
    
    def test_rms_energy_shape(self, synthetic_audio):
        """Test RMS energy returns correct shape."""
        y, sr = synthetic_audio
        rms = extract_rms_energy(y)
        
        assert rms.shape[0] == 1
        assert rms.shape[1] > 0
    
    def test_inter_beat_interval(self, synthetic_audio):
        """Test IBI extraction."""
        y, sr = synthetic_audio
        ibi = extract_inter_beat_interval(y, sr)
        
        assert len(ibi) > 0
        assert all(i > 0 for i in ibi)  # All intervals should be positive
    
    def test_dynamic_range(self, synthetic_audio):
        """Test dynamic range extraction."""
        y, sr = synthetic_audio
        dr = extract_dynamic_range(y)
        
        assert isinstance(dr, float)
        assert dr >= 0  # Dynamic range should be non-negative
    
    def test_onset_rate(self, synthetic_audio):
        """Test onset rate extraction."""
        y, sr = synthetic_audio
        rate = extract_onset_rate(y, sr)
        
        assert isinstance(rate, float)
        assert rate >= 0
    
    def test_all_time_domain_features(self, synthetic_audio):
        """Test extraction of all time-domain features."""
        y, sr = synthetic_audio
        features = extract_all_time_domain_features(y, sr)
        
        expected_keys = ['short_time_energy', 'rms_energy', 'zero_crossing_rate',
                         'inter_beat_interval', 'dynamic_range', 'onset_rate']
        for key in expected_keys:
            assert key in features
            assert not np.any(np.isnan(features[key]))


class TestGrooveFeatures:
    """Tests for groove feature extraction."""
    
    def test_fundamental_frequency_shape(self, synthetic_audio):
        """Test F0 extraction returns correct shape."""
        y, sr = synthetic_audio
        f0 = extract_fundamental_frequency(y, sr)
        
        assert f0.shape[0] == 1
        assert f0.shape[1] > 0
    
    def test_harmonic_energy_ratio(self, synthetic_audio):
        """Test harmonic energy ratio is in valid range."""
        y, sr = synthetic_audio
        ratio = extract_harmonic_energy_ratio(y, sr)
        
        assert isinstance(ratio, float)
        assert 0 <= ratio <= 1
    
    def test_rhythmic_complexity(self, synthetic_audio):
        """Test rhythmic complexity extraction."""
        y, sr = synthetic_audio
        complexity = extract_rhythmic_complexity(y, sr)
        
        assert isinstance(complexity, float)
        assert complexity >= 0
    
    def test_syncopation_index(self, synthetic_audio):
        """Test syncopation index is in valid range."""
        y, sr = synthetic_audio
        syncopation = extract_syncopation_index(y, sr)
        
        assert isinstance(syncopation, float)
        assert 0 <= syncopation <= 1
    
    def test_perceived_loudness(self, synthetic_audio):
        """Test perceived loudness extraction."""
        y, sr = synthetic_audio
        lufs = extract_perceived_loudness(y, sr)
        
        assert isinstance(lufs, float)
        assert lufs < 0  # Typical range is negative LUFS
    
    def test_crest_factor(self, synthetic_audio):
        """Test crest factor extraction."""
        y, sr = synthetic_audio
        crest = extract_crest_factor(y)
        
        assert isinstance(crest, float)
        assert crest >= 1  # Crest factor >= 1
    
    def test_all_groove_features(self, synthetic_audio):
        """Test extraction of all groove features."""
        y, sr = synthetic_audio
        features = extract_all_groove_features(y, sr)
        
        expected_keys = ['fundamental_frequency', 'harmonic_energy_ratio',
                         'rhythmic_complexity', 'syncopation_index',
                         'perceived_loudness', 'crest_factor',
                         'attack_time', 'decay_time', 'beat_strength_dist']
        for key in expected_keys:
            assert key in features
            assert not np.any(np.isnan(features[key]))


class TestStructuralFeatures:
    """Tests for structural feature extraction."""
    
    def test_novelty_curve_shape(self, synthetic_audio):
        """Test novelty curve returns array."""
        y, sr = synthetic_audio
        novelty = extract_novelty_curve(y, sr)
        
        assert novelty.ndim == 1
        assert len(novelty) > 0
    
    def test_novelty_curve_values(self, synthetic_audio):
        """Test novelty values are normalized."""
        y, sr = synthetic_audio
        novelty = extract_novelty_curve(y, sr)
        
        assert np.min(novelty) >= 0
        assert np.max(novelty) <= 1
    
    def test_detect_sections(self, synthetic_audio):
        """Test section detection."""
        y, sr = synthetic_audio
        boundaries, durations = detect_sections(y, sr)
        
        assert len(boundaries) > 0
        assert len(durations) > 0
        assert np.sum(durations) > 0
    
    def test_repetition_score(self, synthetic_audio):
        """Test repetition score is in valid range."""
        y, sr = synthetic_audio
        score = extract_repetition_score(y, sr)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_section_count(self, synthetic_audio):
        """Test section count extraction."""
        y, sr = synthetic_audio
        count = extract_section_count(y, sr)
        
        assert isinstance(count, int)
        assert count >= 1
    
    def test_all_structural_features(self, synthetic_audio):
        """Test extraction of all structural features."""
        y, sr = synthetic_audio
        features = extract_all_structural_features(y, sr)
        
        expected_keys = ['section_count', 'section_duration',
                         'repetition_score', 'novelty_curve']
        for key in expected_keys:
            assert key in features
            assert not np.any(np.isnan(features[key]))


class TestFeatureIntegration:
    """Integration tests for the full feature pipeline."""
    
    def test_pipeline_produces_features(self, synthetic_audio):
        """Test that the complete pipeline produces features."""
        y, sr = synthetic_audio
        
        # Extract all feature categories
        time_domain = extract_all_time_domain_features(y, sr)
        groove = extract_all_groove_features(y, sr)
        structural = extract_all_structural_features(y, sr)
        
        # Verify all have content
        assert len(time_domain) > 0
        assert len(groove) > 0
        assert len(structural) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
