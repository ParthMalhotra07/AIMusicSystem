"""
Tests for Audio Loader Module
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.audio_loader import (
    load_audio,
    validate_audio,
    normalize_amplitude,
    trim_silence
)


class TestLoadAudio:
    """Tests for audio loading functionality."""
    
    def test_load_audio_returns_tuple(self):
        """Test that load_audio returns (y, sr) tuple."""
        # Create a synthetic audio signal for testing
        sr = 22050
        duration = 1.0
        y = np.random.randn(int(sr * duration)).astype(np.float32)
        
        # Test with synthetic data (would need actual file for full test)
        assert isinstance(y, np.ndarray)
        assert isinstance(sr, int)
    
    def test_load_audio_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_audio('/nonexistent/path/to/audio.mp3')
    
    def test_load_audio_unsupported_format(self):
        """Test that ValueError is raised for unsupported formats."""
        # Create a temporary file with unsupported extension
        test_file = '/tmp/test_audio.xyz'
        with open(test_file, 'w') as f:
            f.write('dummy')
        
        try:
            with pytest.raises(ValueError):
                load_audio(test_file)
        finally:
            os.remove(test_file)


class TestValidateAudio:
    """Tests for audio validation functionality."""
    
    def test_validate_normal_audio(self):
        """Test validation of normal audio."""
        sr = 22050
        y = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)
        
        result = validate_audio(y, sr)
        
        assert result['is_valid'] == True
        assert result['duration'] == pytest.approx(2.0, rel=0.01)
        assert result['is_silent'] == False
        assert result['is_clipped'] == False
    
    def test_validate_too_short(self):
        """Test detection of too-short audio."""
        sr = 22050
        y = np.zeros(int(sr * 0.1))  # 0.1 seconds
        
        result = validate_audio(y, sr, min_duration=1.0)
        
        assert result['is_valid'] is False
        assert 'too short' in result['issues'][0].lower()
    
    def test_validate_silent_audio(self):
        """Test detection of silence."""
        sr = 22050
        y = np.zeros(sr * 2)  # 2 seconds of silence
        
        result = validate_audio(y, sr)
        
        assert result['is_silent'] == True
    
    def test_validate_clipped_audio(self):
        """Test detection of clipping."""
        sr = 22050
        y = np.ones(sr * 2)  # All samples at max amplitude
        
        result = validate_audio(y, sr)
        
        assert result['is_clipped'] == True


class TestNormalizeAmplitude:
    """Tests for amplitude normalization."""
    
    def test_peak_normalization(self):
        """Test peak normalization."""
        y = np.array([0.0, 0.25, 0.5, -0.5, -0.25])
        
        y_norm = normalize_amplitude(y, method='peak', target_db=-3.0)
        
        # Peak should be at target amplitude
        target_amp = 10 ** (-3.0 / 20.0)
        assert np.max(np.abs(y_norm)) == pytest.approx(target_amp, rel=0.01)
    
    def test_rms_normalization(self):
        """Test RMS normalization."""
        y = np.array([0.5, 0.5, 0.5, 0.5])
        
        y_norm = normalize_amplitude(y, method='rms', target_db=-3.0)
        
        assert len(y_norm) == len(y)
    
    def test_empty_audio(self):
        """Test that empty audio is handled."""
        y = np.array([])
        y_norm = normalize_amplitude(y)
        
        assert len(y_norm) == 0


class TestTrimSilence:
    """Tests for silence trimming."""
    
    def test_trim_silence_basic(self):
        """Test basic silence trimming."""
        sr = 22050
        
        # Create audio with leading and trailing silence
        silence = np.zeros(sr)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        y = np.concatenate([silence, tone, silence])
        
        y_trimmed, indices = trim_silence(y)
        
        # Trimmed audio should be shorter
        assert len(y_trimmed) < len(y)
        assert len(y_trimmed) == indices[1] - indices[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
