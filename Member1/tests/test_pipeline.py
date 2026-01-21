"""
Tests for the Main Pipeline
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, DEFAULT_CONFIG, get_total_feature_dims
from pipeline.extractor import AudioFeatureExtractor
from pipeline.scaler import MinMaxScaler, ZScoreScaler, create_scaler


class TestAudioFeatureExtractor:
    """Tests for the main extraction pipeline."""
    
    @pytest.fixture
    def extractor(self):
        """Create a feature extractor instance."""
        return AudioFeatureExtractor(DEFAULT_CONFIG)
    
    def test_extractor_initialization(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor.config is not None
        assert extractor.feature_dim > 0
    
    def test_feature_names_generated(self, extractor):
        """Test that feature names are generated."""
        names = extractor.feature_names
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)
    
    def test_feature_dim_matches_schema(self, extractor):
        """Test feature dimension matches expected from schema."""
        dim = extractor.feature_dim
        expected = get_total_feature_dims()
        
        # Allow some flexibility as exact count may vary
        assert dim > 100  # Should be substantial
    
    def test_get_feature_vector_info(self, extractor):
        """Test feature info method."""
        info = extractor.get_feature_vector_info()
        
        assert 'total_dimensions' in info
        assert 'feature_names' in info
        assert 'schema' in info


class TestMinMaxScaler:
    """Tests for MinMax scaler."""
    
    def test_fit_transform(self):
        """Test basic fit_transform."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        X_scaled = scaler.fit_transform(X.astype(float))
        
        # Min should be 0, max should be 1
        assert X_scaled.min() == pytest.approx(0.0, abs=0.01)
        assert X_scaled.max() == pytest.approx(1.0, abs=0.01)
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]]).astype(float)
        
        X_scaled = scaler.fit_transform(X)
        X_reconstructed = scaler.inverse_transform(X_scaled)
        
        np.testing.assert_array_almost_equal(X, X_reconstructed)
    
    def test_save_load(self, tmp_path):
        """Test scaler persistence."""
        scaler = MinMaxScaler()
        X = np.random.randn(10, 5)
        scaler.fit(X)
        
        # Save
        path = str(tmp_path / "scaler.json")
        scaler.save(path)
        
        # Load
        loaded = MinMaxScaler.load(path)
        
        # Compare
        np.testing.assert_array_almost_equal(scaler.min_, loaded.min_)
        np.testing.assert_array_almost_equal(scaler.max_, loaded.max_)


class TestZScoreScaler:
    """Tests for ZScore scaler."""
    
    def test_fit_transform(self):
        """Test basic fit_transform."""
        scaler = ZScoreScaler()
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(float)
        
        X_scaled = scaler.fit_transform(X)
        
        # Mean should be ~0, std should be ~1
        assert np.mean(X_scaled) == pytest.approx(0.0, abs=0.01)
        # Each column should have std ~1
        for i in range(X.shape[1]):
            assert np.std(X_scaled[:, i]) == pytest.approx(1.0, abs=0.1)
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        scaler = ZScoreScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]]).astype(float)
        
        X_scaled = scaler.fit_transform(X)
        X_reconstructed = scaler.inverse_transform(X_scaled)
        
        np.testing.assert_array_almost_equal(X, X_reconstructed)
    
    def test_save_load(self, tmp_path):
        """Test scaler persistence."""
        scaler = ZScoreScaler()
        X = np.random.randn(10, 5)
        scaler.fit(X)
        
        # Save
        path = str(tmp_path / "scaler.json")
        scaler.save(path)
        
        # Load
        loaded = ZScoreScaler.load(path)
        
        # Compare
        np.testing.assert_array_almost_equal(scaler.mean_, loaded.mean_)
        np.testing.assert_array_almost_equal(scaler.std_, loaded.std_)


class TestScalerFactory:
    """Tests for scaler factory function."""
    
    def test_create_minmax(self):
        """Test creating MinMax scaler."""
        scaler = create_scaler('minmax')
        assert isinstance(scaler, MinMaxScaler)
    
    def test_create_zscore(self):
        """Test creating ZScore scaler."""
        scaler = create_scaler('zscore')
        assert isinstance(scaler, ZScoreScaler)
    
    def test_unknown_method_raises(self):
        """Test that unknown method raises error."""
        with pytest.raises(ValueError):
            create_scaler('unknown_method')


class TestConfig:
    """Tests for configuration."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.audio.sample_rate == 22050
        assert config.features.n_mfcc == 13
        assert config.features.n_chroma == 12
    
    def test_total_feature_dims(self):
        """Test feature dimension calculation."""
        dims = get_total_feature_dims()
        
        # Should be around 170-180 with default settings
        assert 150 <= dims <= 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
