"""
Tests for GeoMin library.
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime

import geomin as gm


class TestConfig:
    """Tests for configuration management."""
    
    def test_config_creation(self):
        """Test configuration can be created."""
        config = gm.Config()
        assert config is not None
    
    def test_api_config_defaults(self):
        """Test default API configuration."""
        config = gm.Config()
        assert config.api.earthdata_username is None
        assert config.api.planet_api_key is None
    
    def test_cache_config_defaults(self):
        """Test default cache configuration."""
        config = gm.Config()
        assert config.cache.cache_enabled is True
        assert config.cache.max_cache_size_gb == 50.0


class TestSpectralIndices:
    """Tests for spectral analysis algorithms."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample satellite data for testing."""
        # Create 6-band data: B02, B03, B04, B08, B11, B12
        height, width = 50, 50
        bands = []
        
        for i in range(6):
            band = np.random.rand(height, width) * 0.5 + 0.1
            bands.append(band)
        
        data = xr.concat([
            xr.DataArray(band, dims=['y', 'x']) for band in bands
        ], dim='band')
        
        data = data.assign_coords({
            'band': ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        })
        
        return data
    
    def test_iron_oxide_index(self, sample_data):
        """Test iron oxide index calculation."""
        from geomin.algorithms.spectral import iron_oxide_index
        
        result = iron_oxide_index(sample_data, red='B04', blue='B02')
        
        assert result is not None
        assert result.shape == (50, 50)
        assert 'long_name' in result.attrs
        assert 'Iron Oxide' in result.attrs['long_name']
    
    def test_clay_ratio(self, sample_data):
        """Test clay ratio calculation."""
        from geomin.algorithms.spectral import clay_ratio
        
        result = clay_ratio(sample_data, swir1='B11', swir2='B12')
        
        assert result is not None
        assert result.shape == (50, 50)
        # Clay ratio should be > 0
        assert float(result.mean()) > 0
    
    def test_ndvi(self, sample_data):
        """Test NDVI calculation."""
        from geomin.algorithms.spectral import ndvi
        
        result = ndvi(sample_data, nir='B08', red='B04')
        
        assert result is not None
        assert result.shape == (50, 50)
        # NDVI should be in range [-1, 1]
        assert float(result.min()) >= -1
        assert float(result.max()) <= 1
    
    def test_calculate_all_indices(self, sample_data):
        """Test calculating all available indices."""
        from geomin.algorithms.spectral import calculate_all_indices
        
        indices = calculate_all_indices(sample_data)
        
        assert 'iron_oxide' in indices
        assert 'clay_ratio' in indices
        assert 'ndvi' in indices
    
    def test_mineral_probability(self, sample_data):
        """Test mineral probability calculation."""
        from geomin.algorithms.spectral import mineral_probability
        
        prob = mineral_probability(sample_data, 'iron')
        
        assert prob is not None
        assert prob.shape == (50, 50)
        assert float(prob.min()) >= 0
        assert float(prob.max()) <= 1


class TestChangeDetection:
    """Tests for change detection algorithms."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for change detection."""
        height, width = 50, 50
        
        # Create two images with some differences
        img1 = np.random.rand(6, height, width) * 0.5 + 0.1
        img2 = img1.copy()
        
        # Add change in center region
        img2[:, 20:30, 20:30] = img1[:, 20:30, 20:30] * 0.5
        
        data1 = xr.DataArray(img1, dims=['band', 'y', 'x'])
        data2 = xr.DataArray(img2, dims=['band', 'y', 'x'])
        
        return data1, data2
    
    def test_simple_difference(self, sample_images):
        """Test simple difference change detection."""
        from geomin.models.change_detection import simple_difference
        
        img1, img2 = sample_images
        result = simple_difference(img1, img2)
        
        assert result is not None
        assert result.change_map is not None
        assert result.change_intensity is not None
        assert result.statistics is not None
        assert 'total_changed_pixels' in result.statistics
    
    def test_vegetation_change_detector(self, sample_images):
        """Test vegetation change detection."""
        from geomin.models.change_detection import vegetation_change_detector
        
        img1, img2 = sample_images
        result = vegetation_change_detector(img1, img2)
        
        assert result is not None
        assert result.change_map is not None
    
    def test_pca_change_detector(self, sample_images):
        """Test PCA-based change detection."""
        from geomin.models.change_detection import pca_change_detector
        
        img1, img2 = sample_images
        result = pca_change_detector(img1, img2)
        
        assert result is not None
        assert result.change_map is not None
        assert result.change_intensity is not None


class TestTerrainAnalysis:
    """Tests for terrain analysis algorithms."""
    
    @pytest.fixture
    def sample_dem(self):
        """Create sample DEM for testing."""
        height, width = 50, 50
        
        # Create synthetic terrain
        x = np.linspace(0, 4 * np.pi, width)
        y = np.linspace(0, 4 * np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        elevation = np.sin(X) * np.cos(Y) * 200 + 500
        
        dem = xr.DataArray(
            elevation,
            dims=['y', 'x'],
            coords={
                'y': np.arange(height) * 10,
                'x': np.arange(width) * 10,
            },
            attrs={'units': 'meters'}
        )
        
        return dem
    
    def test_calculate_slope(self, sample_dem):
        """Test slope calculation."""
        from geomin.algorithms.terrain import calculate_slope
        
        slope = calculate_slope(sample_dem)
        
        assert slope is not None
        assert slope.shape == sample_dem.shape
        assert 'long_name' in slope.attrs
        assert slope.attrs['long_name'] == 'Slope'
        # Slope should be in degrees (0-90)
        assert float(slope.min()) >= 0
        assert float(slope.max()) <= 90
    
    def test_calculate_aspect(self, sample_dem):
        """Test aspect calculation."""
        from geomin.algorithms.spectral import iron_oxide_index
        
        from geomin.algorithms.terrain import calculate_aspect
        
        aspect = calculate_aspect(sample_dem)
        
        assert aspect is not None
        assert aspect.shape == sample_dem.shape
        # Aspect should be in range [0, 360] or NaN
        valid_aspect = aspect.values[np.isfinite(aspect.values)]
        if len(valid_aspect) > 0:
            assert valid_aspect.min() >= 0
            assert valid_aspect.max() <= 360
    
    def test_calculate_hillshade(self, sample_dem):
        """Test hillshade calculation."""
        from geomin.algorithms.terrain import calculate_hillshade
        
        hillshade = calculate_hillshade(sample_dem, azimuth=315, altitude=45)
        
        assert hillshade is not None
        assert hillshade.shape == sample_dem.shape
        # Hillshade should be in range [0, 255]
        assert float(hillshade.min()) >= 0
        assert float(hillshade.max()) <= 255
    
    def test_calculate_terrain_metrics(self, sample_dem):
        """Test calculating all terrain metrics."""
        from geomin.algorithms.terrain import calculate_terrain_metrics
        
        metrics = calculate_terrain_metrics(sample_dem)
        
        assert metrics.slope is not None
        assert metrics.aspect is not None
        assert metrics.hillshade is not None


class TestDataLoader:
    """Tests for data loading utilities."""
    
    def test_band_info(self):
        """Test band information definitions."""
        from geomin.core.data_loader import SENTINEL2_BANDS, LANDSAT8_BANDS
        
        assert 'B02' in SENTINEL2_BANDS
        assert 'B04' in SENTINEL2_BANDS
        assert 'B01' in LANDSAT8_BANDS
        
        # Check band info structure
        assert hasattr(SENTINEL2_BANDS['B02'], 'name')
        assert hasattr(SENTINEL2_BANDS['B02'], 'wavelength')
    
    def test_data_loader_class(self):
        """Test DataLoader class exists and has required methods."""
        from geomin.core.data_loader import DataLoader
        
        assert hasattr(DataLoader, 'load')
        assert hasattr(DataLoader, 'load_multiple')
        assert hasattr(DataLoader, 'get_raster_info')


class TestCRSUtilities:
    """Tests for CRS utilities."""
    
    def test_detect_crs(self):
        """Test CRS detection."""
        from geomin.core.crs import detect_crs
        
        # Create data with CRS attribute
        data = xr.DataArray(np.random.rand(10, 10), dims=['y', 'x'])
        data.attrs['spatial_ref'] = 'EPSG:4326'
        
        crs = detect_crs(data)
        assert crs == 'EPSG:4326'
    
    def test_is_valid_crs(self):
        """Test CRS validation."""
        from geomin.core.crs import is_valid_crs
        
        assert is_valid_crs('EPSG:4326') is True
        assert is_valid_crs('EPSG:32611') is True
        assert is_valid_crs('INVALID') is False
    
    def test_get_utm_zone(self):
        """Test UTM zone determination."""
        from geomin.core.crs import get_utm_zone
        
        # Test northern hemisphere
        utm = get_utm_zone(-110, 35)
        assert 'EPSG:32611' in utm  # Zone 11N
        
        # Test southern hemisphere
        utm = get_utm_zone(-110, -35)
        assert 'EPSG:32711' in utm  # Zone 11S


class TestSatelliteClients:
    """Tests for satellite data clients."""
    
    def test_sentinel_client_creation(self):
        """Test Sentinel client can be created."""
        client = gm.SentinelClient()
        assert client is not None
    
    def test_landsat_client_creation(self):
        """Test Landsat client can be created."""
        client = gm.LandsatClient()
        assert client is not None
    
    def test_search_options_validation(self):
        """Test SearchOptions validation."""
        from geomin.satellites.base_client import SearchOptions
        
        # Should raise error without bbox or geometry
        with pytest.raises(ValueError):
            SearchOptions()
        
        # Should work with bbox
        options = SearchOptions(bbox=(-110, 35, -109, 36))
        assert options.bbox == (-110, 35, -109, 36)
    
    def test_search_result_creation(self):
        """Test SearchResult creation and serialization."""
        from geomin.satellites.base_client import SearchResult
        from shapely.geometry import box
        
        result = SearchResult(
            scene_id='test_scene',
            provider='sentinel',
            acquisition_time=datetime.now(),
            cloud_cover=5.0,
            geometry=box(-110, 35, -109, 36),
        )
        
        assert result.scene_id == 'test_scene'
        assert result.cloud_cover == 5.0
        
        # Test serialization
        result_dict = result.to_dict()
        assert result_dict['scene_id'] == 'test_scene'
        
        # Test deserialization
        restored = SearchResult.from_dict(result_dict)
        assert restored.scene_id == 'test_scene'


class TestVisualization:
    """Tests for visualization functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for visualization tests."""
        data = np.random.rand(3, 50, 50)
        return xr.DataArray(data, dims=['band', 'y', 'x'])
    
    def test_plot_mineral_index(self, sample_data):
        """Test mineral index plotting."""
        from geomin.visualization.static import plot_mineral_index
        
        # Create simple index data
        index_data = xr.DataArray(
            np.random.rand(50, 50),
            dims=['y', 'x']
        )
        
        fig = plot_mineral_index(
            index_data,
            'Test Index',
            show=False
        )
        
        assert fig is not None
    
    def test_plot_rgb_composite(self, sample_data):
        """Test RGB composite plotting."""
        from geomin.visualization.static import plot_rgb_composite
        
        fig = plot_rgb_composite(
            sample_data,
            bands=('B04', 'B03', 'B02'),
            show=False
        )
        
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
