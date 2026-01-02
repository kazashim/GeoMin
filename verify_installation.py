#!/usr/bin/env python
"""
Verification script for GeoMin installation.
Tests imports and basic functionality.
"""

import sys

def test_imports():
    """Test that all imports work correctly."""
    print("Testing GeoMin Imports...")
    print("-" * 50)
    
    try:
        # Test main package
        import geomin
        print(f"✓ GeoMin version: {geomin.__version__}")
        
        # Test core module
        from geomin.core.config import Config, get_config
        print("✓ Core config module loaded")
        
        from geomin.core.data_loader import DataLoader
        print("✓ Core data_loader module loaded")
        
        from geomin.core.crs import transform_bbox, get_utm_zone, is_valid_crs
        print("✓ Core CRS module loaded")
        
        # Test satellites module
        from geomin.satellites.base_client import SatClient, SearchResult, SearchOptions
        print("✓ Satellites base module loaded")
        
        from geomin.satellites.landsat import LandsatClient
        print("✓ Landsat client loaded")
        
        from geomin.satellites.sentinel import SentinelClient
        print("✓ Sentinel client loaded")
        
        from geomin.satellites.commercial import PlanetClient, MaxarClient
        print("✓ Commercial clients loaded")
        
        # Test algorithms
        from geomin.algorithms import spectral
        print("✓ Spectral algorithms loaded")
        
        from geomin.algorithms import terrain
        print("✓ Terrain algorithms loaded")
        
        # Test models
        from geomin.models import change_detection
        print("✓ Change detection models loaded")
        
        # Test visualization
        from geomin.visualization import static
        print("✓ Visualization module loaded")
        
        print("-" * 50)
        print("All imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_functionality():
    """Test basic functionality."""
    print("\nTesting Basic Functionality...")
    print("-" * 50)
    
    try:
        # Test configuration
        config = get_config()
        print(f"✓ Configuration created")
        print(f"  - Cache dir: {config.cache.cache_dir}")
        print(f"  - Default CRS: {config.processing.default_crs}")
        
        # Test spectral indices
        import numpy as np
        import xarray as xr
        
        # Create test data
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
        
        print("✓ Test data created")
        
        # Test spectral indices
        iron = spectral.iron_oxide_index(data, red='B04', blue='B02')
        print(f"✓ Iron oxide index: min={float(iron.min()):.3f}, max={float(iron.max()):.3f}")
        
        clay = spectral.clay_ratio(data, swir1='B11', swir2='B12')
        print(f"✓ Clay ratio: min={float(clay.min()):.3f}, max={float(clay.max()):.3f}")
        
        ndvi = spectral.ndvi(data, nir='B08', red='B04')
        print(f"✓ NDVI: min={float(ndvi.min()):.3f}, max={float(ndvi.max()):.3f}")
        
        # Test change detection
        data1 = data.copy()
        data2 = data.copy()
        data2[:, 20:30, 20:30] = data1[:, 20:30, 20:30] * 0.5
        
        result = change_detection.simple_difference(data1, data2)
        print(f"✓ Change detection: {result.statistics['total_changed_pixels']} pixels changed")
        
        # Test terrain analysis
        dem = xr.DataArray(
            np.random.rand(height, width) * 500 + 500,
            dims=['y', 'x']
        )
        
        slope = terrain.calculate_slope(dem)
        print(f"✓ Slope: min={float(slope.min()):.1f}°, max={float(slope.max()):.1f}°")
        
        print("-" * 50)
        print("All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_credentials():
    """Check API credential status."""
    print("\nChecking API Credentials...")
    print("-" * 50)
    
    config = get_config()
    
    providers = [
        ('Copernicus (Sentinel-2)', 
         config.api.copernicus_username and config.api.copernicus_password),
        ('USGS EarthData (Landsat)', 
         config.api.earthdata_username and config.api.earthdata_password),
        ('Planet Labs', 
         config.api.planet_api_key),
        ('Maxar', 
         config.api.maxar_api_key),
    ]
    
    for provider, configured in providers:
        status = "✓ Configured" if configured else "✗ Not configured"
        print(f"  {provider}: {status}")
    
    if not any(configured for _, configured in providers):
        print("\n⚠️  No API credentials configured.")
        print("   See README.md for setup instructions.")
    
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 50)
    print("   GeoMin Verification Script")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test functionality
    if not test_functionality():
        all_passed = False
    
    # Check credentials
    test_api_credentials()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("   ✓ All verification tests passed!")
        print("   GeoMin is ready to use.")
    else:
        print("   ✗ Some tests failed. Check errors above.")
        sys.exit(1)
    print("=" * 50)
    
    print("\n📚 Next steps:")
    print("   1. Configure API credentials (see README.md)")
    print("   2. Run: python geomin/examples/quickstart.py")
    print("   3. Explore the examples directory for more use cases")


if __name__ == "__main__":
    main()
