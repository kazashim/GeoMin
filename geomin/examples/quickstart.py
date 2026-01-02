"""
GeoMin: Geophysics Library for Satellite-Based Mining Detection

This example demonstrates the real-world usage of GeoMin library for:
1. Searching and downloading actual satellite data from multiple providers
2. Loading and processing satellite imagery
3. Calculating mineral spectral indices for exploration
4. Detecting mining activity through change detection
5. Visualizing and exporting results

IMPORTANT: This example requires valid API credentials for satellite data providers.
See the setup guide in README.md for instructions on obtaining credentials.

Author: Kazashim Kuzasuwat
GitHub: https://github.com/kazashim/GeoMin
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# GeoMin imports
import geomin as gm
from geomin.core.config import Config, get_config
from geomin.core.data_loader import DataLoader
from geomin.algorithms.spectral import (
    iron_oxide_index,
    clay_ratio,
    ndvi,
    hydroxyl_index,
    ferrous_index,
    gossan_index,
    calculate_all_indices,
    detect_minerals,
    MINERAL_INDICES,
)
from geomin.algorithms.terrain import (
    calculate_slope,
    calculate_hillshade,
    calculate_terrain_metrics,
    detect_slopes,
)
from geomin.models.change_detection import (
    vegetation_change_detector,
    pca_change_detector,
    detect_mining_activity,
    calculate_change_area,
)
from geomin.visualization.static import (
    plot_mineral_index,
    plot_rgb_composite,
    plot_change_map,
    create_mineral_map,
)
from geomin.satellites.base_client import SearchOptions


def check_api_credentials() -> Dict[str, bool]:
    """
    Check which API credentials are configured.
    
    Returns:
        Dictionary of provider names and their configuration status
    """
    config = get_config()
    
    return {
        'copernicus': bool(config.api.copernicus_username and config.api.copernicus_password),
        'earthdata': bool(config.api.earthdata_username and config.api.earthdata_password),
        'planet': bool(config.api.planet_api_key),
        'maxar': bool(config.api.maxar_api_key),
    }


def setup_api_credentials():
    """
    Display instructions for setting up API credentials.
    """
    print("=" * 70)
    print("API Credential Setup")
    print("=" * 70)
    
    print("\nTo use satellite data APIs, you need to configure credentials.")
    print("You can set them via environment variables or a configuration file.")
    
    print("\n1. COPERNICUS (Sentinel-2 data - FREE):")
    print("   - Register at: https://developer.Copernicus.eu/")
    print("   - Set environment variables:")
    print('     export GEOMIN_COPERNICUS_USERNAME="your_username"')
    print('     export GEOMIN_COPERNICUS_PASSWORD="your_password"')
    
    print("\n2. USGS EARTHDATA (Landsat data - FREE):")
    print("   - Register at: https://ers.cr.usgs.gov/register/")
    print("   - Set environment variables:")
    print('     export GEOMIN_EARTHDATA_USERNAME="your_username"')
    print('     export GEOMIN_EARTHDATA_PASSWORD="your_password"')
    
    print("\n3. PLANET LABS (High-resolution data - COMMERCIAL):")
    print("   - Sign up at: https://www.planet.com/")
    print("   - Get API key from your account")
    print('     export GEOMIN_PLANET_API_KEY="your_api_key"')
    
    print("\n4. MAXAR (Very high-resolution data - COMMERCIAL):")
    print("   - Contact: https://www.maxar.com/")
    print('     export GEOMIN_MAXAR_API_KEY="your_api_key"')


def example_1_search_satellite_data():
    """
    Example 1: Searching and Accessing Satellite Data
    
    This example shows how to:
    - Configure API credentials
    - Search for satellite imagery using multiple providers
    - Filter results by date, cloud cover, and location
    - Select the best scene for your analysis
    """
    print("\n" + "=" * 70)
    print("Example 1: Searching and Accessing Satellite Data")
    print("=" * 70)
    
    # Check credentials
    credentials = check_api_credentials()
    print("\n1. Checking API Credentials:")
    for provider, configured in credentials.items():
        status = "✓ Configured" if configured else "✗ Not configured"
        print(f"   {provider.capitalize()}: {status}")
    
    if not any(credentials.values()):
        setup_api_credentials()
        print("\n   ⚠️  No API credentials found. Using demo mode with sample data.")
        return None
    
    # Define search parameters
    print("\n2. Setting Search Parameters:")
    
    # Example: Copper mining region in Arizona (Morenci Mine area)
    bbox = [-109.5, 33.0, -109.0, 33.5]  # Morenci, AZ
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    max_cloud_cover = 20
    
    print(f"   Search Area (bbox): {bbox}")
    print(f"   Date Range: {start_date.date()} to {end_date.date()}")
    print(f"   Max Cloud Cover: {max_cloud_cover}%")
    
    # Search using Sentinel-2 (recommended for mineral detection)
    print("\n3. Searching Sentinel-2 Data (Copernicus):")
    
    if credentials['copernicus']:
        try:
            client = gm.SentinelClient(use_sentinel_hub=False)
            client.connect()
            
            options = SearchOptions(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                cloud_cover=max_cloud_cover,
                limit=5,
            )
            
            results = client.search(options)
            
            print(f"   Found {len(results)} scenes:")
            for i, result in enumerate(results[:5], 1):
                print(f"   {i}. Scene ID: {result.scene_id}")
                print(f"      Date: {result.acquisition_time.date()}")
                print(f"      Cloud Cover: {result.cloud_cover:.1f}%")
                print(f"      Resolution: {result.resolution}m")
                print(f"      Bands: {len(result.bands)} available")
            
            # Select the best result (least cloud cover)
            best_result = client.get_best_result(results, ['cloud_cover', 'date'])
            print(f"\n   Best Scene: {best_result.scene_id}")
            
            client.disconnect()
            return best_result
            
        except Exception as e:
            print(f"   Error searching Sentinel-2: {e}")
    else:
        print("   ⚠️  Copernicus credentials not configured")
    
    # Try Landsat as alternative
    print("\n4. Searching Landsat Data (USGS):")
    
    if credentials['earthdata']:
        try:
            client = gm.LandsatClient(use_google_cloud=True)  # Use free GCS
            client.connect()
            
            options = SearchOptions(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                cloud_cover=max_cloud_cover,
                limit=5,
            )
            
            results = client.search(options)
            
            print(f"   Found {len(results)} scenes:")
            for i, result in enumerate(results[:3], 1):
                print(f"   {i}. {result.scene_id} - {result.acquisition_time.date()}")
            
            client.disconnect()
            return results[0] if results else None
            
        except Exception as e:
            print(f"   Error searching Landsat: {e}")
    
    return None


def example_2_load_and_process_satellite_data(scene_result):
    """
    Example 2: Loading and Processing Satellite Data
    
    This example shows how to:
    - Download satellite data for a selected scene
    - Load data into xarray DataArray format
    - Preprocess data for analysis
    - Handle different data formats and projections
    """
    print("\n" + "=" * 70)
    print("Example 2: Loading and Processing Satellite Data")
    print("=" * 70)
    
    # Method 1: Load from file (if you have local data)
    print("\n1. Loading Data from Local Files:")
    
    sample_files = {
        'B02': 'data/sentinel2/T35RNL_20231201_B02.jp2',
        'B03': 'data/sentinel2/T35RNL_20231201_B03.jp2',
        'B04': 'data/sentinel2/T35RNL_20231201_B04.jp2',
        'B08': 'data/sentinel2/T35RNL_20231201_B08.jp2',
        'B11': 'data/sentinel2/T35RNL_20231201_B11.jp2',
        'B12': 'data/sentinel2/T35RNL_20231201_B12.jp2',
    }
    
    # Check if files exist
    existing_files = {k: v for k, v in sample_files.items() if Path(v).exists()}
    
    if existing_files:
        print(f"   Found {len(existing_files)} band files")
        
        # Load using DataLoader
        data = DataLoader.create_stack(existing_files, 'sentinel2')
        print(f"   Loaded data shape: {data.shape}")
        print(f"   Bands: {list(data.coords.get('band', []).values)}")
    else:
        print("   ⚠️  No local files found. Loading from API...")
        
        if scene_result:
            try:
                # Download and load
                print("   Downloading satellite data...")
                files = scene_result.download(
                    bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
                )
                print(f"   Downloaded {len(files)} bands")
                
                # Load the data
                data = DataLoader.load(list(files.values())[0])
                print(f"   Loaded data shape: {data.shape}")
            except Exception as e:
                print(f"   Error downloading: {e}")
                print("   ⚠️  Using demo mode with sample data")
                return create_demo_data()
        else:
            print("   ⚠️  No scene selected. Using demo mode with sample data")
            return create_demo_data()
    
    # Method 2: Load from various formats
    print("\n2. Supported Data Formats:")
    
    supported_formats = [
        ('GeoTIFF (.tif/.tiff)', 'Most common format, includes georeference'),
        ('JPEG2000 (.jp2)', 'Sentinel-2 native format'),
        ('NetCDF (.nc)', 'Climate/atmospheric data'),
        ('HDF5 (.h5/.hdf5)', 'MODIS, VIIRS data'),
    ]
    
    for fmt, desc in supported_formats:
        print(f"   • {fmt}: {desc}")
    
    # Method 3: Data preprocessing
    print("\n3. Data Preprocessing:")
    
    # Reproject to UTM if needed
    try:
        if data.rio.crs:
            print(f"   Original CRS: {data.rio.crs}")
            
            # Get UTM zone for the data
            utm_crs = gm.get_utm_zone_from_bbox(data.rio.bounds())
            print(f"   Target UTM CRS: {utm_crs}")
            
            # Reproject (optional)
            # data_reprojected = DataLoader.reproject(data, utm_crs, 10)
    except Exception as e:
        print(f"   CRS handling: {e}")
    
    print("\n4. Data Statistics:")
    
    for band in list(data.coords.get('band', []).values)[:3]:
        band_data = data.sel(band=band)
        print(f"   {band}: min={float(band_data.min()):.4f}, "
              f"max={float(band_data.max()):.4f}, "
              f"mean={float(band_data.mean()):.4f}")
    
    return data


def example_3_mineral_detection(data):
    """
    Example 3: Mineral Detection from Satellite Imagery
    
    This example shows how to:
    - Calculate spectral indices for mineral detection
    - Interpret index values for mineral exploration
    - Create probability maps for mineral occurrence
    - Identify hydrothermal alteration zones
    """
    print("\n" + "=" * 70)
    print("Example 3: Mineral Detection from Satellite Imagery")
    print("=" * 70)
    
    print("\n1. Available Mineral Indices:")
    
    for index_name, index_info in list(MINERAL_INDICES.items())[:6]:
        print(f"\n   {index_info.name} ({index_name}):")
        print(f"      Formula: {index_info.formula}")
        print(f"      Description: {index_info.description}")
        print(f"      Range: {index_info.range}")
        print(f"      Interpretation: {index_info.high_values}")
    
    # Calculate key indices for mineral exploration
    print("\n2. Calculating Mineral Indices:")
    
    try:
        # Iron Oxide Index - highlights iron-rich areas (hematite, goethite)
        print("\n   Calculating Iron Oxide Index...")
        iron_oxide = iron_oxide_index(data, red='B04', blue='B02')
        print(f"   Range: {float(iron_oxide.min()):.3f} - {float(iron_oxide.max()):.3f}")
        print(f"   Interpretation: Values > 1.0 indicate iron oxide presence")
        
        # Clay Ratio - detects clay minerals (hydrothermal alteration)
        print("\n   Calculating Clay Ratio...")
        clay_ratio_idx = clay_ratio(data, swir1='B11', swir2='B12')
        print(f"   Range: {float(clay_ratio_idx.min()):.3f} - {float(clay_ratio_idx.max()):.3f}")
        print(f"   Interpretation: Values > 1.2 indicate clay mineral presence")
        
        # Hydroxyl Index - detects hydroxyl-bearing minerals
        print("\n   Calculating Hydroxyl Index...")
        hydroxyl = hydroxyl_index(data, swir1='B11', swir2='B12')
        print(f"   Range: {float(hydroxyl.min()):.3f} - {float(hydroxyl.max()):.3f}")
        print(f"   Interpretation: High values indicate hydrothermal alteration")
        
        # Ferrous Index - detects ferrous iron minerals
        print("\n   Calculating Ferrous Minerals Index...")
        ferrous = ferrous_index(data, swir='B12', nir='B08')
        print(f"   Range: {float(ferrous.min()):.3f} - {float(ferrous.max()):.3f}")
        print(f"   Interpretation: Values > 1.0 indicate ferrous minerals")
        
        # Gossan Index - oxidized sulfide zones
        print("\n   Calculating Gossan Index...")
        gossan = gossan_index(data)
        print(f"   Range: {float(gossan.min()):.3f} - {float(gossan.max()):.3f}")
        print(f"   Interpretation: High values indicate oxidized sulfide zones")
        
        # NDVI for vegetation context
        print("\n   Calculating NDVI (vegetation context)...")
        ndvi_result = ndvi(data, nir='B08', red='B04')
        print(f"   Range: {float(ndvi_result.min()):.3f} - {float(ndvi_result.max()):.3f}")
        
    except Exception as e:
        print(f"   Error calculating indices: {e}")
        return None
    
    # Calculate all indices at once
    print("\n3. Calculating All Indices...")
    
    all_indices = calculate_all_indices(data)
    print(f"   Calculated {len(all_indices)} indices:")
    for name in all_indices.keys():
        print(f"   • {name}")
    
    # Detect minerals
    print("\n4. Mineral Detection Analysis:")
    
    detections = detect_minerals(data, threshold=0.7)
    print(f"   Detected {len(detections)} mineral types:")
    for mineral, detection in detections.items():
        changed_pixels = int(detection.sum())
        total_pixels = detection.size
        percentage = changed_pixels / total_pixels * 100
        print(f"   • {mineral.capitalize()}: {changed_pixels} pixels ({percentage:.2f}%)")
    
    return iron_oxide, clay_ratio_idx, detections


def example_4_change_detection(data_before, data_after):
    """
    Example 4: Mining Activity Detection Through Change Analysis
    
    This example shows how to:
    - Detect surface changes between two time periods
    - Identify vegetation loss indicative of mining expansion
    - Calculate statistics for changed areas
    - Generate change detection reports
    """
    print("\n" + "=" * 70)
    print("Example 4: Mining Activity Detection")
    print("=" * 70)
    
    print("\n1. Change Detection Methods:")
    
    methods = [
        ('vegetation', 'NDVI-based vegetation change detection'),
        ('pca', 'Principal Component Analysis for change detection'),
        ('simple', 'Simple image differencing'),
    ]
    
    for method, desc in methods:
        print(f"   • {method}: {desc}")
    
    # Vegetation-based detection
    print("\n2. Performing Vegetation Change Detection...")
    
    try:
        change_result = vegetation_change_detector(data_before, data_after)
        
        print(f"\n   Change Statistics:")
        print(f"   • Changed pixels: {change_result.statistics['total_changed_pixels']:,}")
        print(f"   • Change percentage: {change_result.statistics['change_percentage']:.2f}%")
        print(f"   • Mean change intensity: {change_result.statistics['mean_intensity']:.3f}")
        print(f"   • Max change intensity: {change_result.statistics['max_intensity']:.3f}")
        
        # Calculate approximate area (assuming 10m pixels)
        pixel_area_km2 = 0.0001  # 10m x 10m = 100 m² = 0.0001 km²
        changed_area_km2 = calculate_change_area(change_result.change_map, pixel_area_km2)
        print(f"   • Approximate changed area: {changed_area_km2:.4f} km²")
        
    except Exception as e:
        print(f"   Error in vegetation detection: {e}")
        
        print("\n   Trying PCA-based detection...")
        change_result = pca_change_detector(data_before, data_after)
        print(f"   Changed pixels: {change_result.statistics['total_changed_pixels']:,}")
    
    # Mining activity detection
    print("\n3. Mining Activity Analysis:")
    
    mining_result = detect_mining_activity(data_before, data_after, method='vegetation')
    print(f"   Mining activity detected: {mining_result.statistics['change_percentage']:.2f}%")
    
    # Classify change types
    print("\n4. Change Type Classification:")
    
    from geomin.models.change_detection import classify_change_type
    
    try:
        change_types = classify_change_type(data_before, data_after, change_result.change_map)
        
        for change_type, mask in change_types.items():
            changed_pixels = int(mask.sum())
            if changed_pixels > 0:
                print(f"   • {change_type}: {changed_pixels:,} pixels")
    except Exception as e:
        print(f"   Classification error: {e}")
    
    return change_result


def example_5_terrain_analysis():
    """
    Example 5: Terrain Analysis for Mining Site Assessment
    
    This example shows how to:
    - Process digital elevation models (DEM)
    - Calculate slope, aspect, and curvature
    - Generate hillshade visualizations
    - Identify suitable terrain for mining operations
    """
    print("\n" + "=" * 70)
    print("Example 5: Terrain Analysis for Mining")
    print("=" * 70)
    
    # Load DEM data
    print("\n1. Loading Digital Elevation Model:")
    
    dem_path = Path('data/dem/srtm_dem.tif')
    
    if dem_path.exists():
        dem = DataLoader.load(dem_path)
        print(f"   Loaded DEM: {dem.shape}")
    else:
        print("   ⚠️  No DEM file found. Using sample data generation.")
        print("   In production, use data from:")
        print("   • SRTM (30m resolution): https://earthexplorer.usgs.gov/")
        print("   • ALOS World 3D (30m): https://www.eorc.jaxa.jp/ALOS/en/aw3d30/")
        print("   • Copernicus GLO-30 (30m): https://copernicus-dem.eu/")
        return None
    
    print("\n2. Calculating Terrain Metrics:")
    
    # Calculate slope
    print("   Calculating slope...")
    slope = calculate_slope(dem)
    print(f"   Slope range: {float(slope.min()):.1f}° - {float(slope.max()):.1f}°")
    print(f"   Mean slope: {float(slope.mean()):.1f}°")
    
    # Calculate aspect
    print("   Calculating aspect...")
    aspect = calculate_aspect(dem)
    valid_aspect = aspect.values[np.isfinite(aspect.values)]
    if len(valid_aspect) > 0:
        print(f"   Aspect range: {valid_aspect.min():.1f}° - {valid_aspect.max():.1f}°")
    
    # Calculate hillshade
    print("   Generating hillshade...")
    hillshade = calculate_hillshade(dem, azimuth=315, altitude=45)
    print(f"   Hillshade range: {float(hillshade.min()):.0f} - {float(hillshade.max()):.0f}")
    
    # Calculate comprehensive metrics
    print("\n3. Comprehensive Terrain Analysis:")
    
    metrics = calculate_terrain_metrics(dem)
    print(f"   • Slope: {float(metrics.slope.mean()):.1f}° mean")
    print(f"   • Curvature: Available" if metrics.curvature is not None else "   • Curvature: Not calculated")
    print(f"   • TPI: Available" if metrics.TPI is not None else "   • TPI: Not calculated")
    print(f"   • TRI: Available" if metrics.TRI is not None else "   • TRI: Not calculated")
    
    # Identify suitable terrain for mining
    print("\n4. Terrain Suitability Analysis:")
    
    # Mining typically requires moderate slopes (5-35°)
    suitable = (slope >= 5) & (slope <= 35)
    suitable_pct = float(suitable.sum()) / suitable.size * 100
    print(f"   Suitable terrain (5-35°): {suitable_pct:.1f}%")
    
    # Steep terrain (>45°) - may require special equipment
    steep = slope > 45
    steep_pct = float(steep.sum()) / steep.size * 100
    print(f"   Steep terrain (>45°): {steep_pct:.1f}%")
    
    # Very steep (>60°) - likely unsuitable
    very_steep = slope > 60
    very_steep_pct = float(very_steep.sum()) / very_steep.size * 100
    print(f"   Very steep terrain (>60°): {very_steep_pct:.1f}%")
    
    return dem, slope, hillshade


def example_6_visualization(data, indices, change_result):
    """
    Example 6: Visualization and Export
    
    This example shows how to:
    - Create publication-quality maps
    - Generate RGB color composites
    - Visualize mineral distributions
    - Export results to various formats
    """
    print("\n" + "=" * 70)
    print("Example 6: Visualization and Export")
    print("=" * 70)
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n1. Creating Visualizations (output directory: {output_dir})")
    
    # Create RGB composite
    print("   Creating RGB composite...")
    try:
        fig = plot_rgb_composite(
            data,
            bands=('B04', 'B03', 'B02'),
            title='True Color Composite',
            save_path=output_dir / 'true_color_composite.png',
            show=False
        )
        print("   ✓ Saved: true_color_composite.png")
    except Exception as e:
        print(f"   ✗ RGB composite error: {e}")
    
    # Create false color composite (NIR-based)
    print("   Creating false color composite...")
    try:
        fig = plot_rgb_composite(
            data,
            bands=('B08', 'B04', 'B03'),
            title='False Color (NIR-R-G)',
            save_path=output_dir / 'false_color_composite.png',
            show=False
        )
        print("   ✓ Saved: false_color_composite.png")
    except Exception as e:
        print(f"   ✗ False color error: {e}")
    
    # Plot mineral index
    if indices and len(indices) > 0:
        print("   Creating mineral index maps...")
        
        iron_oxide, clay_ratio_idx, detections = indices
        
        try:
            fig = plot_mineral_index(
                iron_oxide,
                'Iron Oxide',
                cmap='Reds',
                title='Iron Oxide Distribution',
                save_path=output_dir / 'iron_oxide_index.png',
                show=False
            )
            print("   ✓ Saved: iron_oxide_index.png")
        except Exception as e:
            print(f"   ✗ Iron oxide plot error: {e}")
        
        try:
            fig = plot_mineral_index(
                clay_ratio_idx,
                'Clay Ratio',
                cmap='Blues',
                title='Clay Mineral Distribution',
                save_path=output_dir / 'clay_ratio_index.png',
                show=False
            )
            print("   ✓ Saved: clay_ratio_index.png")
        except Exception as e:
            print(f"   ✗ Clay ratio plot error: {e}")
    
    # Plot change detection
    if change_result:
        print("   Creating change detection map...")
        
        try:
            fig = plot_change_map(
                change_result,
                background=data,
                save_path=output_dir / 'change_detection.png',
                show=False
            )
            print("   ✓ Saved: change_detection.png")
        except Exception as e:
            print(f"   ✗ Change detection plot error: {e}")
    
    print("\n2. Export Options:")
    
    export_formats = [
        ('GeoTIFF (.tif)', 'Georeferenced raster for GIS software'),
        ('NetCDF (.nc)', 'Multi-dimensional data for analysis'),
        ('PNG/JPEG', 'Visualization images'),
        ('GeoJSON', 'Vector data for boundaries'),
    ]
    
    for fmt, desc in export_formats:
        print(f"   • {fmt}: {desc}")
    
    print("\n✓ Visualization examples completed!")
    print(f"\n   Output files saved to: {output_dir.absolute()}")


def create_demo_data():
    """
    Create demonstration data for testing without API access.
    
    This function generates synthetic satellite-like data that can be used
    for testing and learning the library without actual satellite data.
    """
    print("\n" + "=" * 70)
    print("Creating Demonstration Data")
    print("=" * 70)
    
    print("\nNote: Using synthetic data for demonstration purposes.")
    print("For real analysis, use actual satellite data with API credentials.")
    
    # Create synthetic Sentinel-2 data
    import numpy as np
    import xarray as xr
    
    height, width = 256, 256
    
    # Create terrain-like patterns
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Base terrain
    base = np.sin(X) * np.cos(Y) * 0.5 + 0.5
    
    # Create bands
    blue = base * 0.25 + 0.08
    green = base * 0.35 + 0.12
    red = base * 0.30 + 0.10
    nir = base * 0.45 + 0.18
    swir1 = base * 0.40 + 0.12
    swir2 = base * 0.35 + 0.10
    
    # Add some features (iron oxide zone, clay zone)
    center_y, center_x = height // 2, width // 2
    
    # Iron oxide feature (reddish area)
    Y_idx, X_idx = np.ogrid[:height, :width]
    iron_zone = np.sqrt((X_idx - center_x - 30)**2 + (Y_idx - center_y)**2) < 25
    blue = np.where(iron_zone, blue * 0.7, blue)
    red = np.where(iron_zone, red * 1.5, red)
    
    # Clay feature (bluish area)
    clay_zone = np.sqrt((X_idx - center_x + 30)**2 + (Y_idx - center_y)**2) < 25
    swir1 = np.where(clay_zone, swir1 * 1.3, swir1)
    swir2 = np.where(clay_zone, swir2 * 0.7, swir2)
    
    # Stack bands
    bands = [blue, green, red, nir, swir1, swir2]
    
    data = xr.concat([
        xr.DataArray(band, dims=['y', 'x']) for band in bands
    ], dim='band')
    
    data = data.assign_coords({
        'band': ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    })
    
    data = data.assign_coords({
        'y': np.arange(height),
        'x': np.arange(width)
    })
    
    data.attrs = {
        'long_name': 'Synthetic Satellite Data (Demo)',
        'crs': 'EPSG:4326',
        'resolution': 10,
    }
    
    print(f"\nCreated demo data: {data.shape}")
    print(f"Bands: {list(data.coords['band'].values)}")
    
    # Create before and after data for change detection
    data_before = data.copy()
    
    # Create mining expansion in after data
    mining_mask = np.sqrt((X_idx - center_x)**2 + (Y_idx - center_y)**2) < 20
    data_after = data.copy()
    
    # Reduce vegetation (NDVI) in mining area
    nir_after = data_after.sel(band='B08').values
    nir_after = np.where(mining_mask, nir_after * 0.3, nir_after)
    data_after.loc[dict(band='B08')] = nir_after
    
    # Increase brightness (bare ground)
    for band in ['B02', 'B03', 'B04']:
        band_data = data_after.sel(band=band).values
        band_data = np.where(mining_mask, band_data * 1.2, band_data)
        data_after.loc[dict(band=band)] = band_data
    
    print("Created temporal pair (before/after) for change detection")
    
    return data, data_before, data_after


def main():
    """
    Main function demonstrating GeoMin capabilities.
    
    Run with actual satellite data for real mineral exploration workflows.
    """
    print("\n" + "=" * 70)
    print("   GeoMin - Geophysics Library for Mining Detection")
    print("   Real-World Usage Examples")
    print("=" * 70)
    
    # Check for existing data files
    data_dir = Path('data')
    has_data = data_dir.exists() and any(data_dir.iterdir())
    
    if has_data:
        print("\n✓ Found existing data directory")
    else:
        print("\n📁 No data directory found. Using demonstration mode.")
        print("   To use real data:")
        print("   1. Set up API credentials (see DOCUMENTATION.md)")
        print("   2. Search and download satellite data")
        print("   3. Place data in the 'data/' directory")
    
    # Run examples
    scene_result = example_1_search_satellite_data()
    
    data = example_2_load_and_process_satellite_data(scene_result)
    
    if data is None:
        print("\n⚠️  Using demo data for remaining examples")
        data, data_before, data_after = create_demo_data()
    else:
        # Create temporal pair for change detection
        print("\nCreating demonstration change detection data...")
        _, data_before, data_after = create_demo_data()
    
    indices = example_3_mineral_detection(data)
    
    change_result = example_4_change_detection(data_before, data_after)
    
    terrain_results = example_5_terrain_analysis()
    
    example_6_visualization(data, indices, change_result)
    
    print("\n" + "=" * 70)
    print("   All Examples Completed")
    print("=" * 70)
    
    print("\n📚 Next Steps:")
    print("   1. Read README.md for comprehensive setup guide")
    print("   2. Configure API credentials for real satellite data")
    print("   3. Explore Jupyter notebooks in examples/")
    print("   4. Check API documentation in geomin/satellites/")
    print("   5. Star the repo: https://github.com/kazashim/GeoMin")
    
    print("\n🔗 Useful Links:")
    print("   • GeoMin GitHub: https://github.com/kazashim/GeoMin")
    print("   • Sentinel-2: https://sentinel.esa.int/web/sentinel/missions/sentinel-2")
    print("   • Landsat: https://landsat.gsfc.nasa.gov/")
    print("   • GeoMin Docs: See README.md")
    
    print()


if __name__ == "__main__":
    main()
