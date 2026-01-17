"""
Advanced GeoMin Features Demo
=============================

This script demonstrates the advanced v2.0 features of GeoMin:
- STAC Client for accessing cloud-optimized satellite data
- Anomaly detection for identifying potential mineral deposits
- Cloud masking for preprocessing
- Advanced mineralogy algorithms (Crosta PCA, Spectral Unmixing)

Requirements:
    pip install geomin[stac]

Author: Kazashim Kuzasuwat
License: MIT
"""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import tempfile
import os

# GeoMin imports
from geomin.satellites.stac_client import STACClient, STACConfig
from geomin.models.anomaly_detection import AnomalyDetector, AnomalyResult
from geomin.core.cloud_masking import CloudMasker, CloudMaskResult
from geomin.algorithms.advanced_mineralogy import AdvancedMineralogy, CrostaResult


def generate_synthetic_satellite_data(
    height: int = 200, 
    width: int = 200, 
    n_bands: int = 6
) -> xr.DataArray:
    """
    Generate synthetic satellite imagery for demonstration.
    
    Creates a realistic multi-band image with various features
    including potential mineral anomalies, vegetation, and water.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        n_bands: Number of spectral bands
        
    Returns:
        xarray DataArray with synthetic satellite data
    """
    # Create coordinate arrays
    y_coords = np.arange(height)
    x_coords = np.arange(width)
    
    # Generate base spectral data for different land covers
    # Band order: Blue, Green, Red, NIR, SWIR1, SWIR2
    
    # Background (soil/rock) spectrum
    base_spectrum = np.array([0.18, 0.22, 0.26, 0.32, 0.38, 0.35])
    
    # Vegetation spectrum (high NIR)
    vegetation_spectrum = np.array([0.08, 0.12, 0.10, 0.55, 0.30, 0.25])
    
    # Iron oxide anomaly (high Red and NIR)
    iron_anomaly_spectrum = np.array([0.35, 0.40, 0.45, 0.55, 0.50, 0.48])
    
    # Hydrothermal alteration (high SWIR)
    alteration_spectrum = np.array([0.15, 0.18, 0.22, 0.30, 0.60, 0.55])
    
    # Create coordinate meshgrid
    y, x = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Initialize with background
    data = np.zeros((n_bands, height, width))
    
    # Add base spectrum to all pixels with some noise
    np.random.seed(42)  # For reproducibility
    for b in range(n_bands):
        data[b] = base_spectrum[b] + np.random.normal(0, 0.02, (height, width))
    
    # Add vegetation region (upper left corner)
    veg_mask = (y > 50) & (x > 50) & (y < 120) & (x < 120)
    for b in range(n_bands):
        data[b, veg_mask] = vegetation_spectrum[b] + np.random.normal(0, 0.02, data[b, veg_mask].shape)
    
    # Add iron oxide anomaly (potential mineralization) - right side
    iron_center_y, iron_center_x = 80, 150
    iron_radius = 25
    iron_dist = np.sqrt((y - iron_center_y)**2 + (x - iron_center_x)**2)
    iron_mask = iron_dist < iron_radius
    for b in range(n_bands):
        # Fade edge of anomaly
        fade = 1 - (iron_dist[iron_mask] / iron_radius) * 0.3
        data[b, iron_mask] = (
            iron_anomaly_spectrum[b] * fade + 
            base_spectrum[b] * (1 - fade) +
            np.random.normal(0, 0.01, iron_mask.sum())
        )
    
    # Add hydrothermal alteration zone - bottom center
    alter_center_y, alter_center_x = 150, 100
    alter_radius = 30
    alter_dist = np.sqrt((y - alter_center_y)**2 + (x - alter_center_x)**2)
    alter_mask = alter_dist < alter_radius
    for b in range(n_bands):
        fade = 1 - (alter_dist[alter_mask] / alter_radius) * 0.3
        data[b, alter_mask] = (
            alteration_spectrum[b] * fade + 
            base_spectrum[b] * (1 - fade) +
            np.random.normal(0, 0.01, alter_mask.sum())
        )
    
    # Create xarray DataArray
    band_names = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    
    data_array = xr.DataArray(
        data,
        dims=['band', 'y', 'x'],
        coords={
            'band': band_names,
            'y': y_coords,
            'x': x_coords,
        },
        attrs={
            'long_name': 'Synthetic Sentinel-2 Data',
            'resolution': 10,
            'sensor': 'Synthetic',
            'acquisition_time': datetime.now(),
        }
    )
    
    return data_array


def demo_stac_client():
    """
    Demonstrate STAC client functionality.
    
    Shows how to:
    - Connect to STAC endpoints
    - Search for satellite imagery
    - Filter by date and cloud cover
    """
    print("\n" + "="*60)
    print("STAC Client Demo")
    print("="*60)
    
    # Initialize STAC client with AWS Earth Search endpoint
    print("\n1. Initializing STAC client...")
    client = STACClient(endpoint='aws')
    
    # Connect to catalog
    if client.connect():
        print("   ✓ Connected to AWS Earth Search STAC")
    else:
        print("   ✗ Failed to connect to STAC catalog")
        print("   (This is expected without network access)")
        return
    
    # Get available collections
    print("\n2. Available collections:")
    collections = client.get_collections()
    for coll_id, info in list(collections.items())[:3]:
        print(f"   - {coll_id}: {info.get('title', 'N/A')}")
    
    # Search for Sentinel-2 data (using mock coordinates)
    print("\n3. Searching for Sentinel-2 imagery...")
    from geomin.satellites.base_client import SearchOptions
    from shapely.geometry import box
    
    # Example: Mining district bounding box (Chilean Copper Belt)
    mining_area = box(-70.5, -30, -69, -28)
    
    options = SearchOptions(
        geometry=mining_area,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        cloud_cover=20,
        limit=10
    )
    
    results = client.search(options)
    print(f"   Found {len(results)} scenes matching criteria")
    
    # Show first few results
    for i, result in enumerate(results[:3]):
        print(f"   Scene {i+1}: {result.scene_id}")
        print(f"      Date: {result.acquisition_time}")
        print(f"      Cloud Cover: {result.cloud_cover}%")
    
    # Filter by cloud cover
    print("\n4. Filtering by cloud cover (< 10%)...")
    filtered = client.filter_by_cloud_cover(results, max_cloud_cover=10)
    print(f"   ✓ {len(filtered)} scenes with < 10% cloud cover")
    
    # Disconnect
    client.disconnect()
    print("\n5. Disconnected from STAC catalog")


def demo_anomaly_detection(data: xr.DataArray):
    """
    Demonstrate anomaly detection algorithms.
    
    Shows how to:
    - Use Isolation Forest for anomaly detection
    - Apply RX (Reed-Xiaoli) anomaly detector
    - Use Local Outlier Factor
    - Get top anomaly locations
    """
    print("\n" + "="*60)
    print("Anomaly Detection Demo")
    print("="*60)
    
    detector = AnomalyDetector()
    
    # Define bands to use (standard Sentinel-2 bands)
    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    
    # 1. Isolation Forest
    print("\n1. Isolation Forest Anomaly Detection")
    print("   (Tree-based method for identifying spectral outliers)")
    result_iforest = detector.detect_anomalies(
        data, 
        method='isolation_forest',
        contamination=0.02,
        n_estimators=100,
        bands=bands
    )
    
    print(f"   Method: {result_iforest.statistics['method']}")
    print(f"   Anomalies Detected: {result_iforest.statistics['anomaly_pixels']}")
    print(f"   Anomaly Percentage: {result_iforest.statistics['anomaly_percentage']:.2f}%")
    
    # 2. RX Anomaly Detector
    print("\n2. RX (Reed-Xiaoli) Anomaly Detector")
    print("   (Standard algorithm for remote sensing anomaly detection)")
    result_rx = detector.detect_anomalies(
        data,
        method='rx',
        contamination=0.02,
        bands=bands
    )
    
    print(f"   Method: {result_rx.statistics['method']}")
    print(f"   Anomalies Detected: {result_rx.statistics['anomaly_pixels']}")
    print(f"   Threshold: {result_rx.statistics['threshold']:.4f}")
    
    # 3. Local Outlier Factor
    print("\n3. Local Outlier Factor (LOF)")
    print("   (Density-based method for detecting local anomalies)")
    result_lof = detector.detect_anomalies(
        data,
        method='lof',
        contamination=0.02,
        n_neighbors=20,
        bands=bands
    )
    
    print(f"   Method: {result_lof.statistics['method']}")
    print(f"   Anomalies Detected: {result_lof.statistics['anomaly_pixels']}")
    
    # 4. Get top anomalies
    print("\n4. Top 10 Anomaly Locations")
    y_coords, x_coords = detector.get_top_anomalies(result_rx, n=10)
    
    for i, (y, x) in enumerate(zip(y_coords, x_coords)):
        score = result_rx.anomaly_scores.values[y, x]
        print(f"   #{i+1}: Y={y}, X={x}, Score={score:.4f}")
    
    # 5. Create ranked anomaly map
    print("\n5. Creating Ranked Anomaly Map")
    ranked_map = detector.rank_anomalies(result_rx, n=50)
    print(f"   Ranked map shape: {ranked_map.shape}")
    print(f"   Max rank: {int(ranked_map.max())}")
    
    return result_iforest, result_rx, result_lof


def demo_cloud_masking(data: xr.DataArray):
    """
    Demonstrate cloud masking functionality.
    
    Shows how to:
    - Apply threshold-based cloud detection
    - Use Sentinel-2 specific algorithms
    - Parse Landsat QA bands
    - Apply masks to data
    """
    print("\n" + "="*60)
    print("Cloud Masking Demo")
    print("="*60)
    
    masker = CloudMasker(algorithm='sentinel2')
    
    # Apply cloud masking
    print("\n1. Applying Sentinel-2 Cloud Detection")
    result = masker.mask_clouds(data)
    
    print(f"   Algorithm: {result.statistics['algorithm']}")
    print(f"   Total Pixels: {result.statistics['total_pixels']}")
    print(f"   Cloud Pixels: {result.statistics['cloud_pixels']}")
    print(f"   Cloud Percentage: {result.statistics['cloud_percentage']:.2f}%")
    
    # Show threshold values used
    print("\n2. Detection Thresholds Used:")
    for key, value in result.statistics['thresholds'].items():
        print(f"   - {key}: {value}")
    
    # Apply mask to data
    print("\n3. Applying Cloud Mask to Data")
    masked_data = masker.apply_mask(data, result.mask, fill_value=np.nan)
    print(f"   Original data shape: {data.shape}")
    print(f"   Masked data shape: {masked_data.shape}")
    
    # Get clear pixels
    print("\n4. Extracting Clear Pixels")
    clear_pixels = masker.get_clear_pixels(data, result.mask)
    nan_count = np.isnan(clear_pixels.values).sum()
    print(f"   Pixels set to NaN (clouds): {nan_count}")
    
    return result, masked_data


def demo_advanced_mineralogy(data: xr.DataArray):
    """
    Demonstrate advanced mineralogy algorithms.
    
    Shows how to:
    - Perform Crosta PCA for alteration mapping
    - Calculate Spectral Angle Mapper matches
    - Perform Linear Spectral Unmixing
    - Use reference mineral spectra
    """
    print("\n" + "="*60)
    print("Advanced Mineralogy Demo")
    print("="*60)
    
    analyzer = AdvancedMineralogy()
    
    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    
    # 1. Crosta PCA for Hydroxyl Alteration
    print("\n1. Crosta PCA for Hydroxyl Alteration Mapping")
    print("   (Directed PCA to identify clay and hydroxyl minerals)")
    crosta_result = analyzer.crosta_pca(
        data,
        bands=bands,
        n_components=4,
        target_mineral='hydroxyl'
    )
    
    print(f"   Components extracted: {crosta_result.statistics['n_components']}")
    print(f"   Explained variance ratios:")
    for i, var in enumerate(crosta_result.statistics['explained_variance_ratio']):
        print(f"      PC{i+1}: {var:.4f} ({var*100:.1f}%)")
    
    print(f"   Mineral components identified:")
    for mineral, comp_idx in crosta_result.statistics['mineral_components'].items():
        print(f"      - {mineral}: Component {comp_idx}")
    
    # 2. Iron Oxide Detection
    print("\n2. Crosta PCA for Iron Oxide Detection")
    iron_result = analyzer.crosta_pca(
        data,
        bands=bands,
        n_components=4,
        target_mineral='iron'
    )
    
    print(f"   Mineral components identified:")
    for mineral, comp_idx in iron_result.statistics['mineral_components'].items():
        print(f"      - {mineral}: Component {comp_idx}")
    
    # 3. Spectral Angle Mapper
    print("\n3. Spectral Angle Mapper (SAM) Analysis")
    
    # Get reference spectrum for kaolinite (clay mineral)
    kaolinite_spectrum = analyzer.get_reference_spectrum('kaolinite')
    print(f"   Kaolinite reference spectrum: {kaolinite_spectrum}")
    
    # Calculate SAM
    sam_map = analyzer.spectral_angle_mapper(
        data,
        reference_spectrum=kaolinite_spectrum,
        threshold=0.15
    )
    
    print(f"   SAM map shape: {sam_map.shape}")
    print(f"   Threshold: {sam_map.attrs['threshold']} radians")
    
    # Count potential kaolinite matches
    matches = np.sum(sam_map.values < sam_map.attrs['threshold'])
    print(f"   Potential matches (angle < threshold): {matches}")
    
    # 4. Linear Spectral Unmixing
    print("\n4. Linear Spectral Unmixing")
    
    # Create endmembers for unmixing
    endmembers = analyzer.create_endmember_dict(['kaolinite', 'hematite', 'vegetation', 'soil'])
    print(f"   Endmembers: {list(endmembers.keys())}")
    
    # Perform unmixing
    unmix_result = analyzer.linear_spectral_unmixing(data, endmembers)
    
    print(f"   RMSE: {unmix_result.rmse:.4f}")
    print(f"   Abundance maps shape: {unmix_result.abundances.shape}")
    
    # Show abundance statistics for each endmember
    for endmember in unmix_result.abundances.endmember.values:
        abundance = unmix_result.abundances.sel(endmember=endmember)
        mean_abundance = float(abundance.mean())
        print(f"   - {endmember}: mean abundance = {mean_abundance:.4f}")
    
    # 5. Available Reference Minerals
    print("\n5. Available Reference Minerals:")
    mineral_spectra = ['kaolinite', 'alunite', 'jarosite', 'hematite', 'goethite',
                       'sericite', 'chlorite', 'calcite', 'dolomite', 'quartz']
    print(f"   {', '.join(mineral_spectra)}")
    
    return crosta_result, sam_map, unmix_result


def demo_complete_pipeline():
    """
    Demonstrate a complete mining exploration workflow.
    
    This combines all the advanced features into a typical
    exploration workflow:
    1. Query STAC for data
    2. Preprocess (cloud masking)
    3. Detect anomalies
    4. Map minerals
    """
    print("\n" + "="*60)
    print("Complete Mining Exploration Workflow")
    print("="*60)
    
    print("\nNote: This demo uses synthetic data.")
    print("In production, you would:")
    print("  1. Use STACClient to query real satellite data")
    print("  2. Load data using stackstac for lazy loading")
    print("  3. Apply cloud masking")
    print("  4. Run anomaly detection to find targets")
    print("  5. Use Crosta PCA and SAM for mineral mapping")
    
    # Generate synthetic data
    print("\n1. Generating synthetic satellite data...")
    data = generate_synthetic_satellite_data(height=200, width=200, n_bands=6)
    print(f"   Data shape: {data.shape}")
    print(f"   Bands: {list(data.coords['band'].values)}")
    
    # Apply cloud masking
    print("\n2. Applying cloud masking...")
    masker = CloudMasker(algorithm='sentinel2')
    cloud_result = masker.mask_clouds(data)
    print(f"   Cloud cover: {cloud_result.statistics['cloud_percentage']:.1f}%")
    
    # Detect anomalies
    print("\n3. Running anomaly detection...")
    detector = AnomalyDetector()
    bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    anomaly_result = detector.detect_anomalies(
        data, method='rx', contamination=0.03, bands=bands
    )
    print(f"   Anomalies detected: {anomaly_result.statistics['anomaly_pixels']}")
    
    # Map minerals
    print("\n4. Performing mineral mapping...")
    analyzer = AdvancedMineralogy()
    crosta_result = analyzer.crosta_pca(
        data, bands=bands, n_components=4, target_mineral='hydroxyl'
    )
    print(f"   Hydroxyl alteration components: {list(crosta_result.statistics['mineral_components'].keys())}")
    
    # Get top anomaly coordinates
    print("\n5. Top 5 Priority Anomaly Targets:")
    y_coords, x_coords = detector.get_top_anomalies(anomaly_result, n=5)
    
    # Create a simple priority table
    print("   | Rank |   Y   |   X   |  Score  | Priority |")
    print("   |------|-------|-------|---------|----------|")
    for i, (y, x) in enumerate(zip(y_coords, x_coords)):
        score = anomaly_result.anomaly_scores.values[y, x]
        # Simple priority based on score
        if score > 0.8:
            priority = "HIGH"
        elif score > 0.6:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        print(f"   |  {i+1}   | {y:5d} | {x:5d} | {score:.4f} | {priority:8s} |")
    
    # Export summary
    print("\n6. Workflow Summary:")
    print(f"   - Data: {data.sizes['y']}x{data.sizes['x']} pixels, {data.sizes['band']} bands")
    print(f"   - Cloud pixels: {cloud_result.statistics['cloud_pixels']} ({cloud_result.statistics['cloud_percentage']:.1f}%)")
    print(f"   - Anomaly pixels: {anomaly_result.statistics['anomaly_pixels']}")
    print(f"   - Analysis complete")


def main():
    """Run all demonstrations."""
    print("="*60)
    print("GeoMin v2.0 Advanced Features Demo")
    print("="*60)
    print("\nThis demo showcases the advanced features of GeoMin:")
    print("  - STAC Client for satellite data access")
    print("  - Anomaly Detection for mineral exploration")
    print("  - Cloud Masking for data preprocessing")
    print("  - Advanced Mineralogy algorithms")
    
    # Try STAC demo (requires network)
    try:
        demo_stac_client()
    except Exception as e:
        print(f"\nSTAC demo skipped: {e}")
    
    # Generate synthetic data for offline demos
    print("\nGenerating synthetic satellite data for demonstrations...")
    synthetic_data = generate_synthetic_satellite_data()
    
    # Run demos that use synthetic data
    anomaly_results = demo_anomaly_detection(synthetic_data)
    cloud_result = demo_cloud_masking(synthetic_data)
    mineralogy_results = demo_advanced_mineralogy(synthetic_data)
    
    # Run complete workflow
    demo_complete_pipeline()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("  1. Install STAC dependencies: pip install geomin[stac]")
    print("  2. Get API credentials for your chosen provider")
    print("  3. Try with real data from STAC endpoints")
    print("  4. Explore the full API in geomin/algorithms/")
    print("\nFor more information, see:")
    print("  - README.md for installation and basic usage")
    print("  - DOCUMENTATION.md for detailed API reference")


if __name__ == "__main__":
    main()
