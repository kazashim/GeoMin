# GeoMin: Geophysics Library for Satellite-Based Mining Detection

## Overview

GeoMin is a comprehensive Python library designed for satellite-based mining activity detection and mineral identification. The library provides robust algorithms for processing satellite imagery, detecting spectral signatures associated with minerals, and identifying changes in terrain that indicate mining activities.

## Features

### Satellite Data Access
- **Multi-Satellite Support**: Connect to Landsat, Sentinel-2, and commercial satellite providers
- **Unified API**: Consistent interface for searching and downloading satellite imagery
- **Caching**: Automatic caching of downloaded data for efficient processing

### Mineral Detection
- **Spectral Analysis**: Pre-defined and custom spectral indices for mineral detection
- **Target Minerals**: Iron oxides, clay minerals, sulfides, quartz, and carbonates
- **Probability Mapping**: Generate probability maps for mineral occurrence

### Change Detection
- **Temporal Analysis**: Detect surface changes between different time periods
- **Mining Activity**: Identify vegetation loss, soil exposure, and mine expansion
- **Multiple Methods**: Simple difference, PCA, K-means, and vegetation-based detection

### Terrain Analysis
- **DEM Processing**: Calculate slope, aspect, and curvature from elevation models
- **Hillshade**: Generate shaded relief visualizations
- **TRI/TPI**: Terrain ruggedness and position indices

### Visualization
- **Static Plots**: Publication-quality maps using Matplotlib
- **Interactive Maps**: Web-based visualizations (Folium integration ready)
- **Spectral Profiles**: Plot spectral signatures at specific locations

## Installation

```bash
# Basic installation
pip install geomin

# With GPU support for deep learning
pip install geomin[gpu]

# With interactive visualization
pip install geomin[interactive]

# For development
pip install geomin[dev]
```

## Quick Start

```python
import geomin as gm

# Initialize satellite client
client = gm.SentinelClient()
client.connect()

# Search for imagery
scenes = client.search(
    bbox=[-110.0, 35.0, -109.0, 36.0],
    date='2023-01-01',
    cloud_cover=10
)

# Download and process
data = gm.DataLoader.load(scenes[0])

# Calculate mineral indices
iron_map = gm.algorithms.spectral.iron_oxide_index(data)
clay_map = gm.algorithms.spectral.clay_ratio(data)

# Visualize
gm.viz.plot_mineral_index(iron_map, cmap='magma')
```

## Library Structure

```
geomin/
├── core/                      # Core utilities
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── crs.py                 # Coordinate reference system utilities
│   └── data_loader.py         # Universal data loading
├── satellites/                # Satellite data providers
│   ├── __init__.py
│   ├── base_client.py         # Abstract base class
│   ├── landsat.py             # Landsat 8/9 client
│   ├── sentinel.py            # Sentinel-2 client
│   └── commercial.py          # Planet/Maxar clients
├── algorithms/                # Analysis algorithms
│   ├── __init__.py
│   ├── spectral.py            # Spectral analysis
│   └── terrain.py             # Terrain processing
├── models/                    # ML models
│   ├── __init__.py
│   └── change_detection.py    # Change detection algorithms
├── visualization/             # Visualization tools
│   ├── __init__.py
│   └── static.py              # Matplotlib-based plots
├── examples/                  # Example scripts
│   └── quickstart.py          # Quick start guide
├── tests/                     # Test suite
│   └── test_geomin.py
├── setup.py
├── requirements.txt
└── README.md
```

## API Reference

### Core Module

```python
from geomin.core.config import Config, get_config
from geomin.core.data_loader import DataLoader
from geomin.core.crs import transform_bbox, get_utm_zone
```

### Satellite Clients

```python
from geomin import SentinelClient, LandsatClient, PlanetClient, MaxarClient

# Search for imagery
client = SentinelClient()
results = client.search(options)

# Download data
files = client.download(result, bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])

# Load directly to xarray
data = client.load(result)
```

### Spectral Analysis

```python
from geomin.algorithms import spectral

# Pre-defined indices
iron = spectral.iron_oxide_index(data, red='B04', blue='B02')
clay = spectral.clay_ratio(data, swir1='B11', swir2='B12')
ndvi = spectral.ndvi(data, nir='B08', red='B04')

# Calculate all indices
indices = spectral.calculate_all_indices(data)

# Detect minerals
detections = spectral.detect_minerals(data, threshold=0.7)
```

### Change Detection

```python
from geomin.models import change_detection

# Simple difference
result = change_detection.simple_difference(img1, img2)

# Vegetation-based detection
result = change_detection.vegetation_change_detector(img1, img2)

# PCA-based detection
result = change_detection.pca_change_detector(img1, img2)

# Mining activity detection
result = change_detection.detect_mining_activity(img1, img2)
```

### Terrain Analysis

```python
from geomin.algorithms import terrain

slope = terrain.calculate_slope(dem)
aspect = terrain.calculate_aspect(dem)
hillshade = terrain.calculate_hillshade(dem, azimuth=315, altitude=45)
metrics = terrain.calculate_terrain_metrics(dem)
```

### Visualization

```python
from geomin import visualization as viz

# Plot mineral index
viz.plot_mineral_index(iron_map, 'Iron Oxide')

# Create RGB composite
viz.plot_rgb_composite(data, bands=('B04', 'B03', 'B02'))

# Plot change detection
viz.plot_change_map(change_result, background=data)

# Create mineral map with detections
viz.create_mineral_map(data, detections)
```

## Configuration

GeoMin uses a configuration file or environment variables for API credentials:

```bash
# Environment variables
export GEOMIN_COPERNICUS_USERNAME="your_username"
export GEOMIN_COPERNICUS_PASSWORD="your_password"
export GEOMIN_PLANET_API_KEY="your_api_key"
export GEOMIN_MAXAR_API_KEY="your_api_key"
export GEOMIN_CACHE_DIR="/path/to/cache"
```

## Supported Satellites

| Satellite | Resolution | Bands | Data Source |
|-----------|------------|-------|-------------|
| Sentinel-2A/B | 10-60m | 13 bands | Copernicus Open Access Hub |
| Landsat 8/9 | 15-100m | 11 bands | USGS EarthData / Google Cloud |
| PlanetScope | 3m | 4 bands | Planet Labs API |
| SkySat | 0.5m | RGB+NIR | Planet/Maxar APIs |

## Mineral Detection Capabilities

| Mineral Type | Spectral Signature | Index |
|--------------|-------------------|-------|
| Iron Oxides | Red/Blue absorption | Iron Oxide Ratio |
| Clay Minerals | SWIR absorption | Clay Ratio |
| Hydroxides | OH absorption | Hydroxyl Index |
| Ferrous Minerals | NIR/SWIR | Ferrous Index |
| Gossans | Mixed | Gossan Index |
| Carbonates | SWIR features | Carbonate Index |

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

MIT License - see LICENSE file for details.

## Citation

If you use GeoMin in your research, please cite:

```
Kazashim Kuzasuwat. (2024). GeoMin: Geophysics Library for 
Satellite-Based Mining Detection. Version 0.1.0.
https://github.com/kazashim/GeoMin
```

## Contact

- GitHub: https://github.com/kazashim/GeoMin
- Documentation: See README.md for full documentation
