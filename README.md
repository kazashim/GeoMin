# GeoMin: Geophysics Library for Satellite-Based Mining Detection

GeoMin is a comprehensive Python library designed for satellite-based mining activity detection and mineral identification. The library provides robust algorithms for processing satellite imagery, detecting spectral signatures associated with minerals, and identifying changes in terrain that indicate mining activities.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Setup Guide](#setup-guide)
4. [Quick Start](#quick-start)
5. [API Overview](#api-overview)
6. [Examples](#examples)
7. [Data Sources](#data-sources)
8. [Mineral Detection](#mineral-detection)
9. [Change Detection](#change-detection)
10. [Visualization](#visualization)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [License](#license)

---

## Features

- **Multi-Satellite Support**: Connect to Landsat, Sentinel-2, Planet, and Maxar satellite providers
- **Mineral Detection**: Identify iron oxides, clay minerals, sulfides, and other economically important minerals
- **Mining Activity Detection**: Detect active mining operations through temporal change analysis
- **Spectral Analysis**: Apply custom and pre-defined spectral indices for mineral mapping
- **Terrain Analysis**: Process DEM data for slope, aspect, and terrain characterization
- **Interactive Visualization**: Generate publication-quality maps and visualizations

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/kazashim/GeoMin.git
cd GeoMin

# Install dependencies
pip install -r requirements.txt

# Install GeoMin
pip install -e .
```

### With Optional Dependencies

```bash
# With GPU support for deep learning
pip install -e .[gpu]

# With interactive visualization
pip install -e .[interactive]

# For development and testing
pip install -e .[dev]

# Install all optional dependencies
pip install -e .[gpu,interactive,dev]
```

### System Requirements

- Python 3.8 or higher
- 8 GB RAM (16 GB recommended for large datasets)
- 50 GB free disk space for caching satellite data
- GPU optional (for deep learning features)

---

## Setup Guide

### Step 1: Configure API Credentials

GeoMin requires API credentials to access satellite data. You can configure credentials via environment variables or a configuration file.

#### Environment Variables

Add the following to your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.profile`):

```bash
# Copernicus (Sentinel-2 data - FREE)
export GEOMIN_COPERNICUS_USERNAME="your_username"
export GEOMIN_COPERNICUS_PASSWORD="your_password"

# USGS EarthData (Landsat data - FREE)
export GEOMIN_EARTHDATA_USERNAME="your_username"
export GEOMIN_EARTHDATA_PASSWORD="your_password"

# Planet Labs (High-resolution data - COMMERCIAL)
export GEOMIN_PLANET_API_KEY="your_api_key"

# Maxar (Very high-resolution data - COMMERCIAL)
export GEOMIN_MAXAR_API_KEY="your_api_key"

# Cache directory (optional)
export GEOMIN_CACHE_DIR="/path/to/cache"
```

Apply changes:
```bash
source ~/.bashrc
```

#### Configuration File

Create a JSON configuration file:

```json
{
    "api": {
        "copernicus_username": "your_username",
        "copernicus_password": "your_password",
        "earthdata_username": "your_username",
        "earthdata_password": "your_password",
        "planet_api_key": "your_api_key",
        "maxar_api_key": "your_api_key"
    },
    "cache": {
        "cache_dir": "/path/to/cache",
        "max_cache_size_gb": 50,
        "cache_enabled": true
    },
    "processing": {
        "default_crs": "EPSG:4326",
        "max_workers": 4,
        "use_gpu": false
    }
}
```

Load the configuration:

```python
from pathlib import Path
import geomin as gm

config = gm.Config(config_path=Path("/path/to/config.json"))
```

### Step 2: Obtain API Credentials

#### Copernicus (Sentinel-2) - FREE

1. Register at: https://developer.Copernicus.eu/
2. Verify your email
3. Use your credentials in the environment variables

#### USGS EarthData (Landsat) - FREE

1. Register at: https://ers.cr.usgs.gov/register/
2. Note: The same credentials work for Landsat and other USGS data products

#### Planet Labs - COMMERCIAL

1. Sign up at: https://www.planet.com/
2. Get API key from your account dashboard
3. Requires subscription for commercial use

#### Maxar - COMMERCIAL

1. Contact: https://www.maxar.com/
2. Request API access for your organization
3. Requires enterprise agreement

### Step 3: Verify Installation

```python
import geomin as gm

# Check version
print(gm.__version__)

# Check configuration
config = gm.get_config()
print(f"Cache directory: {config.cache.cache_dir}")

# Check API credentials
from geomin.core.config import get_config
creds = check_api_credentials()
print(creds)
```

---

## Quick Start

### Basic Workflow

```python
import geomin as gm
from datetime import datetime, timedelta

# 1. Initialize satellite client
client = gm.SentinelClient()
client.connect()

# 2. Search for imagery
from geomin.satellites.base_client import SearchOptions

options = gm.SearchOptions(
    bbox=[-110.0, 35.0, -109.0, 36.0],  # Morenci, AZ mining region
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now(),
    cloud_cover=20,
    limit=5
)

results = client.search(options)
print(f"Found {len(results)} scenes")

# 3. Select best scene (least cloud cover)
best = client.get_best_result(results, ['cloud_cover'])
print(f"Selected: {best.scene_id}")

# 4. Download data
files = client.download(best, bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])

# 5. Load into xarray
data = gm.DataLoader.load(list(files.values())[0])
print(f"Data shape: {data.shape}")

# 6. Calculate mineral indices
iron = gm.algorithms.spectral.iron_oxide_index(data, red='B04', blue='B02')
clay = gm.algorithms.spectral.clay_ratio(data, swir1='B11', swir2='B12')

# 7. Visualize
gm.visualization.static.plot_mineral_index(iron, 'Iron Oxide', show=True)
```

---

## API Overview

### Core Module

```python
# Configuration
config = gm.Config()  # Create configuration
config = gm.get_config()  # Get global configuration

# Data loading
data = gm.DataLoader.load("path/to/file.tif")  # Load from file
data = gm.DataLoader.load_multiple(["file1.tif", "file2.tif"])  # Load multiple files
info = gm.DataLoader.get_raster_info("path/to/file.tif")  # Get raster metadata

# CRS utilities
utm_crs = gm.get_utm_zone(-110, 35)  # Get UTM zone
bbox_wgs84 = gm.transform_bbox(bbox_utm, "EPSG:32611", "EPSG:4326")  # Transform
```

### Satellite Clients

```python
# Sentinel-2 (Recommended for mineral detection)
client = gm.SentinelClient()
client.connect()
results = client.search(options)
files = client.download(result, bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])
data = client.load(result)

# Landsat
client = gm.LandsatClient(use_google_cloud=True)  # Use free GCS access
results = client.search(options)

# Planet (High-resolution)
client = gm.PlanetClient()
results = client.search(options)

# Maxar (Very high-resolution)
client = gm.MaxarClient()
results = client.search(options)
```

### Search Options

```python
from geomin.satellites.base_client import SearchOptions

options = SearchOptions(
    bbox=(-110.0, 35.0, -109.0, 36.0),  # minx, miny, maxx, maxy
    geometry=polygon,  # Alternative to bbox
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    cloud_cover=20,  # Maximum cloud cover percentage
    bands=['B04', 'B08', 'B11'],  # Required bands
    min_resolution=10,  # Best resolution in meters
    source='Sentinel-2',  # Specific source
    limit=10  # Maximum results
)
```

---

## Examples

### Example 1: Mineral Detection

```python
import geomin as gm
from geomin.algorithms.spectral import (
    iron_oxide_index, clay_ratio, ndvi, hydroxyl_index
)

# Load satellite data
data = gm.DataLoader.load("sentinel2_scene.tif")

# Calculate multiple indices
iron_oxide = iron_oxide_index(data, red='B04', blue='B02')
clay = clay_ratio(data, swir1='B11', 'swir2='B12')
ndvi = ndvi(data, nir='B08', red='B04')
hydroxyl = hydroxyl_index(data, swir1='B11', swir2='B12')

# Visualize
gm.visualization.static.plot_mineral_index(
    iron_oxide, 
    'Iron Oxide Index',
    cmap='Reds',
    title='Iron Oxide Distribution (Hematite/Goethite)',
    show=True
)
```

### Example 2: Change Detection

```python
import geomin as gm
from geomin.models.change_detection import (
    vegetation_change_detector, pca_change_detector
)

# Load before and after images
before = gm.DataLoader.load("scene_2020.tif")
after = gm.DataLoader.load("scene_2023.tif")

# Detect vegetation loss (mining expansion indicator)
result = vegetation_change_detector(before, after)

print(f"Changed pixels: {result.statistics['total_changed_pixels']}")
print(f"Change percentage: {result.statistics['change_percentage']:.2f}%")

# Visualize
gm.visualization.static.plot_change_map(
    result,
    background=before,
    show=True
)
```

### Example 3: Terrain Analysis

```python
import geomin as gm
from geomin.algorithms.terrain import (
    calculate_slope, calculate_hillshade, calculate_terrain_metrics
)

# Load DEM
dem = gm.DataLoader.load("dem.tif")

# Calculate terrain metrics
slope = calculate_slope(dem)
aspect = calculate_aspect(dem)
hillshade = calculate_hillshade(dem, azimuth=315, altitude=45)

# Comprehensive analysis
metrics = calculate_terrain_metrics(dem)

# Identify suitable terrain (5-35 degrees for mining)
suitable = (slope >= 5) & (slope <= 35)
suitable_pct = suitable.sum() / suitable.size * 100
print(f"Suitable terrain: {suitable_pct:.1f}%")
```

### Example 4: Complete Workflow

```python
import geomin as gm
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# 1. Search and download data
client = gm.SentinelClient()
client.connect()

options = gm.SearchOptions(
    bbox=[-110.0, 35.0, -109.0, 36.0],
    start_date=datetime.now() - timedelta(days=180),
    end_date=datetime.now(),
    cloud_cover=15,
    limit=10
)

results = client.search(options)
best = client.get_best_result(results, ['cloud_cover', 'date'])
files = client.download(best, bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])
data = gm.DataLoader.load(list(files.values())[0])

# 2. Calculate mineral indices
indices = gm.algorithms.spectral.calculate_all_indices(data)

# 3. Detect minerals
detections = gm.algorithms.spectral.detect_minerals(data, threshold=0.7)

# 4. Change detection
before_data = gm.DataLoader.load("historical_scene.tif")
change_result = gm.models.change_detection.vegetation_change_detector(
    before_data, data
)

# 5. Save results
gm.visualization.static.plot_mineral_index(
    indices['iron_oxide'],
    'Iron Oxide',
    save_path=output_dir / 'iron_oxide.png'
)

print(f"Results saved to: {output_dir}")
```

---

## Data Sources

### Sentinel-2 (Recommended for Mineral Detection)

| Property | Value |
|----------|-------|
| Spatial Resolution | 10m (visible), 20m (red edge/SWIR), 60m (aerosols) |
| Spectral Bands | 13 bands (visible to SWIR) |
| Revisit Time | 5 days at equator |
| Data Source | Copernicus Open Access Hub |
| Cost | FREE |

### Landsat 8/9

| Property | Value |
|----------|-------|
| Spatial Resolution | 15m (panchromatic), 30m (multispectral), 100m (thermal) |
| Spectral Bands | 11 bands |
| Revisit Time | 16 days |
| Data Source | USGS EarthData, Google Cloud Storage |
| Cost | FREE |

### Planet Scope

| Property | Value |
|----------|-------|
| Spatial Resolution | 3m |
| Spectral Bands | 4 bands (RGB + NIR) |
| Revisit Time | Daily at equator |
| Data Source | Planet Labs API |
| Cost | Commercial |

### Maxar

| Property | Value |
|----------|-------|
| Spatial Resolution | 0.3m (WorldView), 0.5m (GeoEye) |
| Spectral Bands | 4-8 bands (depending on satellite) |
| Revisit Time | Multiple daily |
| Data Source | Maxar API |
| Cost | Commercial |

---

## Mineral Detection

### Supported Minerals and Detection Methods

| Mineral Type | Spectral Signature | Index | Interpretation |
|--------------|-------------------|-------|----------------|
| Iron Oxides | Red/Blue absorption | Iron Oxide Ratio | High values indicate hematite/goethite |
| Clay Minerals | SWIR absorption | Clay Ratio | High values indicate hydrothermal alteration |
| Hydroxides | OH absorption | Hydroxyl Index | High values indicate clay/alunite |
| Ferrous Minerals | NIR/SWIR features | Ferrous Index | High values indicate magnetite/siderite |
| Gossans | Mixed | Gossan Index | High values indicate oxidized sulfide zones |
| Quartz | SWIR features | Quartz Index | High values indicate siliceous rocks |
| Carbonates | SWIR features | Carbonate Index | High values indicate calcite/dolomite |

### Spectral Index Formulas

```python
# Iron Oxide: B04 / B02
iron = data.sel(band='B04') / data.sel(band='B02')

# Clay Ratio: B11 / B12
clay = data.sel(band='B11') / data.sel(band='B12')

# NDVI: (B08 - B04) / (B08 + B04)
ndvi = (data.sel(band='B08') - data.sel(band='B04')) / \
       (data.sel(band='B08') + data.sel(band='B04'))

# Hydroxyl: (B11 - B12) / (B11 + B12)
hydroxyl = (data.sel(band='B11') - data.sel(band='B12')) / \
           (data.sel(band='B11') + data.sel(band='B12'))
```

---

## Change Detection

### Methods

| Method | Description | Best For |
|--------|-------------|----------|
| Vegetation Change | NDVI-based change detection | Deforestation, land clearing |
| Simple Difference | Pixel-by-pixel subtraction | General changes |
| PCA Change | Principal component analysis | Subtle changes |
| K-Means Change | Clustering-based detection | Class transitions |

### Interpreting Results

```python
# Get change statistics
stats = change_result.statistics

# Changed pixels
changed = stats['total_changed_pixels']

# Percentage of area affected
percentage = stats['change_percentage']

# Mean intensity of changes
intensity = stats['mean_intensity']

# Maximum change magnitude
max_intensity = stats['max_intensity']
```

---

## Visualization

### Available Plot Functions

```python
from geomin.visualization.static import (
    plot_mineral_index,       # Plot spectral index map
    plot_rgb_composite,       # Create RGB color composite
    plot_change_map,          # Visualize change detection
    create_mineral_map,       # Combined mineral detection map
    plot_spectral_profile,    # Plot spectral signatures
    create_comparison_plot,   # Side-by-side comparison
)

# Usage examples
plot_mineral_index(data, 'Iron Oxide', cmap='Reds', show=True)
plot_rgb_composite(data, bands=('B04', 'B03', 'B02'), show=True)
plot_change_map(change_result, background=data, show=True)
```

### Color Maps

Recommended color maps for mineral visualization:

| Index | Recommended Colormap |
|-------|---------------------|
| Iron Oxide | 'Reds', 'Oranges', 'magma' |
| Clay Ratio | 'Blues', 'viridis' |
| NDVI | 'RdYlGn' |
| Change Detection | 'RdYlGn_r', 'jet' |
| General | 'viridis', 'plasma' |

---

## Troubleshooting

### Common Issues

#### Authentication Errors

```
Error: 401 Unauthorized
```

**Solution**: Verify your API credentials are correct and haven't expired.

```python
# Check credentials
from geomin.core.config import get_config
config = get_config()
print(config.api.earthdata_username)  # Verify username is set
```

#### Memory Errors

```
Error: MemoryError when processing large scenes
```

**Solution**: Use dask chunking and process data in tiles.

```python
from geomin.core.config import get_config
config = get_config()
config.processing.dask_chunks = {'x': 256, 'y': 256}
```

#### CRS Errors

```
Error: CRS not defined
```

**Solution**: Set CRS explicitly.

```python
data = data.rio.write_crs('EPSG:4326')
```

#### Download Failures

```
Error: Download timeout or connection error
```

**Solution**: Check internet connection and retry with smaller chunks.

```python
# Retry download
for attempt in range(3):
    try:
        files = client.download(result)
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
```

### Getting Help

1. Check the documentation in `DOCUMENTATION.md`
2. Review example scripts in `geomin/examples/`
3. Run tests with `pytest tests/`
4. Open an issue on GitHub

---

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/kazashim/GeoMin.git
cd geomin
pip install -e .[dev,interactive]
pytest tests/ --cov=geomin --cov-report=html
```

---

## License

GeoMin is distributed under the MIT License. See `LICENSE` file for details.

---

## Citation

If you use GeoMin in your research, please cite:

```
GeoMin Development Team. (2024). GeoMin: Geophysics Library for 
Satellite-Based Mining Detection. Version 0.1.0.
https://github.com/kazashim/GeoMin
```

---

## Contact

- **GitHub**: https://github.com/kazashim/GeoMin
- **Documentation**: See `DOCUMENTATION.md`
- **Issues**: https://github.com/kazashim/GeoMin/issues
