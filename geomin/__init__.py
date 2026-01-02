"""
GeoMin: Geophysics Library for Satellite-Based Mining Detection

A comprehensive Python library for satellite-based mining activity detection 
and mineral identification using spectral analysis and machine learning.

Author: Kazashim Kuzasuwat
GitHub: https://github.com/kazashim/GeoMin
"""

__version__ = "0.1.0"
__author__ = "Kazashim Kuzasuwat"

from .core.config import Config, get_config
from .core.data_loader import DataLoader
from .core.crs import (
    transform_bbox, 
    get_utm_zone, 
    get_utm_zone_from_bbox,
    reproject_raster,
    is_valid_crs,
    get_crs_info
)
from .satellites.base_client import SatClient, SearchResult, SearchOptions
from .satellites.landsat import LandsatClient
from .satellites.sentinel import SentinelClient
from .satellites.commercial import PlanetClient, MaxarClient
from .algorithms import spectral
from .algorithms import terrain
from .models import change_detection
from .visualization import static

__all__ = [
    # Version
    "__version__",
    
    # Core
    "Config",
    "get_config",
    "DataLoader",
    "transform_bbox",
    "get_utm_zone",
    "get_utm_zone_from_bbox",
    "reproject_raster",
    "is_valid_crs",
    "get_crs_info",
    
    # Satellites
    "SatClient",
    "SearchResult",
    "SearchOptions",
    "LandsatClient",
    "SentinelClient",
    "PlanetClient",
    "MaxarClient",
    
    # Algorithms
    "spectral",
    "terrain",
    
    # Models
    "change_detection",
    
    # Visualization
    "static",
]
