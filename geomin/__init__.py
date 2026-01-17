"""
GeoMin: Geophysics Library for Satellite-Based Mining Detection

A comprehensive Python library for satellite-based mining activity detection 
and mineral identification using spectral analysis and machine learning.

Author: Kazashim Kuzasuwat
GitHub: https://github.com/kazashim/GeoMin
"""

__version__ = "0.2.0"
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
from .core.cloud_masking import CloudMasker
from .core.export import GeoExporter
from .satellites.base_client import SatClient, SearchResult, SearchOptions
from .satellites.stac_client import STACClient
from .satellites.landsat import LandsatClient
from .satellites.sentinel import SentinelClient
from .satellites.commercial import PlanetClient, MaxarClient
from .algorithms import spectral
from .algorithms import terrain
from .algorithms import advanced_mineralogy
from .models import change_detection
from .models import anomaly_detection
from .models import deep_learning
from .visualization import static
from .visualization import interactive

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
    "CloudMasker",
    "GeoExporter",
    
    # Satellites
    "SatClient",
    "SearchResult",
    "SearchOptions",
    "STACClient",
    "LandsatClient",
    "SentinelClient",
    "PlanetClient",
    "MaxarClient",
    
    # Algorithms
    "spectral",
    "terrain",
    "advanced_mineralogy",
    
    # Models
    "change_detection",
    "anomaly_detection",
    "deep_learning",
    
    # Visualization
    "static",
    "interactive",
]
