"""
Configuration management for GeoMin library.
Handles API keys, cache paths, and global settings.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class APIConfig:
    """Configuration for satellite API access."""
    
    # USGS/NASA EarthData credentials for Landsat
    earthdata_username: Optional[str] = None
    earthdata_password: Optional[str] = None
    
    # Copernicus Open Access Hub credentials
    copernicus_username: Optional[str] = None
    copernicus_password: Optional[str] = None
    
    # Commercial API keys
    planet_api_key: Optional[str] = None
    maxar_api_key: Optional[str] = None
    
    # Sentinel Hub credentials
    sentinelhub_client_id: Optional[str] = None
    sentinelhub_client_secret: Optional[str] = None


@dataclass
class CacheConfig:
    """Configuration for data caching."""
    
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".geomin" / "cache")
    max_cache_size_gb: float = 50.0
    cache_enabled: bool = True
    auto_cleanup: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for data processing settings."""
    
    default_crs: str = "EPSG:4326"
    target_resolution_m: float = 10.0
    max_workers: int = 4
    use_gpu: bool = False
    dask_chunks: Dict[str, int] = field(default_factory=lambda: {"x": 512, "y": 512})
    
    # Memory settings
    memory_limit_gb: float = 8.0
    oob_processing: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    
    default_colormap: str = "viridis"
    figure_dpi: int = 150
    map_tile_provider: str = "OpenStreetMap"
    interactive_map_port: int = 8765


class Config:
    """
    Global configuration manager for GeoMin library.
    
    Manages API credentials, cache settings, and processing parameters.
    Settings can be loaded from environment variables or config files.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to JSON config file
        """
        self.api = APIConfig()
        self.cache = CacheConfig()
        self.processing = ProcessingConfig()
        self.visualization = VisualizationConfig()
        
        # Load settings from environment and config file
        self._load_from_env()
        if config_path and config_path.exists():
            self._load_from_file(config_path)
        
        # Ensure cache directory exists
        self.cache.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # API configurations
        self.api.earthdata_username = os.getenv("GEOMIN_EARTHDATA_USERNAME")
        self.api.earthdata_password = os.getenv("GEOMIN_EARTHDATA_PASSWORD")
        self.api.copernicus_username = os.getenv("GEOMIN_COPERNICUS_USERNAME")
        self.api.copernicus_password = os.getenv("GEOMIN_COPERNICUS_PASSWORD")
        self.api.planet_api_key = os.getenv("GEOMIN_PLANET_API_KEY")
        self.api.maxar_api_key = os.getenv("GEOMIN_MAXAR_API_KEY")
        self.api.sentinelhub_client_id = os.getenv("GEOMIN_SENTINELHUB_CLIENT_ID")
        self.api.sentinelhub_client_secret = os.getenv("GEOMIN_SENTINELHUB_CLIENT_SECRET")
        
        # Cache configuration
        if os.getenv("GEOMIN_CACHE_DIR"):
            self.cache.cache_dir = Path(os.getenv("GEOMIN_CACHE_DIR"))
        if os.getenv("GEOMIN_CACHE_ENABLED"):
            self.cache.cache_enabled = os.getenv("GEOMIN_CACHE_ENABLED").lower() == "true"
        
        # Processing configuration
        if os.getenv("GEOMIN_DEFAULT_CRS"):
            self.processing.default_crs = os.getenv("GEOMIN_DEFAULT_CRS")
        if os.getenv("GEOMIN_USE_GPU"):
            self.processing.use_gpu = os.getenv("GEOMIN_USE_GPU").lower() == "true"
        if os.getenv("GEOMIN_MAX_WORKERS"):
            self.processing.max_workers = int(os.getenv("GEOMIN_MAX_WORKERS"))
    
    def _load_from_file(self, config_path: Path):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update API settings
            if "api" in config_data:
                api_data = config_data["api"]
                for key, value in api_data.items():
                    if hasattr(self.api, key):
                        setattr(self.api, key, value)
            
            # Update cache settings
            if "cache" in config_data:
                cache_data = config_data["cache"]
                for key, value in cache_data.items():
                    if hasattr(self.cache, key):
                        if key == "cache_dir":
                            value = Path(value)
                        setattr(self.cache, key, value)
            
            # Update processing settings
            if "processing" in config_data:
                proc_data = config_data["processing"]
                for key, value in proc_data.items():
                    if hasattr(self.processing, key):
                        setattr(self.processing, key, value)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    def save(self, config_path: Path):
        """
        Save current configuration to JSON file.
        
        Args:
            config_path: Path to save configuration
        """
        config_data = {
            "api": {
                "earthdata_username": self.api.earthdata_username,
                "copernicus_username": self.api.copernicus_username,
                "planet_api_key": self.api.planet_api_key,
                "maxar_api_key": self.api.maxar_api_key,
            },
            "cache": {
                "cache_dir": str(self.cache.cache_dir),
                "max_cache_size_gb": self.cache.max_cache_size_gb,
                "cache_enabled": self.cache.cache_enabled,
            },
            "processing": {
                "default_crs": self.processing.default_crs,
                "target_resolution_m": self.processing.target_resolution_m,
                "max_workers": self.processing.max_workers,
                "use_gpu": self.processing.use_gpu,
            },
            "visualization": {
                "default_colormap": self.visualization.default_colormap,
                "figure_dpi": self.visualization.figure_dpi,
            },
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_cache_path(self, scene_id: str, band: str) -> Path:
        """
        Generate cache file path for a scene band.
        
        Args:
            scene_id: Satellite scene identifier
            band: Band name or identifier
            
        Returns:
            Path to cached file
        """
        return self.cache.cache_dir / f"{scene_id}_{band}.tif"
    
    def is_api_configured(self, provider: str) -> bool:
        """
        Check if API credentials are configured for a provider.
        
        Args:
            provider: Satellite data provider name
            
        Returns:
            True if credentials are configured
        """
        if provider.lower() in ["landsat", "usgs", "earthdata"]:
            return bool(self.api.earthdata_username and self.api.earthdata_password)
        elif provider.lower() in ["sentinel", "copernicus"]:
            return bool(self.api.copernicus_username and self.api.copernicus_password)
        elif provider.lower() == "planet":
            return bool(self.api.planet_api_key)
        elif provider.lower() == "maxar":
            return bool(self.api.maxar_api_key)
        elif provider.lower() == "sentinelhub":
            return bool(self.api.sentinelhub_client_id and self.api.sentinelhub_client_secret)
        return False


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """
    Get global configuration instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config


def reset_config():
    """Reset global configuration instance."""
    global _config
    _config = None
