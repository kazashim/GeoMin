"""
Base satellite client for GeoMin.
Defines the interface for satellite data providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import json

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, Polygon

from ..core.config import get_config
from ..core.crs import transform_bbox, get_utm_zone_from_bbox


class CloudCoverFilter(Enum):
    """Cloud cover filtering options."""
    NONE = 0
    LOW = 10
    MEDIUM = 20
    HIGH = 40
    ANY = 100


@dataclass
class SearchResult:
    """
    Represents a search result from satellite data provider.
    
    Attributes:
        scene_id: Unique identifier for the scene
        provider: Satellite data provider name
        acquisition_time: When the image was captured
        cloud_cover: Percentage of cloud cover (0-100)
        geometry: Footprint of the scene
        bands: Available spectral bands
        resolution: Ground resolution in meters
        data_size: Size of data in megabytes
        preview_url: URL for preview image
        download_url: URL for downloading data
        metadata: Additional metadata
    """
    scene_id: str
    provider: str
    acquisition_time: datetime
    cloud_cover: float
    geometry: Polygon
    bands: List[str] = field(default_factory=list)
    resolution: Optional[float] = None
    data_size: Optional[float] = None
    preview_url: Optional[str] = None
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'scene_id': self.scene_id,
            'provider': self.provider,
            'acquisition_time': self.acquisition_time.isoformat(),
            'cloud_cover': self.cloud_cover,
            'geometry': self.geometry.__geo_interface__,
            'bands': self.bands,
            'resolution': self.resolution,
            'data_size': self.data_size,
            'preview_url': self.preview_url,
            'download_url': self.download_url,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create from dictionary."""
        from shapely.geometry import shape
        return cls(
            scene_id=data['scene_id'],
            provider=data['provider'],
            acquisition_time=datetime.fromisoformat(data['acquisition_time']),
            cloud_cover=data['cloud_cover'],
            geometry=shape(data['geometry']),
            bands=data.get('bands', []),
            resolution=data.get('resolution'),
            data_size=data.get('data_size'),
            preview_url=data.get('preview_url'),
            download_url=data.get('download_url'),
            metadata=data.get('metadata', {}),
        )
    
    def to_gdf(self) -> gpd.GeoDataFrame:
        """Convert to GeoDataFrame."""
        return gpd.GeoDataFrame(
            [self.to_dict()],
            geometry=[self.geometry],
            crs="EPSG:4326"
        )


@dataclass
class SearchOptions:
    """
    Options for satellite data search.
    
    Attributes:
        bbox: Bounding box (minx, miny, maxx, maxy)
        start_date: Start of time range
        end_date: End of time range
        cloud_cover: Maximum cloud cover percentage
        geometry: Optional custom geometry for search
        bands: Required spectral bands
        min_resolution: Minimum ground resolution
        max_resolution: Maximum ground resolution
        source: Specific source or collection
    """
    bbox: Optional[Tuple[float, float, float, float]] = None
    geometry: Optional[Polygon] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    cloud_cover: float = 20.0
    bands: Optional[List[str]] = None
    min_resolution: Optional[float] = None
    max_resolution: Optional[float] = None
    source: Optional[str] = None
    limit: int = 100
    
    def __post_init__(self):
        """Validate and normalize options."""
        if self.start_date and isinstance(self.start_date, str):
            self.start_date = datetime.fromisoformat(self.start_date)
        if self.end_date and isinstance(self.end_date, str):
            self.end_date = datetime.fromisoformat(self.end_date)
        
        # Ensure bbox or geometry is provided
        if not self.bbox and not self.geometry:
            raise ValueError("Either bbox or geometry must be provided")


class SatClient(ABC):
    """
    Abstract base class for satellite data clients.
    
    Defines the interface that all satellite providers must implement.
    """
    
    PROVIDER_NAME: str = "base"
    DEFAULT_CRS: str = "EPSG:4326"
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize satellite client.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = get_config()
        self._connection_status = False
        
        if config:
            self._apply_config(config)
    
    def _apply_config(self, config: Dict) -> None:
        """Apply configuration from dictionary."""
        for key, value in config.items():
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to satellite data provider.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to provider."""
        pass
    
    @abstractmethod
    def search(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search for satellite imagery matching criteria.
        
        Args:
            options: Search options including area and time range
            
        Returns:
            List of matching scenes
        """
        pass
    
    @abstractmethod
    def download(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Download satellite data for a scene.
        
        Args:
            result: Search result to download
            bands: Specific bands to download
            output_dir: Output directory for files
            
        Returns:
            Dictionary mapping band names to file paths
        """
        pass
    
    @abstractmethod
    def load(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None
    ) -> xr.DataArray:
        """
        Load satellite data directly into xarray.
        
        Args:
            result: Search result to load
            bands: Specific bands to load
            
        Returns:
            DataArray with satellite data
        """
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to provider."""
        return self._connection_status
    
    def _validate_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Validate and normalize bounding box.
        
        Args:
            bbox: (minx, miny, maxx, maxy)
            
        Returns:
            Validated bounding box
        """
        minx, miny, maxx, maxy = bbox
        
        if minx >= maxx:
            raise ValueError("Invalid bbox: minx must be less than maxx")
        if miny >= maxy:
            raise ValueError("Invalid bbox: miny must be less than maxy")
        if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
            raise ValueError("Invalid bbox: longitude out of range")
        if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
            raise ValueError("Invalid bbox: latitude out of range")
        
        return (minx, miny, maxx, maxy)
    
    def _get_geometry_from_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        crs: str = "EPSG:4326"
    ) -> Polygon:
        """
        Create polygon geometry from bounding box.
        
        Args:
            bbox: (minx, miny, maxx, maxy)
            crs: CRS of the bounding box
            
        Returns:
            Shapely polygon
        """
        minx, miny, maxx, maxy = self._validate_bbox(bbox)
        return box(minx, miny, maxx, maxy)
    
    def _get_utm_crs(self, bbox: Tuple[float, float, float, float]) -> str:
        """
        Get UTM CRS for a bounding box.
        
        Args:
            bbox: (minx, miny, maxx, maxy) in WGS84
            
        Returns:
            UTM CRS string
        """
        return get_utm_zone_from_bbox(bbox)
    
    def filter_results(
        self,
        results: List[SearchResult],
        max_cloud_cover: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_resolution: Optional[float] = None,
        required_bands: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Filter search results by criteria.
        
        Args:
            results: List of search results
            max_cloud_cover: Maximum cloud cover percentage
            start_date: Earliest acquisition date
            end_date: Latest acquisition date
            min_resolution: Best (lowest) resolution in meters
            required_bands: Bands that must be available
            
        Returns:
            Filtered list of results
        """
        filtered = results
        
        if max_cloud_cover is not None:
            filtered = [r for r in filtered if r.cloud_cover <= max_cloud_cover]
        
        if start_date is not None:
            filtered = [r for r in filtered if r.acquisition_time >= start_date]
        
        if end_date is not None:
            filtered = [r for r in filtered if r.acquisition_time <= end_date]
        
        if min_resolution is not None:
            filtered = [r for r in filtered if r.resolution is None or r.resolution <= min_resolution]
        
        if required_bands:
            filtered = [
                r for r in filtered
                if all(b in r.bands for b in required_bands)
            ]
        
        return filtered
    
    def sort_results(
        self,
        results: List[SearchResult],
        by: str = 'date',
        ascending: bool = False
    ) -> List[SearchResult]:
        """
        Sort search results by criteria.
        
        Args:
            results: List of search results
            by: Sort key ('date', 'cloud_cover', 'resolution')
            ascending: Sort order
            
        Returns:
            Sorted list of results
        """
        reverse = not ascending
        
        if by == 'date':
            return sorted(results, key=lambda r: r.acquisition_time, reverse=reverse)
        elif by == 'cloud_cover':
            return sorted(results, key=lambda r: r.cloud_cover, reverse=reverse)
        elif by == 'resolution':
            return sorted(results, key=lambda r: r.resolution or float('inf'), reverse=reverse)
        else:
            return results
    
    def get_best_result(
        self,
        results: List[SearchResult],
        criteria: List[str] = ['cloud_cover', 'resolution', 'date']
    ) -> Optional[SearchResult]:
        """
        Get the best result based on multiple criteria.
        
        Args:
            results: List of search results
            criteria: Priority list of criteria to optimize
            
        Returns:
            Best matching result or None
        """
        if not results:
            return None
        
        # Filter out None resolutions
        valid_results = [r for r in results if r.resolution is not None] or results
        
        for criterion in criteria:
            if criterion == 'cloud_cover':
                valid_results = sorted(valid_results, key=lambda r: r.cloud_cover)
            elif criterion == 'resolution':
                valid_results = sorted(valid_results, key=lambda r: r.resolution)
            elif criterion == 'date':
                valid_results = sorted(valid_results, key=lambda r: r.acquisition_time, reverse=True)
        
        return valid_results[0]
    
    def get_coverage(self, results: List[SearchResult]) -> gpd.GeoDataFrame:
        """
        Get combined coverage of search results.
        
        Args:
            results: List of search results
            
        Returns:
            GeoDataFrame with result footprints
        """
        gdf = gpd.GeoDataFrame(
            [r.to_dict() for r in results],
            geometry=[r.geometry for r in results],
            crs="EPSG:4326"
        )
        return gdf
    
    @abstractmethod
    def get_bands(self, scene_id: str) -> Dict[str, Any]:
        """
        Get available bands for a scene.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary of band information
        """
        pass
    
    @abstractmethod
    def get_scene_info(self, scene_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific scene.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene metadata dictionary
        """
        pass
