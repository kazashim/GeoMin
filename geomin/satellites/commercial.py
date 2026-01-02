"""
Commercial satellite data clients for GeoMin.
Provides interfaces to Planet Labs and Maxar for high-resolution imagery.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time

import requests
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon

from .base_client import SatClient, SearchResult, SearchOptions
from ..core.config import get_config
from ..core.data_loader import DataLoader


class PlanetClient(SatClient):
    """
    Client for Planet Labs satellite data.
    
    Provides access to Planet's constellation of Dove, SkySat and Satarie satellites:
    - PlanetScope (Dove): 3m resolution, daily revisit
    - SkySat: 0.5m resolution, frequent revisit
    
    Requires Planet API key for access.
    """
    
    PROVIDER_NAME = "planet"
    DEFAULT_CRS = "EPSG:4326"
    
    # Planet API endpoints
    PLANET_API_BASE = "https://api.planet.com"
    PLANET_DATA_API = f"{PLANET_API_BASE}/data/v1"
    PLANET_ASSETS_API = f"{PLANET_API_BASE}/basemaps/v1"
    
    # PlanetScope band definitions
    PS_BANDS = ['B1', 'B2', 'B3', 'B4']  # Blue, Green, Red, NIR (4-band)
    PS_BANDS_3BAND = ['B1', 'B2', 'B3']  # Blue, Green, Red (3-band)
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Planet client."""
        super().__init__(config)
        self._api_key = self.config.api.planet_api_key
        self.session = None
    
    def connect(self) -> bool:
        """Connect to Planet API."""
        if not self._api_key:
            print("Warning: Planet API key not configured")
            return False
        
        self.session = requests.Session()
        self.session.auth = ("", self._api_key)
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Test connection
        try:
            response = self.session.get(
                f"{self.PLANET_DATA_API}/quick-search",
                params={'_page_size': 1},
                timeout=30
            )
            response.raise_for_status()
            self._connection_status = True
            return True
        except Exception as e:
            print(f"Planet connection failed: {e}")
            self._connection_status = False
            return False
    
    def disconnect(self) -> None:
        """Close connection to Planet API."""
        if self.session:
            self.session.close()
        self._connection_status = False
    
    def search(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search Planet imagery.
        
        Args:
            options: Search options including area and time range
            
        Returns:
            List of matching Planet scenes
        """
        if not self.session:
            self.connect()
        
        # Build Planet query
        query = self._build_query(options)
        
        params = {
            'item_types': ['PSScene', 'SkySatCollect'],
            'search_request': json.dumps({'filter': query, 'sort': {'field': 'acquired', 'direction': 'descending'}}),
            '_page_size': options.limit,
        }
        
        try:
            response = self.session.get(
                f"{self.PLANET_DATA_API}/quick-search",
                params=params,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('features', []):
                properties = item.get('properties', {})
                geometry = self._parse_geojson_geometry(item.get('geometry'))
                
                # Parse acquisition time
                acquired = properties.get('acquired')
                if isinstance(acquired, str):
                    acquisition_time = datetime.fromisoformat(acquired.replace('Z', '+00:00'))
                else:
                    acquisition_time = datetime.now()
                
                # Determine satellite
                instrument = properties.get('instrument', '')
                satellite = 'SkySat' if 'skysat' in instrument.lower() else 'PlanetScope'
                
                # Get cloud cover
                cloud_cover = properties.get('cloud_cover', properties.get('percent_cloud_cover', 0))
                
                # Get resolution
                resolution = 0.5 if satellite == 'SkySat' else 3.0
                
                result = SearchResult(
                    scene_id=item.get('id', ''),
                    provider=self.PROVIDER_NAME,
                    acquisition_time=acquisition_time,
                    cloud_cover=cloud_cover,
                    geometry=geometry,
                    bands=self.PS_BANDS if properties.get('strip_id') else self.PS_BANDS_3BAND,
                    resolution=resolution,
                    metadata={
                        'satellite': satellite,
                        'instrument': instrument,
                        'strip_id': properties.get('strip_id'),
                        'quality_category': properties.get('quality_category'),
                        'sun_azimuth': properties.get('sun_azimuth'),
                        'sun_elevation': properties.get('sun_elevation'),
                        'view_angle': properties.get('view_angle'),
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Planet search failed: {e}")
            return []
    
    def _build_query(self, options: SearchOptions) -> Dict:
        """Build Planet API query filter."""
        filters = []
        
        # Date filter
        date_filter = {
            "type": "DateRangeFilter",
            "field_name": "acquired",
            "config": {}
        }
        if options.start_date:
            date_filter["config"]["gte"] = options.start_date.isoformat()
        if options.end_date:
            date_filter["config"]["lte"] = options.end_date.isoformat()
        if date_filter["config"]:
            filters.append(date_filter)
        
        # Cloud cover filter
        if options.cloud_cover < 100:
            filters.append({
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {"lte": options.cloud_cover / 100}
            })
        
        # Geometry filter
        if options.bbox:
            minx, miny, maxx, maxy = options.bbox
            filters.append({
                "type": "GeoIntersectionFilter",
                "field_name": "geometry",
                "config": {
                    "type": "Polygon",
                    "coordinates": [[
                        [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
                    ]]
                }
            })
        
        # Combine filters
        if len(filters) == 1:
            return filters[0]
        elif len(filters) > 1:
            return {
                "type": "AndFilter",
                "config": filters
            }
        else:
            return {"type": "Filter", "config": {}}
    
    def _parse_geojson_geometry(self, geometry: Dict) -> Polygon:
        """Parse GeoJSON geometry."""
        if not geometry:
            return box(-180, -90, 180, 90)
        
        try:
            coords = geometry.get('coordinates', [])
            if geometry.get('type') == 'Polygon':
                coords = coords[0]
                return Polygon(coords)
            elif geometry.get('type') == 'MultiPolygon':
                from shapely.ops import unary_union
                polys = [Polygon(p[0]) for p in coords]
                return unary_union(polys)
        except Exception:
            pass
        
        return box(-180, -90, 180, 90)
    
    def download(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Download Planet imagery.
        
        Args:
            result: Search result to download
            bands: Specific bands to download
            output_dir: Output directory
            
        Returns:
            Dictionary mapping band names to file paths
        """
        if output_dir is None:
            output_dir = self.config.cache.cache_dir / "planet"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = {}
        
        # Planet provides direct download links
        # This is a simplified implementation
        print(f"Planet download requires activation for {result.scene_id}")
        
        return downloaded
    
    def load(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None
    ) -> xr.DataArray:
        """
        Load Planet data directly into xarray.
        
        Args:
            result: Search result to load
            bands: Specific bands to load
            
        Returns:
            DataArray with Planet data
        """
        files = self.download(result, bands)
        
        if not files:
            raise ValueError("No files downloaded")
        
        # Load first band
        first_band = list(files.keys())[0]
        data = DataLoader.load(files[first_band])
        
        # Stack if multiple bands
        if len(files) > 1:
            bands_data = [data]
            for band_name, filepath in list(files.items())[1:]:
                bands_data.append(DataLoader.load(filepath))
            
            data = xr.concat(bands_data, dim='band')
            data = data.assign_coords({'band': list(files.keys())})
        
        return data
    
    def get_bands(self, scene_id: str) -> Dict[str, Any]:
        """
        Get available bands for Planet imagery.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary of band information
        """
        return {
            'B1': {'name': 'Blue', 'wavelength': 0.47, 'resolution': 3.0},
            'B2': {'name': 'Green', 'wavelength': 0.56, 'resolution': 3.0},
            'B3': {'name': 'Red', 'wavelength': 0.66, 'resolution': 3.0},
            'B4': {'name': 'NIR', 'wavelength': 0.85, 'resolution': 3.0},
        }
    
    def get_scene_info(self, scene_id: str) -> Dict[str, Any]:
        """Get metadata for a Planet scene."""
        return {
            'scene_id': scene_id,
            'provider': 'Planet Labs',
            'resolution': 3.0,
            'swath_width': 24,
            'revisit_time': 'Daily at equator',
        }


class MaxarClient(SatClient):
    """
    Client for Maxar satellite data.
    
    Provides access to Maxar's high-resolution constellation:
    - WorldView: 0.3m resolution
    - GeoEye: 0.5m resolution
    - SkySat: 0.5m resolution (via Maxar acquisition)
    
    Requires API credentials for access.
    """
    
    PROVIDER_NAME = "maxar"
    DEFAULT_CRS = "EPSG:4326"
    
    # Maxar API endpoints
    MAXAR_API_BASE = "https://api.maxar.com"
    MAXAR_CATALOG_API = f"{MAXAR_API_BASE}/catalog"
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Maxar client."""
        super().__init__(config)
        self._api_key = self.config.api.maxar_api_key
        self.session = None
    
    def connect(self) -> bool:
        """Connect to Maxar API."""
        if not self._api_key:
            print("Warning: Maxar API key not configured")
            return False
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        })
        
        # Test connection
        try:
            response = self.session.get(
                f"{self.MAXAR_CATALOG_API}/stac",
                timeout=30
            )
            response.raise_for_status()
            self._connection_status = True
            return True
        except Exception as e:
            print(f"Maxar connection failed: {e}")
            self._connection_status = False
            return False
    
    def disconnect(self) -> None:
        """Close connection to Maxar API."""
        if self.session:
            self.session.close()
        self._connection_status = False
    
    def search(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search Maxar imagery.
        
        Args:
            options: Search options including area and time range
            
        Returns:
            List of matching Maxar scenes
        """
        if not self.session:
            self.connect()
        
        # Build STAC query
        query = {
            "collections": ["maxar"],
            "limit": options.limit,
        }
        
        # Add temporal filter
        if options.start_date or options.end_date:
            datetime_range = "/".join([
                options.start_date.isoformat() if options.start_date else "1900-01-01",
                options.end_date.isoformat() if options.end_date else datetime.now().isoformat()
            ])
            query["datetime"] = datetime_range
        
        # Add spatial filter
        if options.bbox:
            minx, miny, maxx, maxy = options.bbox
            query["bbox"] = [minx, miny, maxx, maxy]
        
        # Add cloud filter
        if options.cloud_cover < 100:
            query["query"] = {
                "eo:cloud_cover": {"lte": options.cloud_cover}
            }
        
        try:
            response = self.session.post(
                f"{self.MAXAR_CATALOG_API}/search",
                json=query,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                geometry = self._parse_geojson_geometry(feature.get('geometry'))
                
                # Parse acquisition time
                acquired = properties.get('datetime', properties.get('acquired'))
                if isinstance(acquired, str):
                    acquisition_time = datetime.fromisoformat(acquired.split('T')[0])
                else:
                    acquisition_time = datetime.now()
                
                # Get cloud cover
                cloud_cover = properties.get('eo:cloud_cover', 0)
                
                # Determine resolution from platform
                platform = properties.get('platform', '').lower()
                if 'worldview' in platform:
                    resolution = 0.3
                elif 'geoeye' in platform:
                    resolution = 0.5
                else:
                    resolution = 0.5
                
                result = SearchResult(
                    scene_id=feature.get('id', ''),
                    provider=self.PROVIDER_NAME,
                    acquisition_time=acquisition_time,
                    cloud_cover=cloud_cover,
                    geometry=geometry,
                    bands=['Pan', 'Blue', 'Green', 'Red', 'NIR'],
                    resolution=resolution,
                    metadata={
                        'platform': properties.get('platform'),
                        'instrument': properties.get('instrument'),
                        'off_nadir': properties.get('view:off_nadir'),
                        'azimuth': properties.get('view:sun_azimuth'),
                        'elevation': properties.get('view:sun_elevation'),
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Maxar search failed: {e}")
            return []
    
    def _parse_geojson_geometry(self, geometry: Dict) -> Polygon:
        """Parse GeoJSON geometry."""
        if not geometry:
            return box(-180, -90, 180, 90)
        
        try:
            coords = geometry.get('coordinates', [])
            if geometry.get('type') == 'Polygon':
                coords = coords[0]
                return Polygon(coords)
        except Exception:
            pass
        
        return box(-180, -90, 180, 90)
    
    def download(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Download Maxar imagery.
        
        Args:
            result: Search result to download
            bands: Specific bands to download
            output_dir: Output directory
            
        Returns:
            Dictionary mapping band names to file paths
        """
        if output_dir is None:
            output_dir = self.config.cache.cache_dir / "maxar"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Maxar download requires order placement
        print(f"Maxar download requires order placement for {result.scene_id}")
        return {}
    
    def load(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None
    ) -> xr.DataArray:
        """
        Load Maxar data directly into xarray.
        
        Args:
            result: Search result to load
            bands: Specific bands to load
            
        Returns:
            DataArray with Maxar data
        """
        files = self.download(result, bands)
        
        if not files:
            raise ValueError("No files downloaded")
        
        return DataLoader.load(list(files.values())[0])
    
    def get_bands(self, scene_id: str) -> Dict[str, Any]:
        """
        Get available bands for Maxar imagery.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary of band information
        """
        return {
            'PAN': {'name': 'Panchromatic', 'wavelength': 0.45, 'resolution': 0.3},
            'Blue': {'name': 'Blue', 'wavelength': 0.48, 'resolution': 1.2},
            'Green': {'name': 'Green', 'wavelength': 0.55, 'resolution': 1.2},
            'Red': {'name': 'Red', 'wavelength': 0.67, 'resolution': 1.2},
            'NIR': {'name': 'Near Infrared', 'wavelength': 0.85, 'resolution': 1.2},
            'RedEdge': {'name': 'Red Edge', 'wavelength': 0.73, 'resolution': 1.2},
        }
    
    def get_scene_info(self, scene_id: str) -> Dict[str, Any]:
        """Get metadata for a Maxar scene."""
        return {
            'scene_id': scene_id,
            'provider': 'Maxar',
            'resolution': 0.3,
            'swath_width': 13,
            'revisit_time': 'Multiple daily at mid-latitudes',
        }
