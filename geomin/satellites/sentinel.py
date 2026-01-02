"""
Sentinel satellite data client for GeoMin.
Connects to Copernicus Open Access Hub and Sentinel Hub for Sentinel-2 data.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time
import hashlib

import requests
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon

from .base_client import SatClient, SearchResult, SearchOptions
from ..core.config import get_config
from ..core.data_loader import DataLoader, SENTINEL2_BANDS
from ..core.crs import detect_crs


class SentinelClient(SatClient):
    """
    Client for Sentinel-2 satellite data.
    
    Supports access to:
    - Sentinel-2A (operational since June 2015)
    - Sentinel-2B (operational since May 2017)
    
    Data sources:
    - Copernicus Open Access Hub (free, requires registration)
    - Sentinel Hub (commercial, with free tier)
    
    Sentinel-2 provides 13 spectral bands at different resolutions:
    - 10m: Blue, Green, Red, NIR
    - 20m: Red Edge 1-4, NIR, SWIR 1-2
    - 60m: Coastal Aerosol, Water Vapor, Cirrus
    """
    
    PROVIDER_NAME = "sentinel"
    DEFAULT_CRS = "EPSG:4326"
    
    # Copernicus Open Access Hub endpoints
    COPERNICUS_API_URL = "https://catalogue.dataspace.copernicus.eu/resto/api"
    COPERNICUS_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/"
    
    # Sentinel Hub endpoints (for commercial API)
    SENTINELHUB_BASE_URL = "https://services.sentinel-hub.com"
    
    # Band definitions for Sentinel-2
    BANDS_10M = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
    BANDS_20M = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']  # Red Edge, NIR, SWIR
    BANDS_60M = ['B01', 'B09', 'B10']  # Coastal, Water Vapor, Cirrus
    
    def __init__(
        self,
        use_sentinel_hub: bool = False,
        config: Optional[Dict] = None
    ):
        """
        Initialize Sentinel client.
        
        Args:
            use_sentinel_hub: Use Sentinel Hub API (requires auth) vs Copernicus (free)
            config: Optional configuration
        """
        super().__init__(config)
        self.use_sentinel_hub = use_sentinel_hub
        self.session = None
        self._access_token = None
        
        # Check for Sentinel Hub credentials
        if not (self.config.api.sentinelhub_client_id and self.config.api.sentinelhub_client_secret):
            self.use_sentinel_hub = False
    
    def connect(self) -> bool:
        """Connect to Sentinel data provider."""
        if self.use_sentinel_hub:
            return self._connect_sentinel_hub()
        else:
            return self._connect_copernicus()
    
    def _connect_copernicus(self) -> bool:
        """Connect to Copernicus Open Access Hub."""
        if not self.config.api.copernicus_username or not self.config.api.copernicus_password:
            print("Warning: Copernicus credentials not configured. Using public endpoint.")
            self._connection_status = True
            return True
        
        self.session = requests.Session()
        self.session.auth = (
            self.config.api.copernicus_username,
            self.config.api.copernicus_password
        )
        
        # Test connection
        try:
            response = self.session.get(
                f"{self.COPERNICUS_API_URL}/collections.json",
                timeout=30
            )
            response.raise_for_status()
            self._connection_status = True
            return True
        except Exception as e:
            print(f"Copernicus connection failed: {e}")
            self._connection_status = False
            return False
    
    def _connect_sentinel_hub(self) -> bool:
        """Connect to Sentinel Hub API."""
        if not self.config.api.sentinelhub_client_id or not self.config.api.sentinelhub_client_secret:
            print("Warning: Sentinel Hub credentials not configured. Using Copernicus.")
            self.use_sentinel_hub = False
            return True
        
        try:
            # OAuth token request
            token_url = f"{self.SENTINELHUB_BASE_URL}/oauth/token"
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.config.api.sentinelhub_client_id,
                'client_secret': self.config.api.sentinelhub_client_secret,
            }
            
            response = requests.post(token_url, data=data, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data.get('access_token')
            
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self._access_token}'
            })
            
            self._connection_status = True
            return True
            
        except Exception as e:
            print(f"Sentinel Hub connection failed: {e}")
            self._connection_status = False
            return False
    
    def disconnect(self) -> None:
        """Close connection to provider."""
        if self.session:
            self.session.close()
        self._access_token = None
        self._connection_status = False
    
    def search(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search for Sentinel-2 imagery.
        
        Args:
            options: Search options including area and time range
            
        Returns:
            List of matching Sentinel-2 scenes
        """
        if self.use_sentinel_hub:
            return self._search_sentinel_hub(options)
        else:
            return self._search_copernicus(options)
    
    def _search_copernicus(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search Copernicus Open Access Hub.
        
        Uses the RESTO API for querying Sentinel-2 data.
        """
        # Build query parameters
        params = {
            'collection': 'Sentinel-2',
            'maxRecords': options.limit,
            'cloudCover': f"[0,{options.cloud_cover}]",
        }
        
        # Add temporal filter
        if options.start_date:
            params['startDate'] = options.start_date.strftime('%Y-%m-%d')
        if options.end_date:
            params['completionDate'] = options.end_date.strftime('%Y-%m-%d')
        
        # Add spatial filter
        if options.bbox:
            minx, miny, maxx, maxy = options.bbox
            params['geometry'] = f"{minx},{miny},{maxx},{maxy}"
        
        try:
            # Make search request
            response = self.session.get(
                f"{self.COPERNICUS_API_URL}/search.json",
                params=params,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                
                # Parse geometry
                geometry = self._parse_geojson_geometry(feature.get('geometry'))
                
                # Parse acquisition time
                acquisition_time = properties.get('acquisitionDate')
                if isinstance(acquisition_time, str):
                    acquisition_time = datetime.fromisoformat(acquisition_time.split('T')[0])
                
                # Get orbit number
                orbit_number = properties.get('relativeOrbitNumber')
                
                result = SearchResult(
                    scene_id=properties.get('id', feature.get('id', '')),
                    provider=self.PROVIDER_NAME,
                    acquisition_time=acquisition_time,
                    cloud_cover=properties.get('cloudCover', 0),
                    geometry=geometry,
                    bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
                           'B8A', 'B09', 'B10', 'B11', 'B12'],
                    resolution=10,  # Native resolution of visible bands
                    data_size=properties.get('size'),
                    preview_url=properties.get('quicklook'),
                    download_url=properties.get('download'),
                    metadata={
                        'satellite': properties.get('satellite'),
                        'productType': properties.get('productType'),
                        'orbit': orbit_number,
                        'tileId': properties.get('tileId'),
                        'processingBaseline': properties.get('processingBaseline'),
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Copernicus search failed: {e}")
            return []
    
    def _search_sentinel_hub(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search Sentinel Hub API.
        
        Provides faster and more flexible searching than Copernicus.
        """
        # Build search request for Sentinel Hub Catalog API
        search_request = {
            "collections": ["sentinel-2-l2a"],
            "datetime": self._format_datetime_range(options.start_date, options.end_date),
            "limit": options.limit,
            "query": {
                "eo:cloud_cover": {"lte": options.cloud_cover},
            }
        }
        
        # Add spatial filter
        if options.bbox:
            minx, miny, maxx, maxy = options.bbox
            search_request["bbox"] = [minx, miny, maxx, maxy]
        
        try:
            catalog_url = f"{self.SENTINELHUB_BASE_URL}/catalog/v1/search"
            
            response = self.session.post(
                catalog_url,
                json=search_request,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for feature in data.get('features', []):
                properties = feature.get('properties', {})
                geometry = self._parse_geojson_geometry(feature.get('geometry'))
                
                # Parse datetime
                datetime_str = properties.get('datetime')
                if isinstance(datetime_str, str):
                    acquisition_time = datetime.fromisoformat(datetime_str.split('T')[0])
                else:
                    acquisition_time = datetime.now()
                
                # Extract scene ID from links
                scene_id = feature.get('id', '')
                
                # Get cloud cover
                cloud_cover = properties.get('eo:cloud_cover', 0)
                
                result = SearchResult(
                    scene_id=scene_id,
                    provider=self.PROVIDER_NAME,
                    acquisition_time=acquisition_time,
                    cloud_cover=cloud_cover,
                    geometry=geometry,
                    bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
                           'B8A', 'B09', 'B10', 'B11', 'B12'],
                    resolution=10,
                    metadata={
                        'links': feature.get('links', []),
                        'assets': list(feature.get('assets', {}).keys()),
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Sentinel Hub search failed: {e}")
            return []
    
    def _format_datetime_range(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> str:
        """Format datetime range for API request."""
        start = start_date.strftime('%Y-%m-%d') if start_date else '1900-01-01'
        end = end_date.strftime('%Y-%m-%d') if end_date else datetime.now().strftime('%Y-%m-%d')
        return f"{start}/{end}"
    
    def _parse_geojson_geometry(self, geometry: Dict) -> Polygon:
        """Parse GeoJSON geometry to Shapely polygon."""
        if not geometry:
            return box(-180, -90, 180, 90)
        
        try:
            import geojson
            geom = geojson.loads(json.dumps(geometry))
            
            if geom.type == 'Polygon':
                coords = list(geom.coordinates[0])
                return Polygon(coords)
            elif geom.type == 'MultiPolygon':
                from shapely.ops import unary_union
                polys = [Polygon(p[0]) for p in geom.coordinates]
                return unary_union(polys)
            else:
                return box(-180, -90, 180, 90)
                
        except Exception:
            return box(-180, -90, 180, 90)
    
    def download(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Download Sentinel-2 data for a scene.
        
        Args:
            result: Search result to download
            bands: Specific bands to download
            output_dir: Output directory
            
        Returns:
            Dictionary mapping band names to file paths
        """
        if output_dir is None:
            output_dir = self.config.cache.cache_dir / "sentinel"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.use_sentinel_hub:
            return self._download_sentinel_hub(result, bands, output_dir)
        else:
            return self._download_copernicus(result, bands, output_dir)
    
    def _download_copernicus(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Path = None
    ) -> Dict[str, Path]:
        """
        Download from Copernicus Open Access Hub.
        
        Returns presigned URLs for download.
        """
        downloaded = {}
        
        if bands is None:
            bands = self.BANDS_10M + self.BANDS_20M + self.BANDS_60M
        
        # Get download URL from metadata
        download_base = result.download_url
        if not download_base:
            print(f"No download URL available for {result.scene_id}")
            return downloaded
        
        # Create manifest URL
        manifest_url = f"{download_base}/manifest.safe"
        
        try:
            # Download manifest and extract band URLs
            response = self.session.get(manifest_url, timeout=60)
            response.raise_for_status()
            
            # Parse manifest (simplified - would need proper XML parsing)
            # For now, construct expected URLs
            for band in bands:
                if band in ['B01', 'B09', 'B10']:
                    resolution = 'R60m'
                elif band in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']:
                    resolution = 'R20m'
                else:
                    resolution = 'R10m'
                
                filename = f"{result.scene_id}_{band}_{resolution}.jp2"
                band_url = f"{download_base}/{filename}"
                output_path = output_dir / filename
                
                if not output_path.exists():
                    print(f"Downloading {band}...")
                    response = self.session.get(band_url, timeout=300, stream=True)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                downloaded[band] = output_path
            
        except Exception as e:
            print(f"Download failed: {e}")
        
        return downloaded
    
    def _download_sentinel_hub(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Path = None
    ) -> Dict[str, Path]:
        """
        Download from Sentinel Hub.
        
        Uses the Batch Statistical API or direct download.
        """
        downloaded = {}
        
        if bands is None:
            bands = self.BANDS_10M
        
        # Get download URL from metadata
        for link in result.metadata.get('links', []):
            if link.get('rel') == 'data' or link.get('rel') == 'download':
                # Request signed URL for download
                download_url = f"{self.SENTINELHUB_BASE_URL}/oauth/token"
                # Implementation would use Sentinel Hub's download API
                break
        
        return downloaded
    
    def load(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        stack_bands: bool = True
    ) -> xr.DataArray:
        """
        Load Sentinel-2 data directly into xarray.
        
        Args:
            result: Search result to load
            bands: Specific bands to load
            stack_bands: Whether to stack bands into single DataArray
            
        Returns:
            DataArray with Sentinel-2 data
        """
        if bands is None:
            bands = self.BANDS_10M + self.BANDS_20M
        
        # Download files
        files = self.download(result, bands)
        
        if not files:
            raise ValueError("No files downloaded")
        
        # Sort files by band name
        sorted_files = {k: files[k] for k in sorted(files.keys())}
        
        if stack_bands:
            # Load and stack all bands
            bands_data = []
            for band_name, filepath in sorted_files.items():
                band_data = DataLoader.load(filepath)
                bands_data.append(band_data)
            
            data = xr.concat(bands_data, dim='band')
            data = data.assign_coords({'band': list(sorted_files.keys())})
        else:
            # Return as dictionary
            return {band: DataLoader.load(filepath) for band, filepath in sorted_files.items()}
        
        return data
    
    def load_tile(
        self,
        tile_id: str,
        date: datetime,
        bands: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> xr.DataArray:
        """
        Load Sentinel-2 data by tile ID.
        
        Args:
            tile_id: MGRS tile identifier (e.g., '10SGD')
            date: Acquisition date
            bands: Bands to load
            output_dir: Output directory for downloads
            
        Returns:
            DataArray with tile data
        """
        # Search for this specific tile
        options = SearchOptions(
            bbox=None,  # Will need to convert tile to bbox
            start_date=date,
            end_date=date,
            cloud_cover=100,
            limit=1
        )
        
        # Search results
        results = self.search(options)
        
        # Filter to specific tile
        tile_results = [r for r in results if r.metadata.get('tileId') == tile_id]
        
        if not tile_results:
            raise ValueError(f"No data found for tile {tile_id} on {date}")
        
        return self.load(tile_results[0], bands)
    
    def get_bands(self, scene_id: str) -> Dict[str, Any]:
        """
        Get available bands for Sentinel-2.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary of band information
        """
        return SENTINEL2_BANDS
    
    def get_scene_info(self, scene_id: str) -> Dict[str, Any]:
        """
        Get metadata for a Sentinel-2 scene.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene metadata dictionary
        """
        return {
            'scene_id': scene_id,
            'satellite': 'Sentinel-2',
            'instrument': 'MSI (Multi-Spectral Instrument)',
            'resolution': {
                'B01, B09, B10': 60,
                'B05, B06, B07, B8A, B11, B12': 20,
                'B02, B03, B04, B08': 10,
            },
            'swath_width': 290,
            'revisit_time': '5 days at equator (2-3 days at mid-latitudes)',
            'bands': {
                **self.get_bands(scene_id),
            },
        }
    
    def create_ndvi(self, data: xr.DataArray) -> xr.DataArray:
        """
        Calculate NDVI from Sentinel-2 data.
        
        Args:
            data: DataArray with B04 (Red) and B08 (NIR) bands
            
        Returns:
            NDVI DataArray
        """
        # Get band indices
        band_names = list(data.coords.get('band', []).values)
        
        if 'B04' not in band_names or 'B08' not in band_names:
            raise ValueError("B04 (Red) and B08 (NIR) bands required for NDVI")
        
        nir = data.sel(band='B08')
        red = data.sel(band='B04')
        
        ndvi = (nir - red) / (nir + red)
        ndvi.attrs['long_name'] = 'Normalized Difference Vegetation Index'
        ndvi.attrs['valid_range'] = [-1, 1]
        
        return ndvi
    
    def create_color_composite(
        self,
        data: xr.DataArray,
        composite: str = 'true_color'
    ) -> xr.DataArray:
        """
        Create color composite image.
        
        Args:
            data: DataArray with spectral bands
            composite: Type of composite ('true_color', 'false_color', 'ndvi')
            
        Returns:
            RGB DataArray
        """
        band_names = list(data.coords.get('band', []).values)
        
        composites = {
            'true_color': ('B04', 'B03', 'B02'),  # Red, Green, Blue
            'false_color': ('B08', 'B04', 'B03'),  # NIR, Red, Green
            'agriculture': ('B08', 'B11', 'B02'),  # NIR, SWIR, Blue
            'geology': ('B12', 'B11', 'B02'),  # SWIR, SWIR, Blue
        }
        
        if composite not in composites:
            raise ValueError(f"Unknown composite: {composite}")
        
        r_band, g_band, b_band = composites[composite]
        
        if r_band not in band_names:
            raise ValueError(f"Band {r_band} not available")
        if g_band not in band_names:
            raise ValueError(f"Band {g_band} not available")
        if b_band not in band_names:
            raise ValueError(f"Band {b_band} not available")
        
        r = data.sel(band=r_band)
        g = data.sel(band=g_band)
        b = data.sel(band=b_band)
        
        # Stack to create RGB image
        composite_image = xr.concat([r, g, b], dim='rgb')
        composite_image = composite_image.assign_coords({
            'rgb': ['Red', 'Green', 'Blue']
        })
        
        return composite_image
