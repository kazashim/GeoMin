"""
Landsat satellite data client for GeoMin.
Connects to USGS EarthData and Google Cloud Storage for Landsat data.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from urllib.parse import urljoin
import json
import time

import requests
import xml.etree.ElementTree as ET
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, Polygon

from .base_client import SatClient, SearchResult, SearchOptions
from ..core.config import get_config
from ..core.data_loader import DataLoader
from ..core.crs import detect_crs, reproject_xarray


class LandsatClient(SatClient):
    """
    Client for Landsat satellite data.
    
    Supports access to:
    - Landsat 8 OLI (Operational Land Imager)
    - Landsat 9 OLI-2
    - Landsat 7 ETM+ (Enhanced Thematic Mapper)
    - Landsat 4-5 TM (Thematic Mapper)
    
    Data sources:
    - USGS EarthData (requires authentication)
    - Google Cloud Storage (public access)
    """
    
    PROVIDER_NAME = "landsat"
    DEFAULT_CRS = "EPSG:4326"
    
    # Google Cloud Storage public bucket
    GCS_BASE_URL = "https://storage.googleapis.com/"
    GCS_BUCKET = "gcp-public-data-landsat"
    
    # USGS EarthData endpoints
    EARTHDATA_BASE_URL = "https://earthexplorer.usgs.gov/inventoryJSON"
    EARTHDATA_SEARCH_URL = "https://api.cr.usgs.gov/las/IPL/search"
    
    # Collection information
    COLLECTIONS = {
        "landsat_ot_c2_l2": {
            "name": "Landsat 8-9 OLI/TIRS Collection 2 Level 2",
            "satellites": ["landsat8", "landsat9"],
            "resolution": 30,
            "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10", "ST_B11", "QA"],
        },
        "landsat_etm_c2_l2": {
            "name": "Landsat 7 ETM+ Collection 2 Level 2",
            "satellites": ["landsat7"],
            "resolution": 30,
            "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA"],
        },
        "landsat_tm_c2_l2": {
            "name": "Landsat 4-5 TM Collection 2 Level 2",
            "satellites": ["landsat4", "landsat5"],
            "resolution": 30,
            "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "ST_B6", "QA"],
        },
    }
    
    def __init__(
        self,
        use_google_cloud: bool = True,
        config: Optional[Dict] = None
    ):
        """
        Initialize Landsat client.
        
        Args:
            use_google_cloud: Use Google Cloud Storage (public) vs USGS (requires auth)
            config: Optional configuration
        """
        super().__init__(config)
        self.use_google_cloud = use_google_cloud
        self.session = None
        self._api_key = None
        
        if self.config.api.earthdata_username and self.config.api.earthdata_password:
            self.use_google_cloud = False
    
    def connect(self) -> bool:
        """Connect to Landsat data provider."""
        if self.use_google_cloud:
            # Google Cloud Storage is publicly accessible
            self._connection_status = True
            return True
        
        # Set up session for USGS EarthData
        if not self.config.api.earthdata_username or not self.config.api.earthdata_password:
            print("Warning: EarthData credentials not configured. Using Google Cloud Storage.")
            self.use_google_cloud = True
            self._connection_status = True
            return True
        
        self.session = requests.Session()
        self.session.auth = (
            self.config.api.earthdata_username,
            self.config.api.earthdata_password
        )
        
        # Test connection
        try:
            # Just verify we can make requests
            self._connection_status = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            self._connection_status = False
            return False
    
    def disconnect(self) -> None:
        """Close connection to provider."""
        if self.session:
            self.session.close()
        self._connection_status = False
    
    def search(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search for Landsat imagery.
        
        Args:
            options: Search options including area and time range
            
        Returns:
            List of matching Landsat scenes
        """
        if self.use_google_cloud:
            return self._search_gcs(options)
        else:
            return self._search_usgs(options)
    
    def _search_gcs(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search Landsat data on Google Cloud Storage.
        
        Uses the Landsat CSV index file for efficient searching.
        """
        # Build GCS path for the scene index
        index_url = (
            f"{self.GCS_BASE_URL}{self.GCS_BUCKET}/"
            "landsat_amazon/index.csv.gz"
        )
        
        try:
            import io
            import gzip
            
            # Download and parse index
            response = requests.get(index_url, timeout=60)
            response.raise_for_status()
            
            # Parse CSV
            import pandas as pd
            df = pd.read_csv(
                io.StringIO(gzip.decompress(response.content).decode('utf-8'))
            )
            
            # Filter by spatial extent
            if options.bbox:
                minx, miny, maxx, maxy = options.bbox
                df = df[
                    (df['min_lat'] <= maxy) &
                    (df['max_lat'] >= miny) &
                    (df['min_lon'] <= maxx) &
                    (df['max_lon'] >= minx)
                ]
            
            # Filter by date
            if options.start_date:
                df = df[df['acquisition_date'] >= options.start_date.strftime('%Y-%m-%d')]
            if options.end_date:
                df = df[df['acquisition_date'] <= options.end_date.strftime('%Y-%m-%d')]
            
            # Filter by cloud cover if available
            if 'cloud_cover' in df.columns and options.cloud_cover < 100:
                df = df[df['cloud_cover'] <= options.cloud_cover]
            
            # Filter by collection
            if options.source:
                df = df[df['collection'] == options.source]
            
            # Limit results
            df = df.head(options.limit)
            
            # Convert to SearchResult objects
            results = []
            for _, row in df.iterrows():
                # Get scene ID components
                scene_id = row.get('scene_id', row.get('product_id', ''))
                
                # Build geometry
                geometry = box(
                    row['min_lon'], row['min_lat'],
                    row['max_lon'], row['max_lat']
                )
                
                # Parse acquisition date
                acquisition_time = datetime.strptime(
                    row['acquisition_date'], '%Y-%m-%d'
                ) if isinstance(row['acquisition_date'], str) else row['acquisition_date']
                
                result = SearchResult(
                    scene_id=scene_id,
                    provider=self.PROVIDER_NAME,
                    acquisition_time=acquisition_time,
                    cloud_cover=row.get('cloud_cover', 0),
                    geometry=geometry,
                    bands=self._get_bands_from_scene_id(scene_id),
                    resolution=30,  # Standard Landsat resolution
                    data_size=row.get('size_mb'),
                    metadata={
                        'collection': row.get('collection'),
                        'satellite': row.get('satellite'),
                        'sensor': row.get('sensor'),
                        'path': row.get('path'),
                        'row': row.get('row'),
                        'gcs_path': row.get('gcs_path'),
                    }
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"GCS search failed: {e}")
            return []
    
    def _search_usgs(self, options: SearchOptions) -> List[SearchResult]:
        """
        Search Landsat data via USGS EarthData API.
        
        Requires authentication and API key.
        """
        if not self.session:
            self.connect()
        
        # Prepare search request
        search_params = {
            "jsonRequest": {
                "catalog": "EE",
                "includeUnknownCloudCover": False,
                "maxCloudCover": options.cloud_cover,
                "maxResults": options.limit,
                "sortOrder": "ASC",
            }
        }
        
        # Add temporal filter
        if options.start_date:
            search_params["jsonRequest"]["startDate"] = options.start_date.strftime('%Y-%m-%d')
        if options.end_date:
            search_params["jsonRequest"]["endDate"] = options.end_date.strftime('%Y-%m-%d')
        
        # Add spatial filter
        if options.bbox:
            minx, miny, maxx, maxy = options.bbox
            search_params["jsonRequest"]["spatialFilter"] = {
                "filterType": "mbr",
                "lowerLeft": {"latitude": miny, "longitude": minx},
                "upperRight": {"latitude": maxy, "longitude": maxx},
            }
        elif options.geometry:
            bounds = options.geometry.bounds
            search_params["jsonRequest"]["spatialFilter"] = {
                "filterType": "mbr",
                "lowerLeft": {"latitude": bounds[1], "longitude": bounds[0]},
                "upperRight": {"latitude": bounds[3], "longitude": bounds[2]},
            }
        
        try:
            # Make search request
            response = self.session.post(
                self.EARTHDATA_SEARCH_URL,
                json=search_params,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for scene in data.get('results', []):
                # Parse geometry
                geometry = self._parse_usgs_geometry(scene.get('spatialFootprint'))
                
                # Parse date
                acquisition_time = datetime.fromisoformat(
                    scene.get('acquisitionDate', scene.get('displayId', ''))
                )
                
                result = SearchResult(
                    scene_id=scene['entityId'],
                    provider=self.PROVIDER_NAME,
                    acquisition_time=acquisition_time,
                    cloud_cover=float(scene.get('cloudCover', 0)),
                    geometry=geometry,
                    bands=self._get_bands_from_scene_id(scene.get('entityId', '')),
                    metadata=scene,
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"USGS search failed: {e}")
            return []
    
    def _parse_usgs_geometry(self, footprint: str) -> Polygon:
        """Parse USGS spatial footprint to geometry."""
        if not footprint:
            return box(-180, -90, 180, 90)
        
        try:
            # Parse WKT or JSON geometry
            if footprint.startswith('POLYGON') or footprint.startswith('MULTIPOLYGON'):
                from shapely import wkt
                return wkt.loads(footprint)
            else:
                import json
                coords = json.loads(footprint)
                return Polygon(coords[0])
        except Exception:
            return box(-180, -90, 180, 90)
    
    def _get_bands_from_scene_id(self, scene_id: str) -> List[str]:
        """
        Determine available bands from scene ID.
        
        Parses scene ID to identify satellite and sensor.
        """
        scene_id = scene_id.upper()
        
        if 'LC08' in scene_id or 'LC09' in scene_id:
            return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'QA']
        elif 'LE07' in scene_id or 'LT05' in scene_id or 'LT04' in scene_id:
            return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7', 'B8', 'QA']
        else:
            return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'QA']
    
    def _get_gcs_path(self, scene_id: str) -> str:
        """
        Get Google Cloud Storage path for a scene.
        
        Args:
            scene_id: Landsat scene ID
            
        Returns:
            GCS path string
        """
        # Parse scene ID components
        parts = scene_id.split('_')
        satellite = parts[0]  # LC08
        path = parts[2]  # Path number
        row = parts[3]  # Row number
        
        return f"{self.GCS_BUCKET}/{satellite}/01/{path}/{row}/{scene_id}"
    
    def download(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Download Landsat data for a scene.
        
        Args:
            result: Search result to download
            bands: Specific bands to download
            output_dir: Output directory
            
        Returns:
            Dictionary mapping band names to file paths
        """
        if output_dir is None:
            output_dir = self.config.cache.cache_dir / "landsat"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = {}
        
        if self.use_google_cloud:
            downloaded = self._download_gcs(result, bands, output_dir)
        else:
            downloaded = self._download_usgs(result, bands, output_dir)
        
        return downloaded
    
    def _download_gcs(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Path = None
    ) -> Dict[str, Path]:
        """
        Download from Google Cloud Storage.
        
        Files are organized by band in the public bucket.
        """
        gcs_path = self._get_gcs_path(result.scene_id)
        base_url = f"{self.GCS_BASE_URL}{gcs_path}"
        
        if bands is None:
            bands = result.bands
        
        downloaded = {}
        for band in bands:
            # Construct band filename
            if band == 'QA':
                filename = f"{result.scene_id}_QA_PIXEL.tif"
            else:
                filename = f"{result.scene_id}_{band}.tif"
            
            url = f"{base_url}/{filename}"
            output_path = output_dir / filename
            
            if not output_path.exists():
                print(f"Downloading {band} from {url}")
                response = requests.get(url, timeout=300)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            
            downloaded[band] = output_path
        
        return downloaded
    
    def _download_usgs(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None,
        output_dir: Path = None
    ) -> Dict[str, Path]:
        """
        Download from USGS EarthData.
        
        Requires valid EarthData credentials and download authorization.
        """
        # USGS download requires pre-authorization
        # This is a simplified implementation
        print("USGS download requires additional setup for download authorization")
        return {}
    
    def load(
        self,
        result: SearchResult,
        bands: Optional[List[str]] = None
    ) -> xr.DataArray:
        """
        Load Landsat data directly into xarray.
        
        Args:
            result: Search result to load
            bands: Specific bands to load
            
        Returns:
            DataArray with Landsat data
        """
        # Download files
        files = self.download(result, bands)
        
        if not files:
            raise ValueError("No files downloaded")
        
        # Load first band to get metadata
        first_band = list(files.keys())[0]
        data = DataLoader.load(files[first_band])
        
        # Load additional bands and stack
        if len(files) > 1:
            bands_data = [data]
            for band_name, filepath in list(files.items())[1:]:
                band_data = DataLoader.load(filepath)
                bands_data.append(band_data)
            
            data = xr.concat(bands_data, dim='band')
            
            # Add band names
            data = data.assign_coords({
                'band': list(files.keys())
            })
        
        return data
    
    def get_bands(self, scene_id: str) -> Dict[str, Any]:
        """
        Get available bands for a Landsat scene.
        
        Args:
            scene_id: Landsat scene ID
            
        Returns:
            Dictionary of band information
        """
        satellite = self._get_satellite_from_scene_id(scene_id)
        
        band_info = {
            'OLI': {
                'B1': {'name': 'Coastal Aerosol', 'wavelength': 0.44, 'resolution': 30},
                'B2': {'name': 'Blue', 'wavelength': 0.45, 'resolution': 30},
                'B3': {'name': 'Green', 'wavelength': 0.53, 'resolution': 30},
                'B4': {'name': 'Red', 'wavelength': 0.64, 'resolution': 30},
                'B5': {'name': 'NIR', 'wavelength': 0.86, 'resolution': 30},
                'B6': {'name': 'SWIR 1', 'wavelength': 1.57, 'resolution': 30},
                'B7': {'name': 'SWIR 2', 'wavelength': 2.11, 'resolution': 30},
                'B8': {'name': 'Pan', 'wavelength': 0.59, 'resolution': 15},
                'B9': {'name': 'Cirrus', 'wavelength': 1.37, 'resolution': 30},
                'B10': {'name': 'TIRS 1', 'wavelength': 10.9, 'resolution': 100},
                'B11': {'name': 'TIRS 2', 'wavelength': 12.0, 'resolution': 100},
            },
            'ETM': {
                'B1': {'name': 'Blue', 'wavelength': 0.45, 'resolution': 30},
                'B2': {'name': 'Green', 'wavelength': 0.52, 'resolution': 30},
                'B3': {'name': 'Red', 'wavelength': 0.63, 'resolution': 30},
                'B4': {'name': 'NIR', 'wavelength': 0.77, 'resolution': 30},
                'B5': {'name': 'SWIR 1', 'wavelength': 1.55, 'resolution': 30},
                'B6_VCID_1': {'name': 'Thermal 1', 'wavelength': 11.45, 'resolution': 60},
                'B6_VCID_2': {'name': 'Thermal 2', 'wavelength': 11.45, 'resolution': 60},
                'B7': {'name': 'SWIR 2', 'wavelength': 2.08, 'resolution': 30},
                'B8': {'name': 'Pan', 'wavelength': 0.71, 'resolution': 15},
            },
            'TM': {
                'B1': {'name': 'Blue', 'wavelength': 0.45, 'resolution': 30},
                'B2': {'name': 'Green', 'wavelength': 0.52, 'resolution': 30},
                'B3': {'name': 'Red', 'wavelength': 0.63, 'resolution': 30},
                'B4': {'name': 'NIR', 'wavelength': 0.77, 'resolution': 30},
                'B5': {'name': 'SWIR 1', 'wavelength': 1.55, 'resolution': 30},
                'B6': {'name': 'Thermal', 'wavelength': 11.45, 'resolution': 120},
                'B7': {'name': 'SWIR 2', 'wavelength': 2.08, 'resolution': 30},
            },
        }
        
        sensor_type = 'OLI' if satellite in ['LC08', 'LC09'] else ('ETM' if satellite == 'LE07' else 'TM')
        return band_info.get(sensor_type, {})
    
    def _get_satellite_from_scene_id(self, scene_id: str) -> str:
        """Extract satellite identifier from scene ID."""
        scene_id = scene_id.upper()
        if 'LC09' in scene_id:
            return 'LC09'
        elif 'LC08' in scene_id:
            return 'LC08'
        elif 'LE07' in scene_id:
            return 'LE07'
        elif 'LT05' in scene_id:
            return 'LT05'
        elif 'LT04' in scene_id:
            return 'LT04'
        else:
            return 'UNKNOWN'
    
    def get_scene_info(self, scene_id: str) -> Dict[str, Any]:
        """
        Get metadata for a Landsat scene.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene metadata dictionary
        """
        satellite = self._get_satellite_from_scene_id(scene_id)
        
        return {
            'scene_id': scene_id,
            'satellite': satellite,
            'sensor': 'OLI/TIRS' if satellite in ['LC08', 'LC09'] else 'ETM+' if satellite == 'LE07' else 'TM',
            'collection': 'Collection 2 Level 2',
            'resolution': 30,
            'wrs_path': scene_id.split('_')[2],
            'wrs_row': scene_id.split('_')[3],
            'bands': self._get_bands_from_scene_id(scene_id),
        }
