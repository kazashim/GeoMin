"""
STAC client for GeoMin.
Provides access to SpatioTemporal Asset Catalog (STAC) endpoints.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box, Polygon

try:
    import pystac_client
    import stackstac
    STAC_AVAILABLE = True
except ImportError:
    STAC_AVAILABLE = False

from .base_client import SatClient, SearchResult, SearchOptions


@dataclass
class STACConfig:
    """Configuration for STAC endpoints."""
    endpoint: str = "https://earth-search.aws.element84.com/v1"
    collections: List[str] = field(default_factory=lambda: ["sentinel-2-l2a"])
    search_kwargs: Dict[str, Any] = field(default_factory=dict)


class STACClient:
    """
    Client for STAC (SpatioTemporal Asset Catalog) endpoints.
    
    Provides access to cloud-optimized satellite data from:
    - AWS Earth Search (Sentinel-2, Landsat)
    - Microsoft Planetary Computer
    - USGS STAC
    - Element84 STAC
    
    Features:
    - Lazy loading with Dask for large regions
    - Automatic chunking for parallel processing
    - Support for Cloud Optimized GeoTIFFs (COG)
    """
    
    # Popular STAC endpoints
    ENDPOINTS = {
        'aws': 'https://earth-search.aws.element84.com/v1',
        'planetary_computer': 'https://planetarycomputer.microsoft.com/api/stac/v1',
        'element84': 'https://api.stac.terrascope.be/v1',
        'usgs': 'https://landsatlook.usgs.gov/stac',
    }
    
    # Supported collections
    COLLECTIONS = {
        'sentinel-2-l2a': {
            'name': 'Sentinel-2 Level-2A',
            'provider': 'Copernicus',
            'license': 'proprietary',
            'resolution': 10,
            'bands': ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'scl'],
        },
        'landsat-c2-l2': {
            'name': 'Landsat Collection 2 Level-2',
            'provider': 'USGS',
            'license': 'proprietary',
            'resolution': 30,
            'bands': ['blue', 'green', 'red', 'nir08', 'swir11', 'swir16', 'qa'],
        },
        'landsat-c2-l1': {
            'name': 'Landsat Collection 2 Level-1',
            'provider': 'USGS',
            'license': 'proprietary',
            'resolution': 30,
            'bands': ['blue', 'green', 'red', 'nir08', 'swir11', 'swir16', 'pan'],
        },
    }
    
    def __init__(
        self,
        endpoint: str = 'aws',
        config: Optional[STACConfig] = None
    ):
        """
        Initialize STAC client.
        
        Args:
            endpoint: Endpoint name ('aws', 'planetary_computer', etc.) or full URL
            config: Optional STACConfig object
        """
        if not STAC_AVAILABLE:
            raise ImportError(
                "pystac-client and stackstac are required for STAC support. "
                "Install with: pip install geomin[stac]"
            )
        
        # Resolve endpoint
        if endpoint in self.ENDPOINTS:
            self.endpoint = self.ENDPOINTS[endpoint]
        else:
            self.endpoint = endpoint
        
        self.config = config or STACConfig(endpoint=self.endpoint)
        self._catalog = None
    
    def connect(self) -> bool:
        """Connect to STAC endpoint."""
        try:
            self._catalog = pystac_client.Client.open(
                self.endpoint,
                **self.config.search_kwargs
            )
            return True
        except Exception as e:
            print(f"Failed to connect to STAC: {e}")
            return False
    
    def disconnect(self) -> None:
        """Close connection."""
        self._catalog = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._catalog is not None
    
    def search(
        self,
        options: SearchOptions
    ) -> List[SearchResult]:
        """
        Search STAC catalog for satellite imagery.
        
        Args:
            options: Search options
            
        Returns:
            List of SearchResult objects
        """
        if not self.is_connected:
            self.connect()
        
        # Build search parameters
        search_params = {
            'collections': self.config.collections,
            'bbox': options.bbox if options.bbox else self._geometry_to_bbox(options.geometry),
            'limit': options.limit,
        }
        
        # Add datetime filter
        if options.start_date or options.end_date:
            start = options.start_date.isoformat() if options.start_date else '1900-01-01'
            end = options.end_date.isoformat() if options.end_date else datetime.now().isoformat()
            search_params['datetime'] = f"{start}/{end}"
        
        # Add query filters
        query = {}
        
        if options.cloud_cover < 100:
            query['eo:cloud_cover'] = {'lte': options.cloud_cover}
        
        if query:
            search_params['query'] = query
        
        try:
            # Execute search
            search = self._catalog.search(**search_params)
            items = list(search.items())
            
            # Convert to SearchResult objects
            results = []
            for item in items:
                result = self._item_to_search_result(item)
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"STAC search failed: {e}")
            return []
    
    def search_text(
        self,
        query: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        datetime_range: Optional[str] = None,
        collections: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[SearchResult]:
        """
        Full-text search across STAC catalog.
        
        Args:
            query: Search query string
            bbox: Optional bounding box filter
            datetime_range: Date range (e.g., "2023-01-01/2023-12-31")
            collections: Optional collection filter
            limit: Maximum results
            
        Returns:
            List of matching SearchResult objects
        """
        if not self.is_connected:
            self.connect()
        
        search_params = {
            'q': query,
            'limit': limit,
        }
        
        if bbox:
            search_params['bbox'] = bbox
        
        if datetime_range:
            search_params['datetime'] = datetime_range
        
        if collections:
            search_params['collections'] = collections
        
        try:
            search = self._catalog.search(**search_params)
            items = list(search.items())
            
            return [self._item_to_search_result(item) for item in items]
            
        except Exception as e:
            print(f"Text search failed: {e}")
            return []
    
    def load_data(
        self,
        results: Union[SearchResult, List[SearchResult]],
        bands: Optional[List[str]] = None,
        chunksize: int = 4096,
        epsg: Optional[int] = None
    ) -> xr.DataArray:
        """
        Load satellite data from STAC directly into xarray.
        
        Uses stackstac for lazy loading with Dask.
        
        Args:
            results: Search results to load
            bands: Specific bands to load
            chunksize: Dask chunk size
            epsg: Target EPSG code (auto-detect if None)
            
        Returns:
            Lazy-loaded xarray DataArray
        """
        if not isinstance(results, list):
            results = [results]
        
        if not STAC_AVAILABLE:
            raise ImportError("stackstac required for data loading")
        
        # Get STAC items from results
        items = []
        for result in results:
            # Re-query to get STAC item
            item = self._catalog.get_item(result.scene_id)
            if item:
                items.append(item)
        
        if not items:
            raise ValueError("No valid STAC items found")
        
        # Determine asset keys
        if bands is None:
            bands = self._get_default_bands(items[0])
        
        # Load with stackstac
        cube = stackstac.stack(
            items,
            assets=bands,
            chunksize=chunksize,
            epsg=epsg,
        )
        
        return cube
    
    def get_collections(self) -> Dict[str, Dict[str, Any]]:
        """Get available collections from catalog."""
        if not self.is_connected:
            self.connect()
        
        collections = {}
        for collection in self._catalog.get_collections():
            collections[collection.id] = {
                'id': collection.id,
                'title': collection.title or collection.id,
                'description': collection.description,
                'license': collection.license,
                'extent': collection.extent.to_dict() if collection.extent else None,
            }
        
        return collections
    
    def get_collection_info(self, collection_id: str) -> Dict[str, Any]:
        """Get information about a specific collection."""
        if not self.is_connected:
            self.connect()
        
        collection = self._catalog.get_collection(collection_id)
        
        if collection:
            return {
                'id': collection.id,
                'title': collection.title,
                'description': collection.description,
                'license': collection.license,
                'keywords': collection.keywords,
                'providers': [
                    {'name': p.name, 'url': p.url}
                    for p in collection.providers
                ],
                'extent': collection.extent.to_dict() if collection.extent else None,
            }
        
        return {}
    
    def _item_to_search_result(self, item) -> SearchResult:
        """Convert STAC item to SearchResult."""
        properties = item.properties
        
        # Parse datetime
        acquired = properties.get('datetime', properties.get('acquired'))
        if isinstance(acquired, str):
            acquisition_time = datetime.fromisoformat(acquired.split('T')[0])
        else:
            acquisition_time = datetime.now()
        
        # Get cloud cover
        cloud_cover = properties.get('eo:cloud_cover', 0)
        
        # Get geometry
        geometry = item.geometry
        
        # Convert to shapely polygon
        if geometry:
            from shapely.geometry import shape
            polygon = shape(geometry)
        else:
            polygon = box(-180, -90, 180, 90)
        
        # Get collection info
        collection_id = item.collection_id
        collection_info = self.COLLECTIONS.get(collection_id, {})
        
        # Get available bands
        bands = list(item.assets.keys())
        
        # Estimate resolution
        resolution = collection_info.get('resolution', 10)
        
        return SearchResult(
            scene_id=item.id,
            provider='stac',
            acquisition_time=acquisition_time,
            cloud_cover=cloud_cover,
            geometry=polygon,
            bands=bands,
            resolution=resolution,
            metadata={
                'collection': collection_id,
                'stac_item': item,
                'assets': list(item.assets.keys()),
                'bbox': item.bbox,
                'links': [link.to_dict() for link in item.links],
            }
        )
    
    def _geometry_to_bbox(self, geometry: Polygon) -> Tuple[float, float, float, float]:
        """Convert shapely geometry to bounding box tuple."""
        if geometry is None:
            return (-180, -90, 180, 90)
        
        bounds = geometry.bounds
        return (bounds[0], bounds[1], bounds[2], bounds[3])
    
    def _get_default_bands(self, item) -> List[str]:
        """Get default bands for a STAC item."""
        collection_id = item.collection_id
        
        if collection_id in self.COLLECTIONS:
            return self.COLLECTIONS[collection_id]['bands']
        
        # Try to infer from common band names
        common_bands = ['blue', 'green', 'red', 'nir08', 'swir16', 'swir22']
        available = [b for b in common_bands if b in item.assets]
        
        if available:
            return available
        
        # Return all asset keys except metadata
        return [k for k in item.assets.keys() if not k.startswith('meta')]
    
    def get_timestamps(self, results: List[SearchResult]) -> List[datetime]:
        """Extract acquisition timestamps from search results."""
        return [r.acquisition_time for r in results]
    
    def sort_by_date(
        self,
        results: List[SearchResult],
        ascending: bool = True
    ) -> List[SearchResult]:
        """Sort search results by acquisition date."""
        return sorted(results, key=lambda r: r.acquisition_time, reverse=not ascending)
    
    def filter_by_date(
        self,
        results: List[SearchResult],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[SearchResult]:
        """Filter results by date range."""
        filtered = results
        
        if start_date:
            filtered = [r for r in filtered if r.acquisition_time >= start_date]
        
        if end_date:
            filtered = [r for r in filtered if r.acquisition_time <= end_date]
        
        return filtered
    
    def filter_by_cloud_cover(
        self,
        results: List[SearchResult],
        max_cloud_cover: float
    ) -> List[SearchResult]:
        """Filter results by maximum cloud cover."""
        return [r for r in results if r.cloud_cover <= max_cloud_cover]
