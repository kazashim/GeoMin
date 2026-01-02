"""
Coordinate Reference System (CRS) utilities for GeoMin.
Handles projection transformations and coordinate conversions.
"""

from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer, CRS as PyprojCRS
from pyproj.exceptions import CRSError
from shapely.geometry import box, Polygon
from shapely.ops import transform
import geopandas as gpd


class CRSError(Exception):
    """Custom exception for CRS-related errors."""
    pass


def detect_crs(data: Union[xr.DataArray, rasterio.DatasetReader]) -> str:
    """
    Detect coordinate reference system from data source.
    
    Args:
        data: xarray DataArray or rasterio dataset
        
    Returns:
        CRS string in EPSG format
        
    Raises:
        CRSError: If CRS cannot be detected
    """
    if isinstance(data, xr.DataArray):
        if 'spatial_ref' in data.attrs:
            return data.attrs['spatial_ref']
        elif hasattr(data, 'rio'):
            try:
                return data.rio.crs.to_string()
            except Exception:
                pass
        # Try to get from coordinates
        for coord in data.coords:
            if 'crs' in str(data.coords[coord].attrs).lower():
                return str(data.coords[coord].attrs.get('crs', 'EPSG:4326'))
    
    elif hasattr(data, 'crs') and data.crs:
        return data.crs.to_string()
    
    # Default to WGS84 if no CRS found
    return "EPSG:4326"


def get_utm_zone(lon: float, lat: float) -> str:
    """
    Get UTM zone for given longitude and latitude.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        
    Returns:
        UTM zone string (e.g., "EPSG:32611" for zone 11N)
    """
    zone_number = int((lon + 180) / 6) + 1
    
    # Determine hemisphere
    if lat >= 0:
        hemisphere = 'north'
        epsg_base = 32600
    else:
        hemisphere = 'south'
        epsg_base = 32700
    
    epsg_code = epsg_base + zone_number
    return f"EPSG:{epsg_code}"


def get_utm_zone_from_bbox(bbox: Tuple[float, float, float, float]) -> str:
    """
    Get optimal UTM zone for a bounding box.
    
    Uses the center of the bounding box to determine zone.
    
    Args:
        bbox: (minx, miny, maxx, maxy) in WGS84
        
    Returns:
        UTM zone string
    """
    minx, miny, maxx, maxy = bbox
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2
    return get_utm_zone(center_lon, center_lat)


def transform_bbox(
    bbox: Tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
    buffer: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box between coordinate systems.
    
    Args:
        bbox: (minx, miny, maxx, maxy) in source CRS
        src_crs: Source CRS string
        dst_crs: Destination CRS string
        buffer: Optional buffer to add (in destination units)
        
    Returns:
        Transformed bounding box (minx, miny, maxx, maxy)
    """
    # Create polygon from bbox
    minx, miny, maxx, maxy = bbox
    polygon = box(minx, miny, maxx, maxy)
    
    # Transform polygon
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    transformed_polygon = transform(transformer.transform, polygon)
    
    # Get bounding box
    minx, miny = transformed_polygon.bounds[:2]
    maxx, maxy = transformed_polygon.bounds[2:]
    
    # Add buffer if specified
    if buffer > 0:
        minx -= buffer
        miny -= buffer
        maxx += buffer
        maxy += buffer
    
    return (minx, miny, maxx, maxy)


def reproject_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    dst_crs: str,
    resampling: str = 'bilinear',
    target_resolution: Optional[float] = None
) -> None:
    """
    Reproject a raster file to a new coordinate system.
    
    Args:
        src_path: Source raster file path
        dst_path: Destination raster file path
        dst_crs: Target CRS string
        resampling: Resampling method (nearest, bilinear, cubic, etc.)
        target_resolution: Optional target resolution in meters
    """
    with rasterio.open(src_path) as src:
        src_crs = src.crs.to_string()
        
        # Calculate transform and dimensions
        resampling_methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'cubic_spline': Resampling.cubic_spline,
            'lanczos': Resampling.lanczos,
            'average': Resampling.average,
        }
        
        resample = resampling_methods.get(resampling, Resampling.bilinear)
        
        if target_resolution:
            # Calculate new dimensions based on target resolution
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height,
                *src.bounds, resolution=target_resolution
            )
        else:
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, src.width, src.height,
                *src.bounds
            )
        
        # Create destination raster
        dst_kwargs = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': src.count,
            'dtype': src.dtypes[0],
            'crs': dst_crs,
            'transform': transform,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }
        
        with rasterio.open(dst_path, 'w', **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resample
                )


def reproject_xarray(
    data: xr.DataArray,
    target_crs: str,
    target_resolution: Optional[float] = None,
    resampling: str = 'bilinear'
) -> xr.DataArray:
    """
    Reproject xarray DataArray to a new coordinate system.
    
    Args:
        data: Input DataArray with geospatial coordinates
        target_crs: Target CRS string
        target_resolution: Optional target resolution in meters
        resampling: Resampling method
        
    Returns:
        Reprojected DataArray
    """
    try:
        import rioxarray  # Enable xarray rio accessor
    except ImportError:
        raise ImportError(
            "rioxarray is required for xarray reprojection. "
            "Install with: pip install rioxarray"
        )
    
    # Ensure CRS is set
    if not data.rio.crs:
        data = data.rio.write_crs(detect_crs(data))
    
    # Reproject
    reprojected = data.rio.reproject(
        target_crs,
        resolution=target_resolution,
        resampling=resampling
    )
    
    return reprojected


def align_rasters(
    raster1: Union[xr.DataArray, str, Path],
    raster2: Union[xr.DataArray, str, Path],
    target_crs: Optional[str] = None,
    target_resolution: Optional[float] = None
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Align two rasters to common grid.
    
    Args:
        raster1: First raster (DataArray or path)
        raster2: Second raster (DataArray or path)
        target_crs: Optional target CRS (uses raster1 CRS if not specified)
        target_resolution: Optional target resolution
        
    Returns:
        Tuple of aligned DataArrays
    """
    # Load rasters if paths provided
    if isinstance(raster1, (str, Path)):
        raster1 = xr.open_rasterio(raster1)
    if isinstance(raster2, (str, Path)):
        raster2 = xr.open_rasterio(raster2)
    
    # Determine target CRS
    if target_crs is None:
        target_crs = detect_crs(raster1)
    
    # Reproject both to target
    aligned1 = reproject_xarray(raster1, target_crs, target_resolution)
    aligned2 = reproject_xarray(raster2, target_crs, target_resolution)
    
    # Align grids using resample
    # Get common extent
    minx = max(aligned1.x.min(), aligned2.x.min())
    maxx = min(aligned1.x.max(), aligned2.x.max())
    miny = max(aligned1.y.min(), aligned2.y.min())
    maxy = min(aligned1.y.max(), aligned2.y.max())
    
    # Clip to common extent
    aligned1 = aligned1.sel(x=slice(minx, maxx), y=slice(maxy, miny))
    aligned2 = aligned2.sel(x=slice(minx, maxx), y=slice(maxy, miny))
    
    return aligned1, aligned2


def create_wgs84_bbox(bbox: Tuple[float, float, float, float], src_crs: str) -> Tuple[float, float, float, float]:
    """
    Convert bounding box to WGS84 coordinates.
    
    Args:
        bbox: (minx, miny, maxx, maxy) in source CRS
        src_crs: Source CRS string
        
    Returns:
        Bounding box in WGS84 (EPSG:4326)
    """
    return transform_bbox(bbox, src_crs, "EPSG:4326")


def is_valid_crs(crs_string: str) -> bool:
    """
    Validate a CRS string.
    
    Args:
        crs_string: CRS string to validate
        
    Returns:
        True if valid CRS
    """
    try:
        PyprojCRS.from_string(crs_string)
        return True
    except CRSError:
        return False


def get_crs_info(crs_string: str) -> dict:
    """
    Get information about a CRS.
    
    Args:
        crs_string: CRS string (EPSG code, WKT, etc.)
        
    Returns:
        Dictionary with CRS information
    """
    try:
        crs = PyprojCRS.from_string(crs_string)
        return {
            'authority': crs.to_epsg() if crs.to_epsg() else None,
            'name': crs.name,
            'type': 'projected' if crs.is_projected else 'geographic',
            'proj4': crs.to_proj4(),
            'wkt': crs.to_wkt(),
            'datum': crs.datum.name if crs.datum else None,
            'ellipsoid': crs.ellipsoid.name if crs.ellipsoid else None,
        }
    except CRSError as e:
        raise CRSError(f"Invalid CRS: {e}")
