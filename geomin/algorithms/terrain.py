"""
Terrain analysis algorithms for GeoMin.
Processes digital elevation models for slope, aspect, and terrain characterization.
"""

from typing import Union, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import xarray as xr


# Type alias
XRData = Union[xr.DataArray, xr.Dataset]


@dataclass
class TerrainMetrics:
    """
    Collection of terrain derivatives.
    
    Attributes:
        slope: Slope in degrees
        aspect: Aspect direction in degrees
        hillshade: Hillshade illumination
        curvature: Surface curvature
        TPI: Topographic Position Index
        TRI: Terrain Ruggedness Index
    """
    slope: xr.DataArray
    aspect: xr.DataArray
    hillshade: xr.DataArray
    curvature: Optional[xr.DataArray] = None
    TPI: Optional[xr.DataArray] = None
    TRI: Optional[xr.DataArray] = None


def calculate_slope(
    dem: xr.DataArray,
    z_scale: float = 1.0,
    method: str = 'horn'
) -> xr.DataArray:
    """
    Calculate slope from DEM.
    
    Uses finite difference method to compute gradient.
    
    Args:
        dem: Digital Elevation Model DataArray
        z_scale: Vertical exaggeration factor
        method: Slope calculation method ('horn', 'simple')
        
    Returns:
        Slope in degrees DataArray
    """
    # Get pixel dimensions
    dy = _get_pixel_size(dem, 'y')
    dx = _get_pixel_size(dem, 'x')
    
    # Handle negative dy (y decreases with latitude)
    dy = abs(dy)
    
    # Get elevation data as numpy array
    z = dem.values.astype(np.float64)
    
    if method == 'horn':
        # Horn's method (3x3 kernel)
        # Weights for 8 neighbors
        weights = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ])
        
        # Calculate gradients
        dz_dx = _convolve(z, weights.T) / (8 * dx)
        dz_dy = _convolve(z, weights) / (8 * dy)
        
    else:
        # Simple central difference
        dz_dx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * dx)
        dz_dy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * dy)
    
    # Apply z-scale
    dz_dx = dz_dx * z_scale
    dz_dy = dz_dy * z_scale
    
    # Calculate slope in degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        slope_deg = np.where(np.isfinite(slope_deg), slope_deg, np.nan)
    
    result = xr.DataArray(
        slope_deg,
        coords=dem.coords,
        dims=dem.dims,
        attrs={
            'long_name': 'Slope',
            'units': 'degrees',
            'method': method,
        }
    )
    
    return result


def calculate_aspect(dem: xr.DataArray) -> xr.DataArray:
    """
    Calculate aspect (exposure direction) from DEM.
    
    Returns direction of maximum slope in degrees (0-360).
    
    Args:
        dem: Digital Elevation Model DataArray
        
    Returns:
        Aspect in degrees DataArray
    """
    # Get pixel dimensions
    dy = _get_pixel_size(dem, 'y')
    dx = _get_pixel_size(dem, 'x')
    dy = abs(dy)
    
    # Get elevation data
    z = dem.values.astype(np.float64)
    
    # Central difference gradients
    dz_dx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * dx)
    dz_dy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * dy)
    
    # Calculate aspect in radians
    with np.errstate(divide='ignore', invalid='ignore'):
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        aspect_deg = np.degrees(aspect_rad)
        
        # Convert to 0-360 range
        aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)
        aspect_deg = np.where(np.isfinite(aspect_deg), aspect_deg, np.nan)
        
        # Flat areas get special value (e.g., -1)
        flat_mask = np.sqrt(dz_dx**2 + dz_dy**2) < 1e-6
        aspect_deg = np.where(flat_mask, np.nan, aspect_deg)
    
    result = xr.DataArray(
        aspect_deg,
        coords=dem.coords,
        dims=dem.dims,
        attrs={
            'long_name': 'Aspect',
            'units': 'degrees',
            'description': 'Direction of maximum slope (0=N, 90=E, 180=S, 270=W)',
        }
    )
    
    return result


def calculate_hillshade(
    dem: xr.DataArray,
    azimuth: float = 315,
    altitude: float = 45
) -> xr.DataArray:
    """
    Calculate hillshade (shaded relief).
    
    Args:
        dem: Digital Elevation Model DataArray
        azimuth: Sun azimuth in degrees (315 = NW)
        altitude: Sun altitude in degrees (45 = 45 degrees up)
        
    Returns:
        Hillshade intensity (0-255) DataArray
    """
    # Convert angles to radians
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    
    # Get pixel dimensions
    dy = _get_pixel_size(dem, 'y')
    dx = _get_pixel_size(dem, 'x')
    dy = abs(dy)
    
    # Get elevation data
    z = dem.values.astype(np.float64)
    
    # Central difference gradients
    dz_dx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * dx)
    dz_dy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * dy)
    
    # Calculate hillshade
    with np.errstate(divide='ignore', invalid='ignore'):
        # Aspect components
        sin_alt = np.sin(alt_rad)
        cos_alt = np.cos(alt_rad)
        
        # Slope calculation
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        
        # Aspect calculation
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        
        # Hillshade formula
        hs = sin_alt * np.cos(slope_rad) + \
             cos_alt * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
        
        # Scale to 0-255
        hs_scaled = ((hs + 1) / 2 * 255).clip(0, 255)
        hs_scaled = np.where(np.isfinite(hs_scaled), hs_scaled, np.nan)
    
    result = xr.DataArray(
        hs_scaled,
        coords=dem.coords,
        dims=dem.dims,
        attrs={
            'long_name': 'Hillshade',
            'units': '0-255',
            'azimuth': azimuth,
            'altitude': altitude,
        }
    )
    
    return result


def calculate_curvature(dem: xr.DataArray) -> xr.DataArray:
    """
    Calculate surface curvature.
    
    Positive curvature = convex (ridges), negative = concave (valleys).
    
    Args:
        dem: Digital Elevation Model DataArray
        
    Returns:
        Curvature DataArray
    """
    # Get pixel dimensions
    dy = _get_pixel_size(dem, 'y')
    dx = _get_pixel_size(dem, 'x')
    dy = abs(dy)
    
    # Get elevation data
    z = dem.values.astype(np.float64)
    
    # Second derivatives
    d2z_dx2 = (np.roll(z, -1, axis=1) - 2*z + np.roll(z, 1, axis=1)) / (dx**2)
    d2z_dy2 = (np.roll(z, -1, axis=0) - 2*z + np.roll(z, 1, axis=0)) / (dy**2)
    
    # Cross derivative
    d2z_dxdy = (np.roll(np.roll(z, -1, axis=0), -1, axis=1) -
                np.roll(np.roll(z, -1, axis=0), 1, axis=1) -
                np.roll(np.roll(z, 1, axis=0), -1, axis=1) +
                np.roll(np.roll(z, 1, axis=0), 1, axis=1)) / (4 * dx * dy)
    
    # Planform curvature
    with np.errstate(divide='ignore', invalid='ignore'):
        p = (d2z_dx2 * (dz_dx**2) + 2*d2z_dxdy*dz_dx*dz_dy + d2z_dy2 * (dz_dy**2)) / ((1 + dz_dx**2 + dz_dy**2)**2)
        p = np.where(np.isfinite(p), p, np.nan)
    
    result = xr.DataArray(
        p,
        coords=dem.coords,
        dims=dem.dims,
        attrs={
            'long_name': 'Planform Curvature',
            'units': '1/m',
            'description': 'Positive = convex, negative = concave',
        }
    )
    
    return result


def calculate_tpi(dem: xr.DataArray, radius: int = 5) -> xr.DataArray:
    """
    Calculate Topographic Position Index.
    
    TPI = local_elevation - mean_surrounding_elevation
    
    Args:
        dem: Digital Elevation Model DataArray
        radius: Analysis radius in pixels
        
    Returns:
        TPI DataArray
    """
    z = dem.values.astype(np.float64)
    
    # Create moving window
    kernel_size = 2 * radius + 1
    
    # Simple mean filter
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    mean_local = _convolve(z, kernel)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        tpi = z - mean_local
        tpi = np.where(np.isfinite(tpi), tpi, np.nan)
    
    result = xr.DataArray(
        tpi,
        coords=dem.coords,
        dims=dem.dims,
        attrs={
            'long_name': 'Topographic Position Index',
            'units': 'meters',
            'radius': radius,
            'description': 'Positive = ridges, negative = valleys',
        }
    )
    
    return result


def calculate_tri(dem: xr.DataArray, radius: int = 1) -> xr.DataArray:
    """
    Calculate Terrain Ruggedness Index.
    
    TRI = sqrt(sum((elevation - neighbor)^2))
    
    Args:
        dem: Digital Elevation Model DataArray
        radius: Analysis radius in pixels
        
    Returns:
        TRI DataArray
    """
    z = dem.values.astype(np.float64)
    
    # Calculate sum of squared differences
    tri = np.zeros_like(z)
    
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(z, di, axis=0), dj, axis=1)
            tri += (z - shifted)**2
    
    tri = np.sqrt(tri)
    
    result = xr.DataArray(
        tri,
        coords=dem.coords,
        dims=dem.dims,
        attrs={
            'long_name': 'Terrain Ruggedness Index',
            'units': 'meters',
            'radius': radius,
            'description': 'Higher values = more rugged terrain',
        }
    )
    
    return result


def calculate_terrain_metrics(dem: xr.DataArray) -> TerrainMetrics:
    """
    Calculate comprehensive terrain metrics.
    
    Args:
        dem: Digital Elevation Model DataArray
        
    Returns:
        TerrainMetrics object with all derivatives
    """
    return TerrainMetrics(
        slope=calculate_slope(dem),
        aspect=calculate_aspect(dem),
        hillshade=calculate_hillshade(dem),
        curvature=calculate_curvature(dem),
        TPI=calculate_tpi(dem),
        TRI=calculate_tri(dem),
    )


def detect_slopes(
    dem: xr.DataArray,
    min_slope: float = 15,
    max_slope: float = 60
) -> xr.DataArray:
    """
    Detect slopes within specified range.
    
    Useful for identifying terrain suitable for mining operations.
    
    Args:
        dem: Digital Elevation Model DataArray
        min_slope: Minimum slope threshold in degrees
        max_slope: Maximum slope threshold in degrees
        
    Returns:
        Boolean mask of slopes in range
    """
    slope = calculate_slope(dem)
    
    mask = (slope >= min_slope) & (slope <= max_slope)
    mask.attrs['long_name'] = 'Slope Detection'
    mask.attrs['min_slope'] = min_slope
    mask.attrs['max_slope'] = max_slope
    
    return mask


def identify_pits_and_peaks(
    dem: xr.DataArray,
    threshold: float = 10
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Identify pits (depressions) and peaks (summits).
    
    Args:
        dem: Digital Elevation Model DataArray
        threshold: TPI threshold for detection
        
    Returns:
        Tuple of (pit_mask, peak_mask)
    """
    tpi = calculate_tpi(dem, radius=5)
    
    pits = tpi < -threshold
    peaks = tpi > threshold
    
    pits.attrs['long_name'] = 'Pit Detection'
    peaks.attrs['long_name'] = 'Peak Detection'
    
    return pits, peaks


def _get_pixel_size(data: xr.DataArray, axis: str) -> float:
    """
    Get pixel size for a coordinate axis.
    
    Args:
        data: DataArray with coordinates
        axis: 'x' or 'y'
        
    Returns:
        Pixel size in meters
    """
    coord = data.coords[axis].values
    
    if len(coord) < 2:
        return 1.0
    
    diffs = np.diff(coord)
    
    # Check if values look like lat/lon (degrees) or meters
    if np.all(np.abs(diffs) < 1):  # Likely degrees
        # Approximate conversion at mid-latitude
        lat = np.mean(coord) if axis == 'y' else np.mean(data.coords['y'].values)
        meters_per_degree = 111320 * np.cos(np.radians(lat))
        
        if axis == 'x':
            return np.median(diffs) * meters_per_degree
        else:
            return np.median(diffs) * 111320
    else:
        # Already in meters
        return np.median(np.abs(diffs))


def _convolve(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply convolution with handling for edges.
    
    Args:
        arr: Input array
        kernel: Convolution kernel
        
    Returns:
        Convolved array
    """
    result = np.zeros_like(arr)
    
    k_rows, k_cols = kernel.shape
    offset_r = k_rows // 2
    offset_c = k_cols // 2
    
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # Get window
            r1 = max(0, i - offset_r)
            r2 = min(arr.shape[0], i + offset_r + 1)
            c1 = max(0, j - offset_c)
            c2 = min(arr.shape[1], j + offset_c + 1)
            
            # Get kernel portion that fits
            kr1 = max(0, offset_r - i)
            kr2 = kr1 + (r2 - r1)
            kc1 = max(0, offset_c - j)
            kc2 = kc1 + (c2 - c1)
            
            window = arr[r1:r2, c1:c2]
            k = kernel[kr1:kr2, kc1:kc2]
            
            if window.shape == k.shape:
                result[i, j] = np.sum(window * k)
            else:
                result[i, j] = arr[i, j]
    
    return result


def profile_curvature(dem: xr.DataArray, direction: str = 'x') -> xr.DataArray:
    """
    Calculate profile curvature along a specific direction.
    
    Args:
        dem: Digital Elevation Model DataArray
        direction: 'x', 'y', or 'maximum'
        
    Returns:
        Profile curvature DataArray
    """
    dy = _get_pixel_size(dem, 'y')
    dx = _get_pixel_size(dem, 'x')
    dy = abs(dy)
    
    z = dem.values.astype(np.float64)
    
    if direction == 'x':
        d2z_dx2 = (np.roll(z, -1, axis=1) - 2*z + np.roll(z, 1, axis=1)) / (dx**2)
        return xr.DataArray(d2z_dx2, coords=dem.coords, dims=dem.dims)
    elif direction == 'y':
        d2z_dy2 = (np.roll(z, -1, axis=0) - 2*z + np.roll(z, 1, axis=0)) / (dy**2)
        return xr.DataArray(d2z_dy2, coords=dem.coords, dims=dem.dims)
    else:
        # Maximum curvature
        d2z_dx2 = (np.roll(z, -1, axis=1) - 2*z + np.roll(z, 1, axis=1)) / (dx**2)
        d2z_dy2 = (np.roll(z, -1, axis=0) - 2*z + np.roll(z, 1, axis=0)) / (dy**2)
        return xr.DataArray(np.maximum(d2z_dx2, d2z_dy2), coords=dem.coords, dims=dem.dims)
