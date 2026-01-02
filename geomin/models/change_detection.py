"""
Change detection algorithms for mining activity monitoring.
Identifies surface changes indicative of mining operations.
"""

from typing import Union, Optional, Tuple, Dict, List
from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Type alias
XRData = Union[xr.DataArray, xr.Dataset]


@dataclass
class ChangeResult:
    """
    Result of change detection analysis.
    
    Attributes:
        change_map: Binary mask of changed areas
        change_intensity: Intensity of change (0-1)
        change_magnitude: Magnitude of change
        change_date: Estimated date of change
        statistics: Statistics about detected changes
    """
    change_map: xr.DataArray
    change_intensity: xr.DataArray
    change_magnitude: xr.DataArray
    change_date: Optional[xr.DataArray] = None
    statistics: Dict = None


def simple_difference(
    image1: xr.DataArray,
    image2: xr.DataArray,
    threshold: float = 0.1
) -> ChangeResult:
    """
    Simple image differencing for change detection.
    
    Computes absolute difference between two images.
    
    Args:
        image1: Earlier image
        image2: Later image
        threshold: Change detection threshold
        
    Returns:
        ChangeResult with detection results
    """
    # Ensure images are aligned
    if not _images_aligned(image1, image2):
        image2 = _align_images(image1, image2)
    
    # Calculate difference
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = np.abs(image2.values - image1.values)
        
        # Normalize to 0-1 range
        max_diff = np.nanmax(diff)
        if max_diff > 0:
            intensity = diff / max_diff
        else:
            intensity = diff
        
        # Apply threshold
        change_map = xr.DataArray(
            intensity > threshold,
            coords=image1.coords,
            dims=image1.dims,
        )
    
    return ChangeResult(
        change_map=change_map,
        change_intensity=xr.DataArray(intensity, coords=image1.coords, dims=image1.dims),
        change_magnitude=xr.DataArray(diff, coords=image1.coords, dims=image1.dims),
        statistics=_compute_change_stats(change_map, intensity),
    )


def ratio_difference(
    image1: xr.DataArray,
    image2: xr.DataArray,
    threshold: float = 0.2
) -> ChangeResult:
    """
    Ratio-based change detection.
    
    Computes ratio of images to detect changes.
    
    Args:
        image1: Earlier image
        image2: Later image
        threshold: Change detection threshold
        
    Returns:
        ChangeResult with detection results
    """
    if not _images_aligned(image1, image2):
        image2 = _align_images(image1, image2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = image2.values / (image1.values + 1e-6)
        
        # Deviation from 1 indicates change
        deviation = np.abs(ratio - 1)
        
        # Clip and normalize
        intensity = np.clip(deviation, 0, 2) / 2
        
        change_map = xr.DataArray(
            intensity > threshold,
            coords=image1.coords,
            dims=image1.dims,
        )
    
    return ChangeResult(
        change_map=change_map,
        change_intensity=xr.DataArray(intensity, coords=image1.coords, dims=image1.dims),
        change_magnitude=xr.DataArray(deviation, coords=image1.coords, dims=image1.dims),
        statistics=_compute_change_stats(change_map, intensity),
    )


def vegetation_change_detector(
    image1: xr.DataArray,
    image2: xr.DataArray,
    threshold: float = -0.1
) -> ChangeResult:
    """
    Detect vegetation loss using NDVI.
    
    Useful for identifying deforestation or land clearing for mining.
    
    Args:
        image1: Earlier image
        image2: Later image
        threshold: NDVI change threshold
        
    Returns:
        ChangeResult with vegetation change detection
    """
    # Calculate NDVI for both images
    from ..algorithms.spectral import ndvi
    
    try:
        ndvi1 = ndvi(image1)
        ndvi2 = ndvi(image2)
        
        # NDVI decrease indicates vegetation loss
        change = ndvi2 - ndvi1
        
        # Negative change = vegetation loss
        change_map = change < threshold
        
        # Normalize intensity
        intensity = np.clip(-change, 0, 1)
        
        return ChangeResult(
            change_map=change_map,
            change_intensity=intensity,
            change_magnitude=change,
            statistics=_compute_change_stats(change_map, intensity),
        )
    except Exception as e:
        print(f"Vegetation change detection failed: {e}")
        return simple_difference(image1, image2, threshold)


def pca_change_detector(
    image1: xr.DataArray,
    image2: xr.DataArray,
    n_components: int = 3
) -> ChangeResult:
    """
    PCA-based change detection.
    
    Uses principal component analysis to detect changes.
    
    Args:
        image1: Earlier image (multi-band)
        image2: Later image (multi-band)
        n_components: Number of PCA components
        
    Returns:
        ChangeResult with detection results
    """
    if not _images_aligned(image1, image2):
        image2 = _align_images(image1, image2)
    
    # Stack images along a new dimension
    stacked = xr.concat([image1, image2], dim='time')
    
    # Reshape for PCA
    n_bands = stacked.sizes.get('band', 1)
    n_time = stacked.sizes.get('time', 2)
    n_pixels = np.prod([stacked.sizes[d] for d in stacked.dims if d not in ['band', 'time']])
    
    # Get data as 2D array
    data_2d = stacked.values.reshape(n_bands * n_time, n_pixels)
    
    # Remove NaN
    valid_mask = np.all(np.isfinite(data_2d), axis=0)
    data_valid = data_2d[:, valid_mask]
    
    # Standardize
    data_mean = np.mean(data_valid, axis=1, keepdims=True)
    data_std = np.std(data_valid, axis=1, keepdims=True)
    data_std[data_std == 0] = 1
    data_normalized = (data_valid - data_mean) / data_std
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, min(data_normalized.shape)))
    pca_result = pca.fit_transform(data_normalized)
    
    # The last component often captures change
    change_component = pca_result[-1, :]
    
    # Reconstruct full array
    change_2d = np.zeros(n_pixels)
    change_2d[valid_mask] = change_component
    change_array = change_2d.reshape(
        [stacked.sizes[d] for d in stacked.dims if d not in ['band', 'time']]
    )
    
    # Normalize
    intensity = (change_array - np.nanmin(change_array)) / (np.nanmax(change_array) - np.nanmin(change_array) + 1e-6)
    
    # Threshold for change detection
    threshold = np.nanmean(intensity) + np.nanstd(intensity)
    change_map = intensity > threshold
    
    # Create DataArrays with correct coordinates
    result_coords = {k: v for k, v in image1.coords.items() if k in image1.dims}
    change_map_da = xr.DataArray(change_map, coords=result_coords, dims=[d for d in image1.dims if d != 'band'])
    intensity_da = xr.DataArray(intensity, coords=result_coords, dims=[d for d in image1.dims if d != 'band'])
    magnitude_da = xr.DataArray(change_array, coords=result_coords, dims=[d for d in image1.dims if d != 'band'])
    
    return ChangeResult(
        change_map=change_map_da,
        change_intensity=intensity_da,
        change_magnitude=magnitude_da,
        statistics=_compute_change_stats(change_map_da, intensity_da),
    )


def kmeans_change_detector(
    image1: xr.DataArray,
    image2: xr.DataArray,
    n_clusters: int = 4
) -> ChangeResult:
    """
    K-means clustering for change detection.
    
    Clusters both images and identifies class transitions.
    
    Args:
        image1: Earlier image
        image2: Later image
        n_clusters: Number of clusters
        
    Returns:
        ChangeResult with detection results
    """
    if not _images_aligned(image1, image2):
        image2 = _align_images(image1, image2)
    
    # Stack images
    stacked = xr.concat([image1, image2], dim='time')
    
    # Reshape for clustering
    n_pixels = np.prod([stacked.sizes[d] for d in stacked.dims if d not in ['band', 'time']])
    data_2d = stacked.values.reshape(-1, n_pixels).T
    
    # Handle NaN
    valid_mask = np.all(np.isfinite(data_2d), axis=1)
    data_valid = data_2d[valid_mask]
    
    # Standardize
    data_mean = np.mean(data_valid, axis=0)
    data_std = np.std(data_valid, axis=0)
    data_std[data_std == 0] = 1
    data_normalized = (data_valid - data_mean) / data_std
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_normalized)
    
    # Detect changes based on label changes
    # First half of features = image1, second half = image2
    n_features = data_valid.shape[1] // 2
    labels1 = labels[:n_pixels // 2]
    labels2 = labels[n_pixels // 2:]
    
    change_mask = labels1 != labels2
    
    # Reconstruct full array
    change_2d = np.zeros(n_pixels, dtype=bool)
    change_2d[valid_mask] = change_mask
    
    # Get coordinates
    result_coords = {k: v for k, v in image1.coords.items() if k in image1.dims}
    shape = [image1.sizes[d] for d in image1.dims if d != 'band']
    change_map = xr.DataArray(
        change_2d.reshape(shape),
        coords=result_coords,
        dims=[d for d in image1.dims if d != 'band'],
    )
    
    return ChangeResult(
        change_map=change_map,
        change_intensity=change_map.astype(float),
        change_magnitude=change_map.astype(float),
        statistics=_compute_change_stats(change_map, change_map.astype(float)),
    )


def time_series_change_detector(
    images: List[xr.DataArray],
    threshold: float = 0.2
) -> ChangeResult:
    """
    Multi-temporal change detection.
    
    Analyzes a time series of images to detect changes.
    
    Args:
        images: List of images sorted by time
        threshold: Change detection threshold
        
    Returns:
        ChangeResult with detection results
    """
    if len(images) < 2:
        raise ValueError("At least 2 images required")
    
    # Stack all images
    stacked = xr.concat(images, dim='time')
    
    # Calculate mean and std over time
    mean_image = stacked.mean(dim='time')
    std_image = stacked.std(dim='time')
    
    # Detect pixels that deviate significantly
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate deviation from mean
        last_image = images[-1]
        deviation = np.abs(last_image.values - mean_image.values)
        
        # Normalize by local variability
        normalized = deviation / (std_image.values + 1e-6)
        
        # Apply threshold
        intensity = np.clip(normalized, 0, 5) / 5
        change_map = intensity > threshold
    
    return ChangeResult(
        change_map=xr.DataArray(change_map, coords=last_image.coords, dims=last_image.dims),
        change_intensity=xr.DataArray(intensity, coords=last_image.coords, dims=last_image.dims),
        change_magnitude=xr.DataArray(deviation, coords=last_image.coords, dims=last_image.dims),
        statistics=_compute_change_stats(
            xr.DataArray(change_map, coords=last_image.coords, dims=last_image.dims),
            xr.DataArray(intensity, coords=last_image.coords, dims=last_image.dims)
        ),
    )


def detect_mining_activity(
    image1: xr.DataArray,
    image2: xr.DataArray,
    method: str = 'vegetation'
) -> ChangeResult:
    """
    Detect mining activity between two images.
    
    Combines multiple indicators:
    - Vegetation loss
    - Soil exposure
    - New infrastructure
    
    Args:
        image1: Earlier image
        image2: Later image
        method: Detection method
        
    Returns:
        ChangeResult with mining activity detection
    """
    if method == 'vegetation':
        result = vegetation_change_detector(image1, image2)
    elif method == 'pca':
        result = pca_change_detector(image1, image2)
    elif method == 'kmeans':
        result = kmeans_change_detector(image1, image2)
    else:
        result = simple_difference(image1, image2)
    
    return result


def classify_change_type(
    image1: xr.DataArray,
    image2: xr.DataArray,
    change_mask: xr.DataArray
) -> Dict[str, xr.DataArray]:
    """
    Classify types of detected changes.
    
    Identifies:
    - Vegetation loss
    - Water body changes
    - Urban development
    - Mine pit expansion
    
    Args:
        image1: Earlier image
        image2: Later image
        change_mask: Binary mask of changed areas
        
    Returns:
        Dictionary of change types to boolean masks
    """
    from ..algorithms.spectral import ndvi, ndwi
    
    classifications = {}
    
    # Vegetation loss
    try:
        ndvi1 = ndvi(image1)
        ndvi2 = ndvi(image2)
        ndvi_change = ndvi2 - ndvi1
        vegetation_loss = (ndvi_change < -0.1) & change_mask
        vegetation_loss.attrs['long_name'] = 'Vegetation Loss'
        classifications['vegetation_loss'] = vegetation_loss
    except Exception:
        pass
    
    # Water changes
    try:
        ndwi1 = ndwi(image1)
        ndwi2 = ndwi(image2)
        water_change = (ndwi2 - ndwi1) > 0.2
        water_change = water_change & change_mask
        water_change.attrs['long_name'] = 'Water Body Changes'
        classifications['water_change'] = water_change
    except Exception:
        pass
    
    # Bare ground/soil exposure
    try:
        # Low NDVI with high brightness
        brightness1 = (image1.sel(band='B04') + image1.sel(band='B03') + image1.sel(band='B02')) / 3
        brightness2 = (image2.sel(band='B04') + image2.sel(band='B03') + image2.sel(band='B02')) / 3
        
        bare_ground = (ndvi2 < 0.2) & (bright2 > brightness1) & change_mask
        bare_ground.attrs['long_name'] = 'Bare Ground / Soil Exposure'
        classifications['bare_ground'] = bare_ground
    except Exception:
        pass
    
    return classifications


def _images_aligned(img1: xr.DataArray, img2: xr.DataArray) -> bool:
    """Check if two images are spatially aligned."""
    for dim in ['x', 'y']:
        if dim in img1.coords and dim in img2.coords:
            if not np.allclose(img1.coords[dim].values, img2.coords[dim].values):
                return False
    return True


def _align_images(target: xr.DataArray, source: xr.DataArray) -> xr.DataArray:
    """Align source image to target image grid."""
    # This is a simplified implementation
    # A full implementation would use interpolation
    return source


def _compute_change_stats(
    change_map: xr.DataArray,
    intensity: xr.DataArray
) -> Dict:
    """Compute statistics for detected changes."""
    change_values = change_map.values
    intensity_values = intensity.values
    
    total_pixels = change_values.size
    changed_pixels = np.nansum(change_values)
    
    if changed_pixels == 0:
        return {
            'total_changed_pixels': 0,
            'change_percentage': 0.0,
            'mean_intensity': 0.0,
            'max_intensity': 0.0,
        }
    
    return {
        'total_changed_pixels': int(changed_pixels),
        'change_percentage': float(changed_pixels / total_pixels * 100),
        'mean_intensity': float(np.nanmean(intensity_values[change_values])),
        'max_intensity': float(np.nanmax(intensity_values)),
    }


def calculate_change_area(
    change_map: xr.DataArray,
    pixel_area_km2: float = 0.0001
) -> float:
    """
    Calculate total area of detected changes.
    
    Args:
        change_map: Binary change mask
        pixel_area_km2: Area of one pixel in km2
        
    Returns:
        Total changed area in km2
    """
    changed_pixels = np.nansum(change_map.values)
    return float(changed_pixels * pixel_area_km2)
