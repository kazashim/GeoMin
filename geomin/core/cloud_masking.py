"""
Cloud masking utilities for GeoMin.
Provides cloud detection and removal for satellite imagery.
"""

from typing import Optional, Union, Dict, Any
from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class CloudMaskResult:
    """Result of cloud masking operation."""
    mask: xr.DataArray
    cloud_probability: Optional[xr.DataArray] = None
    statistics: Dict[str, Any] = None


class CloudMasker:
    """
    Cloud detection and masking for satellite imagery.
    
    Supports multiple algorithms:
    - Simple threshold-based masking
    - Band ratio methods
    - Sentinel-2 specific cloud detection
    - Landsat QA band parsing
    """
    
    # Sentinel-2 cloud detection thresholds
    SENTINEL2_THRESHOLDS = {
        'blue_threshold': 0.3,  # High blue reflectance
        'nir_threshold': 0.4,   # Low NIR in clouds
        'swir_ratio_threshold': 0.75,  # SWIR1/SWIR2 ratio
    }
    
    # Landsat QA bit masks
    LANDSAT_QA_FLAGS = {
        'dilated_cloud': (1 << 1),
        'cirrus': (1 << 2),
        'cloud': (1 << 3),
        'cloud_shadow': (1 << 4),
    }
    
    def __init__(self, algorithm: str = 'threshold', **kwargs):
        """
        Initialize cloud masker.
        
        Args:
            algorithm: Masking algorithm ('threshold', 'sentinel2', 'landsat_qa')
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.thresholds = kwargs.get('thresholds', self.SENTINEL2_THRESHOLDS.copy())
        
        if algorithm == 'sentinel2':
            self._configure_sentinel2(**kwargs)
    
    def _configure_sentinel2(self, **kwargs):
        """Configure Sentinel-2 specific settings."""
        self.thresholds.update({
            'blue_threshold': kwargs.get('blue_threshold', 0.3),
            'nir_threshold': kwargs.get('nir_threshold', 0.4),
            'swir_ratio_threshold': kwargs.get('swir_ratio_threshold', 0.75),
            'cloud_confidence': kwargs.get('cloud_confidence', 0.4),
        })
    
    def mask_clouds(
        self,
        data: xr.DataArray,
        bands: Optional[Dict[str, str]] = None
    ) -> CloudMaskResult:
        """
        Detect and mask clouds in satellite imagery.
        
        Args:
            data: Input DataArray with spectral bands
            bands: Mapping of standard band names to data band names
                  e.g., {'blue': 'B02', 'nir': 'B08', 'swir1': 'B11', 'swir2': 'B12'}
            
        Returns:
            CloudMaskResult with mask and statistics
        """
        if self.algorithm == 'threshold':
            return self._mask_threshold(data, bands)
        elif self.algorithm == 'sentinel2':
            return self._mask_sentinel2(data, bands)
        elif self.algorithm == 'landsat_qa':
            return self._mask_landsat_qa(data, bands)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _mask_threshold(
        self,
        data: xr.DataArray,
        bands: Optional[Dict[str, str]] = None
    ) -> CloudMaskResult:
        """
        Threshold-based cloud detection.
        
        Uses blue band threshold and NIR/SWIR ratio.
        """
        if bands is None:
            bands = {'blue': 'B02', 'green': 'B03', 'red': 'B04', 'nir': 'B08'}
        
        # Get required bands
        blue = self._get_band(data, bands.get('blue', 'B02'))
        nir = self._get_band(data, bands.get('nir', 'B08'))
        swir1 = self._get_band(data, bands.get('swir1', 'B11'))
        swir2 = self._get_band(data, bands.get('swir2', 'B12'))
        
        # Cloud detection criteria
        cloud_mask = np.zeros_like(blue.values, dtype=bool)
        
        # High blue reflectance
        cloud_mask |= blue.values > self.thresholds['blue_threshold']
        
        # Low NIR reflectance
        cloud_mask |= nir.values < self.thresholds['nir_threshold']
        
        # SWIR ratio (clouds have different SWIR characteristics)
        with np.errstate(divide='ignore', invalid='ignore'):
            swir_ratio = swir1.values / (swir2.values + 1e-6)
            cloud_mask |= swir_ratio > self.thresholds['swir_ratio_threshold']
        
        # Create mask DataArray
        mask = xr.DataArray(
            cloud_mask,
            coords=blue.coords,
            dims=blue.dims,
            attrs={'long_name': 'Cloud Mask', 'cloud_mask': True}
        )
        
        # Calculate statistics
        total_pixels = cloud_mask.size
        cloud_pixels = np.sum(cloud_mask)
        cloud_percentage = (cloud_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        statistics = {
            'total_pixels': int(total_pixels),
            'cloud_pixels': int(cloud_pixels),
            'cloud_percentage': float(cloud_percentage),
            'algorithm': 'threshold',
        }
        
        return CloudMaskResult(mask=mask, statistics=statistics)
    
    def _mask_sentinel2(
        self,
        data: xr.DataArray,
        bands: Optional[Dict[str, str]] = None
    ) -> CloudMaskResult:
        """
        Sentinel-2 specific cloud detection.
        
        Uses band combinations optimized for Sentinel-2 spectral characteristics.
        Includes blue reflectance, band ratio, and brightness criteria.
        """
        if bands is None:
            bands = {
                'blue': 'B02',
                'green': 'B03', 
                'red': 'B04',
                'nir': 'B08',
                'swir1': 'B11',
                'swir2': 'B12',
                'cirrus': 'B10'
            }
        
        # Get bands
        blue = self._get_band(data, bands.get('blue', 'B02'))
        green = self._get_band(data, bands.get('green', 'B03'))
        red = self._get_band(data, bands.get('red', 'B04'))
        nir = self._get_band(data, bands.get('nir', 'B08'))
        swir1 = self._get_band(data, bands.get('swir1', 'B11'))
        swir2 = self._get_band(data, bands.get('swir2', 'B12'))
        cirrus = self._get_band(data, bands.get('cirrus', 'B10'))
        
        # Initialize cloud mask
        cloud_mask = np.zeros_like(blue.values, dtype=bool)
        
        # 1. High blue reflectance (clouds reflect strongly in blue)
        cloud_mask |= blue.values > self.thresholds['blue_threshold']
        
        # 2. Cirrus detection (high B10 values indicate cirrus clouds)
        cirrus_threshold = np.percentile(cirrus.values, 95) * 0.5
        cloud_mask |= cirrus.values > max(cirrus_threshold, 0.01)
        
        # 3. Whiteness (low variance between visible bands indicates clouds)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_vis = (blue.values + green.values + red.values) / 3
            std_vis = np.sqrt(
                ((blue.values - mean_vis)**2 + 
                 (green.values - mean_vis)**2 + 
                 (red.values - mean_vis)**2) / 3
            )
            # Low relative variability = cloud
            relative_std = std_vis / (mean_vis + 1e-6)
            cloud_mask |= (relative_std < 0.15) & (mean_vis > 0.2)
        
        # 4. Ratio criteria
        swir_ratio = swir1.values / (swir2.values + 1e-6)
        cloud_mask |= swir_ratio > self.thresholds['swir_ratio_threshold']
        
        # 5. NIR/Red ratio (vegetation has high NIR/Red, clouds don't)
        nir_red_ratio = nir.values / (red.values + 1e-6)
        cloud_mask |= (nir_red_ratio < 0.8) & (blue.values > 0.25)
        
        # Create masks
        mask = xr.DataArray(
            cloud_mask,
            coords=blue.coords,
            dims=blue.dims,
            attrs={'long_name': 'Cloud Mask (Sentinel-2)'}
        )
        
        # Calculate cloud probability for visualization
        cloud_prob = self._calculate_cloud_probability(
            blue, green, red, nir, swir1, swir2, cirrus
        )
        
        # Statistics
        total_pixels = cloud_mask.size
        cloud_pixels = np.sum(cloud_mask)
        cloud_percentage = (cloud_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        statistics = {
            'total_pixels': int(total_pixels),
            'cloud_pixels': int(cloud_pixels),
            'cloud_percentage': float(cloud_percentage),
            'algorithm': 'sentinel2',
            'thresholds': self.thresholds,
        }
        
        return CloudMaskResult(
            mask=mask,
            cloud_probability=cloud_prob,
            statistics=statistics
        )
    
    def _mask_landsat_qa(
        self,
        data: xr.DataArray,
        bands: Optional[Dict[str, str]] = None
    ) -> CloudMaskResult:
        """
        Landsat QA band-based cloud detection.
        
        Uses the quality assessment band for reliable cloud detection.
        """
        if bands is None:
            bands = {'qa': 'QA'}
        
        qa = self._get_band(data, bands.get('qa', 'QA'))
        qa_values = qa.values.astype(np.uint16)
        
        # Create cloud mask from QA bits
        cloud_mask = np.zeros_like(qa_values, dtype=bool)
        
        # Check cloud bit (bit 3)
        cloud_mask |= (qa_values & self.LANDSAT_QA_FLAGS['cloud']) != 0
        
        # Check cloud shadow bit (bit 4)
        cloud_mask |= (qa_values & self.LANDSAT_QA_FLAGS['cloud_shadow']) != 0
        
        # Check dilated cloud bit (bit 1)
        cloud_mask |= (qa_values & self.LANDSAT_QA_FLAGS['dilated_cloud']) != 0
        
        # Check cirrus bit (bit 2)
        cloud_mask |= (qa_values & self.LANDSAT_QA_FLAGS['cirrus']) != 0
        
        mask = xr.DataArray(
            cloud_mask,
            coords=qa.coords,
            dims=qa.dims,
            attrs={'long_name': 'Cloud Mask (Landsat QA)'}
        )
        
        statistics = {
            'total_pixels': int(cloud_mask.size),
            'cloud_pixels': int(np.sum(cloud_mask)),
            'cloud_percentage': float(np.sum(cloud_mask) / cloud_mask.size * 100),
            'algorithm': 'landsat_qa',
        }
        
        return CloudMaskResult(mask=mask, statistics=statistics)
    
    def _calculate_cloud_probability(
        self,
        blue: xr.DataArray,
        green: xr.DataArray,
        red: xr.DataArray,
        nir: xr.DataArray,
        swir1: xr.DataArray,
        swir2: xr.DataArray,
        cirrus: xr.DataArray
    ) -> xr.DataArray:
        """Calculate per-pixel cloud probability score."""
        
        # Normalize bands
        blue_norm = blue.values / (np.nanmax(blue.values) + 1e-6)
        cirrus_norm = cirrus.values / (np.nanmax(cirrus.values) + 1e-6)
        
        # Combined score (0-1 range)
        probability = (
            0.4 * np.clip(blue_norm, 0, 1) +
            0.4 * np.clip(cirrus_norm, 0, 1) +
            0.2 * np.where(swir1.values > swir2.values, 1, 0)
        )
        
        return xr.DataArray(
            probability,
            coords=blue.coords,
            dims=blue.dims,
            attrs={'long_name': 'Cloud Probability', 'range': '0-1'}
        )
    
    def _get_band(self, data: xr.DataArray, band_name: str) -> xr.DataArray:
        """Extract band from DataArray."""
        if 'band' in data.coords:
            band_names = [str(b) for b in data.coords['band'].values]
            if band_name in band_names:
                return data.sel(band=band_name)
            try:
                idx = int(band_name) - 1
                if 0 <= idx < data.sizes['band']:
                    return data.isel(band=idx)
            except ValueError:
                pass
        
        if data.ndim == 2:
            return data
        
        raise ValueError(f"Band {band_name} not found")
    
    def apply_mask(
        self,
        data: xr.DataArray,
        mask: xr.DataArray,
        fill_value: float = np.nan
    ) -> xr.DataArray:
        """
        Apply cloud mask to data.
        
        Args:
            data: Input DataArray
            mask: Cloud mask (True = cloud)
            fill_value: Value to use for masked pixels
            
        Returns:
            Masked DataArray
        """
        masked = data.copy()
        
        # Handle multi-band data
        if 'band' in data.coords:
            for band in data.coords['band'].values:
                masked_data = masked.sel(band=band).values
                masked_data = np.where(mask.values, fill_value, masked_data)
                masked.loc[dict(band=band)] = masked_data
        else:
            masked.values = np.where(mask.values, fill_value, data.values)
        
        return masked
    
    def get_clear_pixels(self, data: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
        """
        Return only clear pixels (invert mask).
        
        Args:
            data: Input DataArray
            mask: Cloud mask
            
        Returns:
            DataArray with clouds set to NaN
        """
        return self.apply_mask(data, mask, fill_value=np.nan)
