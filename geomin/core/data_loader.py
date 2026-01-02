"""
Data loading utilities for GeoMin.
Handles various satellite data formats including GeoTIFF, NetCDF, and HDF5.
"""

from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import json

import numpy as np
import xarray as xr
import rasterio
from rasterio.io import MemoryFile
import rioxarray
import geopandas as gpd
from shapely.geometry import box
import dask.array as da

from .config import get_config
from .crs import detect_crs, reproject_xarray, transform_bbox


@dataclass
class BandInfo:
    """Information about a spectral band."""
    name: str
    wavelength: Optional[float] = None
    wavelength_unit: Optional[str] = None
    band_id: Optional[str] = None
    resolution: Optional[float] = None
    application: Optional[str] = None


# Standard band definitions for common satellites
SENTINEL2_BANDS = {
    'B01': BandInfo('Coastal Aerosol', 0.443, 'μm', 'coastal', 60, 'Aerosol detection'),
    'B02': BandInfo('Blue', 0.492, 'μm', 'blue', 10, 'Soil/vegetation discrimination'),
    'B03': BandInfo('Green', 0.560, 'μm', 'green', 10, 'Vegetation peak reflectance'),
    'B04': BandInfo('Red', 0.665, 'μm', 'red', 10, 'Vegetation chlorophyll absorption'),
    'B05': BandInfo('Red Edge 1', 0.705, 'μm', 'red_edge_1', 20, 'Vegetation stress monitoring'),
    'B06': BandInfo('Red Edge 2', 0.740, 'μm', 'red_edge_2', 20, 'Vegetation stress monitoring'),
    'B07': BandInfo('Red Edge 3', 0.783, 'μm', 'red_edge_3', 20, 'Vegetation stress monitoring'),
    'B08': BandInfo('NIR', 0.842, 'μm', 'nir', 10, 'Biomass content'),
    'B8A': BandInfo('Red Edge 4', 0.865, 'μm', 'red_edge_4', 20, 'Leaf chlorophyll'),
    'B09': BandInfo('Water Vapor', 0.945, 'μm', 'water_vapor', 60, 'Atmospheric water vapor'),
    'B10': BandInfo('Cirrus', 1.375, 'μm', 'cirrus', 60, 'Cirrus cloud detection'),
    'B11': BandInfo('SWIR 1', 1.610, 'μm', 'swir_1', 20, 'Soil/vegetation moisture'),
    'B12': BandInfo('SWIR 2', 2.190, 'μm', 'swir_2', 20, 'Soil/vegetation moisture'),
}

LANDSAT8_BANDS = {
    'B1': BandInfo('Coastal Aerosol', 0.44, 'μm', 'coastal', 30, 'Coastal/aerosol studies'),
    'B2': BandInfo('Blue', 0.48, 'μm', 'blue', 30, 'Bathymetric mapping'),
    'B3': BandInfo('Green', 0.56, 'μm', 'green', 30, 'Peak vegetation'),
    'B4': BandInfo('Red', 0.65, 'μm', 'red', 30, 'Vegetation slopes'),
    'B5': BandInfo('NIR', 0.86, 'μm', 'nir', 30, 'Biomass content'),
    'B6': BandInfo('SWIR 1', 1.57, 'μm', 'swir_1', 30, 'Soil/vegetation moisture'),
    'B7': BandInfo('SWIR 2', 2.20, 'μm', 'swir_2', 30, 'Soil/vegetation moisture'),
    'B8': BandInfo('Pan', 0.59, 'μm', 'pan', 15, 'Sharpen multispectral'),
    'B9': BandInfo('Cirrus', 1.37, 'μm', 'cirrus', 30, 'Cirrus cloud detection'),
    'B10': BandInfo('TIRS 1', 10.9, 'μm', 'thermal', 100, 'Thermal mapping'),
    'B11': BandInfo('TIRS 2', 12.0, 'μm', 'thermal', 100, 'Thermal mapping'),
}


class DataLoader:
    """
    Universal data loader for satellite imagery.
    
    Supports multiple formats including GeoTIFF, NetCDF, and cloud-optimized formats.
    Provides lazy loading with dask for efficient memory management.
    """
    
    SUPPORTED_FORMATS = ['.tif', '.tiff', '.nc', '.h5', '.hdf5', '.jp2']
    
    @staticmethod
    def load(
        source: Union[str, Path, xr.DataArray],
        bands: Optional[List[str]] = None,
        mask_clouds: bool = False,
        reproject_to: Optional[str] = None,
        target_resolution: Optional[float] = None
    ) -> xr.DataArray:
        """
        Load satellite data from file or xarray object.
        
        Args:
            source: File path or xarray DataArray
            bands: Optional list of bands to load
            mask_clouds: Whether to apply cloud masking
            reproject_to: Optional target CRS
            target_resolution: Optional target resolution in meters
            
        Returns:
            xarray DataArray with geospatial metadata
        """
        if isinstance(source, (str, Path)):
            return DataLoader._load_from_file(
                source, bands, mask_clouds, reproject_to, target_resolution
            )
        elif isinstance(source, xr.DataArray):
            return DataLoader._process_xarray(
                source, bands, mask_clouds, reproject_to, target_resolution
            )
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
    
    @staticmethod
    def _load_from_file(
        filepath: Union[str, Path],
        bands: Optional[List[str]] = None,
        mask_clouds: bool = False,
        reproject_to: Optional[str] = None,
        target_resolution: Optional[float] = None
    ) -> xr.DataArray:
        """
        Load data from a file.
        
        Args:
            filepath: Path to data file
            bands: Bands to load
            mask_clouds: Apply cloud masking
            reproject_to: Target CRS
            target_resolution: Target resolution
            
        Returns:
            Loaded DataArray
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix in ['.tif', '.tiff', '.jp2']:
            return DataLoader._load_geotiff(
                filepath, bands, mask_clouds, reproject_to, target_resolution
            )
        elif suffix == '.nc':
            return DataLoader._load_netcdf(
                filepath, bands, mask_clouds, reproject_to, target_resolution
            )
        elif suffix in ['.h5', '.hdf5']:
            return DataLoader._load_hdf5(
                filepath, bands, mask_clouds, reproject_to, target_resolution
            )
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def _load_geotiff(
        filepath: Union[str, Path],
        bands: Optional[List[str]] = None,
        mask_clouds: bool = False,
        reproject_to: Optional[str] = None,
        target_resolution: Optional[float] = None
    ) -> xr.DataArray:
        """
        Load GeoTIFF file with rioxarray.
        
        Args:
            filepath: Path to GeoTIFF
            bands: Bands to load
            mask_clouds: Apply cloud masking
            reproject_to: Target CRS
            target_resolution: Target resolution
            
        Returns:
            DataArray with loaded bands
        """
        # Get config for dask chunks
        config = get_config()
        chunks = config.processing.dask_chunks
        
        # Open with rioxarray
        data = rioxarray.open_rasterio(filepath, chunks=chunks)
        
        # Select bands if specified
        if bands is not None:
            if isinstance(data.band, xr.IndexVariable):
                # Multi-band file - select by index
                band_indices = []
                for band in bands:
                    try:
                        idx = int(band) - 1
                        if 0 <= idx < data.sizes['band']:
                            band_indices.append(idx)
                    except ValueError:
                        # Try to find by band name
                        pass
                if band_indices:
                    data = data.isel(band=band_indices)
        
        # Reproject if needed
        if reproject_to:
            data = reproject_xarray(data, reproject_to, target_resolution)
        
        # Add band information as coordinate
        if 'band' in data.coords:
            data = data.assign_coords({
                'band': [f'B{i+1}' for i in range(data.sizes['band'])]
            })
        
        return data
    
    @staticmethod
    def _load_netcdf(
        filepath: Union[str, Path],
        bands: Optional[List[str]] = None,
        mask_clouds: bool = False,
        reproject_to: Optional[str] = None,
        target_resolution: Optional[float] = None
    ) -> xr.DataArray:
        """
        Load NetCDF file.
        
        Args:
            filepath: Path to NetCDF
            bands: Bands to load
            mask_clouds: Apply cloud masking
            reproject_to: Target CRS
            target_resolution: Target resolution
            
        Returns:
            DataArray with loaded data
        """
        config = get_config()
        chunks = config.processing.dask_chunks
        
        # Open with xarray
        data = xr.open_dataset(filepath, chunks=chunks)
        
        # Convert to DataArray if needed
        if isinstance(data, xr.Dataset):
            # Select first data variable as the primary array
            var_name = list(data.data_vars.keys())[0]
            data = data[var_name]
        
        # Convert to DataArray format
        data = data.to_array()
        
        # Select bands if specified
        if bands is not None:
            band_indices = [i for i, b in enumerate(data.coords['variable'].values) if b in bands]
            if band_indices:
                data = data.isel(variable=band_indices)
        
        # Set CRS if not present
        if not hasattr(data, 'rio') or not data.rio.crs:
            data = data.rio.write_crs(detect_crs(data))
        
        # Reproject if needed
        if reproject_to:
            data = reproject_xarray(data, reproject_to, target_resolution)
        
        return data
    
    @staticmethod
    def _load_hdf5(
        filepath: Union[str, Path],
        bands: Optional[List[str]] = None,
        mask_clouds: bool = False,
        reproject_to: Optional[str] = None,
        target_resolution: Optional[float] = None
    ) -> xr.DataArray:
        """
        Load HDF5 file (e.g., MODIS, VIIRS).
        
        Args:
            filepath: Path to HDF5 file
            bands: Bands to load
            mask_clouds: Apply cloud masking
            reproject_to: Target CRS
            target_resolution: Target resolution
            
        Returns:
            DataArray with loaded data
        """
        import h5py
        
        config = get_config()
        chunks = config.processing.dask_chunks
        
        # Open file
        with h5py.File(filepath, 'r') as f:
            # Get dataset structure (simplified - adapt for specific HDF5 schemas)
            def find_datasets(group, prefix=''):
                datasets = {}
                for key in group.keys():
                    item = group[f"{prefix}{key}"]
                    if isinstance(item, h5py.Dataset):
                        datasets[f"{prefix}{key}"] = item
                    elif isinstance(item, h5py.Group):
                        datasets.update(find_datasets(item, f"{prefix}{key}/"))
                return datasets
            
            datasets = find_datasets(f)
            
            if not datasets:
                raise ValueError("No datasets found in HDF5 file")
            
            # Select first suitable dataset
            dataset_name = list(datasets.keys())[0]
            raw_data = datasets[dataset_name][:]
            
            # Create DataArray
            data = xr.DataArray(
                raw_data,
                dims=['y', 'x'] if raw_data.ndim == 2 else ['band', 'y', 'x'],
                coords={
                    'y': np.arange(raw_data.shape[-2]),
                    'x': np.arange(raw_data.shape[-1]),
                }
            )
        
        # Set CRS
        if not hasattr(data, 'rio') or not data.rio.crs:
            data = data.rio.write_crs(detect_crs(data))
        
        # Reproject if needed
        if reproject_to:
            data = reproject_xarray(data, reproject_to, target_resolution)
        
        return data
    
    @staticmethod
    def _process_xarray(
        data: xr.DataArray,
        bands: Optional[List[str]] = None,
        mask_clouds: bool = False,
        reproject_to: Optional[str] = None,
        target_resolution: Optional[float] = None
    ) -> xr.DataArray:
        """
        Process existing xarray DataArray.
        
        Args:
            data: Input DataArray
            bands: Bands to select
            mask_clouds: Apply cloud masking
            reproject_to: Target CRS
            target_resolution: Target resolution
            
        Returns:
            Processed DataArray
        """
        # Select bands if specified
        if bands is not None and 'band' in data.coords:
            band_names = [b if isinstance(b, str) else f'B{b}' for b in bands]
            if 'band' in data.dims:
                indices = [i for i, b in enumerate(data.coords['band'].values) if b in band_names]
                if indices:
                    data = data.isel(band=indices)
        
        # Reproject if needed
        if reproject_to:
            data = reproject_xarray(data, reproject_to, target_resolution)
        
        return data
    
    @staticmethod
    def load_multiple(
        files: List[Union[str, Path]],
        align: bool = True,
        reproject_to: Optional[str] = None
    ) -> List[xr.DataArray]:
        """
        Load multiple files and optionally align them.
        
        Args:
            files: List of file paths
            align: Whether to align rasters to common grid
            reproject_to: Optional target CRS for alignment
            
        Returns:
            List of loaded DataArrays
        """
        data_arrays = [DataLoader.load(f) for f in files]
        
        if align and len(data_arrays) > 1:
            if reproject_to is None:
                reproject_to = detect_crs(data_arrays[0])
            
            aligned = []
            for data in data_arrays:
                if detect_crs(data) != reproject_to:
                    data = reproject_xarray(data, reproject_to)
                aligned.append(data)
            return aligned
        
        return data_arrays
    
    @staticmethod
    def create_stack(
        band_files: Dict[str, Union[str, Path]],
        stack_name: str = 'satellite_stack'
    ) -> xr.DataArray:
        """
        Create a multi-band stack from individual band files.
        
        Args:
            band_files: Dictionary mapping band names to file paths
            stack_name: Name for the stack
            
        Returns:
            DataArray with all bands stacked
        """
        bands = []
        band_coords = []
        
        for band_name, filepath in band_files.items():
            data = DataLoader.load(filepath)
            # Ensure 2D array for stacking
            if data.ndim == 3:
                data = data.squeeze()
            bands.append(data)
            band_coords.append(band_name)
        
        # Stack along new dimension
        stacked = xr.concat(bands, dim='band')
        stacked = stacked.assign_coords(band=band_coords)
        stacked.name = stack_name
        
        return stacked
    
    @staticmethod
    def get_raster_info(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a raster file without loading full data.
        
        Args:
            filepath: Path to raster file
            
        Returns:
            Dictionary with raster information
        """
        filepath = Path(filepath)
        
        with rasterio.open(filepath) as src:
            info = {
                'path': str(filepath),
                'driver': src.driver,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': str(src.dtypes[0]),
                'crs': src.crs.to_string() if src.crs else None,
                'bounds': src.bounds,
                'transform': list(src.transform),
                'nodata': src.nodata,
                'compression': src.compression if hasattr(src, 'compression') else None,
            }
            
            # Get band descriptions if available
            if src.descriptions:
                info['band_descriptions'] = list(src.descriptions)
            
            # Calculate resolution
            if src.transform[0] > 0:
                info['resolution_x'] = src.transform[0]
            if src.transform[4] < 0:
                info['resolution_y'] = abs(src.transform[4])
            
            # Calculate extent in square kilometers
            pixel_area = abs(src.transform[0]) * abs(src.transform[4]) * 1e-6
            info['total_area_km2'] = pixel_area * src.width * src.height
            
            return info
    
    @staticmethod
    def resample(
        data: xr.DataArray,
        target_resolution: float,
        method: str = 'bilinear'
    ) -> xr.DataArray:
        """
        Resample data to target resolution.
        
        Args:
            data: Input DataArray
            target_resolution: Target resolution in meters
            method: Resampling method
            
        Returns:
            Resampled DataArray
        """
        try:
            import rioxarray
        except ImportError:
            raise ImportError("rioxarray required for resampling")
        
        return data.rio.reproject(
            data.rio.crs,
            resolution=target_resolution,
            resampling=method
        )
    
    @staticmethod
    def clip_to_bbox(
        data: xr.DataArray,
        bbox: Tuple[float, float, float, float],
        crs: str = 'EPSG:4326'
    ) -> xr.DataArray:
        """
        Clip data to bounding box.
        
        Args:
            data: Input DataArray
            bbox: (minx, miny, maxx, maxy) bounding box
            crs: CRS of the bounding box
            
        Returns:
            Clipped DataArray
        """
        # Transform bbox to data CRS if needed
        data_crs = detect_crs(data)
        if data_crs != crs:
            transformed_bbox = transform_bbox(bbox, crs, data_crs)
        else:
            transformed_bbox = bbox
        
        # Clip using rioxarray
        return data.rio.clip_box(*transformed_bbox)
    
    @staticmethod
    def load_cloud_mask(
        filepath: Union[str, Path],
        cloud_classes: List[str] = ['cloud', 'cloud_shadow', 'cirrus']
    ) -> xr.DataArray:
        """
        Load cloud mask from QA file or cloud probability data.
        
        Args:
            filepath: Path to cloud mask file
            cloud_classes: Cloud classes to mask
            
        Returns:
            Boolean DataArray where True = cloud
        """
        mask_data = DataLoader.load(filepath)
        
        if mask_data.dtype == np.uint16 or mask_data.dtype == np.uint8:
            # Sentinel-2 QA mask or similar
            # This is a simplified version - adapt for specific QA formats
            cloud_mask = np.zeros_like(mask_data, dtype=bool)
            
            # Simple threshold-based detection
            cloud_mask = mask_data > 0
            
            return cloud_mask
        else:
            # Already boolean or probability
            return mask_data > 0.5
