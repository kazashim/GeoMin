"""
Spectral analysis algorithms for mineral detection.
Implements various indices and calculations for identifying minerals from satellite imagery.
"""

from typing import Union, Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import xarray as xr


# Type alias for xarray data
XRData = Union[xr.DataArray, xr.Dataset]


@dataclass
class MineralIndex:
    """
    Definition of a mineral spectral index.
    
    Attributes:
        name: Display name of the index
        formula: Formula description
        bands: Required bands
        range: Expected value range
        description: Description of what the index detects
        high_values: What high values indicate
    """
    name: str
    formula: str
    bands: List[str]
    range: Tuple[float, float]
    description: str
    high_values: str


# Pre-defined mineral indices
MINERAL_INDICES = {
    # Iron oxide indices
    'iron_oxide': MineralIndex(
        name='Iron Oxide Ratio',
        formula='B4 / B2',
        bands=['B04', 'B02'],  # Red / Blue
        range=(0.5, 2.0),
        description='Detects iron oxides (hematite, goethite) in soils and rocks',
        high_values='High iron oxide content (rust-colored surfaces)',
    ),
    'ferric_oxide': MineralIndex(
        name='Ferric Oxide Index',
        formula='(B04 - B02) / (B04 + B02)',
        bands=['B04', 'B02'],
        range=(-1, 1),
        description='Measures ferric iron absorption features',
        high_values='Presence of oxidized iron minerals',
    ),
    
    # Clay mineral indices
    'clay_ratio': MineralIndex(
        name='Clay Ratio',
        formula='B11 / B12',
        bands=['B11', 'B12'],  # SWIR1 / SWIR2
        range=(0.5, 2.0),
        description='Detects clay minerals (kaolinite, montmorillonite)',
        high_values='Clay mineral presence (hydrothermal alteration)',
    ),
    'clay_index': MineralIndex(
        name='Normalized Difference Clay Index',
        formula='(B11 - B12) / (B11 + B12)',
        bands=['B11', 'B12'],
        range=(-1, 1),
        description='Normalized index for clay mineral detection',
        high_values='Hydrothermally altered clay zones',
    ),
    
    # Alteration indices
    'gossan': MineralIndex(
        name='Gossan Index',
        formula='(B04 - B03) / (B04 + B03) * (B06 / B12)',
        bands=['B04', 'B03', 'B06', 'B12'],
        range=(0, 2),
        description='Detects gossans (iron-rich cap rocks over ore deposits)',
        high_values='Oxidized sulfide zones (potential ore indicators)',
    ),
    ' hydrothermal_alteration': MineralIndex(
        name='Hydrothermal Alteration Index',
        formula='(B11 + B04) - (B12 + B08)',
        bands=['B11', 'B04', 'B12', 'B08'],
        range=(-1, 1),
        description='Identifies hydrothermal alteration zones',
        high_values='Hydrothermally altered areas',
    ),
    
    # Vegetation indices (for context)
    'ndvi': MineralIndex(
        name='Normalized Difference Vegetation Index',
        formula='(B08 - B04) / (B08 + B04)',
        bands=['B08', 'B04'],
        range=(-1, 1),
        description='Measures vegetation health and coverage',
        high_values='Healthy, dense vegetation',
    ),
    'ndwi': MineralIndex(
        name='Normalized Difference Water Index',
        formula='(B03 - B08) / (B03 + B08)',
        bands=['B03', 'B08'],
        range=(-1, 1),
        description='Detects surface water and vegetation moisture',
        high_values='Surface water or high moisture content',
    ),
    
    # Burn/scarring indices
    'nbr': MineralIndex(
        name='Normalized Burn Ratio',
        formula='(B08 - B12) / (B08 + B12)',
        bands=['B08', 'B12'],
        range=(-1, 1),
        description='Detects burned areas and fire severity',
        high_values='Recently burned areas',
    ),
    'ndmi': MineralIndex(
        name='Normalized Difference Moisture Index',
        formula='(B08 - B11) / (B08 + B11)',
        bands=['B08', 'B11'],
        range=(-1, 1),
        description='Measures vegetation water content',
        high_values='High vegetation moisture',
    ),
    
    # Mineral-specific combinations
    'ferrous': MineralIndex(
        name='Ferrous Minerals Index',
        formula='(B12 / B08)',
        bands=['B12', 'B08'],
        range=(0, 2),
        description='Detects ferrous minerals (magnetite, siderite)',
        high_values='Presence of ferrous iron-bearing minerals',
    ),
    'oxide': MineralIndex(
        name='Iron Oxide Index (Alternative)',
        formula='(B03 / B02) + (B04 / B02)',
        bands=['B03', 'B02', 'B04'],
        range=(0, 4),
        description='Alternative iron oxide detection using multiple bands',
        high_values='Strong iron oxide absorption',
    ),
    
    # Quartz/feldspar detection
    'quartz': MineralIndex(
        name='Quartz Index',
        formula='(B10 / B08)',
        bands=['B10', 'B08'],
        range=(0.5, 1.5),
        description='Detects quartz-rich rocks and sand',
        high_values='High quartz content (siliceous rocks)',
    ),
    
    # Carbonate detection
    'carbonate': MineralIndex(
        name='Carbonate Index',
        formula='(B11 - B12) / (B11 + B12)',
        bands=['B11', 'B12'],
        range=(-1, 1),
        description='Detects carbonate minerals (calcite, dolomite)',
        high_values='Carbonate rock presence',
    ),
}


def _get_band_data(data: xr.DataArray, band_name: str) -> xr.DataArray:
    """
    Extract band data from DataArray.
    
    Args:
        data: Input DataArray
        band_name: Band name or number
        
    Returns:
        Band DataArray
    """
    if 'band' in data.coords:
        # Try to find by name
        band_names = [str(b) for b in data.coords['band'].values]
        if band_name in band_names:
            return data.sel(band=band_name)
        # Try by index
        try:
            idx = int(band_name) - 1
            if 0 <= idx < data.sizes['band']:
                return data.isel(band=idx)
        except ValueError:
            pass
    
    # If no band dimension, assume single band
    if data.ndim == 2:
        return data
    
    raise ValueError(f"Band {band_name} not found in data")


def iron_oxide_index(data: xr.DataArray, red: str = 'B04', blue: str = 'B02') -> xr.DataArray:
    """
    Calculate Iron Oxide Index (B4/B2).
    
    Detects iron oxides (hematite, goethite) which appear rust-red in visible spectrum.
    
    Args:
        data: DataArray with spectral bands
        red: Band name for red spectrum
        blue: Band name for blue spectrum
        
    Returns:
        Iron oxide index DataArray
    """
    red_data = _get_band_data(data, red)
    blue_data = _get_band_data(data, blue)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        index = red_data / blue_data
        index = index.where((blue_data != 0) & np.isfinite(index))
    
    index.attrs['long_name'] = 'Iron Oxide Index'
    index.attrs['description'] = 'Ratio of Red to Blue bands for iron oxide detection'
    index.attrs['valid_range'] = (0.5, 2.0)
    
    return index


def clay_ratio(data: xr.DataArray, swir1: str = 'B11', swir2: str = 'B12') -> xr.DataArray:
    """
    Calculate Clay Ratio (SWIR1/SWIR2).
    
    Detects clay minerals which show absorption in SWIR2 region.
    
    Args:
        data: DataArray with spectral bands
        swir1: Band name for SWIR1 (shortwave infrared)
        swir2: Band name for SWIR2 (shortwave infrared)
        
    Returns:
        Clay ratio DataArray
    """
    swir1_data = _get_band_data(data, swir1)
    swir2_data = _get_band_data(data, swir2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = swir1_data / swir2_data
        ratio = ratio.where((swir2_data != 0) & np.isfinite(ratio))
    
    ratio.attrs['long_name'] = 'Clay Ratio'
    ratio.attrs['description'] = 'Ratio for clay mineral detection'
    ratio.attrs['valid_range'] = (0.5, 2.0)
    
    return ratio


def ndvi(data: xr.DataArray, nir: str = 'B08', red: str = 'B04') -> xr.DataArray:
    """
    Calculate Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        data: DataArray with spectral bands
        nir: Band name for near-infrared
        red: Band name for red
        
    Returns:
        NDVI DataArray
    """
    nir_data = _get_band_data(data, nir)
    red_data = _get_band_data(data, red)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        index = (nir_data - red_data) / (nir_data + red_data)
        index = index.where((nir_data + red_data != 0) & np.isfinite(index))
    
    index.attrs['long_name'] = 'Normalized Difference Vegetation Index'
    index.attrs['valid_range'] = (-1, 1)
    
    return index


def ndwi(data: xr.DataArray, green: str = 'B03', nir: str = 'B08') -> xr.DataArray:
    """
    Calculate Normalized Difference Water Index.
    
    NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        data: DataArray with spectral bands
        green: Band name for green
        nir: Band name for near-infrared
        
    Returns:
        NDWI DataArray
    """
    green_data = _get_band_data(data, green)
    nir_data = _get_band_data(data, nir)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        index = (green_data - nir_data) / (green_data + nir_data)
        index = index.where((green_data + nir_data != 0) & np.isfinite(index))
    
    index.attrs['long_name'] = 'Normalized Difference Water Index'
    index.attrs['valid_range'] = (-1, 1)
    
    return index


def hydroxyl_index(data: xr.DataArray, swir1: str = 'B11', swir2: str = 'B12') -> xr.DataArray:
    """
    Calculate Hydroxyl Index (Al-OH group absorption).
    
    Detects hydroxyl-bearing minerals like clay and alunite.
    
    Args:
        data: DataArray with spectral bands
        swir1: Band name for SWIR1
        swir2: Band name for SWIR2
        
    Returns:
        Hydroxyl index DataArray
    """
    swir1_data = _get_band_data(data, swir1)
    swir2_data = _get_band_data(data, swir2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        index = (swir1_data - swir2_data) / (swir1_data + swir2_data)
        index = index.where((swir1_data + swir2_data != 0) & np.isfinite(index))
    
    index.attrs['long_name'] = 'Hydroxyl Index'
    index.attrs['description'] = 'Detects hydroxyl-bearing minerals'
    index.attrs['valid_range'] = (-0.5, 0.5)
    
    return index


def ferrous_index(data: xr.DataArray, swir: str = 'B12', nir: str = 'B08') -> xr.DataArray:
    """
    Calculate Ferrous Minerals Index.
    
    Detects ferrous iron-bearing minerals (magnetite, siderite).
    
    Args:
        data: DataArray with spectral bands
        swir: Band name for SWIR
        nir: Band name for NIR
        
    Returns:
        Ferrous index DataArray
    """
    swir_data = _get_band_data(data, swir)
    nir_data = _get_band_data(data, nir)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        index = swir_data / nir_data
        index = index.where((nir_data != 0) & np.isfinite(index))
    
    index.attrs['long_name'] = 'Ferrous Minerals Index'
    index.attrs['description'] = 'Detects ferrous iron-bearing minerals'
    index.attrs['valid_range'] = (0.5, 2.0)
    
    return index


def gossan_index(data: xr.DataArray) -> xr.DataArray:
    """
    Calculate Gossan Index for oxidized sulfide detection.
    
    Gossan = (Red - Green) / (Red + Green) * (SWIR1 / SWIR2)
    
    This index helps identify oxidized cap rocks above sulfide deposits.
    
    Args:
        data: DataArray with B04, B03, B06, B12 bands
        
    Returns:
        Gossan index DataArray
    """
    red = _get_band_data(data, 'B04')
    green = _get_band_data(data, 'B03')
    swir1 = _get_band_data(data, 'B06')  # Use B06 for 20m SWIR
    swir2 = _get_band_data(data, 'B12')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Iron oxide component
        iron = (red - green) / (red + green)
        iron = iron.where((red + green != 0) & np.isfinite(iron))
        
        # Clay ratio component
        clay = swir1 / swir2
        clay = clay.where((swir2 != 0) & np.isfinite(clay))
        
        # Combined index
        index = iron * clay
        index = index.where(np.isfinite(index))
    
    index.attrs['long_name'] = 'Gossan Index'
    index.attrs['description'] = 'Detects oxidized sulfide zones (gossans)'
    index.attrs['valid_range'] = (0, 2)
    
    return index


def custom_index(data: xr.DataArray, formula: str, band_map: Dict[str, str]) -> xr.DataArray:
    """
    Calculate custom spectral index from formula.
    
    Supports basic arithmetic with band names.
    
    Args:
        data: DataArray with spectral bands
        formula: Formula string (e.g., "(B08 - B04) / (B08 + B04)")
        band_map: Mapping of band variables to actual band names
        
    Returns:
        Calculated index DataArray
    """
    import re
    
    # Replace band names with data access
    expression = formula
    
    # Get actual band data for each variable
    for var, band in band_map.items():
        band_data = _get_band_data(data, band)
        expression = expression.replace(var, f"({band_data.name})")
    
    # Evaluate expression
    with np.errstate(divide='ignore', invalid='ignore'):
        result = eval(expression)
        result = result.where(np.isfinite(result))
    
    result.attrs['long_name'] = f'Custom Index: {formula}'
    result.attrs['formula'] = formula
    
    return result


def mineral_probability(
    data: xr.DataArray,
    mineral: str,
    method: str = 'threshold'
) -> xr.DataArray:
    """
    Calculate probability map for specific mineral.
    
    Args:
        data: DataArray with spectral bands
        mineral: Mineral type ('iron', 'clay', 'gossan', 'quartz')
        method: Calculation method ('threshold', 'ratio', 'index')
        
    Returns:
        Probability map (0-1) DataArray
    """
    indices = {
        'iron': ('iron_oxide', 0.8),
        'clay': ('clay_ratio', 1.2),
        'gossan': ('gossan_index', 0.5),
        'quartz': ('quartz_index', 1.0),
    }
    
    if mineral not in indices:
        raise ValueError(f"Unknown mineral: {mineral}")
    
    index_name, threshold = indices[mineral]
    
    if index_name == 'iron_oxide':
        index = iron_oxide_index(data)
    elif index_name == 'clay_ratio':
        index = clay_ratio(data)
    elif index_name == 'gossan_index':
        index = gossan_index(data)
    else:
        raise ValueError(f"Unknown index: {index_name}")
    
    # Normalize to probability
    with np.errstate(divide='ignore', invalid='ignore'):
        prob = (index - threshold / 2) / threshold
        prob = prob.clip(0, 1)
        prob = prob.where(np.isfinite(prob), 0)
    
    prob.attrs['long_name'] = f'{mineral.capitalize()} Probability'
    
    return prob


def calculate_all_indices(data: xr.DataArray) -> Dict[str, xr.DataArray]:
    """
    Calculate all available mineral indices.
    
    Args:
        data: DataArray with spectral bands
        
    Returns:
        Dictionary of index names to DataArrays
    """
    indices = {}
    
    try:
        indices['iron_oxide'] = iron_oxide_index(data)
    except Exception as e:
        print(f"Warning: Could not calculate iron_oxide: {e}")
    
    try:
        indices['clay_ratio'] = clay_ratio(data)
    except Exception as e:
        print(f"Warning: Could not calculate clay_ratio: {e}")
    
    try:
        indices['ndvi'] = ndvi(data)
    except Exception as e:
        print(f"Warning: Could not calculate ndvi: {e}")
    
    try:
        indices['ndwi'] = ndwi(data)
    except Exception as e:
        print(f"Warning: Could not calculate ndwi: {e}")
    
    try:
        indices['hydroxyl_index'] = hydroxyl_index(data)
    except Exception as e:
        print(f"Warning: Could not calculate hydroxyl_index: {e}")
    
    try:
        indices['ferrous_index'] = ferrous_index(data)
    except Exception as e:
        print(f"Warning: Could not calculate ferrous_index: {e}")
    
    try:
        indices['gossan_index'] = gossan_index(data)
    except Exception as e:
        print(f"Warning: Could not calculate gossan_index: {e}")
    
    return indices


def detect_minerals(
    data: xr.DataArray,
    threshold: float = 0.7
) -> Dict[str, xr.DataArray]:
    """
    Detect multiple mineral types simultaneously.
    
    Args:
        data: DataArray with spectral bands
        threshold: Probability threshold for detection
        
    Returns:
        Dictionary of mineral names to boolean detection masks
    """
    detections = {}
    
    minerals = ['iron', 'clay', 'gossan']
    
    for mineral in minerals:
        try:
            prob = mineral_probability(data, mineral)
            mask = prob > threshold
            mask.attrs['long_name'] = f'{mineral.capitalize()} Detection'
            detections[mineral] = mask
        except Exception as e:
            print(f"Warning: Could not detect {mineral}: {e}")
    
    return detections
