"""
Spectral and terrain analysis algorithms for GeoMin.
"""

from .spectral import (
    iron_oxide_index,
    clay_ratio,
    ndvi,
    ndwi,
    hydroxyl_index,
    ferrous_index,
    gossan_index,
    custom_index,
    mineral_probability,
    calculate_all_indices,
    detect_minerals,
    MINERAL_INDICES,
)

from .terrain import (
    calculate_slope,
    calculate_aspect,
    calculate_hillshade,
    calculate_curvature,
    calculate_tpi,
    calculate_tri,
    calculate_terrain_metrics,
    detect_slopes,
    identify_pits_and_peaks,
)

__all__ = [
    # Spectral analysis
    "iron_oxide_index",
    "clay_ratio",
    "ndvi",
    "ndwi",
    "hydroxyl_index",
    "ferrous_index",
    "gossan_index",
    "custom_index",
    "mineral_probability",
    "calculate_all_indices",
    "detect_minerals",
    "MINERAL_INDICES",
    
    # Terrain analysis
    "calculate_slope",
    "calculate_aspect",
    "calculate_hillshade",
    "calculate_curvature",
    "calculate_tpi",
    "calculate_tri",
    "calculate_terrain_metrics",
    "detect_slopes",
    "identify_pits_and_peaks",
]
