"""
Machine learning models for mining detection and mineral classification.
"""

from .change_detection import (
    simple_difference,
    ratio_difference,
    vegetation_change_detector,
    pca_change_detector,
    kmeans_change_detector,
    time_series_change_detector,
    detect_mining_activity,
    classify_change_type,
    calculate_change_area,
    ChangeResult,
)

__all__ = [
    # Change detection
    "simple_difference",
    "ratio_difference",
    "vegetation_change_detector",
    "pca_change_detector",
    "kmeans_change_detector",
    "time_series_change_detector",
    "detect_mining_activity",
    "classify_change_type",
    "calculate_change_area",
    "ChangeResult",
]
