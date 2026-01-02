"""
Visualization tools for GeoMin.
Provides matplotlib and folium-based visualization for mineral maps and mining activity.
"""

from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


def plot_mineral_index(
    data: xr.DataArray,
    index_name: str,
    cmap: str = 'magma',
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: Optional[str] = None
) -> plt.Figure:
    """
    Plot a mineral spectral index map.
    
    Args:
        data: DataArray with spectral index values
        index_name: Name of the index for title
        cmap: Colormap name
        figsize: Figure size tuple
        title: Custom title (auto-generated if None)
        save_path: Path to save figure
        show: Whether to display figure
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        colorbar_label: Custom colorbar label
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get data values
    values = data.values
    
    # Handle NaN values
    masked_values = np.ma.masked_invalid(values)
    
    # Determine bounds
    if vmin is None:
        vmin = np.nanpercentile(values, 2)
    if vmax is None:
        vmax = np.nanpercentile(values, 98)
    
    # Plot
    im = ax.imshow(
        masked_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin='upper',
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_label = colorbar_label or f'{index_name} Index'
    cbar.set_label(cbar_label, fontsize=12)
    
    # Title
    if title is None:
        title = f'{index_name.replace("_", " ").title()} Distribution'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_mineral_classification(
    data: xr.DataArray,
    detections: Dict[str, xr.DataArray],
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot mineral detection results.
    
    Args:
        data: Background satellite data (true color)
        detections: Dictionary of mineral name to detection mask
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib Figure
    """
    n_minerals = len(detections)
    
    if n_minerals == 0:
        raise ValueError("No detections provided")
    
    fig, axes = plt.subplots(2, (n_minerals + 1) // 2 + 1, figsize=figsize)
    axes = axes.flatten()
    
    # Background image
    axes[0].imshow(np.transpose(data.values[:3], (1, 2, 0)))
    axes[0].set_title('Satellite Background', fontsize=12, fontweight='bold')
    axes[0].set_axis_off()
    
    # Detection maps
    colors = plt.cm.Set1(np.linspace(0, 1, n_minerals))
    
    for idx, (mineral, detection) in enumerate(detections.items()):
        ax = axes[idx + 1]
        
        # Background
        ax.imshow(np.transpose(data.values[:3], (1, 2, 0)), alpha=0.5)
        
        # Overlay detection
        detection_values = detection.values.astype(float)
        detection_values[~detection_values] = np.nan
        
        ax.imshow(detection_values, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
        ax.set_title(f'{mineral.title()} Detection', fontsize=12, fontweight='bold')
        ax.set_axis_off()
    
    # Hide extra axes
    for idx in range(n_minerals + 1, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_change_map(
    change_result,
    background: Optional[xr.DataArray] = None,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = 'RdYlGn_r',
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot change detection results.
    
    Args:
        change_result: ChangeResult object from change detection
        background: Optional background image
        figsize: Figure size
        cmap: Colormap for intensity
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Background
    if background is not None:
        axes[0].imshow(np.transpose(background.values[:3], (1, 2, 0)))
        axes[0].set_title('Before', fontsize=12)
    axes[0].set_axis_off()
    
    # Change intensity
    intensity = change_result.change_intensity.values
    im = axes[1].imshow(intensity, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Change Intensity', fontsize=12)
    axes[1].set_axis_off()
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Change mask
    change_map = change_result.change_map.values
    axes[2].imshow(change_map, cmap='Reds')
    axes[2].set_title(f"Detected Changes ({change_result.statistics['change_percentage']:.2f}%)", fontsize=12)
    axes[2].set_axis_off()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_rgb_composite(
    data: xr.DataArray,
    bands: Tuple[str, str, str] = ('B04', 'B03', 'B02'),
    stretch: str = 'percentile',
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create true or false color composite image.
    
    Args:
        data: DataArray with spectral bands
        bands: Tuple of (R, G, B) band names
        stretch: Histogram stretch method ('percentile', 'minmax', 'none')
        figsize: Figure size
        title: Custom title
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get bands
    r = _get_band_data(data, bands[0]).values
    g = _get_band_data(data, bands[1]).values
    b = _get_band_data(data, bands[2]).values
    
    # Apply stretch
    if stretch == 'percentile':
        r = _stretch_percentile(r)
        g = _stretch_percentile(g)
        b = _stretch_percentile(b)
    elif stretch == 'minmax':
        r = _stretch_minmax(r)
        g = _stretch_minmax(g)
        b = _stretch_minmax(b)
    
    # Stack and convert to 0-255
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    ax.imshow(rgb)
    ax.set_title(title or f'{" ".join(bands)} Composite', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_mineral_map(
    data: xr.DataArray,
    detections: Dict[str, xr.DataArray],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comprehensive mineral map with legend.
    
    Args:
        data: Background satellite data
        detections: Dictionary of mineral name to detection mask
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Background
    background = np.transpose(data.values[:3], (1, 2, 0))
    ax.imshow(_stretch_percentile(background))
    
    # Color map for different minerals
    mineral_colors = {
        'iron': 'red',
        'clay': 'blue',
        'gossan': 'orange',
        'quartz': 'yellow',
    }
    
    # Plot detections
    legend_elements = []
    for mineral, detection in detections.items():
        color = mineral_colors.get(mineral, 'purple')
        
        # Create mask overlay
        mask = detection.values.astype(float)
        mask[~mask] = np.nan
        
        ax.imshow(mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        
        # Add to legend
        patch = mpatches.Patch(color=color, label=mineral.title(), alpha=0.5)
        legend_elements.append(patch)
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_title('Mineral Detection Map', fontsize=16, fontweight='bold')
    ax.set_axis_off()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_spectral_profile(
    data: xr.DataArray,
    locations: List[Tuple[int, int]],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot spectral profiles at specific locations.
    
    Args:
        data: DataArray with spectral bands
        locations: List of (y, x) pixel coordinates
        labels: Optional labels for each location
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get band names
    band_names = [str(b) for b in data.coords.get('band', np.arange(data.sizes.get('band', 1)))].values
    wavelengths = _get_wavelengths(band_names)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(locations)))
    
    for idx, (y, x) in enumerate(locations):
        # Extract spectrum
        if 'band' in data.dims:
            spectrum = data.isel(y=y, x=x).values
        else:
            spectrum = data.values[y, x]
        
        label = labels[idx] if labels else f'Location {idx + 1}'
        ax.plot(wavelengths, spectrum, color=colors[idx], label=label, linewidth=2)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title('Spectral Profiles', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_comparison_plot(
    images: List[xr.DataArray],
    titles: List[str],
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create side-by-side comparison of multiple images.
    
    Args:
        images: List of DataArrays to compare
        titles: List of titles for each image
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib Figure
    """
    n_images = len(images)
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    for idx, (image, title) in enumerate(zip(images, titles)):
        ax = axes[idx]
        
        # Display image
        if image.ndim == 3 and image.sizes.get('band', 0) >= 3:
            # RGB composite
            display = np.transpose(image.values[:3], (1, 2, 0))
            display = _stretch_percentile(display)
        else:
            # Single band
            display = image.values
            display = np.ma.masked_invalid(display)
        
        ax.imshow(display)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_axis_off()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def _get_band_data(data: xr.DataArray, band_name: str) -> xr.DataArray:
    """Extract band data from DataArray."""
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


def _stretch_percentile(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Apply percentile-based histogram stretch."""
    p2, p98 = np.nanpercentile(arr, (low, high))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled = (arr - p2) / (p98 - p2)
        scaled = np.clip(scaled, 0, 1)
    
    return scaled


def _stretch_minmax(arr: np.ndarray) -> np.ndarray:
    """Apply min-max normalization."""
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scaled = (arr - arr_min) / (arr_max - arr_min)
        scaled = np.clip(scaled, 0, 1)
    
    return scaled


def _get_wavelengths(band_names: List[str]) -> List[float]:
    """Get approximate wavelengths for band names."""
    wavelengths = {
        'B01': 0.443, 'B02': 0.492, 'B03': 0.560, 'B04': 0.665,
        'B05': 0.705, 'B06': 0.740, 'B07': 0.783, 'B08': 0.842,
        'B8A': 0.865, 'B09': 0.945, 'B10': 1.375, 'B11': 1.610, 'B12': 2.190,
        'B1': 0.45, 'B2': 0.52, 'B3': 0.63, 'B4': 0.77,
        'B5': 1.55, 'B6': 11.0, 'B7': 2.11,
    }
    
    return [wavelengths.get(b, float(i + 1) * 0.1) for i, b in enumerate(band_names)]
