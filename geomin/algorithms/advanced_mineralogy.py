"""
Advanced mineralogy algorithms for GeoMin.
Implements Crosta PCA, spectral unmixing, and reference library matching.
"""

from typing import Union, Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


# Type aliases
XRData = Union[xr.DataArray, xr.Dataset]


@dataclass
class CrostaResult:
    """Result of Crosta PCA analysis."""
    components: xr.DataArray
    loadings: np.ndarray
    explained_variance: np.ndarray
    mineral_components: Dict[str, xr.DataArray]
    statistics: Dict[str, Any]


@dataclass
class SpectralUnmixingResult:
    """Result of spectral unmixing analysis."""
    abundances: xr.DataArray
    endmembers: Dict[str, np.ndarray]
    residual: xr.DataArray
    rmse: float
    model: Any


class AdvancedMineralogy:
    """
    Advanced mineral detection algorithms.
    
    Features:
    - Crosta PCA (Directed PCA for alteration mapping)
    - Spectral Angle Mapper (SAM)
    - Linear Spectral Unmixing
    - Reference spectrum matching
    """
    
    # Common hydrothermal alteration mineral signatures
    # Typical wavelength positions (micrometers)
    ALTERATION_MINERALS = {
        'kaolinite': {
            'absorption': 2.17,
            'features': [1.4, 1.8, 2.17, 2.2],
            'type': 'clay',
        },
        'alunite': {
            'absorption': 2.17,
            'features': [1.4, 1.76, 2.17, 2.2],
            'type': 'sulfate',
        },
        'jarosite': {
            'absorption': 2.27,
            'features': [1.4, 1.76, 2.27, 2.4],
            'type': 'sulfate',
        },
        'hematite': {
            'absorption': 0.85,
            'features': [0.55, 0.65, 0.85],
            'type': 'iron_oxide',
        },
        'goethite': {
            'absorption': 0.92,
            'features': [0.55, 0.65, 0.92],
            'type': 'iron_oxide',
        },
        'sericite': {
            'absorption': 2.2,
            'features': [1.4, 2.2, 2.35],
            'type': 'mica',
        },
        'chlorite': {
            'absorption': 2.3,
            'features': [1.4, 1.9, 2.3, 2.35],
            'type': 'phyllosilicate',
        },
        'calcite': {
            'absorption': 2.33,
            'features': [1.4, 1.9, 2.0, 2.33, 2.55],
            'type': 'carbonate',
        },
        'dolomite': {
            'absorption': 2.31,
            'features': [1.4, 1.9, 2.31, 2.52],
            'type': 'carbonate',
        },
    }
    
    # Sentinel-2 band wavelengths (micrometers)
    SENTINEL2_WAVELENGTHS = {
        'B01': 0.443, 'B02': 0.492, 'B03': 0.560, 'B04': 0.665,
        'B05': 0.705, 'B06': 0.740, 'B07': 0.783, 'B08': 0.842,
        'B8A': 0.865, 'B09': 0.945, 'B11': 1.610, 'B12': 2.190,
    }
    
    def __init__(self):
        """Initialize advanced mineralogy analyzer."""
        pass
    
    def crosta_pca(
        self,
        data: xr.DataArray,
        bands: Optional[List[str]] = None,
        n_components: int = 4,
        target_mineral: str = 'hydroxyl'
    ) -> CrostaResult:
        """
        Perform Crosta PCA (Directed PCA) for alteration mapping.
        
        The Crosta technique identifies specific minerals by analyzing
        PCA component loadings to find components that contrast
        absorption features.
        
        Args:
            data: Input DataArray with spectral bands
            bands: Bands to use for analysis
            n_components: Number of PCA components
            target_mineral: Target mineral type ('hydroxyl', 'iron', 'silica')
            
        Returns:
            CrostaResult with components and mineral maps
        """
        # Get bands for analysis
        if bands is None:
            bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        
        # Extract band data
        band_data = []
        for band in bands:
            band_arr = self._get_band(data, band)
            band_data.append(band_arr.values.flatten())
        
        # Stack and clean
        X = np.column_stack(band_data)
        
        # Remove NaN values
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_valid = X[valid_mask]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        # Perform PCA
        pca = PCA(n_components=min(n_components, X_valid.shape[1]))
        components = pca.fit_transform(X_scaled)
        
        # Analyze loadings for target mineral
        mineral_components = self._identify_mineral_components(
            pca.components_, bands, target_mineral
        )
        
        # Reshape components back to image
        result_shape = (n_components,) + data.sizes.get('y', 0), data.sizes.get('x', 0)
        components_reshaped = np.full(result_shape, np.nan)
        
        # Map valid pixels back
        valid_coords = np.where(valid_mask.reshape(data.sizes.get('y', 0), data.sizes.get('x', 0)))
        
        for i in range(n_components):
            comp_2d = np.full(data.sizes.get('y', 0), data.sizes.get('x', 0), np.nan)
            comp_values = components[:, i]
            
            for j, (y_idx, x_idx) in enumerate(zip(valid_coords[0], valid_coords[1])):
                comp_2d[y_idx, x_idx] = comp_values[j]
            
            components_reshaped[i] = comp_2d
        
        # Create DataArray
        coords = {
            'component': np.arange(n_components),
            'y': data.coords.get('y', np.arange(result_shape[1])),
            'x': data.coords.get('x', np.arange(result_shape[2])),
        }
        
        components_da = xr.DataArray(
            components_reshaped,
            dims=['component', 'y', 'x'],
            coords=coords,
            attrs={
                'long_name': 'Crosta PCA Components',
                'explained_variance': pca.explained_variance_ratio_,
            }
        )
        
        statistics = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'target_mineral': target_mineral,
            'mineral_components': list(mineral_components.keys()),
        }
        
        return CrostaResult(
            components=components_da,
            loadings=pca.components_,
            explained_variance=pca.explained_variance_ratio_,
            mineral_components={k: self._create_component_map(
                components_reshaped, v, data, valid_mask
            ) for k, v in mineral_components.items()},
            statistics=statistics
        )
    
    def _identify_mineral_components(
        self,
        loadings: np.ndarray,
        bands: List[str],
        target_mineral: str
    ) -> Dict[str, int]:
        """
        Identify PCA components associated with target minerals.
        
        Args:
            loadings: PCA component loadings
            bands: Band names
            target_mineral: Target mineral type
            
        Returns:
            Dictionary mapping mineral names to component indices
        """
        mineral_components = {}
        
        # Get band indices for key wavelengths
        band_wavelengths = [self.SENTINEL2_WAVELENGTHS.get(b, 0.5) for b in bands]
        
        # Define feature band combinations for different minerals
        if target_mineral == 'hydroxyl':
            # Hydroxyl: SWIR absorption (B11 at 1.6μm, B12 at 2.2μm)
            swir1_idx = band_wavelengths.index(1.610) if 1.610 in band_wavelengths else -1
            swir2_idx = band_wavelengths.index(2.190) if 2.190 in band_wavelengths else -1
            
            for i, loading in enumerate(loadings):
                if swir1_idx >= 0 and swir2_idx >= 0:
                    # Look for contrast between SWIR bands
                    diff = loading[swir2_idx] - loading[swir1_idx]
                    if abs(diff) > 0.3:
                        mineral_components['hydroxyl_alteration'] = i
        
        elif target_mineral == 'iron':
            # Iron: Red absorption (B04 at 0.65μm)
            red_idx = band_wavelengths.index(0.665) if 0.665 in band_wavelengths else -1
            
            for i, loading in enumerate(loadings):
                if red_idx >= 0:
                    if loading[red_idx] < -0.3:  # Negative loading = absorption
                        mineral_components['iron_oxide'] = i
        
        elif target_mineral == 'silica':
            # Silica: High reflectance in SWIR
            for i, loading in enumerate(loadings):
                swir_mean = np.mean([loading[j] for j, w in enumerate(band_wavelengths) if w > 1.5])
                if swir_mean > 0.3:
                    mineral_components['silica'] = i
        
        return mineral_components
    
    def _create_component_map(
        self,
        components: np.ndarray,
        component_idx: int,
        original_data: xr.DataArray,
        valid_mask: np.ndarray
    ) -> xr.DataArray:
        """Create a DataArray for a specific component."""
        comp_data = components[component_idx]
        
        return xr.DataArray(
            comp_data,
            dims=['y', 'x'],
            coords={
                'y': original_data.coords.get('y', np.arange(comp_data.shape[0])),
                'x': original_data.coords.get('x', np.arange(comp_data.shape[1])),
            },
            attrs={
                'long_name': f'Component {component_idx}',
                'component_index': component_idx,
            }
        )
    
    def spectral_angle_mapper(
        self,
        data: xr.DataArray,
        reference_spectrum: np.ndarray,
        band_wavelengths: Optional[List[float]] = None,
        threshold: float = 0.1
    ) -> xr.DataArray:
        """
        Calculate Spectral Angle Mapper similarity.
        
        Measures similarity between pixel spectra and reference spectrum
        using the angle between vectors in n-dimensional space.
        
        Args:
            data: Input DataArray with spectral bands
            reference_spectrum: Reference mineral spectrum
            band_wavelengths: Wavelengths for each band
            threshold: SAM threshold (lower = more similar)
            
        Returns:
            DataArray with SAM values (0 = identical, π/2 = orthogonal)
        """
        if band_wavelengths is None:
            band_wavelengths = [self.SENTINEL2_WAVELENGTHS.get(f'B{i+1}', 0.5) 
                               for i in range(data.sizes.get('band', 6))]
        
        # Normalize reference spectrum
        ref_norm = reference_spectrum / (np.linalg.norm(reference_spectrum) + 1e-6)
        
        # Get band data
        n_bands = data.sizes.get('band', len(band_wavelengths))
        
        # Calculate SAM for each pixel
        sam_values = np.full(data.sizes.get('y', 0), data.sizes.get('x', 0), np.nan)
        
        for y in range(data.sizes.get('y', 0)):
            for x in range(data.sizes.get('x', 0)):
                if 'band' in data.dims:
                    pixel_spectrum = data.isel(y=y, x=x).values
                else:
                    pixel_spectrum = data.values[y, x]
                
                # Normalize pixel spectrum
                pixel_norm = pixel_spectrum / (np.linalg.norm(pixel_spectrum) + 1e-6)
                
                # Calculate angle (in radians)
                dot_product = np.dot(ref_norm, pixel_norm)
                dot_product = np.clip(dot_product, -1, 1)
                angle = np.arccos(dot_product)
                
                sam_values[y, x] = angle
        
        # Create DataArray
        return xr.DataArray(
            sam_values,
            dims=['y', 'x'],
            coords={
                'y': data.coords.get('y', np.arange(sam_values.shape[0])),
                'x': data.coords.get('x', np.arange(sam_values.shape[1])),
            },
            attrs={
                'long_name': 'Spectral Angle Mapper',
                'units': 'radians',
                'threshold': threshold,
                'description': '0 = perfect match, π/2 = no similarity',
            }
        )
    
    def linear_spectral_unmixing(
        self,
        data: xr.DataArray,
        endmembers: Dict[str, np.ndarray]
    ) -> SpectralUnmixingResult:
        """
        Perform linear spectral unmixing.
        
        Decomposes pixel spectra into abundance fractions of endmembers.
        
        Args:
            data: Input DataArray with spectral bands
            endmembers: Dictionary of endmember names to spectra
            
        Returns:
            SpectralUnmixingResult with abundance maps
        """
        # Stack bands
        n_bands = data.sizes.get('band', 6)
        n_pixels = data.sizes.get('y', 0) * data.sizes.get('x', 0)
        
        # Get endmember matrix
        endmember_names = list(endmembers.keys())
        n_endmembers = len(endmember_names)
        
        # Ensure all endmembers have same length
        for name in endmember_names:
            if len(endmembers[name]) != n_bands:
                raise ValueError(f"Endmember {name} has wrong number of bands")
        
        E = np.column_stack([endmembers[name] for name in endmember_names])
        
        # Flatten data
        X = data.values.reshape(n_bands, n_pixels)
        
        # Remove NaN
        valid_mask = np.all(np.isfinite(X), axis=0)
        X_valid = X[:, valid_mask]
        
        # Solve least squares for each pixel
        abundances = np.zeros((n_endmembers, n_pixels))
        
        # Use pseudo-inverse for unmixing
        E_pinv = np.linalg.pinv(E)
        abundances_valid = E_pinv @ X_valid
        
        # Ensure non-negative abundances and sum to 1
        abundances_valid = np.maximum(abundances_valid, 0)
        row_sums = abundances_valid.sum(axis=0, keepdims=True)
        row_sums[row_sums == 0] = 1
        abundances_valid = abundances_valid / row_sums
        
        # Map back to full array
        abundances_full = np.zeros((n_endmembers, n_pixels))
        abundances_full[:, valid_mask] = abundances_valid
        
        # Reshape to image
        y_size = data.sizes.get('y', 0)
        x_size = data.sizes.get('x', 0)
        
        abundance_maps = {}
        for i, name in enumerate(endmember_names):
            abundance_maps[name] = xr.DataArray(
                abundances_full[i].reshape(y_size, x_size),
                dims=['y', 'x'],
                coords={
                    'y': data.coords.get('y', np.arange(y_size)),
                    'x': data.coords.get('x', np.arange(x_size)),
                },
                attrs={
                    'long_name': f'{name} Abundance',
                    'units': 'fraction (0-1)',
                }
            )
        
        # Calculate residual and RMSE
        reconstructed = E @ abundances_valid
        residual = X_valid - reconstructed
        rmse = np.sqrt(np.mean(residual**2))
        
        # Stack abundance maps
        abundances_stacked = xr.concat(
            list(abundance_maps.values()),
            dim='endmember'
        )
        abundances_stacked = abundances_stacked.assign_coords(
            {'endmember': endmember_names}
        )
        
        return SpectralUnmixingResult(
            abundances=abundances_stacked,
            endmembers=endmembers,
            residual=xr.DataArray(
                residual.reshape(n_bands, y_size, x_size),
                dims=['band', 'y', 'x'],
            ),
            rmse=float(rmse),
            model=E,
        )
    
    def get_reference_spectrum(
        self,
        mineral: str,
        source: str = 'usgs'
    ) -> np.ndarray:
        """
        Get reference spectrum for a mineral.
        
        Args:
            mineral: Mineral name (e.g., 'kaolinite', 'hematite')
            source: Spectral library source ('usgs', 'jhk')
            
        Returns:
            Reference spectrum array
        """
        # USGS typical values for Sentinel-2 bands
        mineral_spectra = {
            'kaolinite': np.array([0.15, 0.20, 0.25, 0.35, 0.45, 0.35]),
            'alunite': np.array([0.18, 0.22, 0.28, 0.38, 0.50, 0.40]),
            'jarosite': np.array([0.22, 0.28, 0.35, 0.40, 0.42, 0.38]),
            'hematite': np.array([0.30, 0.35, 0.28, 0.40, 0.50, 0.48]),
            'goethite': np.array([0.28, 0.32, 0.30, 0.42, 0.52, 0.50]),
            'sericite': np.array([0.16, 0.21, 0.26, 0.36, 0.48, 0.42]),
            'chlorite': np.array([0.18, 0.22, 0.25, 0.30, 0.35, 0.32]),
            'calcite': np.array([0.20, 0.25, 0.30, 0.40, 0.45, 0.42]),
            'dolomite': np.array([0.19, 0.24, 0.28, 0.38, 0.44, 0.41]),
            'muscovite': np.array([0.16, 0.20, 0.25, 0.35, 0.46, 0.40]),
            'biotite': np.array([0.12, 0.15, 0.18, 0.25, 0.32, 0.28]),
            'quartz': np.array([0.22, 0.28, 0.35, 0.45, 0.55, 0.52]),
            'feldspar': np.array([0.20, 0.25, 0.30, 0.40, 0.48, 0.45]),
            'vegetation': np.array([0.08, 0.12, 0.10, 0.45, 0.35, 0.25]),
            'soil': np.array([0.18, 0.22, 0.26, 0.32, 0.38, 0.35]),
            'water': np.array([0.05, 0.08, 0.04, 0.02, 0.01, 0.01]),
        }
        
        if mineral.lower() in mineral_spectra:
            return mineral_spectra[mineral.lower()]
        else:
            raise ValueError(f"Unknown mineral: {mineral}. "
                           f"Available: {list(mineral_spectra.keys())}")
    
    def create_endmember_dict(
        self,
        minerals: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Create endmember dictionary from mineral list.
        
        Args:
            minerals: List of mineral names
            
        Returns:
            Dictionary mapping mineral names to spectra
        """
        return {m: self.get_reference_spectrum(m) for m in minerals}
    
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
