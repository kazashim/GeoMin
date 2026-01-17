"""
Anomaly detection algorithms for GeoMin.
Identifies unusual spectral signatures that may indicate mineral deposits.
"""

from typing import Union, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


# Type aliases
XRData = Union[xr.DataArray, xr.Dataset]


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis."""
    anomaly_scores: xr.DataArray
    anomaly_labels: xr.DataArray
    anomaly_mask: xr.DataArray
    contamination: float
    statistics: Dict[str, Any]


class AnomalyDetector:
    """
    Anomaly detection for spectral data.
    
    Identifies pixels that spectrally deviate from the background,
    potentially indicating mineralization or other interesting features.
    
    Methods:
    - Isolation Forest: Tree-based anomaly detection
    - Local Outlier Factor: Density-based detection
    - Mahalanobis Distance: Statistical distance from mean
    - RX Anomaly Detector: Standard RX algorithm for remote sensing
    """
    
    def __init__(self):
        """Initialize anomaly detector."""
        pass
    
    def detect_anomalies(
        self,
        data: xr.DataArray,
        method: str = 'isolation_forest',
        contamination: float = 0.01,
        n_estimators: int = 100,
        bands: Optional[List[str]] = None,
        **kwargs
    ) -> AnomalyResult:
        """
        Detect spectral anomalies in satellite imagery.
        
        Args:
            data: Input DataArray with spectral bands
            method: Detection method ('isolation_forest', 'lof', 'mahalanobis', 'rx')
            contamination: Expected proportion of anomalies (0-1)
            n_estimators: Number of estimators (for tree-based methods)
            bands: Bands to use for analysis
            **kwargs: Additional method-specific parameters
            
        Returns:
            AnomalyResult with scores, labels, and mask
        """
        if method == 'isolation_forest':
            return self._isolation_forest(
                data, contamination, n_estimators, bands
            )
        elif method == 'lof':
            return self._local_outlier_factor(
                data, contamination, bands, **kwargs
            )
        elif method == 'mahalanobis':
            return self._mahalanobis_distance(data, contamination, bands)
        elif method == 'rx':
            return self._rx_anomaly_detector(data, contamination, bands)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _isolation_forest(
        self,
        data: xr.DataArray,
        contamination: float,
        n_estimators: int,
        bands: Optional[List[str]]
    ) -> AnomalyResult:
        """Isolation Forest anomaly detection."""
        # Prepare data
        X, coords = self._prepare_data(data, bands)
        
        # Fit Isolation Forest
        clf = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Get predictions and scores
        labels = clf.fit_predict(X)
        scores = clf.decision_function(X)
        
        # Convert to DataArrays
        anomaly_scores = self._create_result_array(scores, coords, data, 'Anomaly Score')
        anomaly_labels = self._create_result_array(
            labels, coords, data, 'Anomaly Label (-1=anomaly, 1=normal)'
        )
        anomaly_mask = self._create_result_array(
            labels == -1, coords, data, 'Anomaly Mask'
        )
        
        # Statistics
        n_anomalies = np.sum(labels == -1)
        statistics = {
            'method': 'isolation_forest',
            'contamination': contamination,
            'n_estimators': n_estimators,
            'total_pixels': len(labels),
            'anomaly_pixels': int(n_anomalies),
            'anomaly_percentage': float(n_anomalies / len(labels) * 100),
            'mean_score_normal': float(np.mean(scores[labels == 1])),
            'mean_score_anomaly': float(np.mean(scores[labels == -1])),
        }
        
        return AnomalyResult(
            anomaly_scores=anomaly_scores,
            anomaly_labels=anomaly_labels,
            anomaly_mask=anomaly_mask,
            contamination=contamination,
            statistics=statistics
        )
    
    def _local_outlier_factor(
        self,
        data: xr.DataArray,
        contamination: float,
        bands: Optional[List[str]],
        n_neighbors: int = 20
    ) -> AnomalyResult:
        """Local Outlier Factor anomaly detection."""
        X, coords = self._prepare_data(data, bands)
        
        # Fit LOF
        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1
        )
        
        labels = clf.fit_predict(X)
        scores = clf.negative_outlier_factor_
        
        # Convert scores to positive (higher = more anomalous)
        anomaly_scores = -scores
        
        # Create DataArrays
        anomaly_scores_da = self._create_result_array(anomaly_scores, coords, data, 'LOF Score')
        anomaly_labels = self._create_result_array(labels, coords, data, 'LOF Label')
        anomaly_mask = self._create_result_array(labels == -1, coords, data, 'LOF Mask')
        
        statistics = {
            'method': 'local_outlier_factor',
            'contamination': contamination,
            'n_neighbors': n_neighbors,
            'total_pixels': len(labels),
            'anomaly_pixels': int(np.sum(labels == -1)),
        }
        
        return AnomalyResult(
            anomaly_scores=anomaly_scores_da,
            anomaly_labels=anomaly_labels,
            anomaly_mask=anomaly_mask,
            contamination=contamination,
            statistics=statistics
        )
    
    def _mahalanobis_distance(
        self,
        data: xr.DataArray,
        contamination: float,
        bands: Optional[List[str]]
    ) -> AnomalyResult:
        """Mahalanobis distance-based anomaly detection."""
        X, coords = self._prepare_data(data, bands)
        
        # Calculate mean and covariance
        mean = np.mean(X, axis=0)
        cov = np.cov(X.T)
        
        # Handle singular covariance matrix
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Add small regularization
            cov += np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov)
        
        # Calculate Mahalanobis distances
        diff = X - mean
        distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        
        # Set threshold based on contamination
        threshold = np.percentile(distances, (1 - contamination) * 100)
        labels = (distances > threshold).astype(int) * -2 + 1  # -1 for anomalies
        
        anomaly_scores = distances / np.max(distances)  # Normalize to 0-1
        
        # Create DataArrays
        anomaly_scores_da = self._create_result_array(
            anomaly_scores, coords, data, 'Mahalanobis Distance'
        )
        anomaly_labels = self._create_result_array(labels, coords, data, 'Mahalanobis Label')
        anomaly_mask = self._create_result_array(labels == -1, coords, data, 'Mahalanobis Mask')
        
        statistics = {
            'method': 'mahalanobis_distance',
            'contamination': contamination,
            'threshold': float(threshold),
            'total_pixels': len(distances),
            'anomaly_pixels': int(np.sum(labels == -1)),
        }
        
        return AnomalyResult(
            anomaly_scores=anomaly_scores_da,
            anomaly_labels=anomaly_labels,
            anomaly_mask=anomaly_mask,
            contamination=contamination,
            statistics=statistics
        )
    
    def _rx_anomaly_detector(
        self,
        data: xr.DataArray,
        contamination: float,
        bands: Optional[List[str]]
    ) -> AnomalyResult:
        """
        RX (Reed-Xiaoli) Anomaly Detector.
        
        The standard RX algorithm computes the Mahalanobis distance
        of each pixel from the global mean, identifying anomalies
        as pixels with high deviation.
        
        This is a widely used algorithm in remote sensing for
        detecting subtle spectral anomalies.
        """
        X, coords = self._prepare_data(data, bands)
        
        # Global statistics
        mean = np.mean(X, axis=0)
        
        # Robust covariance estimation (using median for robustness)
        # For RX, we use the standard covariance
        cov = np.cov(X.T)
        
        # Regularize if needed
        if np.linalg.cond(cov) > 1e10:
            cov += np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0] * 0.01
        
        cov_inv = np.linalg.inv(cov)
        
        # Calculate RX scores
        diff = X - mean
        rx_scores = np.sum(diff @ cov_inv * diff, axis=1)
        
        # Normalize
        rx_scores = rx_scores / np.max(rx_scores)
        
        # Set threshold
        threshold = 1 - contamination
        threshold_value = np.percentile(rx_scores, threshold * 100)
        
        labels = (rx_scores > threshold_value).astype(int) * -2 + 1
        
        # Create DataArrays
        anomaly_scores_da = self._create_result_array(rx_scores, coords, data, 'RX Score')
        anomaly_labels = self._create_result_array(labels, coords, data, 'RX Label')
        anomaly_mask = self._create_result_array(labels == -1, coords, data, 'RX Mask')
        
        statistics = {
            'method': 'rx_anomaly_detector',
            'contamination': contamination,
            'threshold': float(threshold_value),
            'total_pixels': len(rx_scores),
            'anomaly_pixels': int(np.sum(labels == -1)),
            'description': 'Reed-Xiaoli global anomaly detector',
        }
        
        return AnomalyResult(
            anomaly_scores=anomaly_scores_da,
            anomaly_labels=anomaly_labels,
            anomaly_mask=anomaly_mask,
            contamination=contamination,
            statistics=statistics
        )
    
    def _prepare_data(
        self,
        data: xr.DataArray,
        bands: Optional[List[str]]
    ) -> Tuple[np.ndarray, Dict]:
        """Prepare data for anomaly detection."""
        # Get band data
        if bands:
            band_data = []
            for band in bands:
                band_arr = self._get_band(data, band)
                band_data.append(band_arr.values.flatten())
            X = np.column_stack(band_data)
        else:
            if 'band' in data.dims:
                X = data.values.reshape(data.sizes['band'], -1).T
            else:
                X = data.values.reshape(-1, 1)
        
        # Remove NaN rows
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_valid = X[valid_mask]
        
        # Store coordinates for reconstruction
        coords = {
            'valid_mask': valid_mask,
            'shape': (data.sizes.get('y', 0), data.sizes.get('x', 0)),
            'original_coords': {k: v for k, v in data.coords.items() 
                               if k not in ['band', 'variable']},
        }
        
        return X_valid, coords
    
    def _create_result_array(
        self,
        values: np.ndarray,
        coords: Dict,
        original_data: xr.DataArray,
        name: str
    ) -> xr.DataArray:
        """Create result DataArray with proper coordinates."""
        shape = coords['shape']
        valid_mask = coords['valid_mask']
        
        # Initialize with NaN
        result = np.full(shape, np.nan)
        
        # Fill valid pixels
        result.flat[valid_mask] = values
        
        return xr.DataArray(
            result,
            dims=['y', 'x'],
            coords=coords['original_coords'],
            attrs={
                'long_name': name,
                'anomaly_detection': True,
            }
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
    
    def get_top_anomalies(
        self,
        result: AnomalyResult,
        n: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get coordinates of top N anomalies.
        
        Args:
            result: Anomaly detection result
            n: Number of top anomalies to return
            
        Returns:
            Tuple of (y_coords, x_coords) for top anomalies
        """
        scores = result.anomaly_scores.values
        shape = scores.shape
        
        # Flatten and get indices
        flat_scores = scores.flatten()
        
        # Get indices of top N
        top_indices = np.argsort(flat_scores)[-n:][::-1]
        
        # Convert to 2D coordinates
        y_coords = top_indices // shape[1]
        x_coords = top_indices % shape[1]
        
        return y_coords, x_coords
    
    def rank_anomalies(
        self,
        result: AnomalyResult,
        n: int = 50
    ) -> xr.DataArray:
        """
        Create ranked anomaly map showing top N anomalies.
        
        Args:
            result: Anomaly detection result
            n: Number of top anomalies to highlight
            
        Returns:
            DataArray with anomaly ranks (0 = not anomalous)
        """
        scores = result.anomaly_scores.values
        shape = scores.shape
        
        # Initialize rank array
        ranks = np.zeros(shape, dtype=np.float32)
        
        # Get sorted indices
        flat_scores = scores.flatten()
        sorted_indices = np.argsort(flat_scores)[::-1]
        
        # Assign ranks (1 = most anomalous)
        for rank, idx in enumerate(sorted_indices[:n], 1):
            y = idx // shape[1]
            x = idx % shape[1]
            ranks[y, x] = rank
        
        return xr.DataArray(
            ranks,
            dims=['y', 'x'],
            coords={k: v for k, v in result.anomaly_scores.coords.items() 
                   if k in ['y', 'x']},
            attrs={
                'long_name': 'Anomaly Rank',
                'description': 'Rank of spectral anomalies (1=most anomalous)',
                'total_anomalies': n,
            }
        )
