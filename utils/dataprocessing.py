# Copyright (c) 2025 Dr Pascal Corso
# Academic/Research Usage License
#
# This code is part of the mlibra project collaboration at the Swiss Data Science Center.
# Any use of this code or results derived from it in academic publications 
# requires appropriate attribution through co-authorship or adequate contribution
# acknowledgement as specified in the LICENSE file.

"""
Data processing utilities for MALDI imaging mass spectrometry data.

This module provides comprehensive preprocessing functionality for MALDI-MS
imaging data, including normalization, filtering, coordinate transformation,
and quality control methods.
"""

import numpy as np
import torch
import h5py
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Union, List
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class MALDIPreprocessor:
    """
    Preprocessor for MALDI imaging mass spectrometry data.
    
    Args:
        normalization (str): Normalization method ('standard', 'minmax', 'robust')
        smoothing (bool): Whether to apply smoothing
        outlier_removal (bool): Whether to remove outliers
        quality_threshold (float): Quality threshold for filtering
    """
    def __init__(
        self,
        normalization: str = 'standard',
        smoothing: bool = True,
        outlier_removal: bool = True,
        quality_threshold: float = 0.1
    ):
        self.normalization = normalization
        self.smoothing = smoothing
        self.outlier_removal = outlier_removal
        self.quality_threshold = quality_threshold
        self.scalers = {}
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def fit_transform(
        self,
        maldi_data: np.ndarray,
        coords: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessor to data and transform it.
        
        Args:
            maldi_data: Raw MALDI data [n_features, n_spatial_points]
            coords: Coordinate data [3, n_spatial_points]
            reference: Optional reference data [1, n_spatial_points]
            
        Returns:
            Preprocessed versions of input data
        """
        # Quality control
        if self.outlier_removal:
            mask = self._detect_outliers(maldi_data)
            maldi_data = maldi_data[:, mask]
            coords = coords[:, mask]
            if reference is not None:
                reference = reference[:, mask]
        
        # Normalization
        maldi_data = self._normalize_data(maldi_data)
        coords = self._normalize_coordinates(coords)
        
        # Spatial smoothing
        if self.smoothing:
            maldi_data = self._spatial_smoothing(maldi_data, coords)
        
        return maldi_data, coords, reference
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to MALDI data."""
        if self.normalization == 'standard':
            self.scalers['maldi'] = StandardScaler()
        elif self.normalization == 'minmax':
            self.scalers['maldi'] = MinMaxScaler()
        elif self.normalization == 'robust':
            self.scalers['maldi'] = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
            
        return self.scalers['maldi'].fit_transform(data)
    
    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Normalize coordinate data."""
        self.scalers['coords'] = MinMaxScaler()
        return self.scalers['coords'].fit_transform(coords.T).T
    
    def _detect_outliers(self, data: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers in MALDI data using Z-score method.
        
        Args:
            data: Input data
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean mask for non-outlier points
        """
        z_scores = np.abs(stats.zscore(data, axis=1))
        return np.all(z_scores < z_threshold, axis=0)
    
    def _spatial_smoothing(
        self,
        data: np.ndarray,
        coords: np.ndarray,
        kernel_size: float = 1.0
    ) -> np.ndarray:
        """
        Apply spatial smoothing using coordinate-based Gaussian kernel.
        
        Args:
            data: Input data
            coords: Coordinate data
            kernel_size: Size of Gaussian kernel
            
        Returns:
            Smoothed data
        """
        from scipy.spatial.distance import cdist
        
        # Calculate pairwise distances
        distances = cdist(coords.T, coords.T)
        
        # Compute Gaussian weights
        weights = np.exp(-distances**2 / (2 * kernel_size**2))
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Apply smoothing
        return data @ weights

class CoordinateTransformer:
    """
    Transform coordinates between different coordinate systems.
    """
    @staticmethod
    def cartesian_to_spherical(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        center_x: float = 0,
        center_y: float = 0,
        center_z: float = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert Cartesian to spherical coordinates."""
        x_rel = np.zeros(len(x))
        y_rel = np.zeros(len(y))
        z_rel = np.zeros(len(z))
        theta = np.zeros(len(z))
        phi = np.zeros(len(z))
        
        for i in range(len(x)):
            if x[i] != 0 and y[i] != 0 and z[i] != 0:
                x_rel[i] = x[i] - center_x
                y_rel[i] = y[i] - center_y
                z_rel[i] = z[i] - center_z
        else:
                x_rel[i] = y_rel[i] = z_rel[i] = 0
    
        r = np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)
        
        for i in range(len(x)):
            if r[i] != 0:
                theta[i] = np.arctan2(y_rel[i], x_rel[i])
                phi[i] = np.arccos(np.clip(z_rel[i]/r[i], -1.0, 1.0))
        
        # Handle zeros
        #non_zero = r > 0
        #theta[~non_zero] = 0
        #phi[~non_zero] = 0
        
        return r, theta, phi
    
    @staticmethod
    def spherical_to_cartesian(
        r: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert spherical to Cartesian coordinates."""
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z

class QualityControl:
    """
    Quality control methods for MALDI data.
    """
    @staticmethod
    def check_missing_values(data: np.ndarray) -> Tuple[bool, float]:
        """Check for missing values in data."""
        missing = np.isnan(data).sum()
        total = data.size
        missing_ratio = missing / total
        return missing > 0, missing_ratio
    
    @staticmethod
    def check_zero_values(data: np.ndarray, threshold: float = 0.1) -> bool:
        """Check for excessive zero values."""
        zero_ratio = (data == 0).sum() / data.size
        return zero_ratio > threshold
    
    @staticmethod
    def check_intensity_range(
        data: np.ndarray,
        min_intensity: float = 0,
        max_intensity: float = 1e6
    ) -> bool:
        """Check if intensities are within expected range."""
        return (data.min() >= min_intensity) and (data.max() <= max_intensity)

class DataGenerator:
    """
    Generate synthetic training data for the CVAE.
    """
    def __init__(
        self,
        original_data: np.ndarray,
        num_samples: int = 300,
        noise_type: str = 'mixed'
    ):
        self.original_data = original_data
        self.num_samples = num_samples
        self.noise_type = noise_type
        
    def generate_samples(self) -> np.ndarray:
        """Generate synthetic samples."""
        samples = []
        
        for i in range(self.num_samples):
            coeff = np.random.random()
            
            if self.noise_type == 'gaussian':
                noise = np.random.normal(0, 1, self.original_data.shape)
            elif self.noise_type == 'laplace':
                noise = np.random.laplace(0, 1, self.original_data.shape)
            elif self.noise_type == 'mixed':
                noise1 = stats.laplace.rvs(loc=0, scale=1, size=self.original_data.shape)
                noise1 = (noise1 - noise1.mean()) / noise1.std() # z-score normalisation
                noise2 = np.random.normal(0, 1, self.original_data.shape)
                noise2 = (noise2 - noise2.mean()) / noise2.std() # z-score normalisation
                noise3 = stats.t.rvs(3, loc=0, scale=1, size=self.original_data.shape)
                noise3 = (noise3 - noise3.mean()) / noise3.std() # z-score normalisation
                noise = coeff * noise1 + (1 - coeff) * noise3
            else:
                raise ValueError(f"Unknown noise type: {self.noise_type}")
            
            # Add noise to original data
            new_sample = self.original_data + 0.3 * self.original_data.std() * noise
            samples.append(new_sample)
        
        return np.array(samples)

def save_processed_data(
    data: Dict[str, np.ndarray],
    output_dir: Union[str, Path],
    prefix: str = 'processed'
) -> None:
    """
    Save processed data to H5 files.
    
    Args:
        data: Dictionary of arrays to save
        output_dir: Output directory
        prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, array in data.items():
        filename = output_dir / f"{prefix}_{name}.h5"
        with h5py.File(filename, 'w') as f:
            f.create_dataset('data', data=array, compression='gzip')
        logging.info(f"Saved {name} to {filename}")

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def process_maldi_data(
    df: pd.DataFrame,
    lipid_name: str,
    step: int = 40,
    exclude_sections: List[int] = [4, 12]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Extract and downsample data
    coords = {
        'x': df['x_ccf'][::step].to_numpy(),
        'y': df['y_ccf'][::step].to_numpy(),
        'z': df['z_ccf'][::step].to_numpy()
    }
    coords_array = np.array([coords['x'], coords['y'], coords['z']])

    # Get section number
    sections = df['Section'][::step].to_numpy()
    
    exclude_idx = np.where(np.isin(sections, exclude_sections))[0]
    
    # Get lipid, exclude zeros, minMaxScale
    lipid_data = df[lipid_name][::step].to_numpy()
    exclude_zero_idx = np.where(lipid_data == 0)[0]
    lipid_data_trunc = np.delete(lipid_data, exclude_zero_idx)
    lipid_data = (lipid_data - min(lipid_data_trunc))/(max(lipid_data_trunc) - min(lipid_data_trunc))
    lipid_data_ref = (lipid_data_trunc - min(lipid_data_trunc))/(max(lipid_data_trunc) - min(lipid_data_trunc))
    
    # Create truncated copies
    coords_trunc_array = coords_array.copy()
    
    # Zero out excluded sections
    coords_trunc_array[:,exclude_idx] = 0
    coords_array = np.delete(coords_array, exclude_zero_idx, axis=1)
    coords_trunc_array = np.delete(coords_trunc_array, exclude_zero_idx, axis=1)
    
    return coords_array, coords_trunc_array, lipid_data_ref, lipid_data, sections, exclude_idx, exclude_zero_idx
