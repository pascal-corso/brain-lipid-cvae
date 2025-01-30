# Copyright (c) 2025 Dr Pascal Corso
# Academic/Research Usage License
#
# This code is part of the mlibra project collaboration at the Swiss Data Science Center.
# Any use of this code or results derived from it in academic publications 
# requires appropriate attribution through co-authorship or adequate contribution
# acknowledgement as specified in the LICENSE file.

"""
Data loading and preprocessing utilities for MALDI imaging mass spectrometry data.

This module provides dataset classes and loading utilities for handling MALDI-MS
imaging data with CCF coordinates and reference data. It supports efficient loading
of large datasets and various preprocessing options.
"""

import h5py
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Union
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MALDIDataset(Dataset):
    """
    Dataset class for MALDI imaging mass spectrometry data with CCF coordinates.
    
    Args:
        maldi_matrix: MALDI data matrix
        ccf: CCF coordinate data
        ref: Reference data
        device: Device to store tensors on
        transform: Optional transform to apply to MALDI data
        normalize: Whether to normalize the data
        scale_method: Scaling method ('standard' or 'minmax')
    """
    def __init__(
        self,
        maldi_matrix: np.ndarray,
        ccf: np.ndarray,
        ref: np.ndarray,
        device: str = 'cpu',
        transform: Optional[callable] = None,
        normalize: bool = True,
        scale_method: str = 'minmax'
    ):
        # Initialize scalers
        self.scalers = {}
        if normalize:
            if scale_method == 'standard':
                self.scalers['maldi'] = StandardScaler()
                #self.scalers['ccf'] = StandardScaler()
            else:
                self.scalers['maldi'] = MinMaxScaler()
                #self.scalers['ccf'] = MinMaxScaler()
            
            # Fit and transform data
            maldi_matrix = self.scalers['maldi'].fit_transform(maldi_matrix)
            #ccf = self.scalers['ccf'].fit_transform(ccf)
        
        # Convert to tensors
        self.maldi = torch.tensor(maldi_matrix, device=device).float()
        self.ccf = torch.tensor(ccf, device=device).float()
        self.ref = torch.tensor(ref, device=device).float()
        
        # Verify dimensions
        assert self.ccf.shape[0] == 3, "CCF should have shape [3, n_spatial_points]"
        assert self.ccf.shape[1] == self.maldi.shape[1], "Number of spatial points should match"
        
        self.transform = transform
        
        # Log initialization
        logging.info(f"Dataset initialized with shapes:")
        logging.info(f"MALDI shape: {self.maldi.shape}")
        logging.info(f"CCF shape: {self.ccf.shape}")
        logging.info(f"Reference data shape: {self.ref.shape}")
        
    def __len__(self) -> int:
        return self.maldi.shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset."""
        maldi_sample = self.maldi[idx]
        
        if self.transform:
            maldi_sample = self.transform(maldi_sample)
            
        return maldi_sample, self.ccf

def read_h5_matrix(filename: Union[str, Path]) -> np.ndarray:
    """
    Read a matrix from an H5 file.
    
    Args:
        filename: Path to the H5 file
        
    Returns:
        Numpy array containing the matrix
    """
    try:
        with h5py.File(filename, 'r') as hf:
            matrix = hf['data'][:]
            logging.info(f"Matrix shape: {matrix.shape}")
            logging.info(f"Matrix data type: {matrix.dtype}")
            return matrix
            
    except Exception as e:
        logging.error(f"Error reading matrix from {filename}: {e}")
        raise

def read_h5_coordinates(filename: Union[str, Path]) -> np.ndarray:
    """
    Read coordinate arrays from an H5 file.
    
    Args:
        filename: Path to the H5 file
        
    Returns:
        Coordinate array [3, n_points]
    """
    try:
        with h5py.File(filename, 'r') as hf:
            coords = hf['data'][:]
            
            logging.info(f"Number of points: {len(coords[0,:])}")
            return coords
            
    except Exception as e:
        logging.error(f"Error reading coordinates from {filename}: {e}")
        raise

def load_matrix_efficiently(
    filename: Union[str, Path],
    downsample_factor: int = 50,
    chunk_size: int = 500
) -> np.ndarray:
    """
    Load matrix efficiently using chunking and downsampling.
    
    Args:
        filename: Path to the H5 file
        downsample_factor: Factor by which to downsample the data
        chunk_size: Number of rows to process at once
        
    Returns:
        Downsampled matrix
    """
    with h5py.File(filename, 'r') as f:
        dataset = f['data']
        original_shape = dataset.shape
        final_shape = (original_shape[0], len(range(0, original_shape[1], downsample_factor)))
        
        output = np.zeros(final_shape, dtype=dataset.dtype)
        
        for i in range(0, original_shape[0], chunk_size):
            end_idx = min(i + chunk_size, original_shape[0])
            chunk = dataset[i:end_idx, ::downsample_factor]
            output[i:end_idx] = chunk
            logging.info(f"Processed rows {i} to {end_idx} of {original_shape[0]}")
            
        return output

def create_dataloader(
    maldi_path: Union[str, Path],
    ccf_path: Union[str, Path],
    ref_path: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    downsample_factor: Optional[int] = None,
    chunk_size: int = 500,
    device: str = 'cpu',
    normalize: bool = True,
    scale_method: str = 'standard',
    **kwargs
) -> Tuple[DataLoader, Dict]:
    """
    Create a DataLoader for MALDI data.
    
    Args:
        maldi_path: Path to MALDI matrix H5 file
        ccf_path: Path to CCF coordinates H5 file
        ref_path: Path to reference data H5 file
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        downsample_factor: Optional downsampling factor
        device: Device to store tensors on
        normalize: Whether to normalize the data
        scale_method: Scaling method
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        tuple: (DataLoader, scalers_dict)
    """
    # Load data
    if downsample_factor:
        maldi_matrix = load_matrix_efficiently(maldi_path, downsample_factor)
    else:
        maldi_matrix = read_h5_matrix(maldi_path)
        
    ccf = read_h5_coordinates(ccf_path)
    ref = read_h5_matrix(ref_path).reshape(1, -1) # Make sure ref is a line vector [1, n_spatial_points]
    
    # Create dataset
    dataset = MALDIDataset(
        maldi_matrix=maldi_matrix,
        ccf=ccf,
        ref=ref,
        device=device,
        normalize=normalize,
        scale_method=scale_method
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device=='cuda' else False,
        persistent_workers=True,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
        generator=torch.Generator().manual_seed(42)
    )
    
    return dataloader, dataset.scalers

class DataTransform:
    """
    Transformations for MALDI data.
    
    Args:
        noise_level: Level of Gaussian noise to add
        mask_ratio: Ratio of points to mask
    """
    def __init__(self, noise_level: float = 0.1, mask_ratio: float = 0.0):
        self.noise_level = noise_level
        self.mask_ratio = mask_ratio
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformations to input tensor."""
        if self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            
        if self.mask_ratio > 0:
            mask = torch.rand_like(x) > self.mask_ratio
            x = x * mask
            
        return x

def save_processed_data(
    data: Union[np.ndarray, torch.Tensor],
    filename: Union[str, Path],
    compression: Optional[str] = 'gzip'
) -> None:
    """
    Save processed data to H5 file.
    
    Args:
        data: Data to save
        filename: Output filename
        compression: Compression method
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    with h5py.File(filename, 'w') as f:
        kwargs = {'compression': compression} if compression else {}
        f.create_dataset('data', data=data, **kwargs)
        
    logging.info(f"Saved processed data to {filename}")
