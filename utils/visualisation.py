# Copyright (c) 2025 Dr Pascal Corso
# Academic/Research Usage License
#
# This code is part of the mlibra project collaboration at the Swiss Data Science Center.
# Any use of this code or results derived from it in academic publications 
# requires appropriate attribution through co-authorship or adequate contribution
# acknowledgement as specified in the LICENSE file.

"""
Visualisation utilities for MALDI imaging mass spectrometry data and CVAE results.

This module provides comprehensive visualization tools for analysing MALDI-MS
imaging data, model training progress, and reconstruction quality assessment.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
import logging

class TrainingVisualizer:
    """
    Visualizer for model training progress and metrics.
    
    Args:
        plots_dir: Directory to save plots
        dpi: DPI for saved figures
    """
    def __init__(self, plots_dir: Union[str, Path], dpi: int = 300):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # Initialize tracking variables
        self.epoch_numbers = []
        self.total_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.kl_weight = []
        
        # Set style
        # plt.style.use('seaborn')
        
    def plot_training_progress(
        self,
        metrics: Dict[str, List[float]],
        title: str = 'Training Progress',
        save_name: str = 'training_epoch.png'
    ) -> None:
        """
        Plot training metrics over time.
        
        Args:
            metrics: Dictionary of metric names and values
            title: Plot title
            save_name: Output filename
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(values, label=metric_name)
            ax.set_title(f'{metric_name} vs. Epoch')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.grid(True)
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def update_loss_plots(
        self, 
        epoch_num: int,
        epoch_metrics: Dict[str, List[float]],
        epochs: int
        ) -> None:
            """Helper function to update and save loss plots with separate graphs for each loss"""
            self.epoch_numbers.append(epoch_num + 1)
            self.total_losses.append(epoch_metrics['loss'])
            self.recon_losses.append(epoch_metrics['reconstruction_loss'])
            self.kl_losses.append(epoch_metrics['kl_loss'])
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
            
            # Plot total loss
            ax1.plot(self.epoch_numbers, self.total_losses, 'b-', label='Total Loss')
            ax1.set_title('Total Loss vs. Epoch')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            ax1.legend()
            
            # Plot reconstruction loss
            ax2.plot(self.epoch_numbers, self.recon_losses, 'g-', label='Reconstruction Loss')
            ax2.set_title('Reconstruction Loss vs. Epoch')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
            ax2.legend()
            
            # Plot KL loss
            ax3.plot(self.epoch_numbers, self.kl_losses, 'r-', label='KL Loss')
            ax3.set_title('KL Loss vs. Epoch')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.grid(True)
            ax3.legend()
            
            plt.tight_layout()
            
            # Save the current plot
            plot_path = self.plots_dir / f'loss_plots_epoch_{epoch_num+1}.png'
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            # Save the final plot with a different name
            if epoch_num == epochs - 1:
                final_plot_path = self.plots_dir / 'final_loss_plots.png'
                plt.figure(figsize=(10, 15))
                
                # Total Loss
                plt.subplot(3, 1, 1)
                plt.plot(self.epoch_numbers, self.total_losses, 'b-', label='Total Loss')
                plt.title('Total Loss vs. Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()
                
                # Reconstruction Loss
                plt.subplot(3, 1, 2)
                plt.plot(self.epoch_numbers, self.recon_losses, 'g-', label='Reconstruction Loss')
                plt.title('Reconstruction Loss vs. Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()
                
                # KL Loss
                plt.subplot(3, 1, 3)
                plt.plot(self.epoch_numbers, self.kl_losses, 'r-', label='KL Loss')
                plt.title('KL Loss vs. Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(final_plot_path, dpi=self.dpi, bbox_inches='tight')
                plt.close()
        
    def plot_loss_components(
        self,
        reconstruction_loss: List[float],
        kl_loss: List[float],
        total_loss: List[float],
        save_name: str = 'loss_components.png'
    ) -> None:
        """
        Plot individual loss components.
        
        Args:
            reconstruction_loss: Reconstruction loss values
            kl_loss: KL divergence loss values
            total_loss: Total loss values
            save_name: Output filename
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(reconstruction_loss) + 1)
        
        plt.plot(epochs, reconstruction_loss, 'b-', label='Reconstruction Loss')
        plt.plot(epochs, kl_loss, 'r-', label='KL Loss')
        plt.plot(epochs, total_loss, 'g-', label='Total Loss')
        
        plt.title('Loss Components vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()

class ReconstructionVisualizer:
    """
    Visualizer for model reconstruction quality assessment.
    """
    @staticmethod
    def plot_reconstruction_comparison(
        original: Union[np.ndarray, torch.Tensor],
        reconstructed: Union[np.ndarray, torch.Tensor],
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300
    ) -> None:
        """
        Create scatter plot comparing original vs reconstructed data.
        
        Args:
            original: Original data
            reconstructed: Reconstructed data
            save_path: Optional path to save figure
            dpi: DPI for saved figure
        """
        # Convert to numpy if needed
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
            
        # Flatten arrays
        original = original.flatten()
        reconstructed = reconstructed.flatten()
        
        # Calculate correlation
        r, p_value = stats.pearsonr(original, reconstructed)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(original, reconstructed, alpha=0.5)
        
        # Add perfect reconstruction line
        min_val = min(original.min(), reconstructed.min())
        max_val = max(original.max(), reconstructed.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
                label='Perfect reconstruction')
        
        plt.title(f'Reconstruction vs Original\nPearson r={r:.3f}, p={p_value:.3e}')
        plt.xlabel('Original Data')
        plt.ylabel('Reconstructed Data')
        plt.axis('square')
        plt.grid(True)
        plt.legend()

        plt.show()
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
    @staticmethod
    def plot_spatial_distribution(
        values: Union[np.ndarray, torch.Tensor],
        coordinates: Union[np.ndarray, torch.Tensor],
        title: str = 'Spatial Distribution',
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300,
        cmap: str = 'hot'
    ) -> None:
        """
        Plot spatial distribution of values.
        
        Args:
            values: Values to plot
            coordinates: 3D coordinates
            title: Plot title
            save_path: Optional path to save figure
            dpi: DPI for saved figure
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            coordinates[0], coordinates[1], coordinates[2],
            c=values,
            cmap=cmap,
            alpha=0.5,
            vmin=0, vmax=0.5
        )
        plt.colorbar(scatter)
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close()

class LatentSpaceVisualizer:
    """
    Visualizer for latent space analysis.
    """
    @staticmethod
    def plot_latent_space_2d(
        z: Union[np.ndarray, torch.Tensor],
        labels: Optional[np.ndarray] = None,
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300
    ) -> None:
        """
        Plot 2D projection of latent space.
        
        Args:
            z: Latent vectors
            labels: Optional labels for coloring
            save_path: Optional path to save figure
            dpi: DPI for saved figure
        """
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
            
        # Use PCA if dimensionality > 2
        if z.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z = pca.fit_transform(z)
            
        plt.figure(figsize=(8, 8))
        if labels is not None:
            scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='viridis')
            plt.colorbar(scatter)
        else:
            plt.scatter(z[:, 0], z[:, 1], alpha=0.5)
            
        plt.title('Latent Space Visualization')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
    @staticmethod
    def plot_latent_traversal(
        decoder: torch.nn.Module,
        coordinates: torch.Tensor,
        dim: int,
        n_steps: int = 10,
        range_: Tuple[float, float] = (-3, 3),
        save_path: Optional[Union[str, Path]] = None,
        dpi: int = 300
    ) -> None:
        """
        Plot latent space traversal along a dimension.
        
        Args:
            decoder: Trained decoder model
            coordinates: Coordinate conditioning
            dim: Dimension to traverse
            n_steps: Number of steps
            range_: Range of values to traverse
            save_path: Optional path to save figure
            dpi: DPI for saved figure
        """
        device = next(decoder.parameters()).device
        z_dim = decoder.latent_processor[0].in_features
        
        # Create latent vectors
        z = torch.zeros(n_steps, z_dim, device=device)
        values = torch.linspace(range_[0], range_[1], n_steps, device=device)
        z[:, dim] = values
        
        # Generate reconstructions
        with torch.no_grad():
            reconstructions = decoder(z, coordinates.expand(n_steps, -1, -1))
            
        # Plot reconstructions
        fig, axes = plt.subplots(1, n_steps, figsize=(2*n_steps, 2))
        for i, (ax, recon) in enumerate(zip(axes, reconstructions)):
            im = ax.imshow(recon.cpu().numpy(), cmap='viridis')
            ax.set_title(f'z_{dim}={values[i]:.1f}')
            ax.axis('off')
            
        plt.colorbar(im, ax=axes.ravel().tolist())
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

def save_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_dir: Union[str, Path],
    tag: str = 'reconstruction'
) -> None:
    """
    Save original and reconstructed data comparison plots.
    
    Args:
        original: Original data
        reconstructed: Reconstructed data
        save_dir: Directory to save plots
        tag: Tag for filenames
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Regular comparison plot
    ReconstructionVisualizer.plot_reconstruction_comparison(
        original, reconstructed,
        save_path=save_dir / f'{tag}_comparison.png'
    )
    
    # Log-scale comparison
    plt.figure(figsize=(8, 8))
    plt.loglog(original.flatten(), reconstructed.flatten(), '.', alpha=0.5)
    plt.loglog([1e-6, 1], [1e-6, 1], 'r--', label='Perfect reconstruction')
    plt.xlabel('Original (log scale)')
    plt.ylabel('Reconstructed (log scale)')
    plt.title('Log-Log Reconstruction Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_dir / f'{tag}_comparison_log.png', dpi=300, bbox_inches='tight')
    plt.close()