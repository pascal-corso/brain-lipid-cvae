# Copyright (c) 2025 Dr Pascal Corso
# Academic/Research Usage License
#
# This code is part of the mlibra project collaboration between 
# the Swiss Data Science Center and Ecole Polytechnique Fédérale de Lausanne (EPFL).
# Any use of this code or results derived from it in academic publications 
# requires appropriate attribution through co-authorship or adequate contribution
# acknowledgement as specified in the LICENSE file.

"""
Conditional Variational Autoencoder (CVAE) for mouse brain lipid spatial distribution enhancement.

This module implements the main c-VAE model, combining the enhanced encoder and decoder
with sophisticated training and inference capabilities. The model is specifically designed
for processing and enhancing brain lipid mass spectrometry data with spatial conditioning.
"""

import os
import time
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from .encoder import EnhancedEncoder
from .decoder import EnhancedDecoder
from .losses import ImprovedVAELoss
from utils.visualisation import TrainingVisualizer

class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for brain lipid spatial distribution enhancement.
    
    Args:
        maldi_dim (int): Dimension of MALDI input data
        ccf_dim (int): Dimension of CCF coordinates
        hidden_dim (int): Dimension of hidden layers
        latent_dim (int): Dimension of latent space
        beta (float): Weight of the KL divergence term
        device (str): Device to run the model on
        reduction (str): Loss reduction method
    """
    def __init__(
        self,
        maldi_dim: int,
        ccf_dim: int,
        hidden_dim: int,
        latent_dim: int,
        beta: float = 1.0,
        device: str = "cpu",
        reduction: str = "mean"
    ):
        super().__init__()
        
        # Initialize components
        self.encoder = EnhancedEncoder(maldi_dim, ccf_dim, hidden_dim, latent_dim)
        self.decoder = EnhancedDecoder(maldi_dim, ccf_dim, hidden_dim, latent_dim)
        
        # Model parameters
        self.maldi_dim = maldi_dim
        self.z_dim = latent_dim
        self.beta = beta
        self.device = device
        self.loss_func = ImprovedVAELoss(reduction)
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Move model to device
        self.to(device)

    def forward(self, maldi: torch.Tensor, ccf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CVAE.
        
        Args:
            maldi: MALDI input data [batch_size, n_spatial_points]
            ccf: CCF condition data [batch_size, ccf_dim, n_spatial_points]
            
        Returns:
            tuple: (recon, mu, logvar)
                recon: Reconstructed data
                mu: Mean of latent distribution
                logvar: Log variance of latent distribution
        """
        mu, logvar = self.encoder(maldi, ccf)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, ccf)
        return recon, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Perform the reparameterization trick.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(mu.device)
            return mu + eps * std
        return mu

    def train_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        print_every: int = 100,
        use_mixed_precision: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_dir: Union[str, Path] = "checkpoints",
        total_epochs: Optional[int] = None
    ) -> Tuple[Dict, Dict, str]:
        """
        Train the CVAE model.
        
        Args:
            dataloader: DataLoader containing training data
            optimizer: Optimizer for training
            epochs: Number of training epochs
            print_every: Frequency of progress updates
            use_mixed_precision: Whether to use mixed precision training
            scheduler: Learning rate scheduler
            save_dir: Directory to save checkpoints
            total_epochs: Total number of epochs for training
            
        Returns:
            tuple: (epoch_metrics, epoch_stats, run_id)
        """
        visualizer = TrainingVisualizer(plots_dir='training_plots')
        
        # Setup training infrastructure
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID and create directories
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"CVAE_b{dataloader.batch_size}_e{epochs}_z{self.z_dim}_{timestamp}"
        run_dir = save_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Setup directories and files
        log_file = run_dir / 'training.log'
        model_file = run_dir / 'best_model.pt'
        
        # Initialize training state
        device = next(self.parameters()).device
        scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
        best_loss = float('inf')


        # Disable mixed precision if on CPU
        has_gpu = torch.cuda.is_available()
        if use_mixed_precision or not has_gpu or device == "cpu":
            use_mixed_precision = False
            print("\nMixed precision training disabled (not supported on CPU)")

        # Get reference matrix
        refMatrix = np.vstack([dataloader.dataset.ref for _ in range(dataloader.batch_size)])
        refMatrix = torch.tensor(refMatrix, device=device).float()
        #print('Shape of the reference matrix : ', refMatrix.shape)
        
        # Initialize mixed precision training
        scaler = GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
        epoch_metrics = {
        'loss': 0.0,
        'reconstruction_loss': 0.0,
        'kl_loss': 0.0,
        'latent_reg': 0.0,
        'kl_weight': 0.0
        }
        
        # Training loop
        for epoch in range(epochs):
            
            self.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                              desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_idx, (maldi, ccf) in progress_bar:
                maldi, ccf = maldi.to(device), ccf.to(device)
                
                # Forward pass with optional mixed precision
                if scaler is not None:
                    with autocast():
                        recon, mu, logvar = self(maldi, ccf)
                        loss_dict = self.loss_func(recon, refMatrix, mu, logvar, 
                                                 epoch=epoch, epochs=epochs)
                    
                    # Backward pass with scaled gradients
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss_dict['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon, mu, logvar = self(maldi, ccf)
                    #print('Shape of the reconstructed matrix : ', recon.shape)
                    loss_dict = self.loss_func(recon, refMatrix, mu, logvar, 
                                             epoch=epoch, epochs=epochs)
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss_dict['loss'].backward()
                    optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                # Update progress
                epoch_loss += loss_dict['loss'].item()
                if batch_idx % print_every == 0:
                    progress_bar.set_postfix({
                        'loss': f"{loss_dict['loss'].item():.4f}",
                        'recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                        'kl': f"{loss_dict['kl_loss'].item():.4f}"
                    })

                # Update metrics with proper type handling
                for key, value in loss_dict.items():
                    if torch.is_tensor(value):
                        epoch_metrics[key] += value.item()
                    else:
                        epoch_metrics[key] += float(value)
            
            # Save best model
            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(model_file, epoch, optimizer, loss_dict)

            # Compute epoch averages
            for key in epoch_metrics:
                epoch_metrics[key] /= len(dataloader)

            # Produce plots by calling dedicated method
            visualizer.update_loss_plots(epoch, epoch_metrics, epochs) 
        
        return epoch_metrics, avg_loss, run_id

    @torch.no_grad()
    def predict(self, maldi: torch.Tensor, ccf: torch.Tensor, sample: str = 'mean') -> torch.Tensor:
        """
        Generate predictions with the model.
        
        Args:
            maldi: Input MALDI data
            ccf: Input CCF coordinates
            sample: Sampling method ('mean' or 'sample')
            
        Returns:
            Reconstructed data
        """
        self.eval()
        recon, mu, logvar = self(maldi, ccf)
        return recon if sample == 'mean' else self.reparameterize(mu, logvar)

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        epoch: int,
        optimizer: torch.optim.Optimizer,
        loss_dict: Dict
    ) -> None:
        """
        Save a model checkpoint.
        
        Args:
            filepath: Path to save the checkpoint
            epoch: Current epoch
            optimizer: Optimizer state
            loss_dict: Dictionary of current losses
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_dict['loss'].item(),
            'beta': self.beta,
            'z_dim': self.z_dim
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Union[str, Path], map_location: Optional[str] = None) -> Dict:
        """
        Load a model checkpoint.
        
        Args:
            filepath: Path to the checkpoint
            map_location: Device mapping for loading the checkpoint
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    @torch.no_grad()
    def interpolate(
        self,
        maldi1: torch.Tensor,
        maldi2: torch.Tensor,
        ccf: torch.Tensor,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Generate interpolations between two MALDI samples.
        
        Args:
            maldi1: First MALDI sample
            maldi2: Second MALDI sample
            ccf: CCF coordinates
            steps: Number of interpolation steps
            
        Returns:
            Interpolated samples
        """
        self.eval()
        mu1, logvar1 = self.encoder(maldi1, ccf)
        mu2, logvar2 = self.encoder(maldi2, ccf)
        
        alphas = torch.linspace(0, 1, steps, device=self.device)
        interpolated = []
        
        for alpha in alphas:
            mu = mu1 * (1 - alpha) + mu2 * alpha
            recon = self.decoder(mu, ccf)
            interpolated.append(recon)
        
        return torch.stack(interpolated)

    @torch.no_grad()
    def predict_with_coordinates(self, coords, noise=None):
            """
            Generate reconstructed spatial distribution from noise vector and coordinates
            
            Args:
                coords: tensor of shape [batch_size=1, 3, n_spatial_points]
                noise: Optional pre-defined noise tensor. If None, random noise will be generated
                      Expected shape: [batch_size=1, latent_dim]
            
            Returns:
                mu_inf: Reconstructed mean spatial distribution [batch_size=1, n_spatial_points]
                logvar_inf: Log variance of the reconstruction [batch_size=1, n_spatial_points]
            """
            self.eval()
            batch_size = 1
            
            # Generate random noise if not provided
            if noise is None:
                noise = torch.randn(batch_size, self.z_dim, device=self.device)
            
            # Process latent noise vector
            noise_decoded = self.decoder.latent_decoder(noise)  # [1, hidden_dim]
            
            # Process each coordinate dimension
            coord_decodings = []
            for i, decoder in enumerate(self.decoder.coord_decoder):
                coord = coords[:,i,:]  # [1, n_spatial_points]
                decoded = decoder(coord)  # [1, hidden_dim//3]
                coord_decodings.append(decoded)
            
            # Combine coordinate decodings
            coords_decoded = torch.cat(coord_decodings, dim=1)  # [1, hidden_dim]
            
            # Apply attention to noise decoded
            noise_decoded = noise_decoded.unsqueeze(0)
            noise_decoded, _ = self.decoder.attention(noise_decoded, noise_decoded, noise_decoded)
            noise_decoded = noise_decoded.squeeze(0)
            
            # Combine noise and coordinate information
            combined = torch.cat([noise_decoded, coords_decoded], dim=1)
            
            # Generate final spatial distribution
            h = self.decoder.combined_decoder(combined)
            
            # Generate output with residual connections
            skip_output = self.decoder.skip_connection(noise)
            mu_inf = self.decoder.decoder_mu_residual(h) + skip_output
            logvar_inf = self.decoder.decoder_logvar_residual(h)
            
            return mu_inf, logvar_inf

    @torch.no_grad()
    def predict_with_coordinates_multiple_passes(self, coords, refData, n_inf_full_pass=5, plot_errors=True, noise=None):
        """
        Generate reconstructed spatial distribution with multiple passes through the full VAE
        
        Args:
            coords: tensor of shape [batch_size=1, 3, n_spatial_points]
            refData: Reference data tensor [batch_size=1, n_spatial_points]
            n_inf_full_pass: Number of full passes through the VAE after initial decoder step
            noise: Optional pre-defined noise tensor. If None, random noise will be generated
            plot_errors: Boolean to control whether to plot reconstruction errors
        
        Returns:
            mu_inf: Final reconstructed mean spatial distribution [batch_size=1, n_spatial_points]
            logvar_inf: Final log variance of the reconstruction [batch_size=1, n_spatial_points]
            reconstruction_errors: List of mean reconstruction errors for each iteration
            fig: matplotlib figure if plot_errors=True, else None
        """
        self.eval()
        batch_size = 1
        reconstruction_errors = []
        iteration_numbers = []  # For plotting
        error_values = []      # For plotting
        
        # Initial reconstruction from decoder
        if noise is None:
            noise = torch.randn(batch_size, self.z_dim, device=self.device)
        
        # Get initial reconstruction from decoder using sample method
        recon_inf = self.decoder.forward(noise, coords)
        #print(mu_inf.shape)
        #print(refData.shape)
        
        # Calculate initial reconstruction error
        initial_error = torch.div( torch.mean(torch.abs(recon_inf - refData)), torch.mean(refData) ).item()
        reconstruction_errors.append(('initial', initial_error))
        iteration_numbers.append(0)
        error_values.append(initial_error)
        
        # Multiple passes through the full VAE
        for i in range(n_inf_full_pass):
            # Encode current reconstruction (using mu from previous decoder output)
            mu_z, logvar_z = self.encoder.forward(recon_inf, coords)
            
            # Use reparameterization trick for sampling latent vector
            std_z = torch.exp(0.5 * logvar_z)
            eps = torch.randn_like(std_z)
            z = mu_z + eps * std_z
            
            # Decode the sampled latent vector using sample method
            recon_inf = self.decoder.forward(z, coords)
            
            # Calculate reconstruction error for this iteration
            current_error = torch.div( torch.mean(torch.abs(recon_inf - refData)), torch.mean(refData) ).item()
            reconstruction_errors.append((f'iteration_{i+1}', current_error))
            iteration_numbers.append(i + 1)
            error_values.append(current_error)
            if current_error < error_values[i]:
                recon_inf_final = recon_inf
                #logvar_inf_final = logvar_inf
            
            # Print progress
            print(f'Iteration {i+1}/{n_inf_full_pass}, Mean Error: {current_error:.6f}')
        
        # Create error evolution plot
        fig = None
        if plot_errors:
            fig = plt.figure(figsize=(10, 6))
            plt.plot(iteration_numbers, error_values, 'b-o', linewidth=2, markersize=8)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Mean Relative Error', fontsize=12)
            plt.title('Reconstruction Error Evolution Over Iterations', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add error values as text above points
            for i, err in zip(iteration_numbers, error_values):
                plt.annotate(f'{err:.6f}', 
                            (i, err), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            fontsize=10)
            
            plt.tight_layout()
            
            # Save the plot as PNG
            filename = "ReconstructionError_condVAE.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return recon_inf_final, reconstruction_errors, fig
    
    def plot_error_comparison(errors1, errors2=None, labels=['Base run'], title='Error Comparison'):
        """
        Plot error evolution for one or two runs side by side
        
        Args:
            errors1: List of tuples (iteration_name, error_value) from first run
            errors2: Optional list of tuples from second run for comparison
            labels: List of labels for the runs
            title: Plot title
        
        Returns:
            matplotlib figure
        """
        fig = plt.figure(figsize=(12, 6))
        
        # Plot first run
        iterations1 = range(len(errors1))
        error_values1 = [e[1] for e in errors1]
        plt.plot(iterations1, error_values1, 'b-o', linewidth=2, markersize=8, label=labels[0])
        
        # Plot second run if provided
        if errors2 is not None:
            iterations2 = range(len(errors2))
            error_values2 = [e[1] for e in errors2]
            plt.plot(iterations2, error_values2, 'r-o', linewidth=2, markersize=8, label=labels[1])
            plt.legend()
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Mean Relative Error', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        return fig
