# Copyright (c) 2025 Dr Pascal Corso
# Academic/Research Usage License
#
# This code is part of the mlibra project collaboration at the Swiss Data Science Center.
# Any use of this code or results derived from it in academic publications 
# requires appropriate attribution through co-authorship or adequate contribution
# acknowledgement as specified in the LICENSE file.

"""
Loss functions for the Conditional VAE.

This module implements various loss functions for training the Conditional VAE,
including improved VAE loss with cyclical KL annealing, progressive KL annealing,
and optional regularization terms.
"""

import torch
import math
import torch.nn.functional as F

class ImprovedVAELoss:
    """
    Improved VAE loss with cyclical KL annealing and better numerical stability.
    
    This loss function implements:
    - Cyclical KL annealing
    - Bounded reconstruction loss
    - Improved numerical stability
    - Optional L1 regularization
    
    Args:
        reduction (str): Reduction method ('mean' or 'sum')
        kl_min (float): Minimum KL weight
        kl_max (float): Maximum KL weight
        kl_cycles (int): Number of annealing cycles
        l1_weight (float): Weight for L1 regularization
    """
    def __init__(self, reduction='mean', kl_min=0.0, kl_max=0.5, kl_cycles=3, l1_weight=0.01):
        self.reduction = reduction
        self.kl_min = kl_min
        self.kl_max = kl_max
        self.kl_cycles = kl_cycles
        self.current_step = 0
        self.l1_weight = l1_weight
        
    def get_kl_weight(self):
        """Calculate cyclical KL annealing weight."""
        cycle_progress = (self.current_step % (self.epochs // self.kl_cycles)) / (self.epochs // self.kl_cycles)
        weight = 0.5 * (1 + math.sin(math.pi * (cycle_progress - 0.5)))
        return self.kl_min + (self.kl_max - self.kl_min) * weight
        
    def __call__(self, x_recon, x_target, z_mu, z_logvar, epoch=0, epochs=100):
        """
        Calculate the loss.
        
        Args:
            x_recon (torch.Tensor): Reconstructed data
            x_target (torch.Tensor): Target data
            z_mu (torch.Tensor): Latent mean
            z_logvar (torch.Tensor): Latent log variance
            epoch (int): Current epoch
            epochs (int): Total epochs
            
        Returns:
            dict: Dictionary containing:
                - loss: Total loss
                - reconstruction_loss: Reconstruction component
                - kl_loss: KL divergence component
                - kl_weight: Current KL weight
                - l1_reg: L1 regularization term
        """
        self.epochs = epochs
        self.current_step = epoch
        
        # Reconstruction loss with bounds enforcement
        mse = F.mse_loss(x_recon, x_target, reduction='none')
        bounds_penalty = 10.0 * (F.relu(x_recon - 1) + F.relu(-x_recon))
        recon_loss = mse + bounds_penalty
        
        if self.reduction == 'mean':
            recon_loss = torch.mean(recon_loss)
        else:
            recon_loss = torch.sum(recon_loss)
        
        # KL divergence with improved stability
        kl_div = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
        kl_div = torch.clamp(kl_div, min=0.0, max=500.0)  # Prevent extreme values
        
        if self.reduction == 'mean':
            kl_div = torch.mean(kl_div)
        else:
            kl_div = torch.sum(kl_div)
        
        # L1 regularization on latent space
        l1_reg = self.l1_weight * torch.mean(torch.abs(z_mu))
        
        # Combine losses with cyclical annealing
        kl_weight = self.get_kl_weight()
        total_loss = recon_loss + kl_weight * kl_div + l1_reg
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_div,
            'kl_weight': kl_weight,
            'latent_reg': l1_reg
        }

class ProgressiveKLVAELoss:
    """
    VAE loss with progressive KL annealing.
    
    This loss function implements:
    - Progressive KL annealing
    - Improved numerical stability
    - Spatial regularization
    
    Args:
        reduction (str): Reduction method ('mean' or 'sum')
        initial_kl_weight (float): Initial KL weight
        final_kl_weight (float): Final KL weight
        kl_warmup_epochs (int): Number of epochs for KL warmup
        spatial_weight (float): Weight for spatial regularization
    """
    def __init__(self, reduction='mean', initial_kl_weight=0.0, final_kl_weight=1.0,
                 kl_warmup_epochs=10, spatial_weight=0.1):
        self.reduction = reduction
        self.initial_kl_weight = initial_kl_weight
        self.final_kl_weight = final_kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.spatial_weight = spatial_weight
        
    def get_kl_weight(self, epoch):
        """Calculate progressive KL weight."""
        if epoch >= self.kl_warmup_epochs:
            return self.final_kl_weight
        return self.initial_kl_weight + (self.final_kl_weight - self.initial_kl_weight) * \
               (epoch / self.kl_warmup_epochs)
    
    def spatial_consistency_loss(self, x_recon, coords):
        """Calculate spatial consistency loss."""
        # Calculate pairwise distances in coordinate space
        coord_dists = torch.cdist(coords.transpose(1, 2), coords.transpose(1, 2))
        
        # Calculate pairwise differences in reconstruction space
        recon_diffs = torch.abs(x_recon.unsqueeze(2) - x_recon.unsqueeze(1))
        
        # Weight differences by inverse coordinate distances
        weights = 1.0 / (coord_dists + 1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        return torch.mean(weights * recon_diffs)
    
    def __call__(self, x_recon, x_target, z_mu, z_logvar, coords, epoch=0):
        """
        Calculate the loss.
        
        Args:
            x_recon (torch.Tensor): Reconstructed data
            x_target (torch.Tensor): Target data
            z_mu (torch.Tensor): Latent mean
            z_logvar (torch.Tensor): Latent log variance
            coords (torch.Tensor): Coordinate data
            epoch (int): Current epoch
            
        Returns:
            dict: Dictionary containing loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x_target, reduction=self.reduction)
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
        kl_div = torch.clamp(kl_div, min=0.0, max=50.0)
        
        if self.reduction == 'mean':
            kl_div = torch.mean(kl_div)
        else:
            kl_div = torch.sum(kl_div)
        
        # Spatial consistency loss
        spatial_loss = self.spatial_consistency_loss(x_recon, coords)
        
        # Progressive KL weight
        kl_weight = self.get_kl_weight(epoch)
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_div + self.spatial_weight * spatial_loss
        
        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_div,
            'kl_weight': kl_weight,
            'spatial_loss': spatial_loss
        }

class MVNLogLikelihoodLoss:
    """
    Multivariate Normal negative log likelihood loss.
    
    This loss is particularly useful for handling uncertainty in the reconstruction.
    
    Args:
        reduction (str): Reduction method ('mean' or 'sum')
    """
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        
    def __call__(self, y_true, mu, logvar):
        """
        Calculate negative log likelihood for multivariate normal distribution.
        
        Args:
            y_true (torch.Tensor): Ground truth values
            mu (torch.Tensor): Predicted mean
            logvar (torch.Tensor): Predicted log variance
            
        Returns:
            torch.Tensor: Negative log likelihood loss
        """
        sigma = torch.exp(0.5 * logvar)
        
        # Numerically stable version
        log_likelihood = -0.5 * (
            ((y_true - mu) / sigma).pow(2) +
            logvar +
            math.log(2 * math.pi)
        ).sum(dim=1)  # Sum over all dimensions
        
        if self.reduction == 'mean':
            return -log_likelihood.mean()
        return -log_likelihood.sum()