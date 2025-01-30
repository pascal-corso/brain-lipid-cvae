# Copyright (c) 2025 Dr Pascal Corso
# Academic/Research Usage License
#
# This code is part of the mlibra project collaboration at the Swiss Data Science Center.
# Any use of this code or results derived from it in academic publications 
# requires appropriate attribution through co-authorship or adequate contribution
# acknowledgement as specified in the LICENSE file.

"""
Enhanced decoder module for the Conditional VAE.

This module implements an enhanced decoder architecture with upsampling blocks,
cross-attention mechanisms, and coordinate conditioning for brain lipid spatial
distribution enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ImprovedResidualBlock, EnhancedSEBlock

class EnhancedDecoder(nn.Module):
    """
    Enhanced decoder with multi-scale reconstruction and attention mechanisms.
    
    This decoder is specifically designed for reconstructing brain lipid spatial data
    with coordinate conditioning. It includes upsampling blocks, coordinate processing,
    and attention mechanisms for high-quality spatial reconstruction.
    
    Args:
        n_spatial_points (int): Number of spatial points in output
        ccf_dim (int): Dimension of CCF coordinates
        hidden_dim (int): Dimension of hidden layers
        latent_dim (int): Dimension of latent space
        dropout_rate (float): Dropout rate for regularization
    """
    def __init__(self, n_spatial_points, ccf_dim, hidden_dim, latent_dim, dropout_rate=0.1):
        super(EnhancedDecoder, self).__init__()
        
        # Initial latent processing
        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            ImprovedResidualBlock(hidden_dim * 2, dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Enhanced coordinate processing
        self.coord_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_spatial_points, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                ImprovedResidualBlock(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 3)
            ) for _ in range(3)
        ])
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Upsampling decoder with skip connections
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * (2 ** i)),
                nn.LayerNorm(hidden_dim * (2 ** i)),
                nn.GELU(),
                ImprovedResidualBlock(hidden_dim * (2 ** i), dropout_rate)
            ) for i in range(3)
        ])
        
        # Final output layers
        self.final_processor = nn.Sequential(
            nn.Linear(hidden_dim * 7 + 3*(hidden_dim//3), hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            ImprovedResidualBlock(hidden_dim * 4, dropout_rate),
            nn.Linear(hidden_dim * 4, n_spatial_points)
        )
        
        # Output activation with learned scaling
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_shift = nn.Parameter(torch.zeros(1))
        
    def forward(self, z, coords):
        """
        Forward pass through the decoder.
        
        Args:
            z (torch.Tensor): Latent vector [batch_size, latent_dim]
            coords (torch.Tensor): Coordinate data [batch_size, ccf_dim, n_spatial_points]
            
        Returns:
            torch.Tensor: Reconstructed spatial distribution [batch_size, n_spatial_points]
        """
        # Process latent vector
        latent_features = self.latent_processor(z)
        
        # Process coordinates
        coord_features = []
        for i, processor in enumerate(self.coord_processors):
            coord = coords[:,i,:]
            processed = processor(coord)
            coord_features.append(processed)
        
        coords_processed = torch.cat(coord_features, dim=1)
        
        # Apply cross-attention
        latent_features = latent_features.unsqueeze(1)
        attn_output, _ = self.cross_attention(
            latent_features, latent_features, latent_features
        )
        latent_features = attn_output.squeeze(1)
        
        # Upsampling decoder blocks
        decoder_features = [block(latent_features) for block in self.decoder_blocks]
        combined_features = torch.cat(decoder_features + [coords_processed], dim=1)
        
        # Generate output with learned scaling
        output = self.final_processor(combined_features)
        output = torch.sigmoid(output * self.output_scale + self.output_shift)
        
        return output

    @torch.no_grad()
    def generate_samples(self, coords, num_samples=1, temperature=1.0):
        """
        Generate multiple samples for a given set of coordinates.
        
        Args:
            coords (torch.Tensor): Coordinate data [1, ccf_dim, n_spatial_points]
            num_samples (int): Number of samples to generate
            temperature (float): Sampling temperature (higher = more diverse)
            
        Returns:
            torch.Tensor: Generated samples [num_samples, n_spatial_points]
        """
        batch_size = coords.size(0)
        device = coords.device
        
        # Generate random latent vectors
        z = torch.randn(batch_size * num_samples, self.latent_dim, device=device)
        z = z * temperature
        
        # Expand coordinates for each sample
        coords_expanded = coords.repeat(num_samples, 1, 1)
        
        # Generate samples
        samples = self.forward(z, coords_expanded)
        
        return samples.view(num_samples, batch_size, -1)

    def interpolate_latent(self, z1, z2, coords, steps=10):
        """
        Generate samples by interpolating between two latent vectors.
        
        Args:
            z1 (torch.Tensor): First latent vector [1, latent_dim]
            z2 (torch.Tensor): Second latent vector [1, latent_dim]
            coords (torch.Tensor): Coordinate data [1, ccf_dim, n_spatial_points]
            steps (int): Number of interpolation steps
            
        Returns:
            torch.Tensor: Interpolated samples [steps, n_spatial_points]
        """
        with torch.no_grad():
            alphas = torch.linspace(0, 1, steps, device=z1.device)
            samples = []
            
            for alpha in alphas:
                z = z1 * (1 - alpha) + z2 * alpha
                sample = self.forward(z, coords)
                samples.append(sample)
            
            return torch.cat(samples, dim=0)