# Copyright (c) 2025 Dr Pascal Corso
# Academic/Research Usage License
#
# This code is part of the mlibra project collaboration at the Swiss Data Science Center.
# Any use of this code or results derived from it in academic publications 
# requires appropriate attribution through co-authorship or adequate contribution
# acknowledgement as specified in the LICENSE file.

"""
Enhanced encoder module for the Conditional VAE.

This module implements an enhanced encoder architecture with multi-scale feature extraction,
attention mechanisms, and coordinate processing for brain lipid spatial distribution enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSEBlock(nn.Module):
    """
    Enhanced Squeeze-and-Excitation block with improved channel attention.
    
    Args:
        channels (int): Number of input channels
        reduction (int): Channel reduction factor for the bottleneck
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the SE block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, N]
            
        Returns:
            torch.Tensor: Attention-weighted tensor of same shape as input
        """
        b, c, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ImprovedResidualBlock(nn.Module):
    """
    Enhanced residual block with dual-path architecture and attention.
    
    Args:
        dim (int): Input dimension
        dropout_rate (float): Dropout rate for regularization
    """
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Main path
        self.conv1 = nn.Linear(dim, dim)
        self.conv2 = nn.Linear(dim, dim)
        
        # Spatial attention path
        self.spatial_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.se = EnhancedSEBlock(dim)
        
    def forward(self, x):
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Processed tensor with residual connection
        """
        residual = x
        
        # Main path
        out = self.norm1(x)
        out = F.gelu(self.conv1(out))
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.conv2(out)
        
        # Spatial attention
        spatial_weight = self.spatial_gate(x)
        out = out * spatial_weight
        
        # Apply SE and residual
        out = self.se(out.unsqueeze(-1)).squeeze(-1)
        return residual + out

class EnhancedEncoder(nn.Module):
    """
    Enhanced encoder with multi-scale feature extraction and attention mechanisms.
    
    This encoder is specifically designed for processing brain lipid spatial data
    with coordinate conditioning. It includes multi-scale feature extraction,
    coordinate processing and attention mechanisms.
    
    Args:
        n_spatial_points (int): Number of spatial points in input
        ccf_dim (int): Dimension of CCF coordinates
        hidden_dim (int): Dimension of hidden layers
        latent_dim (int): Dimension of latent space
        dropout_rate (float): Dropout rate for regularization
    """
    def __init__(self, n_spatial_points, ccf_dim, hidden_dim, latent_dim, dropout_rate=0.1):
        super().__init__()
        
        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_spatial_points, hidden_dim * (2 ** i)),
                nn.LayerNorm(hidden_dim * (2 ** i)),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for i in range(3)
        ])
        
        # Enhanced spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            ImprovedResidualBlock(hidden_dim * 4, dropout_rate),
            
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            ImprovedResidualBlock(hidden_dim * 2, dropout_rate),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            ImprovedResidualBlock(hidden_dim, dropout_rate)
        )
        
        # Coordinate processing with attention
        self.coord_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_spatial_points, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                ImprovedResidualBlock(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 3)
            ) for _ in range(3)
        ])
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Final encoding with uncertainty estimation
        self.mu_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3*(hidden_dim//3), latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.logvar_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3*(hidden_dim//3), latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Hardtanh(min_val=-6.0, max_val=2.0)  # Constrain logvar range
        )
        
    def forward(self, x, coords):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input MALDI data [batch_size, n_spatial_points]
            coords (torch.Tensor): Coordinate data [batch_size, 3, n_spatial_points]
            
        Returns:
            tuple: (mu, logvar) of the latent space distribution
                  mu: [batch_size, latent_dim]
                  logvar: [batch_size, latent_dim]
        """
        # Multi-scale feature extraction
        features = [extractor(x) for extractor in self.feature_extractors]
        spatial_features = torch.cat(features, dim=1)
        
        # Spatial encoding
        spatial_encoded = self.spatial_encoder(spatial_features)
        
        # Coordinate processing
        coord_features = []
        for i, processor in enumerate(self.coord_processors):
            coord = coords[:,i,:]  # Process each coordinate dimension
            encoded = processor(coord)
            coord_features.append(encoded)
        
        coords_encoded = torch.cat(coord_features, dim=1)
        
        # Self-attention on spatial features
        spatial_encoded = spatial_encoded.unsqueeze(1)
        attn_output, _ = self.self_attention(
            spatial_encoded, spatial_encoded, spatial_encoded
        )
        spatial_encoded = attn_output.squeeze(1)
        
        # Combine features
        combined = torch.cat([spatial_encoded, coords_encoded], dim=1)
        
        # Generate mu and logvar with uncertainty estimation
        mu = self.mu_encoder(combined)
        logvar = self.logvar_encoder(combined)
        
        return mu, logvar