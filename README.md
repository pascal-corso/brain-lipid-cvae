# Brain Lipids Conditional Variational Autoencoder (c-VAE)

A Conditional Variational Autoencoder (CVAE) implementation for enhancing spatial distribution of mouse brain lipid data. This project uses deep learning to improve the quality and resolution of Matrix-Assisted Laser Desorption/Ionization (MALDI) Time-of-Flight Mass Spectrometry Imaging (MSI) data.

## Overview

This repository contains a PyTorch implementation of a conditional VAE specifically designed for processing and enhancing brain lipid mass spectrometry data. The model incorporates spatial information through CCF (Common Coordinate Framework) coordinates to produce improved spatial distributions.

## Features

- Conditional VAE architecture optimised for spatial data
- Enhanced encoder and decoder with attention mechanisms
- Progressive KL divergence annealing
- Comprehensive data preprocessing pipeline
- Visualisation tools for results analysis
- Support for both CPU and GPU training
- Mixed precision training support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pascal-corso/brain-lipids-cvae.git
cd brain-lipids-cvae
```

2. Create a virtual environment with venv or conda

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The model expects input data in the following format:
- MALDI-MSI data
- CCF coordinates for spatial information
- Reference data for validation

Place your data files in the `data/` directory. Supported formats include:
- H5 files for large matrices
- Parquet file for original MALDI data

## Usage

### Training

```python
from models.cvae import ConditionalVAE
from utils.dataloader import MALDIDataset, create_dataloader

# Configure model parameters
config = {
    'hidden_dim': 512,
    'latent_dim': 256,
    'beta': 0.1,
    'batch_size': 128,
    'learning_rate': 5e-4,
    'weight_decay': 0.01,
    'dropout_rate': 0.1,
    'epochs': 60
}

# Initialize model
model = ConditionalVAE(
    maldi_dim=n_spatial_points,
    ccf_dim=coords_dim,
    hidden_dim=config['hidden_dim'],
    latent_dim=config['latent_dim'],
    beta=config['beta']
)

# Train model
model.train_model(
    dataloader=dataloader,
    optimizer=optimizer,
    epochs=config['epochs']
)
```

### Inference

The Jupyter notebook `3_inference_results.ipynb` in `notebooks/` provides an example for reconstructing spatially enhanced brain lipid data using the trained c-VAE model.

## Project Structure

- `models/`: Neural network architecture
- `utils/`: Helper functions and data processing utilities
- `notebooks/`: Jupyter notebooks for examples and visualization
- `data/`: Directory for dataset storage

## Results Visualisation

The package includes several visualisation tools:
- Training loss plots
- Reconstruction error analysis
- Spatial distribution comparisons
- Log-log scatter plots for quality assessment

## Requirements

See `requirements.txt` for a complete list.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under an Academic Code Usage License - see the LICENSE file for details.


## Acknowledgments

- Thanks to the Laboratory of Brain Development and Biological Data Science as well as Lipid Cell Biology Lab of the Ecole Polytechnique Fédérale de Lausanne (EPFL) for providing the MALDI-MSI data
- Special thanks to Dr Ekaterina Krymova (Swiss Data Science Center) and Dr Daniel Trejo Banos (Swiss Data Science Center) for their input.

## Contact

Dr Pascal Corso - pascal.corso@hotmail.com