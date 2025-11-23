"""
Aissam HAMIDA – Biomedical Engineering Department, National School of Arts and Crafts, Rabat.

MAP (Maximum A Posteriori) Reconstruction with Regularization
This method incorporates prior knowledge through regularization terms
Provides better noise handling and edge preservation than pure ML methods
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.ndimage import gaussian_filter, laplace, sobel
from scipy.optimize import minimize

# User configuration - easy to modify  
input_image = "sinogram.npy"  # Input sinogram file
output_path = "reconstructed_images/"  # Output directory

def load_sinogram_data(file_path):
    """
    Load sinogram data for MAP reconstruction
    MAP is particularly useful for low-count or noisy data
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                print("Loaded sinogram for MAP reconstruction")
                sinogram = data
                image_size = sinogram.shape[0]
                return sinogram, image_size
            elif data.dtype.names is not None:
                if 'sinogram' in data.dtype.names:
                    sinogram = data['sinogram']
                    image_size = data.get('image_size', sinogram.shape[0])
                    return sinogram, image_size
                    
        if hasattr(data, 'item'):
            data_dict = data.item()
            if 'sinogram' in data_dict:
                sinogram = data_dict['sinogram']
                image_size = data_dict.get('image_size', sinogram.shape[0])
                return sinogram, image_size
                
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        print("Generating data with challenging noise conditions...")
        return generate_challenging_data()
    
    return generate_challenging_data()

def generate_challenging_data():
    """
    Generate data with challenging noise conditions where MAP excels
    Includes low counts and fine structures that test regularization
    """
    print("Creating challenging phantom for MAP reconstruction...")
    image_size = 128
    
    # Create phantom with fine details and varying contrast
    phantom = np.zeros((image_size, image_size))
    center = image_size // 2
    
    # Main structures
    y, x = np.ogrid[-center:image_size-center, -center:image_size-center]
    
    # Elliptical body
    body_mask = (x**2)/