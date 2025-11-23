"""
Aissam HAMIDA – Biomedical Engineering Department, National School of Arts and Crafts, Rabat.

Filtered Backprojection (FBP) Implementation for CT Image Reconstruction
This code implements the classical FBP algorithm with all necessary steps
Note: This is the analytical method commonly used in clinical CT scanners
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy.ndimage import rotate
import cv2
from skimage.transform import radon, iradon
import os

input_image = "sinogram.npy"  # Input sinogram file in .npy format
output_path = "reconstructed_images/"  # Output directory for results

def load_sinogram_data(file_path):
    """
    Load sinogram data from .npy file
    Returns sinogram and angles if available in the file
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        # Handle different data formats that might be saved in the .npy file
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                print("Loaded 2D sinogram, generating angles...")
                sinogram = data
                angles = np.linspace(0, 180, sinogram.shape[1], endpoint=False)
                return sinogram, angles
            elif data.ndim == 1:
                # Might be a dictionary or structured array
                if data.dtype.names is not None:
                    if 'sinogram' in data.dtype.names:
                        sinogram = data['sinogram']
                        angles = data['angles'] if 'angles' in data.dtype.names else np.linspace(0, 180, sinogram.shape[1], endpoint=False)
                        return sinogram, angles
        # If we get here, try to handle as dictionary-like object
        if hasattr(data, 'item'):
            data_dict = data.item()
            if 'sinogram' in data_dict:
                sinogram = data_dict['sinogram']
                angles = data_dict.get('angles', np.linspace(0, 180, sinogram.shape[1], endpoint=False))
                return sinogram, angles
    except Exception as e:
        print(f"Error loading sinogram file: {e}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_sinogram()
    
    print("Could not load sinogram, generating synthetic data...")
    return generate_synthetic_sinogram()

def generate_synthetic_sinogram():
    """
    Generate synthetic Shepp-Logan phantom and its sinogram for testing
    This is used when no sinogram file is available or there's loading errors
    """
    from skimage.data import shepp_logan_phantom
    print("Creating synthetic Shepp-Logan phantom...")
    phantom = shepp_logan_phantom()
    phantom = cv2.resize(phantom, (256, 256))  # Resize to standard size
    
    angles = np.linspace(0, 180, 180, endpoint=False)
    sinogram = radon(phantom, theta=angles, circle=True)
    
    print(f"Synthetic sinogram shape: {sinogram.shape}")
    print(f"Angles range: {angles[0]} to {angles[-1]} degrees")
    
    return sinogram, angles

def ramp_filter(size):
    """
    Create ramp filter for frequency domain filtering
    This filter compensates for the blurring in simple backprojection
    The ramp filter is essential for proper FBP reconstruction
    """
    n = np.concatenate((np.arange(1, size//2 + 1), np.arange(size//2, 0, -1)))
    ramp = np.zeros(size)
    ramp[:len(n)] = n
    return ramp

def apply_ramp_filter(sinogram):
    """
    Apply ramp filter to sinogram in frequency domain
    This step is crucial for compensating the 1/r blurring in backprojection
    """
    print("Applying ramp filter to sinogram...")
    filtered_sinogram = np.zeros_like(sinogram)
    
    # Process each projection (each column of the sinogram)
    for i in range(sinogram.shape[1]):
        projection = sinogram[:, i]
        
        # Pad projection to avoid edge artifacts in FFT
        pad_width = 100
        padded_projection = np.pad(projection, (pad_width, pad_width), mode='constant')
        
        # Compute FFT
        fft_proj = fft(padded_projection)
        frequencies = fftfreq(len(padded_projection))
        
        # Create and apply ramp filter
        ramp = np.abs(frequencies)
        filtered_fft = fft_proj * ramp
        
        # Inverse FFT
        filtered_proj = np.real(ifft(filtered_fft))
        
        # Remove padding and store
        filtered_sinogram[:, i] = filtered_proj[pad_width:-pad_width]
    
    print("Ramp filtering completed")
    return filtered_sinogram

def backprojection(filtered_sinogram, angles, reconstruction_size=256):
    """
    Perform backprojection of filtered sinogram
    This step smears each filtered projection back across the image space
    """
    print("Starting backprojection...")
    reconstruction = np.zeros((reconstruction_size, reconstruction_size))
    center = reconstruction_size // 2
    
    # Convert angles to radians for trigonometric functions
    angles_rad = np.deg2rad(angles)
    
    # Create coordinate system
    x = np.arange(reconstruction_size) - center
    y = np.arange(reconstruction_size) - center
    X, Y = np.meshgrid(x, y)
    
    # For each projection angle
    for i, angle in enumerate(angles_rad):
        if i % 30 == 0:  # Progress indicator
            print(f"Processing angle {i}/{len(angles)}...")
        
        # Calculate projection coordinates
        # This is the core of backprojection - mapping image coordinates to sinogram coordinates
        proj_coords = X * np.cos(angle) + Y * np.sin(angle) + center
        
        # Interpolate values from filtered sinogram
        # We use linear interpolation for better quality
        proj_values = np.interp(proj_coords, 
                               np.arange(filtered_sinogram.shape[0]), 
                               filtered_sinogram[:, i],
                               left=0, right=0)
        
        # Accumulate in reconstruction
        reconstruction += proj_values
    
    # Normalize the reconstruction
    reconstruction = reconstruction * np.pi / (2 * len(angles))
    
    print("Backprojection completed")
    return reconstruction

def fbp_reconstruction(sinogram, angles):
    """
    Complete FBP reconstruction pipeline
    This follows the standard FBP workflow used in clinical CT systems
    """
    print("Starting FBP reconstruction...")
    print(f"Sinogram shape: {sinogram.shape}")
    print(f"Number of angles: {len(angles)}")
    
    # Step 1: Preprocessing - normalize sinogram
    print("Step 1: Preprocessing sinogram...")
    sinogram = sinogram.astype(np.float64)
    sinogram = sinogram - np.min(sinogram)  # Ensure non-negative values
    sinogram = sinogram / np.max(sinogram)  # Normalize to [0,1]
    
    # Step 2: Apply ramp filter
    print("Step 2: Frequency domain filtering...")
    filtered_sinogram = apply_ramp_filter(sinogram)
    
    # Step 3: Backprojection
    print("Step 3: Backprojection...")
    reconstruction_size = max(sinogram.shape[0], 256)  # Ensure adequate size
    reconstructed_image = backprojection(filtered_sinogram, angles, reconstruction_size)
    
    # Step 4: Post-processing
    print("Step 4: Post-processing...")
    reconstructed_image = np.clip(reconstructed_image, 0, 1)  # Clip to valid range
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  # Convert to 8-bit
    
    print("FBP reconstruction completed successfully!")
    return reconstructed_image

def save_results(original_sinogram, reconstructed_image, angles):
    """
    Save reconstruction results and create visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save reconstructed image
    output_file = os.path.join(output_path, "fbp_reconstruction.png")
    plt.imsave(output_file, reconstructed_image, cmap='gray')
    print(f"Reconstructed image saved to: {output_file}")
    
    # Save as numpy array
    np.save(os.path.join(output_path, "fbp_reconstruction.npy"), reconstructed_image)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot sinogram
    axes[0,0].imshow(original_sinogram, cmap='gray', aspect='auto', 
                     extent=[0, len(angles), original_sinogram.shape[0], 0])
    axes[0,0].set_title('Input Sinogram')
    axes[0,0].set_xlabel('Projection Angle')
    axes[0,0].set_ylabel('Detector Position')
    
    # Plot reconstructed image
    axes[0,1].imshow(reconstructed_image, cmap='gray')
    axes[0,1].set_title('FBP Reconstructed Image')
    axes[0,1].axis('off')
    
    # Plot profile through center
    center_line = reconstructed_image[reconstructed_image.shape[0]//2, :]
    axes[1,0].plot(center_line)
    axes[1,0].set_title('Horizontal Center Profile')
    axes[1,0].set_xlabel('Pixel Position')
    axes[1,0].set_ylabel('Intensity')
    
    # Plot histogram
    axes[1,1].hist(reconstructed_image.flatten(), bins=50, alpha=0.7)
    axes[1,1].set_title('Intensity Histogram')
    axes[1,1].set_xlabel('Intensity')
    axes[1,1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fbp_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All results saved successfully!")

def main():
    """
    Main function to run the complete FBP reconstruction pipeline
    """
    print("=" * 60)
    print("FBP Image Reconstruction Pipeline")
    print("Aissam HAMIDA - Biomedical Engineering Department")
    print("=" * 60)
    
    try:
        # Load sinogram data
        print("Loading sinogram data...")
        sinogram, angles = load_sinogram_data(input_image)
        
        # Perform FBP reconstruction
        reconstructed_image = fbp_reconstruction(sinogram, angles)
        
        # Save results
        save_results(sinogram, reconstructed_image, angles)
        
        print("\nReconstruction completed successfully!")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()