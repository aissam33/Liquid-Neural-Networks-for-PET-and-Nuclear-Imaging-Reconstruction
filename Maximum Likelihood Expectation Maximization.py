"""
Aissam HAMIDA – Biomedical Engineering Department, National School of Arts and Crafts, Rabat.

MLEM (Maximum Likelihood Expectation Maximization) Implementation for PET Reconstruction
This is an iterative statistical method that provides better quality than FBP for noisy data
Commonly used in PET and SPECT reconstruction where statistics are poor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import os
import time

# User configuration - easy to modify
input_image = "sinogram.npy"  # Input sinogram file
output_path = "reconstructed_images/"  # Output directory

def load_sinogram_data(file_path):
    """
    Load sinogram data from .npy file
    Handles various data formats that might be encountered in practice
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Handle different data organization styles
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                print("Loaded 2D sinogram")
                sinogram = data
                # Estimate system matrix size based on sinogram dimensions
                image_size = sinogram.shape[0]  # Assuming square reconstruction
                return sinogram, image_size
            elif data.dtype.names is not None:
                if 'sinogram' in data.dtype.names:
                    sinogram = data['sinogram']
                    image_size = data.get('image_size', sinogram.shape[0])
                    return sinogram, image_size
        
        # Try dictionary format
        if hasattr(data, 'item'):
            data_dict = data.item()
            if 'sinogram' in data_dict:
                sinogram = data_dict['sinogram']
                image_size = data_dict.get('image_size', sinogram.shape[0])
                return sinogram, image_size
                
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        print("Generating synthetic PET-like data...")
        return generate_synthetic_pet_data()
    
    print("Using synthetic data for demonstration")
    return generate_synthetic_pet_data()

def generate_synthetic_pet_data():
    """
    Generate synthetic PET-like data with realistic noise and statistics
    PET data typically has Poisson statistics and lower counts than CT
    """
    print("Creating synthetic PET phantom...")
    image_size = 128
    # Create a simple phantom with hot and cold regions
    phantom = np.zeros((image_size, image_size))
    
    # Add some hot spots (tumors)
    center = image_size // 2
    y, x = np.ogrid[-center:image_size-center, -center:image_size-center]
    
    # Main body
    mask = x**2 + y**2 < (image_size//3)**2
    phantom[mask] = 1.0
    
    # Hot spots (tumors)
    phantom[center-10:center+10, center-30:center-10] = 2.0  # High uptake region
    phantom[center+20:center+30, center+10:center+25] = 1.5   # Medium uptake
    phantom[center-25:center-15, center+20:center+35] = 0.5   # Low uptake
    
    # Generate noisy sinogram with Poisson statistics
    angles = np.linspace(0, 180, 90, endpoint=False)  # Fewer angles for PET
    from skimage.transform import radon
    clean_sinogram = radon(phantom, theta=angles, circle=True)
    
    # Add Poisson noise (typical for PET)
    max_count = 100  # Low counts typical for PET
    clean_sinogram = clean_sinogram / np.max(clean_sinogram) * max_count
    noisy_sinogram = np.random.poisson(clean_sinogram)
    
    print(f"Synthetic PET sinogram shape: {noisy_sinogram.shape}")
    return noisy_sinogram, image_size

def create_system_matrix(image_size, sinogram_shape):
    """
    Create a simplified system matrix for PET
    In real systems, this would be pre-calculated based on scanner geometry
    This is a simplified version for demonstration purposes
    """
    print("Creating system matrix...")
    num_angles = sinogram_shape[1]
    num_detectors = sinogram_shape[0]
    
    # For demonstration, we'll use a simple forward projection model
    # In practice, this would include detector efficiency, attenuation, etc.
    def forward_project(image):
        from skimage.transform import radon
        return radon(image, theta=np.linspace(0, 180, num_angles, endpoint=False), circle=True)
    
    def back_project(sinogram):
        from skimage.transform import iradon
        return iradon(sinogram, theta=np.linspace(0, 180, num_angles, endpoint=False), circle=True, filter=None)
    
    return forward_project, back_project

def mlem_reconstruction(sinogram, image_size, iterations=20, subset_size=1):
    """
    MLEM reconstruction algorithm with optional ordered subsets (OSEM)
    This is the core iterative reconstruction algorithm used in PET
    """
    print(f"Starting MLEM reconstruction with {iterations} iterations...")
    
    # Initialize reconstruction with uniform image
    # Starting with uniform values helps convergence
    current_estimate = np.ones((image_size, image_size)) * 0.1
    current_estimate = current_estimate.astype(np.float64)
    
    # Get system matrix operators
    forward_project, back_project = create_system_matrix(image_size, sinogram.shape)
    
    # Prepare sinogram (avoid division by zero)
    measured_sinogram = sinogram.astype(np.float64) + 1e-12
    
    # Store reconstruction history for analysis
    reconstruction_history = []
    
    start_time = time.time()
    
    # Main MLEM iteration loop
    for iteration in range(iterations):
        iteration_start = time.time()
        
        # Forward projection: estimate what sinogram we would get from current image
        estimated_sinogram = forward_project(current_estimate)
        estimated_sinogram = np.maximum(estimated_sinogram, 1e-12)  # Avoid division by zero
        
        # Calculate correction factor: measured / estimated
        correction_ratio = measured_sinogram / estimated_sinogram
        
        # Backproject the correction factor
        correction_image = back_project(correction_ratio)
        
        # Update current estimate (MLEM update equation)
        current_estimate = current_estimate * correction_image
        
        # Apply non-negativity constraint (physical constraint)
        current_estimate = np.maximum(current_estimate, 0)
        
        # Store every 5th iteration for analysis
        if iteration % 5 == 0:
            reconstruction_history.append(current_estimate.copy())
        
        iteration_time = time.time() - iteration_start
        if iteration % 5 == 0:
            print(f"Iteration {iteration+1}/{iterations}, Time: {iteration_time:.2f}s")
            
            # Calculate current likelihood for monitoring convergence
            likelihood = calculate_likelihood(measured_sinogram, estimated_sinogram)
            print(f"  Likelihood: {likelihood:.4f}")
    
    total_time = time.time() - start_time
    print(f"MLEM reconstruction completed in {total_time:.2f} seconds")
    
    return current_estimate, reconstruction_history

def calculate_likelihood(measured, estimated):
    """
    Calculate Poisson likelihood for convergence monitoring
    This helps track how well the reconstruction matches the measured data
    """
    # Poisson log-likelihood (simplified)
    likelihood = np.sum(measured * np.log(estimated) - estimated)
    return likelihood

def apply_post_processing(reconstructed_image):
    """
    Apply post-processing to improve image quality
    This includes filtering and normalization steps
    """
    print("Applying post-processing...")
    
    # Gaussian smoothing to reduce noise
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(reconstructed_image, sigma=1.0)
    
    # Normalize to 0-255 range for display
    normalized = smoothed - np.min(smoothed)
    if np.max(normalized) > 0:
        normalized = normalized / np.max(normalized) * 255
    
    return normalized.astype(np.uint8)

def save_mlem_results(original_sinogram, final_reconstruction, history, image_size):
    """
    Save MLEM reconstruction results and create comprehensive analysis
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Apply post-processing to final reconstruction
    final_processed = apply_post_processing(final_reconstruction)
    
    # Save final reconstruction
    output_file = os.path.join(output_path, "mlem_reconstruction.png")
    plt.imsave(output_file, final_processed, cmap='hot')  # Hot colormap for PET
    np.save(os.path.join(output_path, "mlem_reconstruction.npy"), final_reconstruction)
    
    # Create comprehensive analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original sinogram
    axes[0,0].imshow(original_sinogram, cmap='gray', aspect='auto')
    axes[0,0].set_title('Measured Sinogram (PET Data)')
    axes[0,0].set_xlabel('Projection Angle')
    axes[0,0].set_ylabel('Detector Position')
    
    # Final reconstruction
    axes[0,1].imshow(final_processed, cmap='hot')
    axes[0,1].set_title('MLEM Reconstruction')
    axes[0,1].axis('off')
    
    # Convergence plot (if we have history)
    if len(history) > 1:
        iterations = np.arange(0, len(history) * 5, 5)
        profile_values = [hist[image_size//2, image_size//2] for hist in history]
        axes[0,2].plot(iterations, profile_values, 'bo-')
        axes[0,2].set_title('Convergence (Center Pixel)')
        axes[0,2].set_xlabel('Iteration')
        axes[0,2].set_ylabel('Intensity')
    
    # Profile through center
    center_profile = final_processed[image_size//2, :]
    axes[1,0].plot(center_profile)
    axes[1,0].set_title('Horizontal Profile')
    axes[1,0].set_xlabel('Pixel')
    axes[1,0].set_ylabel('Intensity')
    
    # Histogram
    axes[1,1].hist(final_reconstruction.flatten(), bins=50, alpha=0.7, color='red')
    axes[1,1].set_title('Activity Distribution')
    axes[1,1].set_xlabel('Activity Level')
    axes[1,1].set_ylabel('Frequency')
    
    # Iteration progression (if available)
    if len(history) > 3:
        axes[1,2].imshow(history[-1], cmap='hot')  # Last iteration
        axes[1,2].set_title(f'Iteration {len(history)*5}')
        axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'mlem_comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"MLEM results saved to: {output_path}")

def main():
    """
    Main function for MLEM reconstruction pipeline
    """
    print("=" * 60)
    print("MLEM Reconstruction for PET Imaging")
    print("Aissam HAMIDA - Biomedical Engineering Department")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading PET sinogram data...")
        sinogram, image_size = load_sinogram_data(input_image)
        
        # Perform MLEM reconstruction
        print("Starting iterative reconstruction...")
        final_image, history = mlem_reconstruction(sinogram, image_size, iterations=50)
        
        # Save results
        save_mlem_results(sinogram, final_image, history, image_size)
        
        print("\nMLEM reconstruction completed!")
        print("Note: MLEM typically provides better noise handling than FBP for PET data")
        
    except Exception as e:
        print(f"Error in MLEM reconstruction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()