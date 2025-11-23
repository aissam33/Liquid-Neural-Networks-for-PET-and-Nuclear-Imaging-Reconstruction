"""
Aissam HAMIDA – Biomedical Engineering Department, National School of Arts and Crafts, Rabat.

OSEM (Ordered Subset Expectation Maximization) Implementation
This is an accelerated version of MLEM that uses subsets for faster convergence
Widely used in clinical PET systems for faster reconstruction times
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.ndimage import gaussian_filter

# User configuration - easy to modify
input_image = "sinogram.npy"  # Input sinogram file
output_path = "reconstructed_images/"  # Output directory

def load_sinogram_data(file_path):
    """
    Load sinogram data with support for various formats
    OSEM is particularly useful for large datasets common in PET/CT
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                print("Loaded 2D sinogram for OSEM reconstruction")
                sinogram = data
                image_size = sinogram.shape[0]  # Estimate image size
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
        print("Generating synthetic data with realistic noise...")
        return generate_realistic_pet_data()
    
    return generate_realistic_pet_data()

def generate_realistic_pet_data():
    """
    Generate realistic PET data with multiple regions of interest
    OSEM handles this type of data very well due to its subset approach
    """
    print("Creating realistic PET phantom for OSEM testing...")
    image_size = 128
    
    # Create a more complex phantom with multiple uptake regions
    phantom = np.zeros((image_size, image_size))
    center = image_size // 2
    
    # Background tissue (low uptake)
    y, x = np.ogrid[-center:image_size-center, -center:image_size-center]
    body_mask = x**2 + y**2 < (image_size//2.5)**2
    phantom[body_mask] = 0.8
    
    # Tumors with varying uptake levels
    phantom[center-15:center-5, center-40:center-20] = 2.5  # High uptake tumor
    phantom[center+10:center+25, center+15:center+30] = 1.8  # Medium uptake
    phantom[center-30:center-20, center+25:center+40] = 1.2  # Low uptake
    
    # Cold region (defect)
    phantom[center+5:center+20, center-25:center-10] = 0.3   # Cold spot
    
    # Generate sinogram with realistic PET statistics
    angles = np.linspace(0, 180, 120, endpoint=False)  # More angles for better quality
    from skimage.transform import radon
    clean_sinogram = radon(phantom, theta=angles, circle=True)
    
    # Add Poisson noise (characteristic of PET)
    max_count = 50  # Low counts to simulate realistic PET conditions
    clean_sinogram = clean_sinogram / np.max(clean_sinogram) * max_count
    noisy_sinogram = np.random.poisson(clean_sinogram)
    
    print(f"Generated realistic PET sinogram: {noisy_sinogram.shape}")
    return noisy_sinogram, image_size

def create_subset_indices(num_angles, num_subsets):
    """
    Create ordered subsets for OSEM reconstruction
    The ordering helps accelerate convergence compared to random subsets
    """
    indices = np.arange(num_angles)
    subset_indices = []
    
    for i in range(num_subsets):
        subset = indices[i::num_subsets]  # Strided access for good angular coverage
        subset_indices.append(subset)
    
    return subset_indices

def osem_reconstruction(sinogram, image_size, num_subsets=8, iterations=10):
    """
    OSEM reconstruction algorithm - faster convergence than MLEM
    Uses subsets of projections to update the image more frequently
    """
    print(f"Starting OSEM reconstruction with {num_subsets} subsets and {iterations} iterations...")
    
    # Initialize with uniform image
    current_estimate = np.ones((image_size, image_size)) * 0.1
    current_estimate = current_estimate.astype(np.float64)
    
    # Get system matrix
    from skimage.transform import radon, iradon
    num_angles = sinogram.shape[1]
    angles = np.linspace(0, 180, num_angles, endpoint=False)
    
    def forward_project_subset(image, subset_indices):
        """Forward project using only a subset of angles"""
        subset_angles = angles[subset_indices]
        return radon(image, theta=subset_angles, circle=True), subset_angles
    
    def back_project_subset(sinogram_subset, subset_angles):
        """Backproject a subset of sinogram data"""
        return iradon(sinogram_subset, theta=subset_angles, circle=True, filter=None)
    
    # Create ordered subsets
    subset_indices_list = create_subset_indices(num_angles, num_subsets)
    
    # Prepare measured data
    measured_sinogram = sinogram.astype(np.float64) + 1e-12
    
    # Storage for convergence monitoring
    convergence_history = []
    subset_times = []
    
    total_start_time = time.time()
    
    # Main OSEM iteration loop
    for iteration in range(iterations):
        iteration_start = time.time()
        print(f"OSEM Iteration {iteration+1}/{iterations}")
        
        # Process each subset in order
        for subset_num, subset_indices in enumerate(subset_indices_list):
            subset_start = time.time()
            
            # Extract subset of measured data
            measured_subset = measured_sinogram[:, subset_indices]
            
            # Forward project current estimate using subset
            estimated_subset, subset_angles = forward_project_subset(current_estimate, subset_indices)
            estimated_subset = np.maximum(estimated_subset, 1e-12)
            
            # Calculate correction for this subset
            correction_ratio_subset = measured_subset / estimated_subset
            
            # Backproject the correction
            correction_image = back_project_subset(correction_ratio_subset, subset_angles)
            
            # Update current estimate (OSEM update)
            current_estimate = current_estimate * correction_image
            
            # Apply non-negativity constraint
            current_estimate = np.maximum(current_estimate, 0)
            
            subset_time = time.time() - subset_start
            subset_times.append(subset_time)
            
            if subset_num % 4 == 0:  # Reduced output frequency
                current_likelihood = calculate_poisson_likelihood(measured_sinogram, 
                                                                 forward_project_subset(current_estimate, 
                                                                                       np.arange(num_angles))[0])
                print(f"  Subset {subset_num+1}/{num_subsets}, Likelihood: {current_likelihood:.4f}")
        
        # Store convergence info every iteration
        convergence_history.append(current_estimate.copy())
        
        iteration_time = time.time() - iteration_start
        print(f"  Iteration {iteration+1} completed in {iteration_time:.2f}s")
    
    total_time = time.time() - total_start_time
    print(f"OSEM reconstruction completed in {total_time:.2f} seconds")
    print(f"Average subset time: {np.mean(subset_times):.3f}s")
    
    return current_estimate, convergence_history

def calculate_poisson_likelihood(measured, estimated):
    """
    Calculate Poisson likelihood for convergence monitoring
    Important for checking if reconstruction is improving
    """
    # Avoid log(0) issues
    estimated = np.maximum(estimated, 1e-12)
    likelihood = np.sum(measured * np.log(estimated) - estimated)
    return likelihood

def apply_pet_post_processing(image):
    """
    Apply PET-specific post-processing
    Includes smoothing and normalization suitable for PET images
    """
    print("Applying PET-optimized post-processing...")
    
    # Gaussian smoothing - typical for PET to reduce noise
    smoothed = gaussian_filter(image, sigma=1.2)
    
    # Contrast enhancement for better visualization
    from skimage import exposure
    enhanced = exposure.equalize_adapthist(smoothed, clip_limit=0.03)
    
    # Convert to 8-bit for display
    normalized = (enhanced * 255).astype(np.uint8)
    
    return normalized

def save_osem_results(sinogram, reconstruction, history, image_size):
    """
    Save OSEM results with comprehensive analysis
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Apply post-processing
    processed_recon = apply_pet_post_processing(reconstruction)
    
    # Save final image
    plt.imsave(os.path.join(output_path, "osem_reconstruction.png"), 
               processed_recon, cmap='hot')
    np.save(os.path.join(output_path, "osem_reconstruction.npy"), reconstruction)
    
    # Create detailed analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sinogram
    axes[0,0].imshow(sinogram, cmap='gray', aspect='auto')
    axes[0,0].set_title('PET Sinogram Input')
    axes[0,0].set_xlabel('Projection Angle')
    axes[0,0].set_ylabel('Detector')
    
    # Final reconstruction
    axes[0,1].imshow(processed_recon, cmap='hot')
    axes[0,1].set_title('OSEM Reconstruction')
    axes[0,1].axis('off')
    
    # Convergence plot
    if len(history) > 1:
        max_vals = [np.max(hist) for hist in history]
        axes[0,2].plot(range(1, len(history)+1), max_vals, 'ro-')
        axes[0,2].set_title('Maximum Value Convergence')
        axes[0,2].set_xlabel('Iteration')
        axes[0,2].set_ylabel('Max Intensity')
    
    # Activity profile
    center_profile = processed_recon[image_size//2, :]
    axes[1,0].plot(center_profile, 'g-', linewidth=2)
    axes[1,0].set_title('Activity Profile (Center)')
    axes[1,0].set_xlabel('Pixel')
    axes[1,0].set_ylabel('Activity')
    
    # Histogram of uptake values
    axes[1,1].hist(reconstruction.flatten(), bins=50, alpha=0.7, color='orange')
    axes[1,1].set_title('Uptake Value Distribution')
    axes[1,1].set_xlabel('Uptake Level')
    axes[1,1].set_ylabel('Frequency')
    
    # Comparison of early vs late iteration
    if len(history) > 5:
        early = history[2]
        late = history[-1]
        diff = late - early
        im = axes[1,2].imshow(diff, cmap='coolwarm')
        axes[1,2].set_title('Improvement (Late - Early)')
        axes[1,2].axis('off')
        plt.colorbar(im, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'osem_detailed_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OSEM results saved to: {output_path}")

def compare_osem_vs_mlem(sinogram, image_size):
    """
    Compare OSEM with standard MLEM to demonstrate acceleration
    """
    print("\n" + "="*50)
    print("Performance Comparison: OSEM vs MLEM")
    print("="*50)
    
    # Time OSEM reconstruction
    start_time = time.time()
    osem_result, _ = osem_reconstruction(sinogram, image_size, num_subsets=8, iterations=5)
    osem_time = time.time() - start_time
    
    # Time MLEM reconstruction (equivalent iterations)
    start_time = time.time()
    from mlem_code import mlem_reconstruction  # Assuming previous MLEM implementation
    mlem_result, _ = mlem_reconstruction(sinogram, image_size, iterations=5)
    mlem_time = time.time() - start_time
    
    print(f"OSEM time: {osem_time:.2f}s")
    print(f"MLEM time: {mlem_time:.2f}s")
    print(f"Speedup factor: {mlem_time/osem_time:.2f}x")
    
    return osem_result, mlem_result

def main():
    """
    Main OSEM reconstruction pipeline
    """
    print("=" * 60)
    print("OSEM Reconstruction - Accelerated PET Imaging")
    print("Aissam HAMIDA - Biomedical Engineering Department")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading data for OSEM reconstruction...")
        sinogram, image_size = load_sinogram_data(input_image)
        
        # Perform OSEM reconstruction
        final_image, history = osem_reconstruction(sinogram, image_size, 
                                                 num_subsets=12, iterations=8)
        
        # Save results
        save_osem_results(sinogram, final_image, history, image_size)
        
        # Optional: Compare with MLEM
        compare_choice = input("Perform OSEM vs MLEM comparison? (y/n): ")
        if compare_choice.lower() == 'y':
            compare_osem_vs_mlem(sinogram, image_size)
        
        print("\nOSEM reconstruction completed successfully!")
        print("OSEM provides faster convergence than MLEM for clinical applications")
        
    except Exception as e:
        print(f"Error in OSEM reconstruction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()