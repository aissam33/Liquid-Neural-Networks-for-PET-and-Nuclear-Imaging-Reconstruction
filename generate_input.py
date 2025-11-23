import numpy as np
import cv2
import os
import glob
from skimage.transform import resize
import astra
import matplotlib.pyplot as plt

def normalize_medical_image(image_data):
    """
    Normalisation spéciale pour les images médicales avec plage dynamique élevée
    """
    # Méthode 1: Recadrage des valeurs extrêmes et normalisation
    p_low, p_high = np.percentile(image_data, [1, 99])
    image_clipped = np.clip(image_data, p_low, p_high)
    image_normalized = (image_clipped - p_low) / (p_high - p_low + 1e-8)
    
    print(f"Normalisation: recadrage à [{p_low:.2f}, {p_high:.2f}]")
    return image_normalized

def process_single_image(image_path, output_dir, num_angles=180):
    """
    Traite une seule image : lecture, normalisation, redimensionnement, génération de sinogramme
    """
    # Lire l'image en niveaux de gris
    print(f"Lecture de {os.path.basename(image_path)}...")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f" Impossible de lire {image_path}")
        return None
    
    # Normaliser l'image
    image_normalized = normalize_medical_image(image)
    
    # Redimensionner à 128x128
    image_resized = resize(image_normalized, (128, 128), preserve_range=True, anti_aliasing=True)
    
    # Générer le sinogramme avec ASTRA
    sinogram = generate_sinogram_astra(image_resized, num_angles)
    
    if sinogram is not None:
        # Sauvegarder le sinogramme
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.npy'
        output_path = os.path.join(output_dir, output_filename)
        np.save(output_path, sinogram)
        print(f"✓ Sinogramme sauvegardé: {output_filename}")
        return sinogram
    else:
        print(f" Erreur lors de la génération du sinogramme pour {os.path.basename(image_path)}")
        return None

def generate_sinogram_astra(image, num_angles=180):
    """
    Génère un sinogramme en utilisant ASTRA Toolbox
    """
    try:
        # Paramètres de géométrie
        vol_geom = astra.create_vol_geom(image.shape[0], image.shape[1])
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        proj_geom = astra.create_proj_geom('parallel', 1.0, image.shape[1], angles)
        
        # Créer le projecteur
        projector_id = astra.create_projector('linear', proj_geom, vol_geom)
        
        # Générer le sinogramme
        sinogram_id, sinogram = astra.create_sino(image, projector_id)
        
        # Nettoyer la mémoire ASTRA
        astra.data2d.delete(sinogram_id)
        astra.projector.delete(projector_id)
        
        return sinogram
        
    except Exception as e:
        print(f"Erreur ASTRA: {e}")
        return None

def visualize_sinogram_and_reconstruction(sinogram, original_image=None, title="Sinogramme"):
    """
    Visualise le sinogramme et une reconstruction de référence
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Afficher le sinogramme
    im1 = axes[0].imshow(sinogram, cmap='gray', aspect='auto')
    axes[0].set_title(title)
    axes[0].set_xlabel('Position du détecteur')
    axes[0].set_ylabel('Angle de projection')
    plt.colorbar(im1, ax=axes[0])
    
    # Reconstruction de référence avec FBP (skimage)
    if original_image is not None:
        from skimage.transform import iradon
        angles = np.linspace(0, 180, sinogram.shape[0], endpoint=False)
        reconstruction = iradon(sinogram, theta=angles, circle=True)
        
        im2 = axes[1].imshow(reconstruction, cmap='gray')
        axes[1].set_title('Reconstruction FBP de référence')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

def process_all_images(input_folder, output_folder, num_angles=180):
    """
    Traite toutes les images PNG du dossier d'entrée
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # Trouver tous les fichiers PNG
    png_files = glob.glob(os.path.join(input_folder, "*.png"))
    
    if not png_files:
        print(f" Aucun fichier PNG trouvé dans {input_folder}")
        return 0
    
    print(f" Trouvé {len(png_files)} fichiers PNG à traiter")
    print(f" Dossier d'entrée: {input_folder}")
    print(f" Dossier de sortie: {output_folder}")
    print("-" * 50)
    
    processed_count = 0
    first_sinogram = None
    first_image_name = None
    
    # Traiter chaque image
    for i, png_file in enumerate(png_files):
        print(f"\n[{i+1}/{len(png_files)}] Traitement de {os.path.basename(png_file)}...")
        
        sinogram = process_single_image(png_file, output_folder, num_angles)
        
        if sinogram is not None:
            processed_count += 1
            print(f" Traitement de {os.path.basename(png_file)} terminé!")
            
            # Sauvegarder le premier sinogramme pour visualisation
            if first_sinogram is None:
                first_sinogram = sinogram
                first_image_name = os.path.basename(png_file)
    
    # Afficher les statistiques finales
    print("\n" + "=" * 50)
    print(" RAPPORT FINAL")
    print("=" * 50)
    print(f" Images traitées avec succès: {processed_count}/{len(png_files)}")
    print(f" Dossier de sortie: {output_folder}")
    
    # Visualiser le premier sinogramme généré
    if first_sinogram is not None:
        print(f"\n🔍 Visualisation du premier sinogramme généré ({first_image_name})...")
        visualize_sinogram_and_reconstruction(first_sinogram, title=f"Sinogramme - {first_image_name}")
    
    return processed_count

def check_sinogram_quality(sinogram):
    """
    Vérifie la qualité du sinogramme généré
    """
    print(f"📏 Dimensions du sinogramme: {sinogram.shape}")
    print(f"📊 Statistiques du sinogramme:")
    print(f"   Min: {sinogram.min():.6f}")
    print(f"   Max: {sinogram.max():.6f}")
    print(f"   Moyenne: {sinogram.mean():.6f}")
    print(f"   Écart-type: {sinogram.std():.6f}")
    
    if np.isnan(sinogram).any() or np.isinf(sinogram).any():
        print(" ATTENTION: Le sinogramme contient des valeurs NaN ou Inf!")
        return False
    else:
        print("✓ Le sinogramme semble correct")
        return True

# Configuration principale
if __name__ == "__main__":
    # Chemins d'entrée et de sortie
    INPUT_FOLDER = "/Users/aissamhamida/Desktop/Soutenance_PFE_2025/Six_Try_SheppLogan Phantom/images"
    OUTPUT_FOLDER = "/Users/aissamhamida/Desktop/Soutenance_PFE_2025/Six_Try_SheppLogan Phantom/Sinograms_CT"
    
    # Paramètres de génération des sinogrammes
    NUM_ANGLES = 180  # 180 angles de projection entre 0 et π
    
    print(" Démarrage du traitement par lots des images CT...")
    print("=" * 60)
    
    # Traiter toutes les images
    processed_count = process_all_images(INPUT_FOLDER, OUTPUT_FOLDER, NUM_ANGLES)
    
    # Vérification finale
    if processed_count > 0:
        print(f"\n🎉 Traitement terminé avec succès!")
        print(f"📈 {processed_count} sinogrammes générés et sauvegardés dans:")
        print(f"   {OUTPUT_FOLDER}")
        
        # Vérifier la qualité du premier fichier généré
        npy_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.npy"))
        if npy_files:
            first_npy = npy_files[0]
            sinogram_check = np.load(first_npy)
            print(f"\n🔍 Vérification de la qualité du premier sinogramme:")
            check_sinogram_quality(sinogram_check)
    else:
        print("\n Aucune image n'a pu être traitée. Vérifiez les chemins et les fichiers.")