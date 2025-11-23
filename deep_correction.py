import numpy as np
import cv2
import nibabel as nib
import time
import scipy.io as sio
from skimage.transform import iradon, radon, resize
from skimage import util
import random
import astra

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

def verify_model_compatibility(sinogram_shape):
    """
    Vérifie que les dimensions du sinogramme correspondent aux attentes du modèle
    """
    # Le modèle attend: (batch_size, 180, 128, 1) basé sur votre architecture
    expected_shape = (None, 180, 128, 1)  # 180 angles, 128 détecteurs
    
    print(f"\nVérification de compatibilité du modèle:")
    print(f"Le modèle attend: {expected_shape}")
    print(f"Vos sinogrammes: {sinogram_shape}")
    
    if sinogram_shape[1] == expected_shape[1] and sinogram_shape[2] == expected_shape[2]:
        print("✓ Les dimensions correspondent!")
        return True
    else:
        print("✗ Incompatibilité de dimensions!")
        print("Veuillez vérifier vos paramètres de génération de sinogrammes")
        return False

def check_data_distribution(data, name):
    """
    Vérifie la distribution des données pour détecter d'éventuels problèmes
    """
    print(f"\n{name} statistiques:")
    print(f"Min: {data.min():.6f}, Max: {data.max():.6f}")
    print(f"Moyenne: {data.mean():.6f}, Écart-type: {data.std():.6f}")
    
    if data.min() < -0.1 or data.max() > 1.1:
        print("⚠️ Attention: Les données peuvent nécessiter une re-normalisation")
        return False
    
    print("✓ La normalisation semble correcte")
    return True

def Aug_Rotate(aug, augNumber):
    """
    Fonction de rotation augmentée corrigée
    """
    Nim = aug.shape[0]
    ysize = aug.shape[1]
    xsize = aug.shape[2]
    
    tic1 = time.time()
    print('Augmentation des données ... ')

    newAug = np.zeros((Nim * augNumber, aug.shape[1], aug.shape[2], aug.shape[3]))
    
    cont = 0
    
    for n in range(Nim):
        augIm = np.squeeze(aug[n, :, :])
        newAug[cont, :, :, 0] = augIm
        cont += 1
        
        for a in range(augNumber - 1):
            Rotation = random.randint(0, 360)
            M = cv2.getRotationMatrix2D((xsize // 2, ysize // 2), Rotation, 1)
            AugImPos = cv2.warpAffine(augIm, M, (xsize, ysize))
            
            # Gestion des bordures noires
            AugImPos[AugImPos <= 0.01] = 0
            
            newAug[cont, :, :, 0] = AugImPos
            cont += 1
                
    newAug = (newAug - newAug.min()) / (newAug.max() - newAug.min() + 1e-8)
    
    toc1 = time.time()
    print('Augmentation terminée en ', (toc1 - tic1), ' !')
    
    return newAug

def Aug_translate(aug, augNumber):
    """
    Fonction de translation augmentée corrigée
    """
    Nim = aug.shape[0]
    xsize = aug.shape[1]
    ysize = aug.shape[2]
    
    tic1 = time.time()
    print('Augmentation des données ... ')

    newAug = np.zeros((Nim * augNumber, aug.shape[1], aug.shape[2], aug.shape[3]))
    
    cont = 0
    
    for n in range(Nim):
        augIm = np.squeeze(aug[n, :, :])
        newAug[cont, :, :, 0] = augIm
        cont += 1
        
        for a in range(augNumber - 1):
            TrasX = random.randint(-15, 15)
            TrasY = random.randint(-15, 15)
            M = np.float32([[1, 0, TrasX], [0, 1, TrasY]])
    
            AugImPos = cv2.warpAffine(augIm, M, (xsize, ysize))
            AugImPos[AugImPos <= 0.01] = 0
            
            newAug[cont, :, :, 0] = AugImPos
            cont += 1
                
    newAug = (newAug - newAug.min()) / (newAug.max() - newAug.min() + 1e-8)
    
    toc1 = time.time()
    print('Augmentation terminée en ', (toc1 - tic1), ' !')
    
    return newAug

def Aug_Zoom(aug, zoomFactor):
    """
    Fonction de zoom augmentée
    """
    Nim = aug.shape[0]
    xsize = aug.shape[1]
    ysize = aug.shape[2]
    
    tic1 = time.time()
    print('Augmentation par zoom ... ')

    newAug = np.zeros((Nim * (zoomFactor + 1), aug.shape[1], aug.shape[2], aug.shape[3]))
    
    cont = 0
    
    for n in range(Nim):
        augIm = np.squeeze(aug[n, :, :])
        newAug[cont, :, :, 0] = augIm
        cont += 1
        
        for z in range(zoomFactor):
            zoom_level = random.uniform(0.8, 1.2)
            new_size = int(xsize * zoom_level)
            
            # Redimensionner l'image
            resized = cv2.resize(augIm, (new_size, new_size))
            
            # Recadrer ou padding pour retrouver la taille originale
            if zoom_level > 1:
                # Zoom avant - recadrer
                start_x = (new_size - xsize) // 2
                start_y = (new_size - ysize) // 2
                zoomed = resized[start_y:start_y+ysize, start_x:start_x+xsize]
            else:
                # Zoom arrière - padding
                zoomed = np.zeros((ysize, xsize))
                start_x = (xsize - new_size) // 2
                start_y = (ysize - new_size) // 2
                zoomed[start_y:start_y+new_size, start_x:start_x+new_size] = resized
            
            newAug[cont, :, :, 0] = zoomed
            cont += 1
                
    newAug = (newAug - newAug.min()) / (newAug.max() - newAug.min() + 1e-8)
    
    toc1 = time.time()
    print('Zoom augmentation terminée en ', (toc1 - tic1), ' !')
    
    return newAug

# Configuration principale
path = '/Users/aissamhamida/Desktop/Soutenance_PFE_2025/'
file = '333.npy'
imagesfile = 'Happy_normalized'
sinogramfile = 'Sinogram_normalized'

# Paramètres d'augmentation de données
Rotations = 0
Zoom = 0
Translation = 2

# Chemin final pour sauvegarder le nouvel ensemble
pathfinal = '/Users/aissamhamida/Desktop/Soutenance_PFE_2025/'

# IMPORTER VOS IMAGES
Images = np.load(path + file)

# Si c'est une image 2D (H, W)
if Images.ndim == 2:
    Images = np.expand_dims(Images, axis=(0, -1))  # -> (1, H, W, 1)

# Si c'est un volume 3D (N, H, W)
elif Images.ndim == 3:
    Images = np.expand_dims(Images, axis=-1)  # -> (N, H, W, 1)

print("Forme originale:", Images.shape)
print(f"Plage d'intensité originale: {Images.min():.2f} à {Images.max():.2f}")

# NORMALISATION CRITIQUE DES IMAGES MÉDICALES
Images = normalize_medical_image(Images)
print(f"Après normalisation médicale: {Images.min():.6f} à {Images.max():.6f}")

# Redimensionner pour correspondre à l'entrée du modèle (128x128)
if Images.shape[1] != 128 or Images.shape[2] != 128:
    print(f"Redimensionnement des images de {Images.shape[1:3]} à (128, 128)")
    resized_images = np.zeros((Images.shape[0], 128, 128, 1))
    for i in range(Images.shape[0]):
        resized_images[i, :, :, 0] = resize(Images[i, :, :, 0], (128, 128), 
                                           preserve_range=True, anti_aliasing=True)
    Images = resized_images

print("Forme après redimensionnement:", Images.shape)
print('Images importées et redimensionnées')

nIm = Images.shape[0]
ysize = Images.shape[1]   # nombre de lignes
xsize = Images.shape[2]   # nombre de colonnes

# AUGMENTATION DES DONNÉES (commenté pour l'instant)
# Images = Aug_Zoom(Images, Zoom)
# Images = Aug_Rotate(Images, Rotations)
# Images = Aug_translate(Images, Translation)

# GÉNÉRER LES SINOGRAMMES
print('Génération des sinogrammes...')

# Créer les géométries et le projecteur avec des paramètres FIXES
vol_geom = astra.create_vol_geom(ysize, xsize)

# CORRECTION CRITIQUE: Utiliser 180 angles au lieu de max(xsize, ysize)
num_angles = 180
angles = np.linspace(0, np.pi, num_angles, endpoint=False)

proj_geom = astra.create_proj_geom('parallel', 1.0, xsize, angles)
projector_id = astra.create_projector('linear', proj_geom, vol_geom)

# Créer le sinogramme
Sinograms = np.zeros((nIm, num_angles, xsize, 1))

for n in range(nIm):
    im = np.squeeze(Images[n, :, :, 0])   # extraire image 2D
    sinogram_id, sinogram = astra.create_sino(im, projector_id)
    Sinograms[n, :, :, 0] = sinogram
    
    # Nettoyer la mémoire ASTRA
    astra.data2d.delete(sinogram_id)

# Nettoyer le projecteur
astra.projector.delete(projector_id)

# Normaliser les sinogrammes
Sinograms = (Sinograms - Sinograms.min()) / (Sinograms.max() - Sinograms.min() + 1e-8)

print('Sinogrammes générés correctement!')
print(f'Forme des sinogrammes: {Sinograms.shape}')

# Vérifier la compatibilité avec le modèle
verify_model_compatibility(Sinograms.shape)

# SAUVEGARDER VOS DONNÉES
np.save(pathfinal + imagesfile, Images) 
np.save(pathfinal + sinogramfile, Sinograms)

# Vérifier la distribution des données
check_data_distribution(Images, "Images")
check_data_distribution(Sinograms, "Sinogrammes")

print("\nProcessus terminé avec succès!")
print("Utilisez les fichiers *_normalized.npy avec votre modèle DeepPET")