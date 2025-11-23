


"""
Aissam HAMIDA – Biomedical Engineering Department, National School of Arts and Crafts, Rabat.
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt

# Set memory growth to avoid GPU memory issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class SimpleLNNLayer(layers.Layer):
    """Simplified LNN layer using GRU for stability"""
    def __init__(self, units, return_sequences=True, **kwargs):
        super(SimpleLNNLayer, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        
    def build(self, input_shape):
        # Use a GRU layer as base for stability
        self.gru = layers.GRU(self.units, 
                             return_sequences=self.return_sequences,
                             recurrent_activation='sigmoid')
        super(SimpleLNNLayer, self).build(input_shape)
    
    def call(self, inputs):
        return self.gru(inputs)
    
    def get_config(self):
        """CRITIQUE: Cette méthode permet la sérialisation correcte du layer"""
        config = super(SimpleLNNLayer, self).get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """CRITIQUE: Cette méthode permet la désérialisation correcte"""
        return cls(**config)

class Pix2PixLNN:
    def __init__(self, sinogram_shape=(180, 128), image_shape=(128, 128)):
        self.sinogram_shape = sinogram_shape  # (num_angles, num_detectors)
        self.image_shape = image_shape
        self.channels = 1  # Grayscale images
        
        # Build models
        self.generator = self.build_lnn_generator()
        self.discriminator = self.build_discriminator()
        
        # Optimizers
        self.generator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        # Loss functions
        self.loss_obj = keras.losses.BinaryCrossentropy(from_logits=True)
        self.l1_lambda = 100  # Weight for L1 loss
        
    def build_lnn_generator(self):
        """LNN-based generator that processes sinograms as sequences"""
        # Input: sinogram treated as sequence (batch_size, sequence_length=num_angles, input_size=num_detectors)
        inputs = keras.Input(shape=(self.sinogram_shape[0], self.sinogram_shape[1]))
        
        # LNN layers using GRU
        x = SimpleLNNLayer(256, return_sequences=True)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = SimpleLNNLayer(128, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Use global average pooling instead of last timestep for better stability
        encoded = layers.GlobalAveragePooling1D()(x)  # Shape: (batch_size, 128)
        
        # Fully connected layers to transform to image space
        x = layers.Dense(256)(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Dense(1024)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Final layer to output flattened image
        image_size = self.image_shape[0] * self.image_shape[1]
        x = layers.Dense(image_size)(x)
        x = layers.Activation('tanh')(x)
        
        # Reshape to 2D image
        output = layers.Reshape((self.image_shape[0], self.image_shape[1], 1))(x)
        
        return keras.Model(inputs=inputs, outputs=output)
    
    def build_discriminator(self):
        """Discriminator remains unchanged (PatchGAN)"""
        # Inputs
        input_sinogram = keras.Input(shape=(*self.sinogram_shape, 1), name='input_sinogram')
        input_image = keras.Input(shape=(*self.image_shape, 1), name='input_image')
        
        # Resize sinogram if necessary
        if self.sinogram_shape != self.image_shape:
            input_sinogram_resized = layers.Resizing(self.image_shape[0], self.image_shape[1])(input_sinogram)
        else:
            input_sinogram_resized = input_sinogram
        
        # Concatenate inputs
        x = layers.Concatenate()([input_sinogram_resized, input_image])
        
        # Convolutional layers
        x = layers.Conv2D(64, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(512, 4, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Output layer
        output = layers.Conv2D(1, 4, strides=1, padding='same')(x)
        
        return keras.Model(inputs=[input_sinogram, input_image], outputs=output)
    
    def generator_loss(self, disc_generated_output, gen_output, target):
        """Generator loss function"""
        gan_loss = self.loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
        
        # L1 loss
        target = tf.cast(target, tf.float32)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        
        total_gen_loss = gan_loss + (self.l1_lambda * l1_loss)
        
        return total_gen_loss, gan_loss, l1_loss
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        """Discriminator loss function"""
        real_loss = self.loss_obj(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
        
        total_disc_loss = real_loss + generated_loss
        
        return total_disc_loss
    
    @tf.function
    def train_step(self, sinogram, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate image - sinogram is already in sequence format
            gen_output = self.generator(sinogram, training=True)
            
            # For discriminator, we need to add channel dimension to sinogram
            sinogram_with_channel = tf.expand_dims(sinogram, -1)
            
            # Discriminator outputs
            disc_real_output = self.discriminator([sinogram_with_channel, target], training=True)
            disc_generated_output = self.discriminator([sinogram_with_channel, gen_output], training=True)
            
            # Calculate losses
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)
        
        # Calculate gradients
        generator_gradients = gen_tape.gradient(gen_total_loss,
                                              self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   self.discriminator.trainable_variables)
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                   self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                       self.discriminator.trainable_variables))
        
        return gen_total_loss, disc_loss, gen_gan_loss, gen_l1_loss

def smart_resize(data, target_shape):
    """Smart resizing that maintains aspect ratio when possible"""
    if data.shape == target_shape:
        return data
    
    # Use OpenCV for better interpolation
    resized = cv2.resize(data, (target_shape[1], target_shape[0]), 
                        interpolation=cv2.INTER_AREA)
    return resized

def prepare_sinogram_sequences(sinograms, sequence_format=True):
    """Prepare sinograms as sequences for LNN processing"""
    if sequence_format:
        # Sinograms are already in sequence format: (batch_size, num_angles, num_detectors)
        return sinograms
    else:
        # Reshape to sequence format if needed
        batch_size = sinograms.shape[0]
        num_angles, num_detectors = sinograms.shape[1], sinograms.shape[2]
        return sinograms.reshape(batch_size, num_angles, num_detectors)

def load_and_preprocess_data(sinogram_files, image_files, sino_target_shape=(180, 128), img_target_shape=(128, 128)):
    """Load and preprocess sinogram and image pairs for LNN"""
    sinograms = []
    images = []
    
    print(f"Loading {len(sinogram_files)} pairs...")
    
    for i, (sinogram_file, image_file) in enumerate(zip(sinogram_files, image_files)):
        try:
            # Load data
            sinogram = np.load(sinogram_file)
            image = np.load(image_file)
            
            # Remove singleton dimensions
            if len(sinogram.shape) > 2:
                sinogram = sinogram.squeeze()
            if len(image.shape) > 2:
                image = image.squeeze()
            
            # Verify data is not empty
            if sinogram.size == 0 or image.size == 0:
                print(f"Warning: Empty data in file {i}, skipping...")
                continue
            
            # Resize to target shapes
            sinogram_resized = smart_resize(sinogram, sino_target_shape)
            image_resized = smart_resize(image, img_target_shape)
            
            # Normalize to [-1, 1] for tanh activation
            sinogram_norm = (sinogram_resized - sinogram_resized.min()) / (sinogram_resized.max() - sinogram_resized.min() + 1e-8)
            sinogram_norm = sinogram_norm * 2 - 1
            sinogram_norm = sinogram_norm.astype(np.float32)
            
            image_norm = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min() + 1e-8)
            image_norm = image_norm * 2 - 1
            image_norm = image_norm.astype(np.float32)
            
            sinograms.append(sinogram_norm)
            images.append(image_norm)
            
        except Exception as e:
            print(f"Error loading file {i}: {e}")
            continue
    
    print(f"Successfully loaded {len(sinograms)} pairs")
    return np.array(sinograms), np.array(images)

def generate_images(model, test_sinogram, test_image, epoch, save_dir="training_progress"):
    """Generate and save sample images during training"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate prediction - test_sinogram is already in sequence format
    prediction = model.generator(test_sinogram[np.newaxis, ...], training=False)[0]
    
    # Denormalize from [-1, 1] to [0, 1]
    display_sinogram = (test_sinogram + 1) / 2.0
    display_image = (test_image + 1) / 2.0
    display_prediction = (prediction + 1) / 2.0
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # For sinogram display - reshape to 2D for visualization
    # Sinogram shape: (num_angles, num_detectors) - already 2D
    if len(display_sinogram.shape) == 1:
        # If somehow 1D, reshape it
        sinogram_display = display_sinogram.reshape(-1, 1)
    else:
        sinogram_display = display_sinogram
    
    axes[0].imshow(sinogram_display, cmap='gray', aspect='auto')
    axes[0].set_title('Input Sinogram')
    axes[0].axis('off')
    
    axes[1].imshow(display_prediction.numpy().squeeze(), cmap='gray')
    axes[1].set_title('Generated Image (LNN)')
    axes[1].axis('off')
    
    axes[2].imshow(display_image.squeeze(), cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:04d}.png'))
    plt.close()

# --- FONCTION POUR SAUVEGARDER CORRECTEMENT ---
def save_model_correctly(model, filepath):
    """Sauvegarde le modèle avec les custom layers correctement enregistrés"""
    # Méthode 1: Sauvegarde standard avec custom_objects
    model.save(filepath, save_format='h5')
    print(f"✅ Modèle sauvegardé: {filepath}")

# --- FONCTION POUR CHARGER CORRECTEMENT ---
def load_model_correctly(filepath):
    """Charge le modèle avec les custom layers correctement enregistrés"""
    custom_objects = {
        'SimpleLNNLayer': SimpleLNNLayer
    }
    return keras.models.load_model(filepath, custom_objects=custom_objects, compile=False)

def main():
    # Configuration
    checkpoint_dir = "pix2pix_lnn_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    sinogram_dir = "/Users/aissamhamida/Desktop/Soutenance_PFE_2025/Second try training the model/content333/sinograms/"
    image_dir = "/Users/aissamhamida/Desktop/Soutenance_PFE_2025/Second try training the model/content333/original_slices/"
    
    # Training parameters
    batch_size = 4
    epochs = 100
    sinogram_shape = (180, 128)  # (num_angles, num_detectors)
    image_shape = (128, 128)
    validation_split = 0.2
    
    # Get list of files
    sinogram_files = sorted(glob.glob(os.path.join(sinogram_dir, "*.npy")))
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
    
    # Verify files match
    if len(sinogram_files) != len(image_files):
        print("Warning: Number of sinogram and image files don't match!")
        min_length = min(len(sinogram_files), len(image_files))
        sinogram_files = sinogram_files[:min_length]
        image_files = image_files[:min_length]
    
    print(f"Found {len(sinogram_files)} sinogram-image pairs")
    
    if len(sinogram_files) == 0:
        print("Error: No training files found!")
        return
    
    # Split data into training and validation
    train_sino, val_sino, train_img, val_img = train_test_split(
        sinogram_files, image_files, test_size=validation_split, random_state=42
    )
    
    print(f"Training samples: {len(train_sino)}, Validation samples: {len(val_sino)}")
    
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_and_preprocess_data(train_sino, train_img, sinogram_shape, image_shape)
    
    # Load validation data
    print("Loading validation data...")
    X_val, y_val = load_and_preprocess_data(val_sino, val_img, sinogram_shape, image_shape)
    
    # Prepare sinograms as sequences for LNN (no channel dimension needed)
    X_train_seq = prepare_sinogram_sequences(X_train)
    X_val_seq = prepare_sinogram_sequences(X_val)
    
    # Add channel dimension only to images (not to sinogram sequences)
    y_train = y_train[..., np.newaxis].astype(np.float32)
    y_val = y_val[..., np.newaxis].astype(np.float32)
    
    print(f"Training data shapes - Sinogram sequences: {X_train_seq.shape}, Images: {y_train.shape}")
    print(f"Validation data shapes - Sinogram sequences: {X_val_seq.shape}, Images: {y_val.shape}")
    print(f"Data types - X_train: {X_train_seq.dtype}, y_train: {y_train.dtype}")
    
    # Verify data is not empty
    if len(X_train_seq) == 0 or len(y_train) == 0:
        print("Error: No training data loaded!")
        return
    
    # Initialize Pix2Pix LNN model
    pix2pix_lnn = Pix2PixLNN(sinogram_shape, image_shape)
    
    # Print model summaries
    print("\nLNN Generator Summary:")
    pix2pix_lnn.generator.summary()
    
    print("\nDiscriminator Summary:")
    pix2pix_lnn.discriminator.summary()
    
    # Test forward pass
    print("Testing forward pass...")
    test_output = pix2pix_lnn.generator(X_train_seq[:1])
    print(f"LNN Generator output shape: {test_output.shape}")
    print(f"LNN Generator output dtype: {test_output.dtype}")
    
    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Training loop
    print("Starting training...")
    
    # Sample for generating progress images
    sample_idx = min(5, len(X_val_seq) - 1)
    sample_sinogram = X_val_seq[sample_idx]
    sample_image = y_val[sample_idx]
    
    # Training history
    history = {
        'gen_total_loss': [],
        'disc_loss': [],
        'gen_gan_loss': [],
        'gen_l1_loss': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        epoch_gen_loss = []
        epoch_disc_loss = []
        epoch_gen_gan_loss = []
        epoch_gen_l1_loss = []
        
        # Training step for each batch
        batch_count = 0
        for batch, (sinogram_batch, image_batch) in enumerate(train_dataset):
            try:
                gen_total_loss, disc_loss, gen_gan_loss, gen_l1_loss = pix2pix_lnn.train_step(sinogram_batch, image_batch)
                
                epoch_gen_loss.append(gen_total_loss.numpy())
                epoch_disc_loss.append(disc_loss.numpy())
                epoch_gen_gan_loss.append(gen_gan_loss.numpy())
                epoch_gen_l1_loss.append(gen_l1_loss.numpy())
                
                if batch % 10 == 0:
                    print(f"Batch {batch}: Gen Loss: {gen_total_loss:.4f}, Disc Loss: {disc_loss:.4f}")
                
                batch_count += 1
            except tf.errors.OutOfRangeError:
                break  # End of dataset
        
        if batch_count == 0:
            print("No batches processed in this epoch!")
            continue
            
        # Calculate epoch averages
        avg_gen_loss = np.mean(epoch_gen_loss)
        avg_disc_loss = np.mean(epoch_disc_loss)
        avg_gen_gan_loss = np.mean(epoch_gen_gan_loss)
        avg_gen_l1_loss = np.mean(epoch_gen_l1_loss)
        
        # Store history
        history['gen_total_loss'].append(avg_gen_loss)
        history['disc_loss'].append(avg_disc_loss)
        history['gen_gan_loss'].append(avg_gen_gan_loss)
        history['gen_l1_loss'].append(avg_gen_l1_loss)
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"Generator Total Loss: {avg_gen_loss:.4f}")
        print(f"Discriminator Loss: {avg_disc_loss:.4f}")
        print(f"Generator GAN Loss: {avg_gen_gan_loss:.4f}")
        print(f"Generator L1 Loss: {avg_gen_l1_loss:.4f}")
        
        # Generate sample images every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            try:
                generate_images(pix2pix_lnn, sample_sinogram, sample_image, epoch + 1)
            except Exception as e:
                print(f"Error generating images: {e}")
            
            # Save checkpoint AVEC LA NOUVELLE MÉTHODE
            generator_path = os.path.join(checkpoint_dir, f'lnn_generator_epoch_{epoch+1}.h5')
            discriminator_path = os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch+1}.h5')
            
            try:
                save_model_correctly(pix2pix_lnn.generator, generator_path)
                save_model_correctly(pix2pix_lnn.discriminator, discriminator_path)
                print(f"Checkpoint saved at epoch {epoch + 1}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
    
    # Save final models AVEC LA NOUVELLE MÉTHODE
    final_generator_path = "final_lnn_generator.h5"
    final_discriminator_path = "final_discriminator.h5"
    
    try:
        save_model_correctly(pix2pix_lnn.generator, final_generator_path)
        save_model_correctly(pix2pix_lnn.discriminator, final_discriminator_path)
        print(f"\n=== LNN Training completed successfully ===")
        print(f"Final LNN generator saved as: {final_generator_path}")
        print(f"Final discriminator saved as: {final_discriminator_path}")
        
        # Test de chargement immédiat pour vérifier
        print("\n🔍 Test de chargement du modèle sauvegardé...")
        test_model = load_model_correctly(final_generator_path)
        print("✅ Modèle chargé avec succès! Il sera utilisable plus tard.")
        
    except Exception as e:
        print(f"Error saving final models: {e}")
    
    # Plot training history
    try:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['gen_total_loss'])
        plt.title('LNN Generator Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 2)
        plt.plot(history['disc_loss'])
        plt.title('Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 3)
        plt.plot(history['gen_gan_loss'])
        plt.title('LNN Generator GAN Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(2, 2, 4)
        plt.plot(history['gen_l1_loss'])
        plt.title('LNN Generator L1 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('lnn_training_history.png')
        plt.close()
        print("Training history plot saved successfully.")
    except Exception as e:
        print(f"Error plotting training history: {e}")

if __name__ == "__main__":
    main()









