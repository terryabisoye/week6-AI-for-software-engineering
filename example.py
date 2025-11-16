EDGE AI PROTOTYPE

Train a lightweight image classification model (e.g., recognizing recyclable items).
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

class RecyclableItemsClassifier:
    def __init__(self, num_classes, img_size=IMG_SIZE):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create a lightweight model using MobileNetV2"""
        # Load pre-trained MobileNetV2 without top layers
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model initially
        base_model.trainable = False
        
        # Create the model
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        
        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomContrast(0.2)(x)
        
        # Preprocessing for MobileNetV2
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def prepare_dataset(self, image_dir):
        """Prepare dataset from directory structure"""
        data_dir = Path(image_dir)
        
        # Create dataset from directory
        train_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE
        )
        
        val_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE
        )
        
        # Get class names
        self.class_names = train_ds.class_names
        print(f"Found classes: {self.class_names}")
        
        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def train(self, train_ds, val_ds, epochs=EPOCHS):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Train the model
        print("Training model...")
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr]
        )
        
        return self.history
    
    def fine_tune(self, train_ds, val_ds, epochs=10):
        """Fine-tune the model by unfreezing some layers"""
        base_model = self.model.layers[4]  # Get the MobileNetV2 base
        base_model.trainable = True
        
        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE/10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Fine-tuning model...")
        history_fine = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        
        return history_fine
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train Loss')
        ax2.plot(self.history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def save_model(self, filepath='recyclable_classifier.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_image(self, image_path):
        """Predict class for a single image"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Load and preprocess image
        img = keras.utils.load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return self.class_names[predicted_class], confidence


# Example usage
if __name__ == "__main__":
    """
    Directory structure should be:
    data/
        ├── trash_bags/
        │   ├── image1.jpg
        │   ├── image2.jpg
        ├── recyclable_plastic/
        │   ├── image1.jpg
        ├── recyclable_paper/
        │   ├── image1.jpg
    """
    
    # Initialize classifier
    # Number of classes depends on your dataset structure
    classifier = RecyclableItemsClassifier(num_classes=3)
    
    # Create model
    model = classifier.create_model()
    model.summary()
    
    # Prepare dataset (replace with your data directory)
    # train_ds, val_ds = classifier.prepare_dataset('data/')
    
    # Train the model
    # classifier.train(train_ds, val_ds, epochs=20)
    
    # Optional: Fine-tune
    # classifier.fine_tune(train_ds, val_ds, epochs=10)
    
    # Plot training history
    # classifier.plot_training_history()
    
    # Save model
    # classifier.save_model('recyclable_classifier.h5')
    
    # Predict on new image
    # predicted_class, confidence = classifier.predict_image('test_image.jpg')
    # print(f"Predicted: {predicted_class} (Confidence: {confidence:.2%})")
