import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from model import create_emotion_model
from prepare_dataset import prepare_data
import gc

def train_emotion_model(data_dir='data', batch_size=32, epochs=20):
    """Train the emotion detection model with a custom training loop"""
    
    if not os.path.exists('models'):
        os.makedirs('models')

    print("Loading and preparing dataset...")
    X_train, X_test, y_train, y_test = prepare_data(data_dir)
    
    print("\nData shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nNormalizing data...")
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    print("\nCreating and compiling model...")
    model = create_emotion_model()
    
    print("\nStarting training...")
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        print("\nTraining completed successfully!")

        print("\nSaving model...")
        model.save('models/emotion_model_final.keras')
        print("Model saved successfully!")

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("\nTraining history plot saved as 'training_history.png'")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        raise
    finally:
        gc.collect()

if __name__ == "__main__":
    train_emotion_model()
