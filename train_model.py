"""
MNIST Digit Classification with TensorFlow
Train a neural network to recognize handwritten digits
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_and_preprocess_data():
    """Load and preprocess the MNIST dataset"""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to 0-1 range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for the neural network (flatten 28x28 to 784)
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Input shape: {x_train.shape[1]} features")
    
    return (x_train, y_train), (x_test, y_test)

def build_model():
    """Build a neural network model"""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """Train the model"""
    print("\nStarting training...")
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return history

def save_model(model):
    """Save the trained model"""
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'mnist_model.keras')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join('models', 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {plot_path}")
    plt.show()

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("MNIST Digit Classification - Training Pipeline")
    print("=" * 60)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    # Save model
    save_model(model)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
