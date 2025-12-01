"""
Make predictions using the trained MNIST model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

def load_model():
    """Load the trained model"""
    model_path = os.path.join('models', 'mnist_model.keras')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run train_model.py first."
        )
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return model

def load_test_data():
    """Load test data"""
    print("Loading test data...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize
    x_test = x_test.astype('float32') / 255.0
    
    return x_test, y_test

def predict_single_image(model, image, true_label=None):
    """Make prediction on a single image"""
    # Reshape for model input
    image_flat = image.reshape(1, 28 * 28)
    
    # Make prediction
    predictions = model.predict(image_flat, verbose=0)
    predicted_label = np.argmax(predictions[0])
    confidence = predictions[0][predicted_label]
    
    return predicted_label, confidence, predictions[0]

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """Visualize predictions on random test samples"""
    # Select random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        image = x_test[idx]
        true_label = y_test[idx]
        
        # Make prediction
        predicted_label, confidence, _ = predict_single_image(model, image, true_label)
        
        # Plot
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(
            f'True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.2%}',
            color=color,
            fontsize=10
        )
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join('models', 'predictions_sample.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPredictions visualization saved to: {plot_path}")
    plt.show()

def evaluate_model(model, x_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model on entire test set...")
    
    # Reshape test data
    x_test_flat = x_test.reshape(-1, 28 * 28)
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
    
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions on all test data
    predictions = model.predict(x_test_flat, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate per-class accuracy
    print("\nPer-digit accuracy:")
    for digit in range(10):
        mask = y_test == digit
        accuracy = np.mean(predicted_labels[mask] == y_test[mask])
        count = np.sum(mask)
        print(f"  Digit {digit}: {accuracy:.4f} ({accuracy*100:.2f}%) - {count} samples")

def predict_custom_input():
    """Interactive prediction mode"""
    model = load_model()
    x_test, y_test = load_test_data()
    
    print("\n" + "=" * 60)
    print("Interactive Prediction Mode")
    print("=" * 60)
    
    while True:
        try:
            idx = input("\nEnter test image index (0-9999) or 'q' to quit: ")
            
            if idx.lower() == 'q':
                break
            
            idx = int(idx)
            
            if idx < 0 or idx >= len(x_test):
                print(f"Invalid index. Please enter a number between 0 and {len(x_test)-1}")
                continue
            
            image = x_test[idx]
            true_label = y_test[idx]
            
            # Make prediction
            predicted_label, confidence, probabilities = predict_single_image(
                model, image, true_label
            )
            
            # Display results
            print(f"\nTrue label: {true_label}")
            print(f"Predicted label: {predicted_label}")
            print(f"Confidence: {confidence:.2%}")
            print(f"\nAll probabilities:")
            for i, prob in enumerate(probabilities):
                bar = '█' * int(prob * 50)
                print(f"  {i}: {prob:.4f} {bar}")
            
            # Show image
            plt.figure(figsize=(4, 4))
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            color = 'green' if predicted_label == true_label else 'red'
            plt.title(
                f'True: {true_label} | Predicted: {predicted_label}\nConfidence: {confidence:.2%}',
                color=color,
                fontsize=12
            )
            plt.show()
            
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def main():
    """Main prediction pipeline"""
    print("=" * 60)
    print("MNIST Digit Classification - Prediction")
    print("=" * 60)
    
    # Load model
    model = load_model()
    
    # Load test data
    x_test, y_test = load_test_data()
    
    # Evaluate model
    evaluate_model(model, x_test, y_test)
    
    # Visualize sample predictions
    visualize_predictions(model, x_test, y_test, num_samples=10)
    
    # Interactive mode
    print("\n" + "=" * 60)
    response = input("Enter interactive prediction mode? (y/n): ")
    if response.lower() == 'y':
        predict_custom_input()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
