# MNIST Digit Classification with TensorFlow

A simple machine learning project that trains a neural network to recognize handwritten digits using TensorFlow and the MNIST dataset.

## Project Overview

This project demonstrates a complete ML pipeline:
- Data loading and preprocessing
- Neural network model building
- Model training with early stopping
- Model evaluation and predictions
- Visualization of results

## Dataset

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9):
- 60,000 training images
- 10,000 test images
- Image size: 28x28 pixels

## Model Architecture

- **Input Layer**: 784 features (28x28 flattened)
- **Hidden Layer 1**: 128 neurons, ReLU activation, 20% dropout
- **Hidden Layer 2**: 64 neurons, ReLU activation, 20% dropout
- **Output Layer**: 10 neurons (one per digit), Softmax activation

## Project Structure

```
ml-tensorflow-project/
│
├── data/                      # Data directory (MNIST auto-downloads)
├── models/                    # Saved models and plots
│   ├── mnist_model.keras     # Trained model
│   ├── training_history.png  # Training plots
│   └── predictions_sample.png # Prediction visualizations
│
├── train_model.py            # Training script
├── predict.py                # Prediction script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

Run the training script to train the neural network:

```bash
python train_model.py
```

This will:
- Load and preprocess the MNIST dataset
- Build the neural network
- Train for up to 20 epochs (with early stopping)
- Save the trained model to `models/mnist_model.keras`
- Generate training history plots

Expected accuracy: ~97-98% on test set

### 2. Make Predictions

Run the prediction script to test the trained model:

```bash
python predict.py
```

This will:
- Load the trained model
- Evaluate performance on the test set
- Show predictions on 10 random samples
- Optionally enter interactive mode to test specific images

### Interactive Mode

In interactive mode, you can:
- Enter any test image index (0-9999)
- See the true label, predicted label, and confidence
- View probability distribution for all digits
- Visualize the image

## Results

The model typically achieves:
- **Test Accuracy**: ~97-98%
- **Training Time**: 2-5 minutes (CPU)
- **Model Size**: ~400 KB

## Key Features

- ✅ Simple and easy to understand
- ✅ Well-commented code
- ✅ Reproducible results (random seeds set)
- ✅ Early stopping to prevent overfitting
- ✅ Dropout layers for regularization
- ✅ Comprehensive evaluation metrics
- ✅ Interactive prediction mode
- ✅ Visualization of training and predictions

## Requirements

- Python 3.8+
- TensorFlow 2.15+
- NumPy
- Matplotlib
- scikit-learn

## Next Steps

To extend this project, consider:
- Try different model architectures (CNN, deeper networks)
- Experiment with hyperparameters (learning rate, batch size)
- Add data augmentation
- Implement confusion matrix visualization
- Try other datasets (Fashion-MNIST, CIFAR-10)
- Deploy the model as a web application

## License

This project is for educational purposes.
