# 🤖 AI Check Recognition System

An intelligent check processing system that combines computer vision and deep learning to automatically recognize and validate check amounts. Built with TensorFlow, OpenCV, and Flask.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **🎯 MNIST Digit Recognition**: 97-98% accuracy on digit classification
- **🖼️ Image Preprocessing**: Automatic rotation correction, noise removal, contrast enhancement
- **✂️ Digit Segmentation**: Intelligent contour detection and digit extraction
- **✅ Amount Validation**: Business logic validation with confidence thresholds
- **🌐 Web Interface**: Beautiful, responsive Flask web application
- **💱 Multi-Currency Support**: USD, INR, EUR, GBP and more
- **🔍 Interactive Tools**: Manual cropping tool for precise amount extraction

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Karthik24041995/ML-Check-recognition-Tensorflow.git
cd ML-Check-recognition-Tensorflow
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the MNIST model** (if not already trained)
```bash
python train_model.py
```

4. **Run the web application**
```bash
python app.py
```

5. **Open your browser**
```
http://localhost:5000
```

## 📁 Project Structure

```
ai-check-recognition/
│
├── 📂 data/                          # Dataset directory
├── 📂 models/                        # Trained models and plots
│   ├── mnist_model.keras            # Trained MNIST model (97-98% accuracy)
│   ├── training_history.png         # Training metrics visualization
│   └── predictions_sample.png       # Sample predictions
│
├── 📂 templates/                     # Flask HTML templates
│   └── index.html                   # Main web interface
│
├── 📂 static/                        # Static assets
│   ├── 📂 css/
│   │   └── style.css               # Modern gradient styling
│   └── 📂 js/
│       └── app.js                  # Frontend JavaScript
│
├── 📂 uploads/                       # User uploaded images
│
├── 🔧 Core Modules
│   ├── train_model.py               # MNIST model training
│   ├── predict.py                   # Prediction and evaluation
│   ├── preprocess_image.py          # Image preprocessing pipeline
│   ├── digit_segmentation.py        # Digit detection and extraction
│   ├── amount_validator.py          # Business logic validation
│   ├── crop_amount.py               # Interactive cropping tool
│   └── app.py                       # Flask web application
│
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # This file
├── 📄 README_CHECK_RECOGNITION.md   # Detailed documentation
└── 📄 .gitignore                     # Git ignore rules
```

## 🎯 Usage Examples

### 1. Train Custom Model

```bash
python train_model.py
```

### 2. Test Model Predictions

```bash
python predict.py
```

### 3. Crop Amount Region (for better accuracy)

```bash
python crop_amount.py check_image.jpg
```

### 4. Use as Python Module

```python
from preprocess_image import ImagePreprocessor
from digit_segmentation import DigitSegmenter
from amount_validator import AmountValidator
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/mnist_model.keras')

# Process check
preprocessor = ImagePreprocessor()
binary = preprocessor.preprocess_pipeline('check.jpg')

segmenter = DigitSegmenter()
digits = segmenter.segment_digits(binary)
prepared = segmenter.prepare_for_model(digits)

# Predict and validate
validator = AmountValidator(currency='INR')
result = validator.validate_complete(predictions, confidences)
print(f"Amount: {result['amount_formatted']}")
```

## 🔧 Configuration

### Validation Settings

```python
validator = AmountValidator(
    min_amount=0.01,
    max_amount=100000.00,
    min_confidence=0.7,
    currency='INR'  # USD, EUR, GBP
)
```

## 🧪 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 97-98% |
| Training Time | 2-5 minutes (CPU) |
| Model Size | ~400 KB |
| Inference Time | <100ms per digit |

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow 2.15+, Keras
- **Computer Vision**: OpenCV, PIL
- **Web Framework**: Flask 3.0+
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: NumPy, scikit-learn

## 🌍 Real-World Applications

- 🏦 Banking: Automated check processing
- 📊 Accounting: Digital expense management
- 🏪 Retail: Payment processing
- 🏥 Healthcare: Insurance claims
- 🏛️ Government: Tax document processing

## 🤝 Contributing

Contributions welcome! Please submit a Pull Request.

## 📝 License

MIT License - see LICENSE file for details.

## 📞 Support

- Open an issue on [GitHub](https://github.com/Karthik24041995/ML-Check-recognition-Tensorflow/issues)
- Star the repository if you find it helpful!

---

⭐ **Star this repository if you found it helpful!**
