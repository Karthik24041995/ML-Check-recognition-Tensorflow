# Check Amount Recognition System

An AI-powered system to automatically recognize and validate check amounts using TensorFlow and computer vision.

## 🎯 Features

- **Image Preprocessing**: Automatic rotation correction, noise removal, contrast enhancement
- **Digit Segmentation**: Intelligent detection and extraction of individual digits
- **AI Recognition**: MNIST-trained neural network for digit recognition
- **Amount Validation**: Business logic validation with confidence thresholds
- **Web Interface**: User-friendly Flask web application
- **Real-time Processing**: Upload and process checks instantly

## 📁 Project Structure

```
ml-tensorflow-project/
│
├── data/                          # Data directory
├── models/                        # Trained models
│   ├── mnist_model.keras         # Trained MNIST model
│   ├── training_history.png      # Training plots
│   └── predictions_sample.png    # Sample predictions
│
├── uploads/                       # Uploaded check images
├── templates/                     # HTML templates
│   └── index.html                # Main web interface
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css            # Styles
│   └── js/
│       └── app.js               # Frontend logic
│
├── train_model.py                # Train MNIST model
├── predict.py                    # Make predictions
├── preprocess_image.py           # Image preprocessing module
├── digit_segmentation.py         # Digit detection module
├── amount_validator.py           # Validation module
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
└── README_CHECK_RECOGNITION.md   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)

```bash
python train_model.py
```

Expected output: ~97-98% test accuracy

### 3. Run the Web Application

```bash
python app.py
```

Open your browser to: `http://localhost:5000`

## 💡 How It Works

### Complete Pipeline

```
Check Image → Preprocessing → Segmentation → Recognition → Validation → Results
```

### 1. **Image Preprocessing** (`preprocess_image.py`)

Handles real-world image challenges:
- **Rotation Correction**: Auto-detects and fixes skewed images
- **Noise Removal**: Gaussian, median, and bilateral filtering
- **Contrast Enhancement**: CLAHE for better visibility
- **Binarization**: Adaptive thresholding for digit extraction
- **Morphological Operations**: Cleaning and smoothing

```python
from preprocess_image import preprocess_check_image

# Preprocess a check image
binary_image = preprocess_check_image('check.jpg', show_steps=True)
```

### 2. **Digit Segmentation** (`digit_segmentation.py`)

Extracts individual digits:
- **Contour Detection**: Finds digit regions
- **Filtering**: Removes noise and non-digit objects
- **Sorting**: Orders digits left-to-right
- **MNIST Formatting**: Resizes to 28x28 pixels

```python
from digit_segmentation import segment_check_amount

# Extract digits
digits = segment_check_amount(binary_image, show_visualization=True)
```

### 3. **AI Recognition** (Uses trained MNIST model)

- Neural network predicts each digit (0-9)
- Returns prediction + confidence score
- 97-98% accuracy on clean digits

### 4. **Amount Validation** (`amount_validator.py`)

Business logic validation:
- **Confidence Threshold**: Minimum 70% confidence
- **Amount Range**: $0.01 - $100,000
- **Format Validation**: Proper numeric format
- **Anomaly Detection**: Repeated digits, unusual patterns

```python
from amount_validator import validate_amount

# Validate recognized amount
result = validate_amount(predictions, confidences)
print(result['amount_formatted'])  # $123.45
```

## 🌐 Web Interface

### Features

- **Drag & Drop Upload**: Easy file upload
- **Real-time Processing**: Instant results
- **Visual Feedback**: See detection boxes and preprocessing
- **Validation Status**: Clear pass/fail indicators
- **Detailed Results**: Confidence scores per digit

### Usage

1. Open `http://localhost:5000`
2. Upload or drag a check image
3. Click "Process Check"
4. View results with confidence scores

## 📊 API Endpoints

### `POST /upload`

Upload and process a check image.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

**Response:**
```json
{
  "success": true,
  "predictions": [1, 2, 3, 4, 5],
  "confidences": [0.95, 0.88, 0.92, 0.85, 0.90],
  "validation": {
    "is_valid": true,
    "amount": 123.45,
    "amount_formatted": "$123.45",
    "confidence": {
      "average": 0.90,
      "min": 0.85,
      "max": 0.95
    }
  }
}
```

### `GET /health`

Check application health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🔧 Configuration

### Validation Settings

Edit `amount_validator.py`:

```python
validator = AmountValidator(
    min_amount=0.01,        # Minimum check amount
    max_amount=100000.00,   # Maximum check amount
    min_confidence=0.7      # Minimum confidence threshold
)
```

### Segmentation Parameters

Edit `digit_segmentation.py`:

```python
digits = segmenter.segment_digits(
    binary_image,
    min_area=100,           # Minimum digit area (pixels)
    max_area=10000,         # Maximum digit area (pixels)
    show_visualization=True
)
```

## 📝 Module Usage Examples

### Standalone Image Processing

```python
from preprocess_image import ImagePreprocessor
from digit_segmentation import DigitSegmenter
from amount_validator import AmountValidator
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/mnist_model.keras')

# Preprocess
preprocessor = ImagePreprocessor()
binary = preprocessor.preprocess_pipeline('check.jpg')

# Segment
segmenter = DigitSegmenter()
digits = segmenter.segment_digits(binary)
prepared = segmenter.prepare_for_model(digits)

# Predict
predictions = []
confidences = []
for digit in prepared:
    pred = model.predict(digit['flattened'])
    predictions.append(int(pred[0].argmax()))
    confidences.append(float(pred[0].max()))

# Validate
validator = AmountValidator()
result = validator.validate_complete(predictions, confidences)
print(f"Amount: {result['amount_formatted']}")
print(f"Valid: {result['is_valid']}")
```

## 🎨 Customization

### Adding Custom Validation Rules

```python
class CustomValidator(AmountValidator):
    def validate_business_hours(self):
        """Only process during business hours"""
        from datetime import datetime
        hour = datetime.now().hour
        return 9 <= hour <= 17
```

### Custom Preprocessing

```python
preprocessor = ImagePreprocessor()

# Custom pipeline
gray = preprocessor.convert_to_grayscale(image)
enhanced = preprocessor.enhance_contrast(gray)
denoised = preprocessor.remove_noise(enhanced, method='bilateral')
binary = preprocessor.binarize(denoised, method='otsu')
```

## 🔍 Troubleshooting

### No Digits Detected

- Check image quality and lighting
- Adjust segmentation parameters (min_area, max_area)
- Try different preprocessing methods

### Low Confidence Scores

- Improve image quality
- Retrain model with more data
- Adjust validation threshold

### Web App Won't Start

- Ensure model is trained: `python train_model.py`
- Check port 5000 is available
- Verify all dependencies installed

## 🚀 Production Considerations

### Security
- Add authentication/authorization
- Validate file sizes and types
- Scan uploads for malware
- Use HTTPS in production

### Performance
- Add caching for repeated requests
- Use queue system for batch processing
- Optimize image processing pipeline
- Consider GPU acceleration

### Monitoring
- Log all transactions
- Track accuracy metrics
- Monitor processing times
- Alert on validation failures

## 📈 Future Enhancements

- [ ] Support for multiple check formats
- [ ] OCR for payee name and date
- [ ] Signature verification
- [ ] Database integration
- [ ] Batch processing API
- [ ] Mobile app integration
- [ ] Multi-language support
- [ ] Fraud detection features

## 🤝 Real-World Applications

- **Banking**: Automated check processing
- **Accounting**: Digital expense management
- **Retail**: Payment processing
- **Insurance**: Claims processing
- **Government**: Tax document processing

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Built with TensorFlow, OpenCV, and Flask

---

**Need Help?** Open an issue or check the documentation in each module.
