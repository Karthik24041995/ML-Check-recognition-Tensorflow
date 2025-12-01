"""
Flask Web Application for Check Amount Recognition
Upload check images and get automatic amount recognition
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import base64
from io import BytesIO
from PIL import Image

# Import our custom modules
from preprocess_image import ImagePreprocessor
from digit_segmentation import DigitSegmenter
from amount_validator import AmountValidator

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None


def load_model():
    """Load the trained MNIST model"""
    global model
    model_path = os.path.join('models', 'mnist_model.keras')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run train_model.py first."
        )
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    if len(image.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(image)
    else:
        # Color
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"


def process_check_image(image_path):
    """
    Complete pipeline to process check image
    
    Args:
        image_path: Path to uploaded check image
    
    Returns:
        Dictionary with recognition results
    """
    try:
        # Step 1: Preprocess image
        preprocessor = ImagePreprocessor()
        binary_image = preprocessor.preprocess_pipeline(image_path, show_steps=False)
        
        # Step 2: Segment digits
        segmenter = DigitSegmenter()
        digits = segmenter.segment_digits(binary_image, show_visualization=False)
        
        if not digits:
            return {
                'success': False,
                'error': 'No digits detected in image',
                'preprocessed_image': image_to_base64(binary_image)
            }
        
        # Step 3: Prepare digits for model
        prepared_digits = segmenter.prepare_for_model(digits)
        
        # Step 4: Predict each digit
        predictions = []
        confidences = []
        
        for digit_data in prepared_digits:
            pred = model.predict(digit_data['flattened'], verbose=0)
            predicted_digit = np.argmax(pred[0])
            confidence = pred[0][predicted_digit]
            
            predictions.append(int(predicted_digit))
            confidences.append(float(confidence))
        
        # Step 5: Validate amount
        # Get currency from request (default to USD)
        currency = request.form.get('currency', 'USD')
        validator = AmountValidator(min_confidence=0.6, currency=currency)
        validation_result = validator.validate_complete(predictions, confidences)
        
        # Step 6: Create visualization
        # Draw boxes on original image
        original = cv2.imread(image_path)
        for box in segmenter.digit_boxes:
            x, y, w, h = box
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return {
            'success': True,
            'predictions': predictions,
            'confidences': confidences,
            'validation': validation_result,
            'original_image': image_to_base64(original),
            'preprocessed_image': image_to_base64(binary_image),
            'num_digits': len(predictions)
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        })
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        result = process_check_image(filepath)
        
        # Clean up uploaded file
        # os.remove(filepath)  # Commented out for debugging
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


if __name__ == '__main__':
    print("=" * 60)
    print("Check Amount Recognition Web Application")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    try:
        load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease run train_model.py first to train the model.")
        exit(1)
    
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
