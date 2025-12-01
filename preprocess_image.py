"""
Image Preprocessing Module for Check Amount Recognition
Handles real-world image issues: rotation, noise, lighting, etc.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class ImagePreprocessor:
    """Preprocesses check images for digit recognition"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
    
    def load_image(self, image_path):
        """Load image from file path"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        return self.original_image
    
    def load_from_array(self, image_array):
        """Load image from numpy array"""
        self.original_image = image_array
        return self.original_image
    
    def convert_to_grayscale(self, image):
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def correct_rotation(self, image, angle=None):
        """
        Correct image rotation using deskewing
        If angle is None, automatically detect rotation
        """
        gray = self.convert_to_grayscale(image)
        
        if angle is None:
            # Automatic rotation detection using Hough Transform
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle_deg = np.degrees(theta) - 90
                    if -45 < angle_deg < 45:
                        angles.append(angle_deg)
                
                if angles:
                    angle = np.median(angles)
                else:
                    angle = 0
            else:
                angle = 0
        
        # Rotate image
        if abs(angle) > 0.5:  # Only rotate if angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def remove_noise(self, image, method='gaussian'):
        """
        Remove noise from image
        Methods: 'gaussian', 'median', 'bilateral'
        """
        gray = self.convert_to_grayscale(image)
        
        if method == 'gaussian':
            # Gaussian blur - good for general noise
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        elif method == 'median':
            # Median blur - good for salt-and-pepper noise
            denoised = cv2.medianBlur(gray, 5)
        elif method == 'bilateral':
            # Bilateral filter - preserves edges while removing noise
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        else:
            denoised = gray
        
        return denoised
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        gray = self.convert_to_grayscale(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def binarize(self, image, method='adaptive'):
        """
        Convert image to binary (black and white)
        Methods: 'adaptive', 'otsu', 'simple'
        """
        gray = self.convert_to_grayscale(image)
        
        if method == 'adaptive':
            # Adaptive threshold - works well with varying lighting
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )
        elif method == 'otsu':
            # Otsu's method - automatically determines threshold
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        elif method == 'simple':
            # Simple threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            binary = gray
        
        return binary
    
    def morphological_operations(self, image, operation='close'):
        """
        Apply morphological operations to clean up binary image
        Operations: 'close', 'open', 'dilate', 'erode'
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        if operation == 'close':
            # Closing - removes small holes
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            # Opening - removes small objects
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'dilate':
            # Dilation - expands white regions
            result = cv2.dilate(image, kernel, iterations=1)
        elif operation == 'erode':
            # Erosion - shrinks white regions
            result = cv2.erode(image, kernel, iterations=1)
        else:
            result = image
        
        return result
    
    def preprocess_pipeline(self, image_input, show_steps=False):
        """
        Complete preprocessing pipeline
        
        Args:
            image_input: File path or numpy array
            show_steps: If True, display intermediate steps
        
        Returns:
            Preprocessed binary image ready for digit segmentation
        """
        # Load image
        if isinstance(image_input, str):
            image = self.load_image(image_input)
        else:
            image = self.load_from_array(image_input)
        
        steps = {'Original': image.copy()}
        
        # Step 1: Correct rotation
        rotated = self.correct_rotation(image)
        steps['Rotated'] = rotated.copy()
        
        # Step 2: Convert to grayscale
        gray = self.convert_to_grayscale(rotated)
        steps['Grayscale'] = gray.copy()
        
        # Step 3: Enhance contrast
        enhanced = self.enhance_contrast(gray)
        steps['Enhanced Contrast'] = enhanced.copy()
        
        # Step 4: Remove noise
        denoised = self.remove_noise(enhanced, method='bilateral')
        steps['Denoised'] = denoised.copy()
        
        # Step 5: Binarize
        binary = self.binarize(denoised, method='adaptive')
        steps['Binary'] = binary.copy()
        
        # Step 6: Morphological operations
        cleaned = self.morphological_operations(binary, operation='close')
        steps['Cleaned'] = cleaned.copy()
        
        self.processed_image = cleaned
        
        # Display steps if requested
        if show_steps:
            self.display_steps(steps)
        
        return cleaned
    
    def display_steps(self, steps):
        """Display preprocessing steps"""
        n_steps = len(steps)
        fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(15, 6))
        axes = axes.ravel()
        
        for idx, (title, image) in enumerate(steps.items()):
            if len(image.shape) == 3:
                # Color image
                axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                # Grayscale image
                axes[idx].imshow(image, cmap='gray')
            axes[idx].set_title(title)
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(steps), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def crop_roi(self, image, x, y, w, h):
        """Crop region of interest from image"""
        return image[y:y+h, x:x+w]
    
    def resize_to_mnist(self, image):
        """Resize image to MNIST format (28x28)"""
        # Ensure image is grayscale
        gray = self.convert_to_grayscale(image)
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize to 0-1 range
        normalized = resized.astype('float32') / 255.0
        
        return normalized


def preprocess_check_image(image_path, show_steps=False):
    """
    Convenience function to preprocess a check image
    
    Args:
        image_path: Path to check image
        show_steps: Whether to display preprocessing steps
    
    Returns:
        Preprocessed binary image
    """
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess_pipeline(image_path, show_steps=show_steps)
    return processed


# Example usage
if __name__ == "__main__":
    print("Image Preprocessor Module")
    print("=" * 60)
    print("This module provides functions to preprocess check images:")
    print("- Rotation correction")
    print("- Noise removal")
    print("- Contrast enhancement")
    print("- Binarization")
    print("- Morphological operations")
    print("\nUsage example:")
    print("  from preprocess_image import preprocess_check_image")
    print("  processed = preprocess_check_image('check.jpg', show_steps=True)")
