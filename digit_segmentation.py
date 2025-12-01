"""
Digit Segmentation Module for Check Amount Recognition
Detects and extracts individual digits from preprocessed check images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocess_image import ImagePreprocessor


class DigitSegmenter:
    """Segments individual digits from check amount images"""
    
    def __init__(self):
        self.image = None
        self.contours = []
        self.digit_boxes = []
        self.digits = []
    
    def find_contours(self, binary_image):
        """Find contours in binary image"""
        # Find contours
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        self.contours = contours
        return contours
    
    def filter_digit_contours(self, contours, min_area=100, max_area=10000,
                             min_aspect=0.2, max_aspect=2.0):
        """
        Filter contours to keep only those likely to be digits
        
        Args:
            contours: List of contours
            min_area: Minimum contour area
            max_area: Maximum contour area
            min_aspect: Minimum aspect ratio (width/height)
            max_aspect: Maximum aspect ratio
        
        Returns:
            Filtered list of bounding boxes (x, y, w, h)
        """
        digit_boxes = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter based on size and aspect ratio
            if (min_area < area < max_area and
                min_aspect < aspect_ratio < max_aspect):
                digit_boxes.append((x, y, w, h))
        
        return digit_boxes
    
    def sort_boxes_left_to_right(self, boxes):
        """Sort bounding boxes from left to right"""
        return sorted(boxes, key=lambda b: b[0])
    
    def merge_overlapping_boxes(self, boxes, overlap_threshold=0.3):
        """Merge boxes that overlap significantly"""
        if not boxes:
            return []
        
        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [boxes[0]]
        
        for current in boxes[1:]:
            previous = merged[-1]
            
            # Calculate overlap
            x1 = max(previous[0], current[0])
            x2 = min(previous[0] + previous[2], current[0] + current[2])
            
            if x2 > x1:  # Boxes overlap
                overlap = x2 - x1
                min_width = min(previous[2], current[2])
                
                if overlap / min_width > overlap_threshold:
                    # Merge boxes
                    x_min = min(previous[0], current[0])
                    y_min = min(previous[1], current[1])
                    x_max = max(previous[0] + previous[2], current[0] + current[2])
                    y_max = max(previous[1] + previous[3], current[1] + current[3])
                    merged[-1] = (x_min, y_min, x_max - x_min, y_max - y_min)
                else:
                    merged.append(current)
            else:
                merged.append(current)
        
        return merged
    
    def extract_digits(self, image, boxes, padding=5):
        """
        Extract individual digit images from bounding boxes
        
        Args:
            image: Original binary image
            boxes: List of bounding boxes
            padding: Padding around each digit
        
        Returns:
            List of extracted digit images
        """
        digits = []
        h, w = image.shape[:2]
        
        for x, y, box_w, box_h in boxes:
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + box_w + padding)
            y2 = min(h, y + box_h + padding)
            
            # Extract digit
            digit = image[y1:y2, x1:x2]
            digits.append(digit)
        
        return digits
    
    def segment_digits(self, binary_image, min_area=100, max_area=10000,
                       show_visualization=False):
        """
        Complete digit segmentation pipeline
        
        Args:
            binary_image: Preprocessed binary image
            min_area: Minimum digit area
            max_area: Maximum digit area
            show_visualization: Whether to display results
        
        Returns:
            List of extracted digit images
        """
        self.image = binary_image
        
        # Find contours
        contours = self.find_contours(binary_image)
        
        # Filter to get likely digit contours
        boxes = self.filter_digit_contours(
            contours,
            min_area=min_area,
            max_area=max_area
        )
        
        # Merge overlapping boxes
        boxes = self.merge_overlapping_boxes(boxes)
        
        # Sort left to right
        boxes = self.sort_boxes_left_to_right(boxes)
        
        self.digit_boxes = boxes
        
        # Extract digit images
        digits = self.extract_digits(binary_image, boxes)
        self.digits = digits
        
        # Visualize if requested
        if show_visualization:
            self.visualize_segmentation(binary_image, boxes, digits)
        
        return digits
    
    def visualize_segmentation(self, image, boxes, digits):
        """Visualize segmentation results"""
        # Create figure
        fig = plt.figure(figsize=(15, 8))
        
        # Show original image with boxes
        ax1 = plt.subplot(2, 1, 1)
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        for idx, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display_image,
                str(idx),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        ax1.imshow(display_image)
        ax1.set_title(f'Detected Digits: {len(boxes)} boxes found')
        ax1.axis('off')
        
        # Show individual digits
        if digits:
            n_digits = len(digits)
            for idx, digit in enumerate(digits):
                ax = plt.subplot(2, n_digits, n_digits + idx + 1)
                ax.imshow(digit, cmap='gray')
                ax.set_title(f'Digit {idx}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_for_model(self, digits, target_size=(28, 28)):
        """
        Prepare extracted digits for MNIST model
        
        Args:
            digits: List of digit images
            target_size: Target size (width, height)
        
        Returns:
            List of normalized digit images ready for model
        """
        prepared_digits = []
        
        for digit in digits:
            # Add padding to make square
            h, w = digit.shape
            max_dim = max(h, w)
            
            # Create square canvas
            square = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            # Center the digit
            y_offset = (max_dim - h) // 2
            x_offset = (max_dim - w) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = digit
            
            # Resize to target size
            resized = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to 0-1 range
            normalized = resized.astype('float32') / 255.0
            
            # Flatten for model (if needed)
            flattened = normalized.reshape(1, -1)
            
            prepared_digits.append({
                'image': normalized,
                'flattened': flattened,
                'original': digit
            })
        
        return prepared_digits
    
    def segment_amount_region(self, image, amount_region=None):
        """
        Extract the amount region from a full check image
        
        Args:
            image: Full check image
            amount_region: Tuple (x, y, w, h) or None for auto-detection
        
        Returns:
            Cropped amount region
        """
        if amount_region is not None:
            x, y, w, h = amount_region
            return image[y:y+h, x:x+w]
        
        # Auto-detection (simple heuristic - typically in lower right)
        h, w = image.shape[:2]
        
        # Assume amount is in bottom-right quarter
        amount_region = image[h//2:, w//2:]
        
        return amount_region


def segment_check_amount(binary_image, show_visualization=False):
    """
    Convenience function to segment digits from check amount
    
    Args:
        binary_image: Preprocessed binary image
        show_visualization: Whether to display results
    
    Returns:
        List of prepared digit images
    """
    segmenter = DigitSegmenter()
    digits = segmenter.segment_digits(
        binary_image,
        show_visualization=show_visualization
    )
    prepared = segmenter.prepare_for_model(digits)
    return prepared


# Example usage
if __name__ == "__main__":
    print("Digit Segmentation Module")
    print("=" * 60)
    print("This module segments individual digits from check images:")
    print("- Contour detection")
    print("- Bounding box filtering")
    print("- Digit extraction")
    print("- MNIST format preparation")
    print("\nUsage example:")
    print("  from digit_segmentation import segment_check_amount")
    print("  from preprocess_image import preprocess_check_image")
    print("  ")
    print("  binary = preprocess_check_image('check.jpg')")
    print("  digits = segment_check_amount(binary, show_visualization=True)")
    print("=" * 60)
    
    # Demo with synthetic data
    print("\nDemo: Creating synthetic digit image...")
    demo_image = np.zeros((100, 300), dtype=np.uint8)
    
    # Draw some digit-like shapes
    cv2.rectangle(demo_image, (10, 20), (40, 80), 255, -1)
    cv2.rectangle(demo_image, (60, 20), (90, 80), 255, -1)
    cv2.rectangle(demo_image, (110, 20), (140, 80), 255, -1)
    cv2.rectangle(demo_image, (160, 20), (190, 80), 255, -1)
    cv2.rectangle(demo_image, (210, 20), (240, 80), 255, -1)
    
    # Segment
    segmenter = DigitSegmenter()
    digits = segmenter.segment_digits(demo_image, min_area=50, show_visualization=True)
    print(f"\nDetected {len(digits)} digits from demo image")
