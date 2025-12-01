"""
Interactive tool to crop the amount region from a check image
"""

import cv2
import sys

# Global variables for cropping
drawing = False
start_point = None
end_point = None
image = None
clone = None


def crop_amount_region(image_path, output_path='amount_cropped.jpg'):
    """
    Interactive tool to select and crop amount region
    
    Usage:
        python crop_amount.py check_image.jpg
    
    Instructions:
        1. Click and drag to select the amount region
        2. Press 'r' to reset selection
        3. Press 'c' to crop and save
        4. Press 'q' to quit
    """
    global image, clone
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    clone = image.copy()
    
    # Create window
    cv2.namedWindow('Crop Amount Region')
    cv2.setMouseCallback('Crop Amount Region', mouse_callback)
    
    print("\n" + "="*60)
    print("Amount Region Cropping Tool")
    print("="*60)
    print("\nInstructions:")
    print("  1. Click and drag to select the amount region")
    print("  2. Press 'r' to reset selection")
    print("  3. Press 'c' to crop and save")
    print("  4. Press 'q' to quit")
    print("="*60)
    
    while True:
        cv2.imshow('Crop Amount Region', image)
        key = cv2.waitKey(1) & 0xFF
        
        # Reset
        if key == ord('r'):
            image = clone.copy()
            print("Selection reset")
        
        # Crop
        elif key == ord('c'):
            if start_point and end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                
                # Ensure correct order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Crop
                cropped = clone[y1:y2, x1:x2]
                
                # Save
                cv2.imwrite(output_path, cropped)
                print(f"\n✓ Amount region saved to: {output_path}")
                print(f"  Dimensions: {x2-x1} x {y2-y1} pixels")
                print(f"\nNow upload '{output_path}' to the web interface!")
                
                # Show cropped image
                cv2.imshow('Cropped Amount', cropped)
                cv2.waitKey(2000)
                break
            else:
                print("Please select a region first")
        
        # Quit
        elif key == ord('q'):
            print("Exiting without saving")
            break
    
    cv2.destroyAllWindows()


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing rectangle"""
    global start_point, end_point, drawing, image, clone
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            image = clone.copy()
            cv2.rectangle(image, start_point, (x, y), (0, 255, 0), 2)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        image = clone.copy()
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python crop_amount.py <check_image_path>")
        print("\nExample:")
        print('  python crop_amount.py "Screenshot 2025-12-01 090257.png"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    crop_amount_region(image_path)
