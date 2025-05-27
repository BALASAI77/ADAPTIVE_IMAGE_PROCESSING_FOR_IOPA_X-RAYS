import logging
import numpy as np
import cv2
from pathlib import Path

class Config:
    """Configuration settings."""
    DATA_DIR = Path("C:/Users/BALA SAI/OneDrive/Desktop/cavity_processing/data")
    OUTPUT_DIR = Path("C:/Users/BALA SAI/OneDrive/Desktop/cavity_processing/output")
    SUPPORTED_FORMATS = ['.dcm', '.rvg', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    THRESHOLDS = {
        'brightness': {'low': 40, 'high': 200},  # Tightened for better classification
        'contrast': {'low': 25, 'high': 85},     # Narrowed to reduce misclassification
        'sharpness': {'low': 60, 'high': 500},   # Adjusted for accuracy
        'noise': {'low': 5, 'high': 20}          # Refined noise range
    }
    VISUALIZATION = {
        'show_plots': False,
        'dpi': 300
    }
    LOG_FILE = "preprocessing.log"

def setup_logging(log_file):
    """Configure logging."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Logging initialized")

def normalize_to_8bit(image):
    """Normalize image to 8-bit range."""
    if image.dtype != np.uint8:
        image_min, image_max = np.min(image), np.max(image)
        if image_max > image_min:
            image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
    return image

def calculate_gradient_map(image):
    """Calculate gradient magnitude map."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobel_x**2 + sobel_y**2)