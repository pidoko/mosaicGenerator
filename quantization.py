"""
Colour Quantizer by Peter Chibuikem Idoko
2025-02-03
Uses K-Means Clustering to apply colour quantization for improved efficiency.

Enhancements:
- Logging for better debugging.
- Structured error handling.
- Dynamic parameters for better configurability.
- Supports multiple image formats.
"""

import os
import cv2
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    filename="quantizer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Colour Quantizer script started.")

# Constants
DEFAULT_K = 8  # Default number of clusters (colors)
RESIZE_DIM = (300, 300) # Fixed resolution of all images
INPUT_FOLDER = "scraped_photos"
OUTPUT_FOLDER = "quantized_photos"
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to quantize images may return None
def quantize_image(image_path: str, K: int = DEFAULT_K, resize_dim: tuple = RESIZE_DIM) -> Optional[np.ndarray]:
    """
    Applies colour quantization using K-Means clustering.

    Args:
        image_path (str): Path to the image file.
        K (int): Number of clusters (colors) to reduce the image to.
        resize_dim (tuple): Target resolution (width, height) for resizing.

    Returns:
        np.ndarray: The quantized image, or None if an error occurs.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Error loading image: {image_path}")
            return None
        
        # Resize image to fixed resolution (300x300)
        image = cv2.resize(image, resize_dim)

        # Convert image to RGB (OpenCV loads images in BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape image into a 2D array of pixels (Each pixel is a 3D point [R, G, B])
        pixel_values = image.reshape((-1, 3)).astype(np.float32)

        # Define stopping criteria and apply K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers back to uint8 (color values)
        centers = np.uint8(centers)
        
        # Replace each pixel value with its nearest centroid color
        quantized_image = centers[labels.flatten()]
        
        # Reshape back to the original image shape
        quantized_image = quantized_image.reshape(image.shape)

        return quantized_image
    
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

def process_images(input_folder: str = INPUT_FOLDER, output_folder: str = OUTPUT_FOLDER, K: int = DEFAULT_K):
    """
    Processes all images in the input folder and applies color quantization.

    Args:
        input_folder (str): Directory containing images to process.
        output_folder (str): Directory to save quantized images.
        K (int): Number of colors (clusters) for quantization.
    """
    logging.info(f"Processing images from {input_folder}, saving to {output_folder}")

    total_images = len([f for f in os.listdir(input_folder) if f.lower().endswith(SUPPORTED_FORMATS)])
    processed_count = 0  # Track the number of images processed

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(SUPPORTED_FORMATS):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            quantized_img = quantize_image(input_path, K)
            
            if quantized_img is not None:
                cv2.imwrite(output_path, cv2.cvtColor(quantized_img, cv2.COLOR_RGB2BGR))
                processed_count += 1

                # Log progress to both console and log file
                log_message = f"[{processed_count}/{total_images}] Quantized & saved: {output_path}"
                print(log_message)  # Console output
                logging.info(log_message)  # Log file entry

    logging.info("Color quantization complete.")
    print("Color quantization complete!")  # Console notification

if __name__ == "__main__":
    process_images()
