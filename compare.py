"""
Program to compare the similarity of a 300x300 input and mosaic image using three metrics:
MSE: Pixel-level accuracy.
SSIM: Perceptual similarity.
Histogram Similarity: Color distribution match.

Author: Peter Chibuikem Idoko
"""

import cv2
import os
import numpy as np
import logging
import csv
from skimage.metrics import structural_similarity as ssim

# Configure logging
logging.basicConfig(
    filename="similarity.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants
INPUT_IMAGE = "scraped_photos/image_18.jpg"  # Path to the original scraped image
MOSAIC_IMAGE = "output/reconstructed_image.jpg"  # Path to the final mosaic image
IMAGE_SIZE = 300  # Images are resized to 300x300

def compute_mse(imageA, imageB):
    """
    Computes the Mean Squared Error (MSE) between two images.
    Lower MSE means the images are more similar.

    Args:
        imageA (numpy.ndarray): First image.
        imageB (numpy.ndarray): Second image.

    Returns:
        float: Mean Squared Error value.
    """
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def compute_ssim(imageA, imageB):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    SSIM ranges from -1 to 1, where 1 means identical images.

    Args:
        imageA (numpy.ndarray): First image.
        imageB (numpy.ndarray): Second image.

    Returns:
        float: SSIM score.
    """
    ssim_scores = [
        ssim(imageA[:, :, i], imageB[:, :, i]) for i in range(3)
    ]
    return np.mean(ssim_scores)  # Average SSIM over RGB channels

def compute_histogram_similarity(imageA, imageB):
    """Compares color histograms using Chi-Square (more sensitive)."""
    histA = cv2.calcHist([imageA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([imageB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    histA = cv2.normalize(histA, histA).flatten()
    histB = cv2.normalize(histB, histB).flatten()
    
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CHISQR)

def measure_similarity(image1_path, image2_path):
    """
    Measures the similarity between two images using MSE, SSIM, and histogram comparison.

    Args:
        image1_path (str): Path to the first image (original quantized image).
        image2_path (str): Path to the second image (mosaic image).

    Returns:
        dict: A dictionary containing similarity scores.
    """
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error: Could not load one or both images.")
        return None

    # Resize images to ensure they are the same size
    image1 = cv2.resize(image1, (IMAGE_SIZE, IMAGE_SIZE))
    image2 = cv2.resize(image2, (IMAGE_SIZE, IMAGE_SIZE))

    # Compute similarity metrics
    mse_value = compute_mse(image1, image2)
    ssim_value = compute_ssim(image1, image2)
    hist_sim_value = compute_histogram_similarity(image1, image2)

    # Generate log message
    log_message = (
        f"\nImage Similarity Analysis\n"
        f"Mean Squared Error (MSE): {mse_value:.2f} (Lower is better)\n"
        f"Structural Similarity Index (SSIM): {ssim_value:.3f} (Closer to 1 is better)\n"
        f"Histogram Similarity: {hist_sim_value:.3f} (Closer to 1 is better)\n"
    )

    # Print and log the results
    print(log_message)
    logging.info(log_message)

    # Return results as a dictionary
    return {
        "MSE": mse_value,
        "SSIM": ssim_value,
        "Histogram Similarity": hist_sim_value
    }

def log_results_to_csv(results: dict, output_csv: str = "similarity_results.csv") -> None:
    """
    Logs the computed similarity metrics to a CSV file.

    Args:
        results (dict): A dictionary containing similarity metrics (MSE, SSIM, Histogram Similarity).
        output_csv (str): Path to the CSV file where results should be saved. Default is "similarity_results.csv".

    Returns:
        None

    Edge Cases:
        - If the results dictionary is empty or None, the function does nothing.
        - If the CSV file does not exist, it creates one with headers.
        - If an I/O error occurs, it logs the issue and prevents data loss.
    """
    if not results:
        logging.warning("No results to log. Skipping CSV logging.")
        return

    file_exists = os.path.isfile(output_csv)

    try:
        with open(output_csv, mode="a", newline="") as csv_file:
            fieldnames = ["MSE", "SSIM", "Histogram Similarity"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            # Write header only if the file is newly created
            if not file_exists:
                writer.writeheader()

            writer.writerow(results)
            logging.info(f"Similarity results logged to {output_csv}")

    except IOError as e:
        logging.error(f"Error writing to CSV file {output_csv}: {e}")

if __name__ == "__main__":
    output = measure_similarity(INPUT_IMAGE, MOSAIC_IMAGE)
    log_results_to_csv(output)
