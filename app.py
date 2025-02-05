"""
Grid Divider - Photomosaic Generator
Author: Peter Chibuikem Idoko
Date: 2025-02-03

This script creates a photomosaic by segmenting an input image into grid tiles
and replacing each tile with the closest-matching image from a folder.
"""

import os
import cv2
import numpy as np
import logging
import gradio as gr
from scipy.spatial import distance
from scipy.spatial import KDTree
from joblib import Parallel, delayed

# Constants
DATABASE_FOLDER = "quantized_photos"  # Folder containing preprocessed images
OUTPUT_IMAGE = "output/reconstructed_image.jpg"  # Output file format

# Define Mosaic Properties
IMAGE_SIZE = 300  # Input image size (ensures consistency)
TILE_SIZE = 2  # Each tile size in pixels (smaller = more detail)
GRID_SIZE = IMAGE_SIZE // TILE_SIZE  # Auto-calculate number of tiles per row/column

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_average_color(image: np.ndarray) -> tuple:
    """
    Computes the average RGB color of an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        tuple: (R, G, B) integer values representing the average color.

    Edge Cases:
        - If the image is empty or None, it raises a ValueError.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image provided for color averaging.")

    avg_color = cv2.mean(image)[:3]  # Get mean RGB values
    return tuple(map(int, avg_color))  # Convert to integer RGB


def load_database_images_kdtree(database_folder: str) -> KDTree:
    image_data = []
    image_paths = []

    for filename in os.listdir(database_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(database_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                continue  # Skip unreadable images
            
            avg_color = get_average_color(image)
            image_data.append(avg_color)
            image_paths.append(img_path)

    if not image_data:
        return None, []

    kd_tree = KDTree(image_data)
    return kd_tree, image_paths, image_data  # Return KD-Tree and paths
'''
def load_database_images(database_folder: str) -> list:
    """
    Loads all images from the database and calculates their average color.

    Args:
        database_folder (str): Path to the folder containing database images.

    Returns:
        list: A list of tuples (image_path, avg_color).

    Edge Cases:
        - Skips unreadable or missing images.
        - Returns an empty list if no valid images are found.
    """
    image_data = []
    
    for filename in os.listdir(database_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(database_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                logging.warning(f"Skipping unreadable image: {img_path}")
                continue
            
            avg_color = get_average_color(image)
            image_data.append((img_path, avg_color))

    return image_data'''

def find_closest_image(query_color, kd_tree, image_paths, image_data):
    """Finds the image with the closest average color."""
    _, index = kd_tree.query(query_color)  # Find nearest color in O(log N)
    return image_paths[index], image_data[index]

'''
def find_closest_image(avg_color: tuple, image_database: list) -> str:
    """
    Finds the closest matching image in the database based on Euclidean color distance.

    Args:
        avg_color (tuple): The average (R, G, B) color of the tile.
        image_database (list): A list of (image_path, avg_color) tuples.

    Returns:
        str: Path to the closest matching image.

    Edge Cases:
        - If the database is empty, returns None.
    """
    if not image_database:
        return None

    closest_match = min(image_database, key=lambda img: distance.euclidean(avg_color, img[1]))
    return closest_match[0]  # Return the path
'''


def reconstruct_image(input_img, tile_size, image_size, database_folder):
    """
    Gradio-compatible function to process an image and generate a mosaic.

    Args:
        input_img (np.ndarray): The input image uploaded by the user.
        tile_size (int): Size of each tile in pixels.
        image_size (int): Size to resize the input image.
        database_folder (str): Path to the database of tiles.

    Returns:
        np.ndarray: The reconstructed photomosaic image.
    """
    GRID_SIZE = image_size // tile_size  # Auto-calculate grid size

    # Convert Gradio input to OpenCV format
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # Resize input image
    input_img = cv2.resize(input_img, (image_size, image_size))

    # Load database images with KD-Tree
    kd_tree, image_paths, image_data = load_database_images_kdtree(database_folder)
    if kd_tree is None:
        logging.error("No images found in the database for replacement.")
        return None

    # Create a blank canvas for the reconstructed image
    reconstructed_image = np.zeros_like(input_img)

    def process_tile(row, col):
        """Processes a single tile using KD-Tree to find the best match."""
        x_start, y_start = col * tile_size, row * tile_size
        x_end, y_end = x_start + tile_size, y_start + tile_size

        cell = input_img[y_start:y_end, x_start:x_end]
        avg_color = get_average_color(cell)

        # Find the closest matching image using KD-Tree
        closest_image_path, _ = find_closest_image(avg_color, kd_tree, image_paths, image_data)

        if closest_image_path:
            closest_image = cv2.imread(closest_image_path)
            closest_image = cv2.resize(closest_image, (tile_size, tile_size))
            return x_start, y_start, closest_image
        return None

    # Parallel processing for speed-up
    results = Parallel(n_jobs=-1)(
        delayed(process_tile)(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE)
    )

    # Apply the processed tiles
    for result in results:
        if result:
            x_start, y_start, tile_image = result
            reconstructed_image[y_start:y_start+tile_size, x_start:x_start+tile_size] = tile_image

    # Convert to RGB for Gradio
    return cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)


examples = [
    ["scraped_photos/image_1002.jpg"],  # Preloaded image files
    ["scraped_photos/image_1003.jpg"],
    ["scraped_photos/image_1004.jpg"],
]

def refresh_interface():
    """Resets the UI without clearing input fields."""
    return gr.update(), gr.update(), gr.update(), gr.update()

# GRADIO INTERFACE
with gr.Blocks() as interface:
    gr.Markdown("Mosaic Generator")
    gr.Markdown("Upload an image, adjust parameters, and generate a mosaic.")

    with gr.Row():
        input_image = gr.Image(type="numpy", label="Upload an Image")
        output_image = gr.Image(type="numpy", label="Mosaic Output")

    tile_size_slider = gr.Slider(minimum=2, maximum=50, value=TILE_SIZE, step=1, label="Tile Size (px)")
    image_size_slider = gr.Slider(minimum=100, maximum=1000, value=IMAGE_SIZE, step=50, label="Image Size (px)")
    database_folder_input = gr.Textbox(value=DATABASE_FOLDER, label="Tile Database Path")

    generate_button = gr.Button("Generate Mosaic")
    refresh_button = gr.Button("Refresh")

    generate_button.click(
        fn=reconstruct_image,
        inputs=[input_image, tile_size_slider, image_size_slider, database_folder_input],
        outputs=[output_image]
    )

    refresh_button.click(
        fn=refresh_interface,
        inputs=[],
        outputs=[input_image, tile_size_slider, image_size_slider, database_folder_input]
    )

    # Add Examples Section
    gr.Examples(
        examples=examples,
        inputs=[input_image],
        outputs=[output_image],
        label="Try with Example Images",
    )

interface.launch()