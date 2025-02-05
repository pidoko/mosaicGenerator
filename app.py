"""
Interactive Image Mosaic Generator
Author: Peter Chibuikem Idoko
Date: 2025-02-03

This program creates a mosaic by segmenting an input image into grid tiles
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

def load_database_images_kdtree(database_folder: str) -> tuple[KDTree, list, list]:
    """
    Loads images from the specified database folder, computes their average colors, 
    and builds a KD-Tree for efficient nearest-neighbor searches.

    Args:
        database_folder (str): Path to the folder containing the database images.

    Returns:
        tuple: A tuple containing:
            - KDTree: A KD-Tree built from the average colors of the images.
            - list: A list of image file paths corresponding to the colors in the KD-Tree.
            - list: A list of average colors (RGB tuples) used in the KD-Tree.

    Edge Cases:
        - Skips unreadable or corrupted images.
        - Returns (None, []) if no valid images are found in the database.
        - Converts image colors to integer tuples to avoid floating-point precision issues.

    Efficiency:
        - Building the KD-Tree takes O(N log N), where N is the number of images.
        - Querying the KD-Tree for nearest neighbors takes O(log N), significantly faster than O(N) linear search.
    """
    image_data = []
    image_paths = []

    for filename in os.listdir(database_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(database_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                continue  # Skip unreadable images
            
            avg_color = get_average_color(image)  # Returns (R, G, B)
            image_data.append(avg_color)
            image_paths.append(img_path)

    if not image_data:
        return None, []

    kd_tree = KDTree(image_data)  # Build KD-Tree from image colors
    return kd_tree, image_paths, image_data  # Return KD-Tree, paths, and colors


def find_closest_image(query_color: tuple[int, int, int], kd_tree: KDTree, image_paths: list[str], image_data: list[tuple[int, int, int]]) -> tuple[str, tuple[int, int, int]]:
    """
    Finds the image with the closest average color using a KD-Tree.

    Args:
        query_color (tuple[int, int, int]): The target color in (R, G, B) format.
        kd_tree (KDTree): A KD-Tree containing the average colors of database images.
        image_paths (list[str]): A list of file paths corresponding to the images in the database.
        image_data (list[tuple[int, int, int]]): A list of average colors (R, G, B) for the images.

    Returns:
        tuple: A tuple containing:
            - str: The file path of the image with the closest matching color.
            - tuple[int, int, int]: The average color of the matched image.

    Edge Cases:
        - Assumes `kd_tree` is properly built and contains valid data.
        - If `query_color` is not in `image_data`, the function returns the closest match.
        - Handles nearest-neighbor search efficiently in O(log N) time.
    """
    _, index = kd_tree.query(query_color)  # Find nearest color in O(log N)
    return image_paths[index], image_data[index]

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

    def process_tile(row: int, col: int) -> tuple[int, int, np.ndarray] | None:
        """
        Processes a single tile from the input image, finds the closest matching image 
        using a KD-Tree, and returns the tile's position and corresponding replacement image.

        Args:
            row (int): The row index of the tile in the mosaic grid.
            col (int): The column index of the tile in the mosaic grid.

        Returns:
            tuple[int, int, np.ndarray] | None: A tuple containing:
                - int: The x-coordinate (column) of the tile's top-left corner.
                - int: The y-coordinate (row) of the tile's top-left corner.
                - np.ndarray: The selected tile image resized to the correct tile size.
            Returns None if no matching image is found.

        Edge Cases:
            - Handles cases where the input image is not properly loaded.
            - Ensures the selected image is resized to match the tile size.
            - If no valid image is found, returns None to avoid errors in reconstruction.
        
        Efficiency:
            - Uses a KD-Tree for nearest-neighbor search, making color matching O(log N).
            - Operates in parallel (if used in a multi-threaded/multi-process pipeline) to speed up reconstruction.
        """
        x_start, y_start = col * tile_size, row * tile_size
        x_end, y_end = x_start + tile_size, y_start + tile_size

        cell = input_img[y_start:y_end, x_start:x_end]  # Extract the tile region
        avg_color = get_average_color(cell)  # Compute its average color

        # Find the closest matching image using KD-Tree
        closest_image_path, _ = find_closest_image(avg_color, kd_tree, image_paths, image_data)

        if closest_image_path:
            closest_image = cv2.imread(closest_image_path)
            closest_image = cv2.resize(closest_image, (tile_size, tile_size))
            return x_start, y_start, closest_image

        return None

    """
    Parallel processing for faster tile processing.

    This step uses joblib's Parallel and delayed functions to distribute 
    the workload across multiple CPU cores. Each tile in the image grid is 
    processed in parallel, significantly improving performance.

    Implementation:
        - `n_jobs=-1` ensures that all available CPU cores are utilized.
        - `delayed(process_tile)(row, col)` schedules each tile for processing.
        - The list comprehension iterates over all rows and columns in the grid.

    Efficiency:
        - Reduces runtime from O(N) sequential processing to approximately O(N / C),
          where C is the number of CPU cores.
        - Automatically distributes workload, making it scalable for larger images.

    Edge Cases:
        - Ensures that all tiles are processed even if some return None.
        - Handles failures in individual tile processing without stopping execution.
    """
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
    """
    Refreshes the user interface without clearing input fields.

    This function updates all UI elements to reflect any changes while preserving 
    the existing input values. It is useful for refreshing outputs without requiring 
    users to re-enter data.

    Returns:
        tuple: A tuple of `gr.update()` calls, one for each UI component that needs refreshing.

    Edge Cases:
        - Ensures UI elements are updated without resetting user input.
        - Prevents unnecessary clearing of fields while allowing the interface to be refreshed.
        - May require additional `gr.update()` calls if more UI components are added in the future.
    """
    return gr.update(), gr.update(), gr.update(), gr.update()

"""
Gradio Interface for the Mosaic Generator.

This interface allows users to upload an image, adjust parameters, and generate 
a photomosaic using images from a specified database. It provides an interactive 
UI for users to experiment with different settings and generate high-quality mosaics.

### Features:
- **Image Upload:** Users can upload an image as input.
- **Tile Size Adjustment:** A slider to modify the tile size.
- **Image Size Adjustment:** A slider to control the final mosaic size.
- **Tile Database Selection:** A textbox to specify the folder containing tile images.
- **Generate Mosaic Button:** Processes the image and constructs the mosaic.
- **Refresh Button:** Updates the UI without clearing input fields.
- **Example Images Section:** Allows users to try preloaded example images.

### Components:
- **Markdown Headers:** Display instructions and descriptions.
- **Image Inputs/Outputs:** 
  - `input_image`: User-uploaded image.
  - `output_image`: Generated photomosaic.
- **Sliders:**
  - `tile_size_slider`: Controls the size of mosaic tiles.
  - `image_size_slider`: Controls the overall size of the generated mosaic.
- **Textbox:** 
  - `database_folder_input`: Specifies the folder containing database images.
- **Buttons:**
  - `generate_button`: Calls `reconstruct_image()` to generate the mosaic.
  - `refresh_button`: Calls `refresh_interface()` to update the UI.
- **Example Section:** Allows users to load predefined example images.

### Button Functionality:
- **Generate Button:**
  - Calls `reconstruct_image()`
  - Inputs: `[input_image, tile_size_slider, image_size_slider, database_folder_input]`
  - Outputs: `[output_image]`
- **Refresh Button:**
  - Calls `refresh_interface()`
  - Inputs: `[]`
  - Outputs: `[input_image, tile_size_slider, image_size_slider, database_folder_input]`

### Edge Cases:
- Ensures users can modify tile size and image resolution dynamically.
- Prevents the UI from resetting unnecessarily when refreshing.
- Provides example images for quick testing without requiring uploads.
"""
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

"""
Launches the Gradio interface.

This function starts the Gradio web-based UI, allowing users to interact with 
the Mosaic Generator. Once launched, users can upload images, adjust parameters, 
and generate mosaics through a browser.

### Behavior:
- Initializes and serves the Gradio interface on a local or public URL.
- Provides an interactive UI for processing and visualizing image mosaics.
- Runs a web server that remains active until manually stopped.

### Edge Cases:
- If Gradio is running in a notebook, it may default to inline display.
- If launched in a standalone script, it opens a browser window.
- Requires all components (buttons, sliders, images) to be correctly defined before launching.
"""
interface.launch()