---
title: mosaicGenerator
app_file: app.py
sdk: gradio
sdk_version: 5.13.2
---
# Interactive Image Mosaic Generator 

## **Overview**
This project creates a **mosaic** by segmenting an input image into a grid of tiles and replacing each tile with the closest-matching image from a predefined database.

## **Installation**
### **Prerequisites**
Ensure you have Python **3.8+** and install the required dependencies:
```bash
pip install gradio opencv-python numpy scipy joblib argparse scikit-image   
```

## **Usage**
### **Basic Execution**
To run the mosaic generator:
```bash
python mosaic_generator.py
```

### **Command-Line Arguments**
You can modify the script to accept command-line arguments using argparse:
```bash
python mosaic_generator.py --input path/to/image.jpg --database path/to/database --output output.jpg --tile-size 5
```

### **Configuration Options**
```bash
IMAGE_SIZE: Size of the input image (default: 300x300).
TILE_SIZE: Size of individual tiles (smaller = more detail).
GRID_SIZE: Automatically calculated based on TILE_SIZE.
DATABASE_FOLDER: Path to the folder containing the image tiles.
```

### **Performance Improvements**
Uses parallel processing and a KD Tree for faster execution.

## Author 
**Peter Chibuikem Idoko**
## Date 
**2025-02-03**