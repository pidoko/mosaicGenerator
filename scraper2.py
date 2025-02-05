# Web Scraper by Peter Chibuikem Idoko
# 2025-02-03
# Scraped a 1000 images to be quantized before use in mosaic generator

import os
import time
import logging
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager as driver
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as EC

# Collect info, warning and error logs in scraper.log
logging.basicConfig(
    filename="scraper.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Constants for this scraper
BASE_URL = "https://www.shopify.com/stock-photos/photos"
OUTPUT_DIR = "scraped_photos"
MAX_IMAGES = 1000 
NUM_PAGES = 16
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WebScraper/1.0)"}

# Initialize Selenium's WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920x1080")

# Function to handle clicking next
def click_next_page():
    """Clicks the Next button to navigate pages, returning False if no button exists"""
    try:
        next_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Next')]")))
        next_button.click()
        logging.info("Navigated to the next page.")
        return True
    except (NoSuchElementException, TimeoutError, ElementClickInterceptedException):
        logging.warning("Next button not found or not clickable.")
        return False
    
# Function to download images efficiently
def download_image(img_url, img_name):
    """Downloads an image with error handling"""
    try:
        response = requests.get(img_url, headers=HEADERS, stream=True, timeout=10)
        response.raise_for_status()
        with open(os.path.join(OUTPUT_DIR, img_name), "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        logging.info(f"Downloaded image: {img_name}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {img_url}: {e}")
        return False

# Start scraping
downloaded = len(os.listdir(OUTPUT_DIR))  # Track existing images
# Loop through pages and images and track progress
for page in range(NUM_PAGES):
    logging.info(f"Processing page {page + 1}/{NUM_PAGES}...")
    
    # Extract image elements
    images = driver.find_elements(By.TAG_NAME, "img")

    # Download images
    for img in images:
        if downloaded >= MAX_IMAGES:
            logging.info("Reached max image limit.")
            break
        img_url = img.get_attribute("src")
        if img_url and img_url.startswith("https"):
            img_name = f"image_{downloaded + 1}.jpg"
            if download_image(img_url, img_name):
                downloaded += 1
    
    if not click_next_page():
        logging.info("No more pages available.")
        break

driver.quit()
logging.info("Web scraping completed successfully!")
print("Web scraping completed. Check scraper.log for details.")
