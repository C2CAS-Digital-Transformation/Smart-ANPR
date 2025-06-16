import os
import csv
from pathlib import Path
import cv2

# Directory containing images
IMAGE_DIR = r"D:\Work\Projects\ANPR_3\combined_training_data"
# Output CSV file
CSV_PATH = r"D:\Work\Projects\ANPR_3\combined_training_data_sizes.csv"

# Supported image extensions
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

def get_image_size(image_path):
    img = cv2.imread(str(image_path))
    if img is not None:
        h, w = img.shape[:2]
        return w, h
    return None, None

def main():
    image_dir = Path(IMAGE_DIR)
    
    # Use rglob to find all images in the directory and its subdirectories
    image_files = [f for f in image_dir.rglob('*') if f.suffix.lower() in IMAGE_EXTS]
    
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'width', 'height', 'subdirectory'])
        for img_path in image_files:
            width, height = get_image_size(img_path)
            if width is not None and height is not None:
                # Get the subdirectory name relative to the image_dir
                subdirectory = img_path.parent.name
                writer.writerow([img_path.name, width, height, subdirectory])
            else:
                print(f"Warning: Could not read image {img_path}")
    print(f"CSV saved to {CSV_PATH}")

if __name__ == "__main__":
    main() 