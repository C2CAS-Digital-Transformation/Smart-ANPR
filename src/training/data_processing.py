import os
import shutil
import re
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
import difflib
from sklearn.model_selection import train_test_split

# Paths
SOURCE_DIR = r"D:\\Work\\Projects\\ANPR_3\\Cleaned_OCR_Data"
OUTPUT_DIR = r"D:\\Work\\Projects\\ANPR_3\\combined_training_data"

# Output subdirectories for EasyOCR format
ALL_DATA_TEMP_DIR = os.path.join(OUTPUT_DIR, "all_processed_images") # Temporary holding for all processed images
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train_data")
VAL_DIR = os.path.join(OUTPUT_DIR, "val_data")
TEST_DIR = os.path.join(OUTPUT_DIR, "test_data")

# Create directories
os.makedirs(ALL_DATA_TEMP_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Patterns for license plates (adjust as needed for Indian context)
# General Indian license plate format: AA00AA0000, AA00A0000, AA000000
LICENSE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$')
# Stricter validation for "final" labels might be useful
STRICT_LICENSE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{1,2}(?:[A-Z]{1,3})?[0-9]{4}$')

# State codes for fuzzy matching (example, expand as needed)
STATE_CODES = [
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "AN", "CH", "DD", "DL", "JK", "LA", "LD", "PY"
]

CONFUSED_CHARS = {
    '0': ['O', 'D'], 'O': ['0', 'D'], 'D': ['0', 'O'],
    '1': ['I', 'L'], 'I': ['1', 'L'], 'L': ['1', 'I'],
    '2': ['Z'], 'Z': ['2'],
    '5': ['S'], 'S': ['5'],
    '8': ['B'], 'B': ['8'],
    'G': ['6'], '6': ['G']
}

# Define separate target sizes for single-line and multi-line plates
SINGLE_LINE_TARGET_WIDTH = 448
SINGLE_LINE_TARGET_HEIGHT = 64
MULTI_LINE_TARGET_WIDTH = 256
MULTI_LINE_TARGET_HEIGHT = 128

# Aspect ratio thresholds to distinguish between single and multi-line plates
# These values may need tuning based on your dataset analysis
SINGLE_LINE_ASPECT_RATIO_THRESHOLD = 3.0  # Typical for long, thin plates
MULTI_LINE_ASPECT_RATIO_THRESHOLD = 2.0   # Typical for more square-like plates

# Update constants for better data distribution
MIN_IMAGES_PER_STATE = 300  # Increased for better state coverage
TARGET_IMAGES_PER_STATE = 500  # Target number of images per state
MAX_IMAGES_PER_STATE = 1000  # Cap to prevent over-representation
TWO_LINE_PLATE_CHANCE = 0.5  # Increased for better multi-line coverage
QUALITY_THRESHOLD = 0.4  # Increased quality threshold

# --- Added helper for consistent deskewing with test-time pipeline ---
def deskew_image(image):
    """Deskew the image using Hough transform."""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold to get binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find all non-zero points
    coords = np.column_stack(np.where(thresh > 0))
    
    # Find minimum rotated rectangle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle
    if angle < -45:
        angle = 90 + angle
    
    # Rotate the image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def resize_and_pad(image, target_size):
    """
    Resize an image to a target size while maintaining aspect ratio by padding.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    target_w, target_h = target_size
    h, w = image.shape[:2]
    
    # Calculate scale and new dimensions
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Create a new image with a white background and paste the resized image
    padded_img = np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    # Calculate pasting position (center)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return padded_img

# --- Enhanced Image and Label Processing Functions ---

def advanced_image_preprocessing(image):
    """Apply advanced preprocessing to improve image quality."""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply deskewing
    deskewed = deskew_image(gray)
    
    # Denoise the grayscale image first
    denoised = cv2.fastNlMeansDenoising(deskewed, h=10)
    
    # Enhance contrast on the denoised image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Apply adaptive thresholding as the final step
    final_thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    return final_thresh

def finalize_image_for_training(image):
    """
    Finalize image preprocessing for training, adapting to aspect ratio.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply advanced preprocessing
    processed = advanced_image_preprocessing(image)
    
    # Determine layout based on aspect ratio
    h, w = processed.shape[:2]
    aspect_ratio = w / h if h > 0 else 0
    
    if aspect_ratio > SINGLE_LINE_ASPECT_RATIO_THRESHOLD:
        # Likely a single-line plate
        target_size = (SINGLE_LINE_TARGET_WIDTH, SINGLE_LINE_TARGET_HEIGHT)
    else:
        # Likely a multi-line or square plate
        target_size = (MULTI_LINE_TARGET_WIDTH, MULTI_LINE_TARGET_HEIGHT)
        
    # Resize with padding
    final_image = resize_and_pad(processed, target_size)
    
    # Convert back to PIL Image
    pil_image = Image.fromarray(final_image)
    
    return pil_image

def detect_text_regions(image):
    """Detect text regions in license plate image"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Use morphological operations to find text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [gray]  # Return original if no text regions found
    
    # Get bounding rectangles
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda x: x[1])  # Sort by y-coordinate
    
    # Group boxes into lines (if multi-line)
    lines = []
    current_line = []
    
    for box in boxes:
        x, y, w, h = box
        if not current_line:
            current_line.append(box)
        else:
            # Check if this box is on the same line as previous boxes
            last_y = current_line[-1][1]
            if abs(y - last_y) < gray.shape[0] * 0.3:  # Same line
                current_line.append(box)
            else:  # New line
                lines.append(current_line)
                current_line = [box]
    
    if current_line:
        lines.append(current_line)
    
    return lines

def get_base_label(filename):
    """Extracts the base license plate number from a filename."""
    name_part = Path(filename).stem
    # Remove suffixes like _1, _2 or (1), (2)
    name_part = re.sub(r'_\d+$', '', name_part)
    name_part = re.sub(r'\s*\(\d+\)$', '', name_part)
    return name_part.upper()

def normalize_license_number(text):
    """Enhanced normalization with better fuzzy logic"""
    text = text.upper().replace(" ", "").replace("-", "")
    
    # Remove common OCR artifacts
    text = text.replace("_", "").replace(".", "").replace(",", "")
    
    # State code corrections
    state_corrections = {
        "IS": "TS", "T5": "TS", "15": "TS",
        "KL": "KL", "KA": "KA", "TN": "TN",
        "MH": "MH", "GJ": "GJ", "HR": "HR",
        "DL": "DL", "UP": "UP", "MP": "MP",
        "WB": "WB", "PB": "PB", "RJ": "RJ"
    }
    
    # Apply state corrections
    for wrong, correct in state_corrections.items():
        if text.startswith(wrong):
            text = correct + text[len(wrong):]
            break
    
    # Character-by-character correction based on position
    corrected_text = ""
    for i, char in enumerate(text):
        if i < 2:  # State code - should be letters
            if char.isdigit():
                # Try to convert digits to similar letters
                digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'}
                corrected_text += digit_to_letter.get(char, char)
            else:
                corrected_text += char
        elif i == 2 or i == 3:  # District numbers - should be digits
            if not char.isdigit():
                letter_to_digit = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6', 'Z': '2'}
                corrected_text += letter_to_digit.get(char, char)
            else:
                corrected_text += char
        else:
            corrected_text += char
    
    text = corrected_text
    
    # Fuzzy match state codes
    if len(text) >= 2 and text[:2] not in STATE_CODES:
        first_two = text[:2]
        closest_match = difflib.get_close_matches(first_two, STATE_CODES, n=1, cutoff=0.6)
        if closest_match:
            text = closest_match[0] + text[2:]
    
    return text

def is_good_label(label):
    """Enhanced validation for license plate patterns"""
    if not label or len(label) < 6:
        return False
    
    # Check basic pattern
    if not LICENSE_PATTERN.match(label):
        return False
    
    # Additional checks
    if len(label) > 13:  # Too long
        return False
    
    # Check state code
    if label[:2] not in STATE_CODES:
        return False
    
    return True

# --- Enhanced Augmentation and Synthetic Data ---

def augment_image(img_pil, num_augmentations=3):
    """Augment image with various techniques for robustness."""
    if not isinstance(img_pil, Image.Image):
        img_pil = Image.fromarray(img_pil)

    augmented_images = []
    
    for _ in range(num_augmentations):
        try:
            # Start with original image
            aug_img = np.array(img_pil)
            
            # Apply random augmentations
            
            # 1. Geometric transformations (30% chance)
            if random.random() < 0.3:
                rows, cols = aug_img.shape[:2] if len(aug_img.shape) == 2 else aug_img.shape[:2]
                
                # Small rotation (-1.5 to 1.5 degrees)
                angle = random.uniform(-1.5, 1.5)
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                aug_img = cv2.warpAffine(aug_img, M, (cols, rows), borderValue=255)
                
                # Small translation
                tx = random.randint(-3, 3)
                ty = random.randint(-3, 3)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                aug_img = cv2.warpAffine(aug_img, M, (cols, rows), borderValue=255)
            
            # 2. Brightness and contrast (40% chance)
            if random.random() < 0.4:
                aug_img_pil = Image.fromarray(aug_img) if len(aug_img.shape) == 2 else Image.fromarray(aug_img)
                
                # Brightness
                enhancer = ImageEnhance.Brightness(aug_img_pil)
                brightness_factor = random.uniform(0.9, 1.1)
                aug_img_pil = enhancer.enhance(brightness_factor)
                
                # Contrast
                enhancer = ImageEnhance.Contrast(aug_img_pil)
                contrast_factor = random.uniform(0.9, 1.1)
                aug_img_pil = enhancer.enhance(contrast_factor)
                
                aug_img = np.array(aug_img_pil)
            
            # 3. Blur (10% chance, mild)
            if random.random() < 0.1:
                blur_kernel = 3
                aug_img = cv2.GaussianBlur(aug_img, (blur_kernel, blur_kernel), 0)
            
            # 4. Noise (20% chance)
            if random.random() < 0.2:
                noise = np.random.normal(0, 5, aug_img.shape).astype(np.uint8)
                aug_img = cv2.add(aug_img, noise)
            
            # 5. Minor perspective distortion (10% chance, mild)
            if random.random() < 0.1:
                rows, cols = aug_img.shape[:2]
                # Small perspective change
                pts1 = np.float32([[0,0], [cols,0], [0,rows], [cols,rows]])
                offset = random.randint(2, 5)
                pts2 = np.float32([[offset,offset], [cols-offset,offset], 
                                  [offset,rows-offset], [cols-offset,rows-offset]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                aug_img = cv2.warpPerspective(aug_img, M, (cols, rows), borderValue=255)
            
            # Convert back to PIL and finalize
            if len(aug_img.shape) == 2:
                final_aug_img = Image.fromarray(aug_img, mode='L')
            else:
                final_aug_img = Image.fromarray(aug_img)
            
            # Apply final processing
            final_aug_img = finalize_image_for_training(final_aug_img)
            if final_aug_img is not None:
                augmented_images.append(final_aug_img)
                
        except Exception as e:
            print(f"Error during simple augmentation: {e}")
    
    return augmented_images

def generate_enhanced_synthetic_plate_image(text, two_lines=False):
    """Generate a high-quality synthetic license plate image with more realism."""
    if two_lines:
        width, height = MULTI_LINE_TARGET_WIDTH, MULTI_LINE_TARGET_HEIGHT
        return enhanced_multiline_plate_generation(text)
    else:
        width, height = SINGLE_LINE_TARGET_WIDTH, SINGLE_LINE_TARGET_HEIGHT

    # Base image
    plate_color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
    image = Image.new('RGB', (width, height), color=plate_color)
    draw = ImageDraw.Draw(image)
    
    # Add subtle texture to background
    for _ in range(random.randint(50, 150)):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        noise_color = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in plate_color)
        draw.point((x, y), fill=noise_color)
    
    # Font settings
    try:
        # Try multiple font paths
        font_paths = [
            "arial.ttf", "Arial.ttf", "/Windows/Fonts/arial.ttf",
            "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        font = None
        for font_path in font_paths:
            try:
                font_size = int(height * 0.5) if not two_lines else int(height * 0.3)
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        if font is None:
            font_size = int(height * 0.5) if not two_lines else int(height * 0.3)
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Text color (usually black or dark blue)
    text_colors = [(0, 0, 0), (0, 0, 139), (25, 25, 112)]
    text_color = random.choice(text_colors)
    
    if two_lines and len(text) > 4:
        # Enhanced two-line splitting logic
        # Common Indian formats: TS07EC1234 -> TS07 / EC1234
        if len(text) >= 8:
            # Find the best split point
            if text[4].isalpha():  # XX00AAYYYY format
                split_point = 4
                while split_point < len(text) - 2 and text[split_point].isalpha():
                    split_point += 1
            else:  # XX00YYYY format
                split_point = 4
        else:
            split_point = len(text) // 2
        
        line1 = text[:split_point]
        line2 = text[split_point:]
        
        # Calculate positions for two lines
        if hasattr(draw, 'textbbox'):
            bbox1 = draw.textbbox((0, 0), line1, font=font)
            bbox2 = draw.textbbox((0, 0), line2, font=font)
            text_width1 = bbox1[2] - bbox1[0]
            text_height1 = bbox1[3] - bbox1[1]
            text_width2 = bbox2[2] - bbox2[0]
            text_height2 = bbox2[3] - bbox2[1]
        else:
            text_width1, text_height1 = draw.textsize(line1, font=font)
            text_width2, text_height2 = draw.textsize(line2, font=font)
        
        # Position lines with proper spacing
        y_spacing = height // 3
        y_pos1 = y_spacing - text_height1 // 2
        y_pos2 = 2 * y_spacing - text_height2 // 2
        
        x_pos1 = (width - text_width1) // 2
        x_pos2 = (width - text_width2) // 2
        
        # Add drop shadow for better readability
        shadow_offset = 1
        draw.text((x_pos1 + shadow_offset, y_pos1 + shadow_offset), line1, 
                 font=font, fill=(128, 128, 128))
        draw.text((x_pos2 + shadow_offset, y_pos2 + shadow_offset), line2, 
                 font=font, fill=(128, 128, 128))
        
        # Draw main text
        draw.text((x_pos1, y_pos1), line1, font=font, fill=text_color)
        draw.text((x_pos2, y_pos2), line2, font=font, fill=text_color)
        
    else:  # Single line
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = draw.textsize(text, font=font)
        
        x_pos = (width - text_width) // 2
        y_pos = (height - text_height) // 2
        
        # Add drop shadow
        shadow_offset = 1
        draw.text((x_pos + shadow_offset, y_pos + shadow_offset), text, 
                 font=font, fill=(128, 128, 128))
        
        # Draw main text
        draw.text((x_pos, y_pos), text, font=font, fill=text_color)
    
    # Add border (like real license plates)
    border_color = (0, 0, 0)
    border_width = 2
    draw.rectangle([0, 0, width-1, height-1], outline=border_color, width=border_width)
    
    # Convert to array and apply realistic effects
    img_np = np.array(image)
    
    # Apply simple realistic distortions
    if random.random() < 0.3:  # 30% chance for rotation
        rows, cols = img_np.shape[:2] if len(img_np.shape) == 2 else img_np.shape[:2]
        angle = random.uniform(-1, 1)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_np = cv2.warpAffine(img_np, M, (cols, rows), borderValue=255)
    
    if random.random() < 0.2:  # 20% chance for perspective
        rows, cols = img_np.shape[:2]
        offset = random.randint(1, 3)
        pts1 = np.float32([[0,0], [cols,0], [0,rows], [cols,rows]])
        pts2 = np.float32([[offset,offset], [cols-offset,offset], 
                          [offset,rows-offset], [cols-offset,rows-offset]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = cv2.warpPerspective(img_np, M, (cols, rows), borderValue=255)
    
    if random.random() < 0.1:  # 10% chance for blur
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
    
    if random.random() < 0.1:  # 10% chance for noise
        noise = np.random.normal(0, 3, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
    
    # Convert back to PIL and finalize
    final_img_pil = Image.fromarray(img_np).convert('L')
    return finalize_image_for_training(final_img_pil)

def generate_random_plate_text(state_code):
    """Generate a random license plate number for a given state."""
    # Different Indian license plate formats
    formats = [
        "{state}{dd}{letters}{dddd}",  # TS07EC1234 (most common)
        "{state}{dd}{letter}{dddd}",   # KA05M1234
        "{state}{dd}{dddd}",           # DL8C1234 (older format)
        "{state}{d}{letter}{dddd}",    # UP9C1234
    ]
    
    format_choice = random.choice(formats)
    
    # Generate components
    dd = f"{random.randint(1, 99):02d}"
    d = f"{random.randint(1, 9)}"
    dddd = f"{random.randint(1, 9999):04d}"
    letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(1, 2)))
    
    result = format_choice.format(
        state=state_code,
        dd=dd,
        d=d,
        dddd=dddd,
        letter=letter,
        letters=letters
    )
    
    return result

def enhanced_multiline_plate_generation(text):
    """
    Generate a more realistic two-line license plate with better text splitting.
    """
    width, height = MULTI_LINE_TARGET_WIDTH, MULTI_LINE_TARGET_HEIGHT
    
    # Intelligent splitting of the license plate number
    if len(text) < 8:  # Not enough characters for a standard multi-line plate
        return None
        
    # Split point is typically after the state and district code (4 chars)
    line1 = text[:4]
    line2 = text[4:]

    # Create a base image with a realistic plate color
    plate_color = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
    image = Image.new('RGB', (width, height), plate_color)
    draw = ImageDraw.Draw(image)
    
    # Font loading with multiple fallbacks
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf", 
        "/System/Library/Fonts/Arial.ttf",
        "arial.ttf", "Arial.ttf"
    ]
    
    font = None
    base_font_size = int(height * 0.35)
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, base_font_size)
            break
        except:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # Text color (black or dark blue for contrast)
    text_color = random.choice([(0, 0, 0), (0, 0, 139)])
    
    # Add spacing for readability (Indian standard)
    if len(line1) >= 4:
        line1 = line1[:2] + " " + line1[2:]  # XX ##
    if len(line2) >= 3:
        if line2[0:2].isalpha():  # XX ####
            line2 = line2[:2] + " " + line2[2:]
        elif line2[0].isalpha():  # X ####
            line2 = line2[0] + " " + line2[1:]
    
    # Calculate text dimensions
    if hasattr(draw, 'textbbox'):
        bbox1 = draw.textbbox((0, 0), line1, font=font)
        bbox2 = draw.textbbox((0, 0), line2, font=font)
        text_width1, text_height1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        text_width2, text_height2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    else:
        text_width1, text_height1 = draw.textsize(line1, font=font)
        text_width2, text_height2 = draw.textsize(line2, font=font)
    
    # Position lines with proper Indian license plate spacing
    available_height = height - 2 * 2  # Assuming border_thickness is 2
    line_spacing = available_height // 3
    
    y_pos1 = 2 + line_spacing - text_height1 // 2
    y_pos2 = 2 + 2 * line_spacing - text_height2 // 2
    
    x_pos1 = (width - text_width1) // 2
    x_pos2 = (width - text_width2) // 2
    
    # Add subtle shadow for depth
    shadow_offset = 2
    shadow_color = (128, 128, 128)
    
    # Draw shadows
    draw.text((x_pos1 + shadow_offset, y_pos1 + shadow_offset), line1, 
             font=font, fill=shadow_color)
    draw.text((x_pos2 + shadow_offset, y_pos2 + shadow_offset), line2, 
             font=font, fill=shadow_color)
    
    # Draw main text
    draw.text((x_pos1, y_pos1), line1, font=font, fill=text_color)
    draw.text((x_pos2, y_pos2), line2, font=font, fill=text_color)
    
    # Add "INDIA" text at bottom (common in Indian plates)
    if random.random() < 0.3:  # 30% chance
        india_font_size = int(height * 0.08)
        try:
            india_font = ImageFont.truetype(font_paths[0], india_font_size)
        except:
            india_font = font
        
        india_text = "INDIA"
        if hasattr(draw, 'textbbox'):
            india_bbox = draw.textbbox((0, 0), india_text, font=india_font)
            india_width = india_bbox[2] - india_bbox[0]
        else:
            india_width, _ = draw.textsize(india_text, font=india_font)
        
        india_x = (width - india_width) // 2
        india_y = height - int(height * 0.15)
        
        draw.text((india_x, india_y), india_text, font=india_font, fill=(0, 0, 0))
    
    # Convert to numpy and apply realistic effects
    img_np = np.array(image)
    
    # Apply simple realistic distortions
    if random.random() < 0.3:  # 30% chance for rotation
        rows, cols = img_np.shape[:2] if len(img_np.shape) == 2 else img_np.shape[:2]
        angle = random.uniform(-1, 1)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_np = cv2.warpAffine(img_np, M, (cols, rows), borderValue=255)
    
    if random.random() < 0.2:  # 20% chance for perspective
        rows, cols = img_np.shape[:2]
        offset = random.randint(1, 3)
        pts1 = np.float32([[0,0], [cols,0], [0,rows], [cols,rows]])
        pts2 = np.float32([[offset,offset], [cols-offset,offset], 
                          [offset,rows-offset], [cols-offset,rows-offset]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = cv2.warpPerspective(img_np, M, (cols, rows), borderValue=255)
    
    if random.random() < 0.1:  # 10% chance for blur
        img_np = cv2.GaussianBlur(img_np, (3, 3), 0)
    
    if random.random() < 0.1:  # 10% chance for noise
        noise = np.random.normal(0, 3, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
    
    # Convert back to PIL and finalize
    final_img_pil = Image.fromarray(img_np).convert('L')
    return finalize_image_for_training(final_img_pil)

def is_variant_image(filename):
    """Check if a filename suggests it's a variant of another image."""
    stem = Path(filename).stem
    # Check for patterns like _1, _2, _copy, (1), (2), etc.
    variant_patterns = [
        r'_\d+$',           # Ends with _1, _2, etc.
        r'_copy\d*$',       # Ends with _copy, _copy1, etc.
        r'\s*\(\d+\)$',     # Ends with (1), (2), etc.
        r'_duplicate\d*$',  # Ends with _duplicate, _duplicate1, etc.
        r'_angle\d*$',      # Ends with _angle, _angle1, etc.
    ]
    
    for pattern in variant_patterns:
        if re.search(pattern, stem, re.IGNORECASE):
            return True
    return False

def select_best_primary_image(files):
    """Select the best primary image from a group, preferring non-variants"""
    if not files:
        return None
    
    # Separate primary images from variants
    primary_images = [f for f in files if not is_variant_image(f)]
    variant_images = [f for f in files if is_variant_image(f)]
    
    # If we have primary images, choose the shortest name (likely the main one)
    if primary_images:
        return sorted(primary_images, key=len)[0]
    
    # If only variants exist, choose the first variant (but this shouldn't happen often)
    if variant_images:
        return sorted(variant_images, key=len)[0]
    
    return files[0]  # Fallback

def assess_image_quality(img_path):
    """Assess image quality to determine if it adds training value"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return 0.0
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Quality metrics
        # 1. Image size (bigger is generally better for OCR)
        size_score = min(1.0, (gray.shape[0] * gray.shape[1]) / (100 * 50))  # Normalize by minimum useful size
        
        # 2. Contrast (higher contrast is better for text)
        contrast_score = gray.std() / 128.0  # Normalize by max possible std
        
        # 3. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 1000.0)  # Normalize
        
        # 4. Text region detection (more text regions = better)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = len([c for c in contours if cv2.contourArea(c) > 20])
        text_score = min(1.0, text_regions / 10.0)  # Normalize
        
        # Combined quality score
        quality_score = (size_score * 0.2 + contrast_score * 0.3 + 
                        sharpness_score * 0.3 + text_score * 0.2)
        
        return min(1.0, quality_score)
        
    except Exception as e:
        print(f"Error assessing quality for {img_path}: {e}")
        return 0.5  # Default medium quality

def detect_multiline_layout(img_path):
    """Detect if an image contains a multi-line license plate layout"""
    try:
        # Read image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, None
        
        # Apply preprocessing to enhance text detection
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Apply adaptive thresholding to create binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours that could be text
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio (likely to be text)
        text_contours = []
        img_height, img_width = img.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (not too small, not too large)
            area = cv2.contourArea(contour)
            if area < (img_width * img_height * 0.005):  # Too small
                continue
            if area > (img_width * img_height * 0.8):    # Too large
                continue
                
            # Filter by aspect ratio (width should be reasonable for text)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 8:  # Not text-like
                continue
                
            text_contours.append((x, y, w, h))
        
        if len(text_contours) < 2:  # Need at least 2 text regions for multi-line
            return False, None
        
        # Sort contours by y-coordinate (top to bottom)
        text_contours.sort(key=lambda c: c[1])
        
        # Group contours into lines based on y-coordinate proximity
        lines = []
        current_line = [text_contours[0]]
        
        for i in range(1, len(text_contours)):
            current_y = text_contours[i][1]
            prev_y = text_contours[i-1][1]
            
            # If y-coordinates are close, add to current line
            if abs(current_y - prev_y) < img_height * 0.15:  # 15% of image height
                current_line.append(text_contours[i])
            else:
                # Start new line
                lines.append(current_line)
                current_line = [text_contours[i]]
        
        lines.append(current_line)  # Add the last line
        
        # Check if we have multiple lines with reasonable spacing
        if len(lines) >= 2:
            # Calculate vertical spacing between lines
            line1_y = sum(c[1] for c in lines[0]) / len(lines[0])
            line2_y = sum(c[1] for c in lines[1]) / len(lines[1])
            
            vertical_spacing = abs(line2_y - line1_y)
            
            # Check if spacing is reasonable for license plate
            if vertical_spacing > img_height * 0.2 and vertical_spacing < img_height * 0.8:
                return True, len(lines)
        
        return False, None
        
    except Exception as e:
        print(f"Error detecting multi-line layout for {img_path}: {e}")
        return False, None

def process_unique_license_plates(image_groups):
    """Process image groups to get only the best primary image per license plate"""
    unique_processed_data = []
    state_counts = defaultdict(int)
    
    print(f"Processing {len(image_groups)} unique license plates...")
    
    for label, files in tqdm(image_groups.items(), desc="Selecting best primary images"):
        if not files:
            continue
            
        state_code = label[:2]
        if state_code not in STATE_CODES:
            continue
            
        # Skip if we already have enough images for this state
        if state_counts[state_code] >= MAX_IMAGES_PER_STATE:
            continue
        
        # Step 1: Select the best primary image (non-variant)
        primary_file = select_best_primary_image(files)
        if not primary_file:
            continue
        
        # Step 2: Assess image quality
        img_path = os.path.join(SOURCE_DIR, primary_file)
        quality_score = assess_image_quality(img_path)
        
        # Step 3: Only process high-quality images
        if quality_score < QUALITY_THRESHOLD:
            continue
        
        # Step 4: Detect if this is a multi-line plate
        is_multiline, num_lines = detect_multiline_layout(img_path)
        
        # Step 5: Process the selected primary image
        try:
            img = Image.open(img_path)
            
            # Apply enhanced preprocessing
            processed_img = finalize_image_for_training(img.copy())
            if processed_img is not None:
                # Mark the image type based on detection
                image_type = f"primary_multiline_{num_lines}L" if is_multiline else "primary_single"
                
                unique_processed_data.append((
                    processed_img, 
                    label, 
                    f"{image_type}_{primary_file}",
                    quality_score,
                    is_multiline
                ))
                
                state_counts[state_code] += 1
                
                # Add high-quality variants if needed
                if state_counts[state_code] < TARGET_IMAGES_PER_STATE:
                    variant_files = [f for f in files if f != primary_file and not is_variant_image(f)]
                    for variant_file in variant_files:  # No slice, include all
                        variant_path = os.path.join(SOURCE_DIR, variant_file)
                        variant_quality = assess_image_quality(variant_path)
                        
                        if variant_quality > quality_score + 0.2:
                            try:
                                variant_img = Image.open(variant_path)
                                processed_variant = finalize_image_for_training(variant_img.copy())
                                if processed_variant is not None:
                                    variant_is_multiline, variant_num_lines = detect_multiline_layout(variant_path)
                                    variant_type = f"variant_multiline_{variant_num_lines}L" if variant_is_multiline else "variant_single"
                                    
                                    unique_processed_data.append((
                                        processed_variant,
                                        label,
                                        f"{variant_type}_{variant_file}",
                                        variant_quality,
                                        variant_is_multiline
                                    ))
                                    state_counts[state_code] += 1
                                    
                                    if state_counts[state_code] >= TARGET_IMAGES_PER_STATE:
                                        break
                            except Exception as e:
                                print(f"Error processing variant {variant_file}: {e}")
                
        except Exception as e:
            print(f"Error processing primary image {primary_file}: {e}")
            continue
    
    return unique_processed_data

def process_multiline_variants(image_groups):
    """Process image groups and create multi-line variants for training"""
    enhanced_data = []
    state_counts = defaultdict(int)
    
    # First get unique, high-quality primary images with multi-line detection
    unique_data = process_unique_license_plates(image_groups)
    print(f"Selected {len(unique_data)} high-quality unique images")
    
    # Count real multi-line vs single-line images
    real_multiline_count = 0
    real_singleline_count = 0
    
    # Convert to the expected format and add synthetic multi-line versions
    for processed_img, label, file_info, quality_score, is_multiline in unique_data:
        state_code = label[:2]
        
        # Add the original processed image
        enhanced_data.append((processed_img, label, file_info))
        state_counts[state_code] += 1
        
        if is_multiline:
            real_multiline_count += 1
        else:
            real_singleline_count += 1
            
            # Add synthetic multi-line version for single-line plates
            if len(label) >= 8 and quality_score > 0.6:
                multi_line_img = enhanced_multiline_plate_generation(label)
                if multi_line_img is not None:
                    enhanced_data.append((multi_line_img, label, f"synthetic_multiline_{file_info}"))
                    state_counts[state_code] += 1
    
    print(f"\n📊 MULTI-LINE DETECTION SUMMARY:")
    print(f"   Real multi-line plates found: {real_multiline_count}")
    print(f"   Real single-line plates found: {real_singleline_count}")
    print(f"   Synthetic multi-line generated: {len([x for x in enhanced_data if 'synthetic_multiline' in x[2]])}")
    
    return enhanced_data, state_counts

def main():
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(all_files)} initial image files in {SOURCE_DIR}")

    # 1. Group images by base label and normalize labels
    image_groups = defaultdict(list)
    for f_name in tqdm(all_files, desc="Grouping images by base label"):
        base_label = get_base_label(f_name)
        normalized_label = normalize_license_number(base_label)
        if is_good_label(normalized_label):
             image_groups[normalized_label].append(f_name)
        else:
            corrected_base = normalize_license_number(base_label)
            if is_good_label(corrected_base):
                image_groups[corrected_base].append(f_name)

    # 2. Enhanced processing with multi-line support
    print(f"Found {len(image_groups)} unique normalized license plates.")
    processed_data, state_counts = process_multiline_variants(image_groups)
    print(f"Generated {len(processed_data)} images after multi-line processing.")

    # 3. Enhanced data augmentation - Focused on underrepresented states
    augmented_data_to_add = []
    for pil_img, label, file_info in tqdm(processed_data, desc="Generating focused augmentations"):
        state_code = label[:2]
        current_count = state_counts[state_code]
        
        # Determine number of augmentations based on current state count
        if current_count < MIN_IMAGES_PER_STATE:
            num_augs = 3  # More augmentations for underrepresented states
        elif current_count < TARGET_IMAGES_PER_STATE:
            num_augs = 2  # Moderate augmentations
        else:
            num_augs = 1  # Minimal augmentations for well-represented states
            
        augs = augment_image(pil_img, num_augmentations=num_augs)
        for i, aug_img in enumerate(augs):
            if state_counts[state_code] < MAX_IMAGES_PER_STATE:
                augmented_data_to_add.append((aug_img, label, f"aug_{i}_{file_info}"))
                state_counts[state_code] += 1
    
    processed_data.extend(augmented_data_to_add)
    print(f"Total images after focused augmentation: {len(processed_data)}")

    # 4. Synthetic data generation - Focus on underrepresented states
    synthetic_data_to_add = []
    for state_code in tqdm(STATE_CODES, desc="Generating focused synthetic data"):
        current_count = state_counts[state_code]
        num_to_generate = max(0, TARGET_IMAGES_PER_STATE - current_count)
        
        if num_to_generate == 0:
            continue
        
        for i in range(num_to_generate):
            if state_counts[state_code] >= MAX_IMAGES_PER_STATE:
                break
                
            synthetic_text = generate_random_plate_text(state_code)
            if not is_good_label(synthetic_text):
                synthetic_text = normalize_license_number(synthetic_text)
                if not is_good_label(synthetic_text):
                    continue

            # Generate single-line version
            single_img = generate_enhanced_synthetic_plate_image(synthetic_text, two_lines=False)
            if single_img is not None:
                synthetic_data_to_add.append((single_img, synthetic_text, f"synthetic_single_{state_code}_{i}"))
                state_counts[state_code] += 1
            
            # 50% chance for multi-line version
            if random.random() < TWO_LINE_PLATE_CHANCE and len(synthetic_text) >= 8:
                multi_img = enhanced_multiline_plate_generation(synthetic_text)
                if multi_img is not None and state_counts[state_code] < MAX_IMAGES_PER_STATE:
                    synthetic_data_to_add.append((multi_img, synthetic_text, f"synthetic_multi_{state_code}_{i}"))
                    state_counts[state_code] += 1

    processed_data.extend(synthetic_data_to_add)
    print(f"Total images after focused synthetic generation: {len(processed_data)}")
    
    # Quality summary
    primary_count = len([x for x in processed_data if "primary_" in x[2]])
    variant_count = len([x for x in processed_data if "variant_" in x[2]])
    synthetic_count = len([x for x in processed_data if "synthetic_" in x[2]])
    augmented_count = len([x for x in processed_data if "aug_" in x[2]])
    
    print(f"\n📊 FOCUSED TRAINING DATA SUMMARY:")
    print(f"   Primary images (unique plates): {primary_count}")
    print(f"   High-quality variants: {variant_count}")
    print(f"   Synthetic images: {synthetic_count}")
    print(f"   Augmented images: {augmented_count}")
    print(f"   Total focused dataset: {len(processed_data)}")
    
    # State distribution summary
    print("\n📊 STATE DISTRIBUTION SUMMARY:")
    for state_code in sorted(STATE_CODES):
        count = state_counts[state_code]
        print(f"   {state_code}: {count} images")
    
    # Continue with the rest of the processing...
    
    # --- Save all processed images to a temporary flat directory and create a master list ---
    master_image_list = []  # list of (new_filename, label)
    print(f"Saving all {len(processed_data)} processed images to {ALL_DATA_TEMP_DIR}...")
    
    for i, (img_data, label, original_file_info) in tqdm(enumerate(processed_data), desc="Saving processed images"):
        new_filename = f"img_{i:06d}_{label.replace(' ','_')}.png"  # Ensure .png for PIL save
        save_path = os.path.join(ALL_DATA_TEMP_DIR, new_filename)
        try:
            if isinstance(img_data, Image.Image):
                img_data.save(save_path, "PNG")
                master_image_list.append({"filename": new_filename, "words": label})
            else:
                print(f"Warning: img_data for {original_file_info} is not a PIL Image object. Skipping.")
        except Exception as e:
            print(f"Error saving image {save_path}: {e}")

    if not master_image_list:
        print("No images were processed and saved. Exiting.")
        return

    master_df = pd.DataFrame(master_image_list)
    print(f"Successfully saved {len(master_df)} images to {ALL_DATA_TEMP_DIR}.")

    # Split data into Train, Validation, Test sets
    print("Splitting data into train, validation, and test sets...")
    labels = master_df["words"]
    filenames = master_df["filename"]

    # Check if stratification is possible for the first split
    labels_value_counts = labels.value_counts()
    can_stratify_first_split = True
    if labels_value_counts.empty or (labels_value_counts < 2).any():
        can_stratify_first_split = False
        print("Warning: Cannot stratify the first train/temp split due to classes with < 2 samples. Using random split.")

    # Split: 80% train, 10% validation, 10% test
    # First split into train (80%) and temp (20% for val+test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        filenames, labels, test_size=0.2, random_state=42,
        stratify=labels if can_stratify_first_split else None
    )

    # Check if stratification is possible for the second split (val/test from temp)
    if not temp_labels.empty:
        temp_labels_value_counts = temp_labels.value_counts()
        can_stratify_second_split = True
        if temp_labels_value_counts.empty or (temp_labels_value_counts < 2).any():
            can_stratify_second_split = False
            print("Warning: Cannot stratify the temp to val/test split due to classes with < 2 samples. Using random split.")
        
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.5, random_state=42,
            stratify=temp_labels if can_stratify_second_split else None
        )
    else:  # Handle case where temp_files might be empty after first split (e.g. very small dataset)
        print("Warning: temp_labels is empty after the first split. Validation and test sets will be empty.")
        val_files, test_files, val_labels, test_labels = pd.Series(dtype='object'), pd.Series(dtype='object'), pd.Series(dtype='object'), pd.Series(dtype='object')
    
    print(f"Train set size: {len(train_files)}")
    print(f"Validation set size: {len(val_files)}")
    print(f"Test set size: {len(test_files)}")

    # Function to copy files and create labels.csv
    def organize_split_set(target_dir, file_list, label_list):
        os.makedirs(target_dir, exist_ok=True)
        data_for_df = []
        for img_file, img_label in tqdm(zip(file_list, label_list), desc=f"Organizing {Path(target_dir).name}", total=len(file_list)):
            source_path = os.path.join(ALL_DATA_TEMP_DIR, img_file)
            dest_path = os.path.join(target_dir, img_file)
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
                data_for_df.append({"filename": img_file, "words": img_label})
            else:
                print(f"Warning: Source image {source_path} not found during split organization.")
        
        if data_for_df:
            df = pd.DataFrame(data_for_df)
            df.to_csv(os.path.join(target_dir, "labels.csv"), index=False, header=True)  # EasyOCR format often has header
        else:
             print(f"Warning: No data to write for labels.csv in {target_dir}")

    # Organize train, validation, and test sets
    organize_split_set(TRAIN_DIR, train_files, train_labels)
    organize_split_set(VAL_DIR, val_files, val_labels)
    organize_split_set(TEST_DIR, test_files, test_labels)

    print("Enhanced data processing complete. Output directories:")
    print(f"  Training data: {TRAIN_DIR}")
    print(f"  Validation data: {VAL_DIR}")
    print(f"  Test data: {TEST_DIR}")
    print(f"  Enhanced features: Multi-line support, attention mechanisms, realistic augmentation")
    
    # Optional: Clean up the temporary flat directory
    # shutil.rmtree(ALL_DATA_TEMP_DIR) 
    # print(f"Cleaned up temporary directory: {ALL_DATA_TEMP_DIR}")

if __name__ == '__main__':
    main()
    print("Data processing script finished.") 