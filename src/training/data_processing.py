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

# --- Centralized Path Configuration ---
class Paths:
    PROJECT_ROOT = Path(r"D:\Work\Projects\ANPR")
    RAW_OCR_DATA = PROJECT_ROOT / "data" / "raw_ocr"
    PROCESSED_OCR_DATA = PROJECT_ROOT / "data" / "combined_training_data_ocr"
    REJECTED_DIR = PROJECT_ROOT / "data" / "rejected_data"

# Paths
SOURCE_DIR = Paths.RAW_OCR_DATA
OUTPUT_DIR = Paths.PROCESSED_OCR_DATA
REJECTED_DIR = Paths.REJECTED_DIR

# Output subdirectories for EasyOCR format
ALL_DATA_TEMP_DIR = OUTPUT_DIR / "all_processed_images"
TRAIN_DIR = OUTPUT_DIR / "train_data"
VAL_DIR = OUTPUT_DIR / "val_data"
TEST_DIR = OUTPUT_DIR / "test_data"

# Create directories
os.makedirs(ALL_DATA_TEMP_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

# Global log for rejected files
REJECTED_FILES_LOG = []

# Patterns for license plates (adjust as needed for Indian context)
# General Indian license plate format: AA00AA0000, AA00A0000, AA000000
LICENSE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$')
# New Bharat Series (BH) plate format: 22BH1234A
BH_LICENSE_PATTERN = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$')
# Stricter validation for "final" labels might be useful
STRICT_LICENSE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{1,2}(?:[A-Z]{1,3})?[0-9]{4}$')

# State codes for fuzzy matching (example, expand as neede  d)
STATE_CODES = [
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "AN", "CH", "DD", "DL", "JK", "LA", "LD", "PY"
]
ALL_STATE_CODES_FOR_SYNTHETIC_DATA = STATE_CODES + ["BH"]

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

# Aspect ratio sanity check thresholds for final validation
MIN_SANE_ASPECT_RATIO = 0.9  # Loosened to allow for square-like multi-line plates
MAX_SANE_ASPECT_RATIO = 8.0 # Stricter to remove overly long, noisy images
MIN_PLATE_CHARS = 5 # Stricter: A plausible plate must have at least 5 character-like objects
REQUIRED_BORDER_MARGIN_PX = 4 # Require at least 4 pixels of padding around the text block
MIN_TEXT_AREA_RATIO = 0.25  # The text block should occupy at least 25% of the image area

# Update constants for better data distribution
MIN_IMAGES_PER_STATE = 300  # Increased for better state coverage
TARGET_IMAGES_PER_STATE = 500  # Target number of images per state
MAX_IMAGES_PER_STATE = 1000  # Cap to prevent over-representation
TWO_LINE_PLATE_CHANCE = 0.5  # Increased for better multi-line coverage
QUALITY_THRESHOLD = 0.4  # Increased quality threshold

# --- Added helper for consistent deskewing with test-time pipeline ---
def deskew_image(image):
    """
    Ultra-conservative deskew function that prioritizes text preservation over perfect alignment.
    Enhanced to prevent ANY text cutoff during rotation.
    """
    # Convert PIL Image to a numpy array we can work with
    image_np = np.array(image)
    
    # Convert to grayscale if needed
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()
    
    # Apply threshold to get a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours that could be characters
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = gray.shape
    char_contours = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Use lenient filters to find potential character shapes
        contour_aspect_ratio = ch / cw if cw > 0 else 0
        if (0.8 < contour_aspect_ratio < 6.0) and (0.3 < ch / h < 1.0):
            char_contours.append(cnt)
            
    # SAFETY CHECK 1: Need enough characters for reliable angle detection
    if len(char_contours) < MIN_PLATE_CHARS - 1:
        return image_np  # Return original, un-rotated image
    
    # Combine all character contour points
    all_char_points = np.vstack([c for c in char_contours])
    
    # Find the minimum area rectangle that encloses the text block
    angle = cv2.minAreaRect(all_char_points)[-1]
    
    # Adjust the angle for rotation
    if angle < -45:
        angle = 90 + angle
    
    # SAFETY CHECK 2: ULTRA-CONSERVATIVE angle limits to prevent ANY text cutoff
    max_safe_angle = 8.0  # Reduced from 15° to 8° for maximum safety
    if abs(angle) > max_safe_angle:
        return image_np  # Return original if rotation would be risky
    
    # SAFETY CHECK 3: Skip tiny angles that don't matter
    min_angle_threshold = 1.5  # Increased from 1.0° to 1.5° to be more selective
    if abs(angle) < min_angle_threshold:
        return image_np  # Skip rotation for very small angles
    
    # SAFETY CHECK 4: Calculate the current text bounding box
    text_bbox = cv2.boundingRect(all_char_points)
    text_x, text_y, text_w, text_h = text_bbox
    
    # SAFETY CHECK 5: Ensure text region is reasonable size relative to image
    text_area_ratio = (text_w * text_h) / (w * h)
    if text_area_ratio < 0.15 or text_area_ratio > 0.85:  # Tightened bounds
        return image_np  # Skip rotation if text region seems unreasonable
    
    # SAFETY CHECK 6: Verify text is not already too close to edges
    edge_safety_margin = 15  # Increased margin for safety
    if (text_x < edge_safety_margin or 
        text_y < edge_safety_margin or 
        (text_x + text_w) > (w - edge_safety_margin) or 
        (text_y + text_h) > (h - edge_safety_margin)):
        return image_np  # Text already too close to edges - don't risk rotation
    
    # Calculate rotation transformation
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # SAFETY CHECK 7: Predict where text corners will be after rotation
    text_corners = np.array([
        [text_x, text_y],
        [text_x + text_w, text_y],
        [text_x + text_w, text_y + text_h],
        [text_x, text_y + text_h]
    ], dtype=np.float32)
    
    # Apply rotation matrix to text corners
    ones = np.ones(shape=(len(text_corners), 1))
    text_corners_hom = np.hstack([text_corners, ones])
    rotated_corners = M.dot(text_corners_hom.T).T
    
    # SAFETY CHECK 8: Ensure ALL rotated corners stay well within image boundaries
    safe_margin = 10  # Increased safety margin
    for corner in rotated_corners:
        x, y = corner
        if x < safe_margin or x > (w - safe_margin) or y < safe_margin or y > (h - safe_margin):
            return image_np  # Return original if ANY corner would be too close to edge
    
    # SAFETY CHECK 9: Additional validation - check if rotation would expand text beyond safe bounds
    rotated_text_x_coords = [corner[0] for corner in rotated_corners]
    rotated_text_y_coords = [corner[1] for corner in rotated_corners]
    
    rotated_text_width = max(rotated_text_x_coords) - min(rotated_text_x_coords)
    rotated_text_height = max(rotated_text_y_coords) - min(rotated_text_y_coords)
    
    # Don't rotate if it would make text too wide or tall for the image
    if (rotated_text_width > (w * 0.9) or rotated_text_height > (h * 0.8)):
        return image_np  # Text would become too large after rotation
    
    # All safety checks passed - perform the rotation
    rotated = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def validate_character_completeness(image_np, expected_label):
    """
    Check if the number of detected characters reasonably matches the expected label.
    This helps detect cases where the label is complete but the image shows partial text.
    """
    # Remove spaces and special characters from label for counting
    clean_label = ''.join(c for c in expected_label if c.isalnum())
    expected_char_count = len(clean_label)
    
    # Apply threshold to get binary image for character detection
    thresh = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = image_np.shape
    detected_chars = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        contour_aspect_ratio = ch / cw if cw > 0 else 0
        size_ratio_to_plate_height = ch / h if h > 0 else 0
        
        # More lenient character detection for counting
        if (0.5 < contour_aspect_ratio < 8.0) and (0.2 < size_ratio_to_plate_height < 1.0):
            # Filter out very small noise
            if cw > 3 and ch > 8:
                detected_chars.append((x, y, cw, ch))
    
    detected_count = len(detected_chars)
    
    # Allow some tolerance for character detection variations
    # But reject if we're missing more than 20% of expected characters
    min_acceptable_chars = max(MIN_PLATE_CHARS, int(expected_char_count * 0.8))
    max_acceptable_chars = expected_char_count + 3  # Allow for some noise detection
    
    if detected_count < min_acceptable_chars:
        return False, f"Too few characters detected: {detected_count} vs expected ~{expected_char_count}"
    
    if detected_count > max_acceptable_chars:
        return False, f"Too many characters detected: {detected_count} vs expected ~{expected_char_count} (noise)"
    
    return True, f"Character count acceptable: {detected_count} characters detected"

def detect_post_processing_cutoff(image_np):
    """
    Additional validation to detect text cutoff that might have occurred during processing.
    Returns True if cutoff is detected, False if image looks good.
    """
    # Apply threshold to get binary image
    thresh = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = image_np.shape
    char_contours = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        contour_aspect_ratio = ch / cw if cw > 0 else 0
        size_ratio_to_plate_height = ch / h if h > 0 else 0
        
        # Detect potential character contours
        if (0.5 < contour_aspect_ratio < 8.0) and (0.2 < size_ratio_to_plate_height < 1.0):
            if cw > 3 and ch > 5:  # Very small noise filter
                char_contours.append((x, y, cw, ch))
    
    if len(char_contours) < 3:
        return True  # Too few characters - likely cutoff
    
    # Check for characters that are cut off at the edges
    cutoff_margin = 6  # Pixels from edge to check for cutoff
    cutoff_detected = False
    
    for x, y, cw, ch in char_contours:
        # Check if character extends to the very edge (indicating cutoff)
        if (x <= cutoff_margin or  # Left edge cutoff
            y <= cutoff_margin or  # Top edge cutoff
            (x + cw) >= (w - cutoff_margin) or  # Right edge cutoff
            (y + ch) >= (h - cutoff_margin)):  # Bottom edge cutoff
            cutoff_detected = True
            break
    
    # Additional check: if text block occupies almost the entire image width/height
    if char_contours:
        min_x = min(x for x, y, cw, ch in char_contours)
        max_x = max(x + cw for x, y, cw, ch in char_contours)
        min_y = min(y for x, y, cw, ch in char_contours)
        max_y = max(y + ch for x, y, cw, ch in char_contours)
        
        text_width_ratio = (max_x - min_x) / w
        text_height_ratio = (max_y - min_y) / h
        
        # If text takes up more than 95% of image width/height, it's likely cut off
        if text_width_ratio > 0.95 or text_height_ratio > 0.90:
            cutoff_detected = True
    
    return cutoff_detected

def detect_text_orientation(image_np):
    """
    Detect if text in the image is upside down by analyzing character patterns.
    Returns: 'normal', 'upside_down', or 'unclear'
    """
    # Apply threshold to get binary image
    thresh = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours that could be characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = image_np.shape
    char_contours = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        contour_aspect_ratio = ch / cw if cw > 0 else 0
        size_ratio_to_plate_height = ch / h if h > 0 else 0
        
        # Filter for character-like contours
        if (0.5 < contour_aspect_ratio < 8.0) and (0.2 < size_ratio_to_plate_height < 1.0):
            if cw > 3 and ch > 8:  # Filter out very small noise
                char_contours.append((x, y, cw, ch))
    
    if len(char_contours) < 3:  # Need at least 3 characters for orientation analysis
        return 'unclear'
    
    # Analyze character shapes to detect orientation
    # For each character, check the distribution of black pixels
    orientation_scores = {'normal': 0, 'upside_down': 0}
    
    for x, y, cw, ch in char_contours:
        # Extract character region
        char_region = thresh[y:y+ch, x:x+cw]
        
        if char_region.size == 0:
            continue
            
        # Divide character into top and bottom halves
        mid_y = ch // 2
        top_half = char_region[:mid_y, :]
        bottom_half = char_region[mid_y:, :]
        
        # Count black pixels (text) in each half
        top_pixels = cv2.countNonZero(top_half) if top_half.size > 0 else 0
        bottom_pixels = cv2.countNonZero(bottom_half) if bottom_half.size > 0 else 0
        
        total_pixels = top_pixels + bottom_pixels
        if total_pixels == 0:
            continue
            
        # For normal text, there's usually more weight at the bottom
        # (due to baseline, descenders, etc.)
        bottom_ratio = bottom_pixels / total_pixels if total_pixels > 0 else 0.5
        
        # Normal text typically has bottom_ratio between 0.4-0.7
        # Upside-down text would have this reversed
        if bottom_ratio > 0.6:
            orientation_scores['normal'] += 1
        elif bottom_ratio < 0.4:
            orientation_scores['upside_down'] += 1
    
    # Also check overall text positioning in the image
    # Normal license plates typically have text in the center or lower half
    y_coords = [y + ch/2 for x, y, cw, ch in char_contours]
    avg_y = sum(y_coords) / len(y_coords) if y_coords else h/2
    y_position_ratio = avg_y / h
    
    # If text is very high in the image, it might be upside down
    if y_position_ratio < 0.25:  # Text in top quarter
        orientation_scores['upside_down'] += 2
    elif y_position_ratio > 0.4:  # Text in lower portion (normal)
        orientation_scores['normal'] += 1
    
    # Determine orientation based on scores
    if orientation_scores['normal'] > orientation_scores['upside_down'] + 1:
        return 'normal'
    elif orientation_scores['upside_down'] > orientation_scores['normal'] + 1:
        return 'upside_down'
    else:
        return 'unclear'

def correct_upside_down_text(image_np):
    """
    Check if text is upside down and rotate 180 degrees if needed.
    Returns corrected image or None if orientation is unclear.
    """
    orientation = detect_text_orientation(image_np)
    
    if orientation == 'upside_down':
        # Rotate 180 degrees to correct upside-down text
        corrected = cv2.rotate(image_np, cv2.ROTATE_180)
        
        # Verify the correction worked
        corrected_orientation = detect_text_orientation(corrected)
        if corrected_orientation == 'normal':
            return corrected
        else:
            # If correction didn't help, reject the image
            return None
    elif orientation == 'normal':
        return image_np
    else:  # unclear orientation
        return None

def is_plausible_plate(image_pil, expected_label=None):
    """
    A single, robust function to validate an image using content analysis.
    ENHANCED: Now includes deskewing WITHIN validation to catch cutoff issues.
    """
    # 1. Convert to grayscale numpy array for processing
    try:
        img_np = np.array(image_pil.convert('L'))
        if img_np.ndim != 2 or img_np.shape[0] == 0 or img_np.shape[1] == 0:
            return None, "Invalid image data"
    except Exception as e:
        return None, f"Cannot convert image: {e}"

    # 2. Correct orientation if the image is taller than it is wide
    h, w = img_np.shape
    if h > w:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img_np.shape # Update dimensions after rotation

    # 2.5. Check and correct upside-down text
    corrected_img = correct_upside_down_text(img_np)
    if corrected_img is None:
        return None, "Rejected: Text orientation unclear or correction failed"
    img_np = corrected_img

    # NEW STEP 3: Apply ultra-conservative deskewing WITHIN validation
    # This ensures we catch any cutoff issues BEFORE accepting the image
    deskewed_img = deskew_image(img_np)
    
    # NEW STEP 4: Check for post-deskewing cutoff
    if detect_post_processing_cutoff(deskewed_img):
        return None, "Rejected: Text cutoff detected after deskewing"
    
    # Continue with the rest of validation using the deskewed image
    img_np = deskewed_img

    # 5. Quick sanity checks for aspect ratio and minimum size
    if h == 0: return None, "Height is zero"
    aspect_ratio = w / h
    if not (MIN_SANE_ASPECT_RATIO <= aspect_ratio <= MAX_SANE_ASPECT_RATIO):
        return None, f"Implausible aspect ratio: {aspect_ratio:.2f}"
    if w < 50 or h < 20:
        return None, f"Image too small: {w}x{h}"

    # 6. Binarize image and find all character-like contours
    thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_contours = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        contour_aspect_ratio = ch / cw if cw > 0 else 0
        size_ratio_to_plate_height = ch / h if h > 0 else 0
        if (1.0 < contour_aspect_ratio < 5.0) and (0.35 < size_ratio_to_plate_height < 0.95):
            char_contours.append((x, y, cw, ch))

    # 7. Check for minimum number of characters (rejects fragments)
    if len(char_contours) < MIN_PLATE_CHARS:
        return None, f"Rejected: Fragment found ({len(char_contours)} chars)"

    # 8. Check character alignment to reject vertical text layouts.
    x_coords = [c[0] + c[2] / 2 for c in char_contours]
    y_coords = [c[1] + c[3] / 2 for c in char_contours]
    std_x = np.std(x_coords)
    std_y = np.std(y_coords)

    # For a valid plate (even multi-line), the horizontal spread of characters
    # should be greater than the vertical spread.
    if std_x < std_y:
        return None, f"Rejected: Vertical text layout (std_x: {std_x:.1f} < std_y: {std_y:.1f})"

    # 9. Calculate the bounding box of the entire text block
    min_x = min(c[0] for c in char_contours)
    min_y = min(c[1] for c in char_contours)
    max_x = max(c[0] + c[2] for c in char_contours)
    max_y = max(c[1] + c[3] for c in char_contours)
    text_block_w = max_x - min_x
    text_block_h = max_y - min_y

    # 10. Check for bad crops by ensuring a border exists around the text
    if (min_x <= REQUIRED_BORDER_MARGIN_PX or
        min_y <= REQUIRED_BORDER_MARGIN_PX or
        max_x >= (w - REQUIRED_BORDER_MARGIN_PX) or
        max_y >= (h - REQUIRED_BORDER_MARGIN_PX)):
        return None, "Rejected: Bad crop (text touching edge)"

    # 11. Check if the text block occupies a reasonable area of the image
    image_area = w * h
    text_area = text_block_w * text_block_h
    text_area_ratio = text_area / image_area if image_area > 0 else 0
    if text_area_ratio < MIN_TEXT_AREA_RATIO:
        return None, f"Rejected: Text area too small ({text_area_ratio:.2f})"

    # 12. ENHANCED: Check for incomplete characters at edges (indicates text cutoff)
    edge_margin = 8  # Pixels from edge to check for partial characters
    incomplete_chars_detected = False
    
    for x, y, cw, ch in char_contours:
        char_center_x = x + cw // 2
        char_center_y = y + ch // 2
        
        # Check if character is cut off at any edge
        # Left edge cutoff
        if x <= edge_margin and char_center_x < w * 0.15:
            incomplete_chars_detected = True
            break
        # Right edge cutoff  
        if (x + cw) >= (w - edge_margin) and char_center_x > w * 0.85:
            incomplete_chars_detected = True
            break
        # Top edge cutoff
        if y <= edge_margin and char_center_y < h * 0.15:
            incomplete_chars_detected = True
            break
        # Bottom edge cutoff
        if (y + ch) >= (h - edge_margin) and char_center_y > h * 0.85:
            incomplete_chars_detected = True
            break
    
    if incomplete_chars_detected:
        return None, "Rejected: Incomplete characters detected at edges (text cutoff)"

    # 13. ENHANCED: Check character count consistency with expected label
    if expected_label:
        char_count_valid, char_count_msg = validate_character_completeness(img_np, expected_label)
        if not char_count_valid:
            return None, f"Rejected: {char_count_msg}"

    # If all checks pass, return the DESKEWED image (not the original)
    # This way the final image has already been safely processed
    return Image.fromarray(img_np), "Plausible Plate"

def correct_orientation_and_validate(image_pil):
    """
    Checks for vertical orientation, rotates if needed, and validates the final aspect ratio.
    Returns the corrected PIL image or None if its shape is invalid.
    """
    img_np = np.array(image_pil)
    
    # If the image has no dimensions, reject it
    if img_np.ndim < 2 or img_np.shape[0] == 0 or img_np.shape[1] == 0:
        return None
        
    h, w = img_np.shape[:2]

    # If the image is vertically oriented (taller than it is wide), rotate it
    if h > w:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Update dimensions after rotation
        h, w = img_np.shape[:2]

    # After potential rotation, perform a sanity check on the aspect ratio
    if h == 0: return None # Avoid division by zero
    final_aspect_ratio = w / h

    if not (MIN_SANE_ASPECT_RATIO <= final_aspect_ratio <= MAX_SANE_ASPECT_RATIO):
        # The shape is not plausible for a license plate. Reject it.
        return None

    # Return the corrected image as a PIL Image
    return Image.fromarray(img_np)

def is_green_plate(image_np):
    """Detects if a plate is green (for EVs) by checking HSV color space."""
    if len(image_np.shape) < 3:
        return False # Not a color image

    # Convert RGB (from PIL) to HSV for color analysis
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    # Define a robust range for green color in HSV
    # This covers various shades of green under different lighting
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Create a mask to isolate green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Calculate the percentage of green pixels in the image
    green_percentage = (cv2.countNonZero(mask) / image_np.size) * 100
    
    # If more than 25% of the image is green, classify it as a green plate
    return green_percentage > 25

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

#def advanced_image_preprocessing(image):
#    """Apply advanced preprocessing to improve image quality."""
#    # Convert PIL Image to numpy array if needed
#    if isinstance(image, Image.Image):
#        image = np.array(image.convert("RGB")) # Ensure 3 channels for color check
#    
#    # Check for green plate (white text on green bg) and invert if necessary
#    if is_green_plate(image):
#        # Inverting makes the white text black, standardizing it with other plates
#        image = cv2.bitwise_not(image)
#    
#    # Convert to grayscale if needed
#    if len(image.shape) == 3:
#        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    else:
#        gray = image.copy()
#    
#    # Apply deskewing
#    deskewed = deskew_image(gray)
#    
#    # Denoise the grayscale image first
#    denoised = cv2.fastNlMeansDenoising(deskewed, h=10)
#    
#    # Enhance contrast on the denoised image
#    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#    enhanced = clahe.apply(denoised)
#    
#    # Apply adaptive thresholding as the final step
#    final_thresh = cv2.adaptiveThreshold(
#        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#        cv2.THRESH_BINARY, 11, 2
#    )
#    
#    return final_thresh

def finalize_image_for_training(image):
    """
    Finalize image preprocessing for training, adapting to aspect ratio.
    SIMPLIFIED: Deskewing now happens in validation, so this just does denoising and resizing.
    """
    if isinstance(image, Image.Image):
        # Ensure image is grayscale
        if image.mode != 'L':
            image = image.convert('L')
        image_np = np.array(image)
    else:
        image_np = image
        if len(image_np.shape) > 2 and image_np.shape[2] > 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # REMOVED: deskew_image call (now happens in validation)
    # Apply gentle denoising only
    denoised_np = cv2.fastNlMeansDenoising(image_np, h=10, 
                                          templateWindowSize=7, 
                                          searchWindowSize=21)
    
    processed = denoised_np
    
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
    """Enhanced normalization with better fuzzy logic for Indian and BH plates."""
    text = text.upper().replace(" ", "").replace("-", "")
    
    # Remove common OCR artifacts
    text = text.replace("_", "").replace(".", "").replace(",", "")

    # If it's a BH plate, do simpler normalization and return
    # Use a softer check for normalization purposes
    if re.match(r'^[0-9]{2}BH', text):
        # Allow for some errors in the numeric parts for correction
        bh_match = re.match(r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$', text)
        if bh_match:
            return text # It's already in the correct format
        # Potentially add more specific BH correction logic here if needed
        return text

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

    # Check for BH plate format first
    if BH_LICENSE_PATTERN.match(label):
        return True
    
    # Check basic pattern for standard plates
    if not LICENSE_PATTERN.match(label):
        return False
    
    # Additional checks for standard plates
    if len(label) > 13:  # Too long
        return False
    
    # Check state code for standard plates
    if label[:2] not in STATE_CODES:
        return False
    
    return True

# --- Enhanced Augmentation and Synthetic Data ---

def add_motion_blur(image, max_kernel_size=5):
    """Applies motion blur to an image."""
    image_np = np.array(image)
    kernel_size = random.randint(3, max_kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel = np.zeros((kernel_size, kernel_size))
    
    # Randomly choose horizontal or vertical motion
    if random.random() > 0.5:
        # Horizontal
        kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    else:
        # Vertical
        kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(image_np, -1, kernel)
    return Image.fromarray(blurred)

def add_salt_and_pepper_noise(image, amount=0.005):
    """Adds salt and pepper noise to an image."""
    output = np.array(image)
    # Salt
    num_salt = np.ceil(amount * output.size * 0.5)
    # Use output.shape and handle dimensions of size 1
    coords = [np.random.randint(0, max(1, i - 1), int(num_salt)) for i in output.shape]
    output[tuple(coords)] = 255
    # Pepper
    num_pepper = np.ceil(amount * output.size * 0.5)
    coords = [np.random.randint(0, max(1, i - 1), int(num_pepper)) for i in output.shape]
    output[tuple(coords)] = 0
    return Image.fromarray(output)

def add_random_shadow(image):
    """Adds a random shadow effect to an image."""
    img_np = np.array(image.convert('RGB'))
    h, w, _ = img_np.shape
    
    # Create a transparent overlay
    overlay = img_np.copy()
    output = img_np.copy()
    shadow_intensity = random.uniform(0.4, 0.7)
    
    # Define polygon for shadow
    x1, x2 = random.randint(0, w), random.randint(0, w)
    
    # Make the shadow a polygon
    pts = np.array([[(x1, 0), (x2, 0), (x2, h), (x1, h)]])
    
    # Apply the shadow
    cv2.fillPoly(overlay, pts, (0, 0, 0))
    cv2.addWeighted(overlay, shadow_intensity, output, 1 - shadow_intensity, 0, output)
    
    return Image.fromarray(output).convert('L')

def add_random_line(image):
    """Adds a random line (scratch) to an image."""
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(0, w), random.randint(0, h)
    color = random.randint(0, 50) # Dark scratch
    thickness = random.randint(1, 2)
    cv2.line(img_np, (x1, y1), (x2, y2), (color), thickness)
    return Image.fromarray(img_np)

def augment_image_advanced(img_pil):
    """
    Apply a comprehensive set of advanced augmentations based on best practices.
    """
    if not isinstance(img_pil, Image.Image):
        img_pil = Image.fromarray(img_pil)

    # 1. Geometric transformations
    if random.random() < 0.7:
        img_np = np.array(img_pil)
        rows, cols = img_np.shape[:2]
        angle = random.uniform(-7, 7)
        M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_np = cv2.warpAffine(img_np, M_rot, (cols, rows), borderValue=255)
        
        offset = random.randint(2, 7)
        pts1 = np.float32([[0,0], [cols,0], [0,rows], [cols,rows]])
        pts2 = np.float32([[offset,offset], [cols-offset,offset], [offset,rows-offset], [cols-offset,rows-offset]])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = cv2.warpPerspective(img_np, M_persp, (cols, rows), borderValue=255)
        img_pil = Image.fromarray(img_np)

    # 2. Brightness and Contrast
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(random.uniform(0.8, 1.2))
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 3. Blur
    if random.random() < 0.3:
        if random.random() > 0.5:
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        else:
            img_pil = add_motion_blur(img_pil)

    # 4. Noise
    if random.random() < 0.3:
        img_pil = add_salt_and_pepper_noise(img_pil)
            
    # 5. Occlusions / Weather
    if random.random() < 0.2:
        img_pil = add_random_shadow(img_pil)
    if random.random() < 0.1:
        img_pil = add_random_line(img_pil)
    
    # 6. Morphological Operations
    if random.random() < 0.2:
        img_np = np.array(img_pil)
        kernel = np.ones((3,3), np.uint8)
        if random.random() > 0.5:
            img_np = cv2.erode(img_np, kernel, iterations=1)
        else:
            img_np = cv2.dilate(img_np, kernel, iterations=1)
        img_pil = Image.fromarray(img_np)
        
    return img_pil


def augment_image(img_pil, num_augmentations=3):
    """
    Creates multiple augmented versions of an image using the advanced pipeline.
    """
    augmented_images = []
    for _ in range(num_augmentations):
        try:
            # Each augmentation starts from the original image to avoid compounding effects
            aug_pil = augment_image_advanced(img_pil.copy())
            
            # Final processing step
            final_img = finalize_image_for_training(aug_pil)
            if final_img:
                augmented_images.append(final_img)
        except Exception as e:
            print(f"Skipping augmentation due to error: {e}")
            
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
    
    # Convert back to PIL and apply advanced augmentations for realism
    img_pil = Image.fromarray(img_np).convert('L')
    final_img_pil = augment_image_advanced(img_pil)
    
    return finalize_image_for_training(final_img_pil)

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

def generate_random_plate_text(state_code):
    """Generate a random license plate number for a given state or 'BH' series."""
    # Handle BH plate generation separately
    if state_code == "BH":
        year = f"{random.randint(21, 25):02d}" # YY for 2021-2025
        four_digit_num = f"{random.randint(0, 9999):04d}"
        alpha = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.choice([1, 2])))
        return f"{year}BH{four_digit_num}{alpha}"

    # Different Indian license plate formats for states
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
    """
    Simplified and more robust processing loop.
    Uses is_plausible_plate as the single gatekeeper for image quality.
    """
    unique_processed_data = []
    state_counts = defaultdict(int)
    
    print(f"Processing {len(image_groups)} unique license plates with robust validation...")
    
    for label, files in tqdm(image_groups.items(), desc="Validating images"):
        state_code = label[:2]
        if BH_LICENSE_PATTERN.match(label):
             state_code = "BH"
        elif state_code not in STATE_CODES:
            continue
            
        if state_counts[state_code] >= MAX_IMAGES_PER_STATE:
            continue
        
        # Select the best primary image to represent the group
        primary_file = select_best_primary_image(files)
        if not primary_file:
            continue
        
        # --- Main Validation Step ---
        img_path = os.path.join(SOURCE_DIR, primary_file)
        try:
            img = Image.open(img_path)
            
            # Use our single, robust function to validate the image
            validated_img, reason = is_plausible_plate(img, label)
            
            if validated_img is None:
                # If the best image is implausible, reject the entire group
                rejection_reason = f"{reason} on primary image {primary_file}"
                for f in files:
                    source_path = Path(SOURCE_DIR) / f
                    if source_path.exists():
                        dest_path = Path(REJECTED_DIR) / f
                        shutil.copy(source_path, dest_path)
                REJECTED_FILES_LOG.append({'file': primary_file, 'reason': rejection_reason})
                continue # Skip to next license plate group
            
            # If validated, process it for training
            processed_img = finalize_image_for_training(validated_img.copy())
            if processed_img:
                final_w, final_h = processed_img.size
                is_multiline = (final_w / final_h) < SINGLE_LINE_ASPECT_RATIO_THRESHOLD
                image_type = "primary_multiline" if is_multiline else "primary_single"
                
                unique_processed_data.append((
                    processed_img, 
                    label, 
                    f"{image_type}_{primary_file}",
                    1.0, # Quality is now handled by validation
                    is_multiline
                ))
                state_counts[state_code] += 1
                
                # Also check variants from the same group
                if state_counts[state_code] < TARGET_IMAGES_PER_STATE:
                    variant_files = [f for f in files if f != primary_file]
                    for variant_file in variant_files:
                        try:
                            variant_path = os.path.join(SOURCE_DIR, variant_file)
                            variant_img = Image.open(variant_path)
                            validated_variant, _ = is_plausible_plate(variant_img, label)
                            
                            if validated_variant is not None:
                                processed_variant = finalize_image_for_training(validated_variant.copy())
                                if processed_variant:
                                    final_vw, final_vh = processed_variant.size
                                    variant_is_multiline = (final_vw / final_vh) < SINGLE_LINE_ASPECT_RATIO_THRESHOLD
                                    variant_type = "variant_multiline" if variant_is_multiline else "variant_single"
                                    
                                    unique_processed_data.append((
                                        processed_variant, label, f"{variant_type}_{variant_file}", 1.0, variant_is_multiline
                                    ))
                                    state_counts[state_code] += 1
                                if state_counts[state_code] >= TARGET_IMAGES_PER_STATE:
                                    break
                        except Exception:
                            continue # Ignore variants that fail to open/process

        except Exception as e:
            # Handle cases where the primary image can't be opened or processed
            rejection_reason = f"Error opening or processing {primary_file}: {e}"
            for f in files:
                source_path = Path(SOURCE_DIR) / f
                if source_path.exists():
                    dest_path = Path(REJECTED_DIR) / f
                    shutil.copy(source_path, dest_path)
            REJECTED_FILES_LOG.append({'file': primary_file, 'reason': rejection_reason})
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
        if BH_LICENSE_PATTERN.match(label):
            state_code = "BH"

        # Add the original processed image
        enhanced_data.append((processed_img, label, file_info))
        state_counts[state_code] += 1
        
        if is_multiline:
            real_multiline_count += 1
        else:
            real_singleline_count += 1
            
            # Add synthetic multi-line version for single-line plates (not for BH plates)
            if len(label) >= 8 and quality_score > 0.6 and not BH_LICENSE_PATTERN.match(label):
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
            else:
                # Move invalid file to rejected directory
                reason = f"Invalid Label ('{base_label}' -> '{corrected_base}')"
                source_path = Path(SOURCE_DIR) / f_name
                dest_path = Path(REJECTED_DIR) / f_name
                if source_path.exists():
                    shutil.move(source_path, dest_path)
                    REJECTED_FILES_LOG.append({'file': f_name, 'reason': reason})

    # 2. Enhanced processing with multi-line support
    print(f"Found {len(image_groups)} unique normalized license plates.")
    processed_data, state_counts = process_multiline_variants(image_groups)
    print(f"Generated {len(processed_data)} images after multi-line processing.")

    # 3. Enhanced data augmentation - Focused on underrepresented states
    augmented_data_to_add = []
    for pil_img, label, file_info in tqdm(processed_data, desc="Generating focused augmentations"):
        state_code = label[:2]
        if BH_LICENSE_PATTERN.match(label):
            state_code = "BH"
        
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

    # 4. Synthetic data generation - Focus on underrepresented states and BH plates
    synthetic_data_to_add = []
    for state_code in tqdm(ALL_STATE_CODES_FOR_SYNTHETIC_DATA, desc="Generating focused synthetic data"):
        current_count = state_counts.get(state_code, 0) # Use .get for BH plates
        
        if state_code == "BH":
            # Generate a smaller, fixed number of BH plates as they are less common
            num_to_generate = 150
        else:
            num_to_generate = max(0, TARGET_IMAGES_PER_STATE - current_count)
        
        if num_to_generate == 0:
            continue
        
        for i in range(num_to_generate):
            # For BH plates, update the main counter, not state_counts
            if state_code == "BH":
                if state_counts.get("BH", 0) >= 300: # Set a cap for BH plates
                    break
            elif state_counts[state_code] >= MAX_IMAGES_PER_STATE:
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
                if state_code == "BH":
                    state_counts["BH"] = state_counts.get("BH", 0) + 1
                else:
                    state_counts[state_code] += 1
            
            # 50% chance for multi-line version for non-BH plates
            if random.random() < TWO_LINE_PLATE_CHANCE and len(synthetic_text) >= 8 and state_code != "BH":
                multi_img = enhanced_multiline_plate_generation(synthetic_text)
                if multi_img is not None and state_counts.get(state_code, 0) < MAX_IMAGES_PER_STATE:
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
    
    # Rejection summary
    if REJECTED_FILES_LOG:
        print(f"\n🚫 Moved {len(REJECTED_FILES_LOG)} invalid or low-quality images to: {REJECTED_DIR}")
        try:
            rejected_df = pd.DataFrame(REJECTED_FILES_LOG)
            log_path = Path(REJECTED_DIR) / "rejection_log.csv"
            rejected_df.to_csv(log_path, index=False)
            print(f"   A log of rejected files has been saved to: {log_path}")
        except Exception as e:
            print(f"   Could not save rejection log: {e}")

    # State distribution summary
    print("\n📊 STATE DISTRIBUTION SUMMARY:")
    for state_code in sorted(ALL_STATE_CODES_FOR_SYNTHETIC_DATA):
        count = state_counts.get(state_code, 0)
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