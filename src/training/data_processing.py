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
from imgaug import augmenters as iaa
import imgaug as ia
from sklearn.model_selection import train_test_split

# Paths
SOURCE_DIR = r"D:\Work\Projects\ANPR\data\raw_ocr"
OUTPUT_DIR = r"D:\Work\Projects\ANPR\data\combined_training_data_ocr"

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
    'G': ['6'], '6': ['G'],
    'Q': ['O'], 'O': ['Q'],
    'R': ['P'], 'P': ['R'],
    'Y': ['V'], 'V': ['Y'],
    'Z': ['2'], '2': ['Z'],
    'W': ['M'], 'M': ['W'],
    'X': ['K'], 'K': ['X'],
    'J': ['I'], 'I': ['J'],
    'N': ['M'], 'M': ['N'],
    'T': ['D'], 'D': ['T'],
    'F': ['P'], 'P': ['F']
}

# Target image size for EasyOCR (adjust as needed)
TARGET_WIDTH = 600
TARGET_HEIGHT = 150
TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT

MIN_IMAGES_PER_STATE = 100 # For synthetic data generation
TWO_LINE_PLATE_CHANCE = 0.2 # Chance to convert a plate to two lines

# --- Image and Label Processing Functions ---

def get_base_label(filename):
    """Extracts the base license plate number from a filename."""
    name_part = Path(filename).stem
    # Remove suffixes like _1, _2 or (1), (2)
    name_part = re.sub(r'_\d+$', '', name_part)
    name_part = re.sub(r'\s*\(\d+\)$', '', name_part)
    return name_part.upper()

def detect_plate_type(pil_img):
    """Heuristically detect plate type based on average color."""
    img_np = np.array(pil_img.resize((100, 40)))  # Resize for speed
    avg_color = img_np.mean(axis=(0, 1))
    r, g, b = avg_color

    if r > 200 and g > 200 and b > 200: return 'private'
    elif r > 200 and g > 200 and b < 100: return 'commercial'
    elif g > 150 and r < 100: return 'electric'
    elif b > 100 and r < 100: return 'embassy'
    elif r > 180 and g < 100: return 'vip'
    return 'unknown'

def normalize_license_number(text):
    """Applies fuzzy logic and corrections to license numbers."""
    text = text.upper().replace(" ", "") # Remove spaces, convert to uppercase
    
    # Specific common corrections
    if text.startswith("IS") and len(text) > 2 and text[2].isdigit():
        text = "TS" + text[2:]

    # Character-by-character correction based on CONFUSED_CHARS
    # This can be complex. Let's try a simpler position-based approach for now.
    # Example: If it looks like a state code, ensure it's letters.
    # If it looks like numbers, ensure they are digits.
    
    corrected_text = ""
    for i, char in enumerate(text):
        # Basic check for state code part (first 2 chars)
        if i < 2:
            if char.isdigit():
                for letter, alts in CONFUSED_CHARS.items():
                    if char in alts and not letter.isdigit():
                        char = letter
                        break
        # Basic check for numeric parts (e.g. positions 2,3 and last 4)
        # This is a simplification. A more robust regex based segmentation would be better.
        elif (2 <= i <= 3) or (len(text) - 4 <= i < len(text)):
            if not char.isdigit():
                for digit, alts in CONFUSED_CHARS.items():
                    if char in alts and digit.isdigit():
                        char = digit
                        break
        corrected_text += char
    text = corrected_text

    # Fuzzy match state codes if the first two characters are not a valid state code
    if len(text) >= 2 and text[:2] not in STATE_CODES:
        first_two = text[:2]
        closest_match = difflib.get_close_matches(first_two, STATE_CODES, n=1, cutoff=0.7)
        if closest_match:
            text = closest_match[0] + text[2:]
            
    return text

def is_good_label(label):
    """Validates if a label matches the general license plate pattern."""
    return bool(LICENSE_PATTERN.match(label))

def finalize_image_for_training(pil_img, target_size=(TARGET_WIDTH, TARGET_HEIGHT)):
    """Converts a PIL image to RGB, resizes with aspect ratio, and pads."""
    if pil_img is None:
        return None
    try:
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Resize with aspect ratio preservation
        img_w, img_h = pil_img.size
        img_aspect = img_w / img_h

        if img_aspect > TARGET_ASPECT_RATIO: # Wider than target, fit to width
            new_w = target_size[0]
            new_h = int(new_w / img_aspect)
        else: # Taller than target (or same aspect), fit to height
            new_h = target_size[1]
            new_w = int(new_h * img_aspect)
            
        # Ensure new dimensions are at least 1
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        # On Pillow versions >= 9.0.0, Image.LANCZOS is Image.Resampling.LANCZOS
        # On Pillow versions >= 10.0.0, Image.Resampling.LANCZOS is default for resize.
        # We use Pillow>=8.0.0, so Image.LANCZOS should be available.
        try:
            resample_filter = Image.LANCZOS
        except AttributeError: # For Pillow 9.0.0+ up to 9.5.0
             resample_filter = Image.Resampling.LANCZOS

        pil_img = pil_img.resize((new_w, new_h), resample_filter)
        
        # Pad to target_size
        # Create a new image with black background
        new_img = Image.new('RGB', target_size, color='black') 
        
        # Calculate position for pasting the resized image (center)
        paste_x = (target_size[0] - new_w) // 2
        paste_y = (target_size[1] - new_h) // 2
        
        new_img.paste(pil_img, (paste_x, paste_y))
        return new_img
    except Exception as e:
        print(f"Error processing image for training: {e}")
        return None

# --- Augmentation and Synthetic Data ---

# Define augmentation sequence
ia.seed(1)
seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
    iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2), per_channel=0.2)),
    iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),
    iaa.Sometimes(0.3, iaa.MotionBlur(k=3, angle=[-15, 15])),
    iaa.Sometimes(0.3, iaa.SaltAndPepper(0.01)),
    iaa.Sometimes(0.2, iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-5, 5),
        shear=(-3, 3)
    )),
    iaa.Sometimes(0.1, iaa.PerspectiveTransform(scale=(0.01, 0.05)))
], random_order=True)

def augment_image(img_pil, num_augmentations=2):
    """Applies augmentations to a PIL image."""
    augmented_images = []
    if img_pil is None:
        return augmented_images
        
    img_np = np.array(img_pil)
    # No need to convert to 3-channel as input is already RGB
    
    for _ in range(num_augmentations):
        try:
            aug_img_np = seq(image=img_np)
            aug_img_pil = Image.fromarray(aug_img_np)
            # Finalize to ensure it's grayscale and correct size for training
            final_aug_img = finalize_image_for_training(aug_img_pil)
            if final_aug_img:
                 augmented_images.append(final_aug_img)
        except Exception as e:
            print(f"Error during augmentation: {e}")
    return augmented_images

def generate_synthetic_plate_image(text, plate_type='private', width=TARGET_WIDTH, height=TARGET_HEIGHT, two_lines=False):
    """Generates a synthetic license plate image with the given text."""
    try:
        colors = {
            'private': ((255, 255, 255), (0, 0, 0)),       # white bg, black text
            'commercial': ((255, 255, 0), (0, 0, 0)),      # yellow bg, black text
            'electric': ((0, 128, 0), (255, 255, 255)),    # green bg, white text
            'embassy': ((0, 0, 255), (255, 255, 255)),     # blue bg, white text
            'vip': ((255, 0, 0), (255, 255, 255)),         # red bg, white text
        }
        bg_color, text_color = colors.get(plate_type, ((220,220,220), (0,0,0)))

        # Create a base image (light gray or textured)
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Font (ensure a common, clear font is available or specify path)
        try:
            font_path = "arial.ttf" # Try to use Arial
            font_size = int(height * 0.6) if not two_lines else int(height * 0.35)
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font_size = int(height * 0.6) if not two_lines else int(height * 0.35)
            font = ImageFont.load_default() # Fallback to default
            print(f"Arial font not found, using default. Text may not render optimally.")

        if two_lines and len(text) > 4:
            # Simplistic split: after state code and 1-2 digits
            split_point = 4 
            if len(text) > 5 and text[4].isalpha(): # e.g., TS07AB1234 -> TS07 AB1234
                 split_point = 4 # Default, e.g. XX00 YYY0000
                 # Find a good split point, typically after first 4 chars
                 # Or if there's a letter after 4th char: XX00A0000 -> XX00 A0000
                 if len(text) > 4 and text[4].isalpha():
                    if len(text) > 5 and text[5].isalpha(): # XX00AAYYYY
                        split_point = 5
                    else: # XX00AYYYY
                        split_point = 4
                 elif len(text) > 3 and text[3].isalpha(): # XX0AAYYYY
                    split_point = 4

            line1 = text[:split_point]
            line2 = text[split_point:]

            # Pillow version check for textbbox and textlength
            if hasattr(draw, 'textbbox'): # Pillow 8.0.0+
                bbox1 = draw.textbbox((0,0), line1, font=font)
                bbox2 = draw.textbbox((0,0), line2, font=font)
                text_width1 = bbox1[2] - bbox1[0]
                text_height1 = bbox1[3] - bbox1[1]
                text_width2 = bbox2[2] - bbox2[0]
                # text_height2 = bbox2[3] - bbox2[1] # not used for y_pos calc here
            else: # Older Pillow
                text_width1, text_height1 = draw.textsize(line1, font=font)
                text_width2, _ = draw.textsize(line2, font=font)


            y_pos1 = (height // 2 - text_height1) // 2 + int(height*0.05)
            y_pos2 = height // 2 + (height // 2 - text_height1) // 2 - int(height*0.05)

            x_pos1 = (width - text_width1) // 2
            x_pos2 = (width - text_width2) // 2
            
            draw.text((x_pos1, y_pos1), line1, font=font, fill=text_color)
            draw.text((x_pos2, y_pos2), line2, font=font, fill=text_color)

        else: # Single line
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0,0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width, text_height = draw.textsize(text, font=font)
            
            x = (width - text_width) / 2
            y = (height - text_height) / 2
            draw.text((x, y), text, font=font, fill=text_color)

        # Add some noise/distortions to make it more realistic
        img_np = np.array(img) # Already RGB
        img_np = seq(image=img_np) # Apply some augmentations
        
        final_pil_img = Image.fromarray(img_np)
        return finalize_image_for_training(final_pil_img) # Ensure final format
    except Exception as e:
        print(f"Error generating synthetic image for '{text}': {e}")
        return None

def generate_random_plate_text(state_code):
    """Generates a random Indian license plate number for a given state."""
    if random.random() < 0.15:  # 15% BH series
        return f"{str(random.randint(10,99))}BH{str(random.randint(1000,9999))}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}"

    part1 = str(random.randint(0, 99)).zfill(2) # Digits after state
    
    # Middle letters (0 to 3)
    num_mid_letters = random.randint(0, 3)
    part2 = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=num_mid_letters))
    
    part3 = str(random.randint(0, 9999)).zfill(4) # Last 4 digits
    return f"{state_code}{part1}{part2}{part3}"


# --- Main Processing Logic ---
def main():
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(all_files)} initial image files in {SOURCE_DIR}")

    # 1. Group images by base label and normalize labels
    image_groups = defaultdict(list)
    for f_name in tqdm(all_files, desc="Grouping images by base label"):
        base_label = get_base_label(f_name)
        normalized_label = normalize_license_number(base_label)
        if is_good_label(normalized_label): # Only process if normalized label is valid
             image_groups[normalized_label].append(f_name)
        else:
            # Try to correct filename based label before discarding
            corrected_base = normalize_license_number(base_label)
            if is_good_label(corrected_base):
                image_groups[corrected_base].append(f_name)
            # else:
            # print(f"Skipping {f_name}, invalid base label after normalization: {normalized_label} / {base_label}")


    processed_data = [] # List of tuples: (image_path_or_pil, label, original_filename, plate_type)
    
    # 2. Filter duplicates, select valuable images
    print(f"Found {len(image_groups)} unique normalized license plates.")
    for label, files in tqdm(image_groups.items(), desc="Processing license plates"):
        if not files:
            continue

        # Keep the original image (shortest name, assuming it's the primary one)
        # And a few variants (_1, _2, etc.) if they exist and are different enough
        # For simplicity, we'll keep the first one and then check others
        # A more sophisticated approach would use image hashing (e.g. pHash) to check similarity
        
        sorted_files = sorted(files, key=len) # Shorter names first
        
        # Process the first (presumably main) image
        main_img_path = os.path.join(SOURCE_DIR, sorted_files[0])
        try:
            img = Image.open(main_img_path)
            img_rgb = img.convert('RGB')
            plate_type = detect_plate_type(img_rgb)
            final_img = finalize_image_for_training(img_rgb.copy())
            if final_img:
                processed_data.append((final_img, label, sorted_files[0], plate_type))
        except Exception as e:
            print(f"Error opening or processing main image {main_img_path}: {e}")
            continue # Skip this group if main image fails

        # Keep a few variants if they offer value (e.g. different angles)
        # Simple heuristic: keep up to 2 additional variants if their names differ significantly (e.g. _1, _2)
        variants_kept = 0
        for other_file_name in sorted_files[1:]:
            if variants_kept >= 2: # Limit number of variants
                break
            if '_' in Path(other_file_name).stem or '(' in Path(other_file_name).stem : # Heuristic for angled/duplicate shots
                try:
                    var_img_path = os.path.join(SOURCE_DIR, other_file_name)
                    var_img = Image.open(var_img_path)
                    var_img_rgb = var_img.convert('RGB')
                    # Assume variant has same plate type
                    final_var_img = finalize_image_for_training(var_img_rgb.copy())
                    if final_var_img:
                        processed_data.append((final_var_img, label, other_file_name, plate_type))
                        variants_kept += 1
                except Exception as e:
                    print(f"Error opening or processing variant image {var_img_path}: {e}")
    
    print(f"Selected {len(processed_data)} valuable images after initial filtering.")

    # 3. Data Augmentation (for real images)
    augmented_data_to_add = []
    for pil_img, label, _, plate_type in tqdm(processed_data, desc="Generating augmentations for real images"):
        # Augment original images (not synthetic ones yet)
        # Create 1 or 2 augmentations per valuable image
        num_augs = random.randint(1,2) 
        augs = augment_image(pil_img, num_augmentations=num_augs)
        for aug_img in augs:
            augmented_data_to_add.append((aug_img, label, "augmented", plate_type))
    
    processed_data.extend(augmented_data_to_add)
    print(f"Total images after augmentation: {len(processed_data)}")

    # 4. Synthetic Data Generation
    # Generate for each plate type to balance the dataset
    plate_types_to_generate = ['private', 'commercial', 'electric', 'embassy', 'vip']
    type_counts = Counter(item[3] for item in processed_data if item[3] != 'unknown')
    
    synthetic_data_to_add = []
    for plate_type in tqdm(plate_types_to_generate, desc="Generating synthetic data by type"):
        num_existing = type_counts.get(plate_type, 0)
        num_to_generate = max(0, MIN_IMAGES_PER_STATE - num_existing)
        
        for i in range(num_to_generate):
            state_code = random.choice(STATE_CODES)
            synthetic_text = generate_random_plate_text(state_code)
            if not is_good_label(synthetic_text): # Validate synthetic text
                synthetic_text = normalize_license_number(synthetic_text) # Try to fix
                if not is_good_label(synthetic_text):
                    continue

            two_lines = random.random() < TWO_LINE_PLATE_CHANCE
            synthetic_img = generate_synthetic_plate_image(synthetic_text, plate_type=plate_type, two_lines=two_lines)
            if synthetic_img:
                synthetic_data_to_add.append((synthetic_img, synthetic_text, f"synthetic_{plate_type}_{i}", plate_type))

    processed_data.extend(synthetic_data_to_add)
    print(f"Total images after synthetic data generation: {len(processed_data)}")

    # 5. Generate Two-Line Variants from existing single-line plates
    # To avoid excessive data, do this for a fraction of existing real plates
    two_line_variants_to_add = []
    num_two_line_to_generate = int(len(processed_data) * 0.1) # Generate for 10% of current dataset
    
    # Create a temporary list of (image, label, type) for candidates for two-line conversion
    # Prioritize real, non-augmented images if possible
    candidate_indices = [
        i for i, (_, _, f_type, _) in enumerate(processed_data) 
        if f_type != "augmented" and not f_type.startswith("synthetic")
    ]
    if not candidate_indices: # If only augmented/synthetic, use all
        candidate_indices = list(range(len(processed_data)))

    random.shuffle(candidate_indices)
    
    for idx in tqdm(candidate_indices[:num_two_line_to_generate], desc="Generating two-line variants"):
        pil_img, label, original_fname_type, plate_type = processed_data[idx]
        
        # Use the label to generate a two-line synthetic image
        # Avoid re-augmenting already augmented/synthetic data too much with this specific transform
        # if original_fname_type != "augmented" and not original_fname_type.startswith("synthetic"):
        if len(label) > 4: # Ensure label is long enough to split
            two_line_img = generate_synthetic_plate_image(label, plate_type=plate_type, two_lines=True)
            if two_line_img:
                two_line_variants_to_add.append((two_line_img, label, f"twoline_{Path(str(original_fname_type)).stem}", plate_type))
    
    processed_data.extend(two_line_variants_to_add)
    print(f"Total images after two-line variant generation: {len(processed_data)}")
    
    # --- Save all processed images to a temporary flat directory and create a master list ---
    master_image_list = [] # list of dicts with plate_type
    print(f"Saving all {len(processed_data)} processed images to {ALL_DATA_TEMP_DIR}...")
    
    for i, (img_data, label, original_file_info, plate_type) in tqdm(enumerate(processed_data), desc="Saving processed images"):
        new_filename = f"img_{i:06d}_{label.replace(' ','_')}.png" # Ensure .png for PIL save
        save_path = os.path.join(ALL_DATA_TEMP_DIR, new_filename)
        try:
            if isinstance(img_data, Image.Image):
                img_data.save(save_path, "PNG")
                master_image_list.append({"filename": new_filename, "words": label, "plate_type": plate_type})
            else:
                print(f"Warning: img_data for {original_file_info} is not a PIL Image object. Skipping.")
        except Exception as e:
            print(f"Error saving image {save_path}: {e}")

    if not master_image_list:
        print("No images were processed and saved. Exiting.")
        return

    master_df = pd.DataFrame(master_image_list)
    print(f"Successfully saved {len(master_df)} images to {ALL_DATA_TEMP_DIR}.")

    # 6. Split data into Train, Validation, Test sets
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
    else: # Handle case where temp_files might be empty after first split (e.g. very small dataset)
        print("Warning: temp_labels is empty after the first split. Validation and test sets will be empty.")
        val_files, test_files, val_labels, test_labels = pd.Series(dtype='object'), pd.Series(dtype='object'), pd.Series(dtype='object'), pd.Series(dtype='object')
    
    print(f"Train set size: {len(train_files)}")
    print(f"Validation set size: {len(val_files)}")
    print(f"Test set size: {len(test_files)}")

    # Function to copy files and create labels.csv
    def organize_split_set(target_dir, file_list, label_list):
        os.makedirs(target_dir, exist_ok=True)
        data_for_df = []
        file_to_metadata = master_df.set_index('filename').to_dict('index')

        for img_file in tqdm(file_list, desc=f"Organizing {Path(target_dir).name}", total=len(file_list)):
            source_path = os.path.join(ALL_DATA_TEMP_DIR, img_file)
            dest_path = os.path.join(target_dir, img_file)
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
                metadata = file_to_metadata.get(img_file, {})
                data_for_df.append({
                    "filename": img_file, 
                    "words": metadata.get('words'), 
                    "plate_type": metadata.get('plate_type')
                })
            else:
                print(f"Warning: Source image {source_path} not found during split organization.")
        
        if data_for_df:
            df = pd.DataFrame(data_for_df)
            df.to_csv(os.path.join(target_dir, "labels.csv"), index=False, header=True)
        else:
             print(f"Warning: No data to write for labels.csv in {target_dir}")


    # Organize train, validation, and test sets
    organize_split_set(TRAIN_DIR, train_files, train_labels)
    organize_split_set(VAL_DIR, val_files, val_labels)
    organize_split_set(TEST_DIR, test_files, test_labels)

    print("Data processing complete. Output directories:")
    print(f"  Training data: {TRAIN_DIR}")
    print(f"  Validation data: {VAL_DIR}")
    print(f"  Test data: {TEST_DIR}")
    
    # Optional: Clean up the temporary flat directory
    # shutil.rmtree(ALL_DATA_TEMP_DIR) 
    # print(f"Cleaned up temporary directory: {ALL_DATA_TEMP_DIR}")


if __name__ == '__main__':
    main()
    print("Data processing script finished.") 