#!/usr/bin/env python3
"""
Script to process annotated images, detect license plates, perform OCR,
and save the cropped plate images with recognized text as filenames.

This script uses the models and processing logic from the main application.
"""

import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from pathlib import Path
import time
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List
import re
from collections import deque, defaultdict


# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not available. YOLO detection will be disabled.")
    YOLO_AVAILABLE = False

# Add beam search decoding
try:
    from fast_ctc_decode import beam_search
    BEAM_SEARCH_AVAILABLE = True
except ImportError:
    BEAM_SEARCH_AVAILABLE = False
    # No logger here yet logging.warning("fast_ctc_decode not available. Using greedy decoding.")

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"batch_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration (from src/main.py)
class Config:
    # Model paths (using user provided paths)
    YOLO_MODEL_PATH = r"D:\Work\Projects\ANPR\models\detection\yolo11n_anpr\weights\best.pt"
    CRNN_MODEL_PATH = r"D:\Work\Projects\ANPR\models\ocr\crnn_v1\best_multiline_crnn_epoch292_acc0.9304.pth"
    # CRNN_MODEL_PATH = r"D:\Work\Projects\ANPR\models\ocr\crnn_v2\best_multiline_crnn_epoch51_acc0.9966.pth"
    CRNN_MODEL_PATH_ALT = r"D:\Work\Projects\ANPR_3\saved_models\stable_crnn_v6\checkpoint_epoch_310_acc0.923.pth"
        
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.3  # Reasonable threshold for better detection
    NMS_THRESHOLD = 0.45
    MIN_PLATE_AREA = 400  # Minimum area for license plate (increased to reduce false positives)
    PLATE_CONFIDENCE_THRESHOLD = 0.2  # Separate threshold for plates
    
    # OCR parameters  
    OCR_IMG_HEIGHT = 64
    OCR_IMG_WIDTH = 256
    MIN_OCR_CONFIDENCE = 0.3  # Lowered from 0.5 to 0.3 for debugging

@dataclass
class Detection:
    """Class to store detection information"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float
    frame_id: int
    detection_type: str = "plate"  # "car" or "plate"
    plate_type: str = "unknown"  # "green", "white", "red", "unknown"
    
class ImprovedBidirectionalLSTM(nn.Module):
    """Improved Bidirectional LSTM layer"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(ImprovedBidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_tensor):
        recurrent, _ = self.rnn(input_tensor)
        output = self.dropout_layer(recurrent)
        output = self.linear(output)
        return output

class CustomCRNN(nn.Module):
    """Custom CRNN model for license plate recognition"""
    def __init__(self, img_height, n_classes, n_hidden=256, dropout_rate=0.3, input_channels=3):
        super(CustomCRNN, self).__init__()
        self.input_channels = input_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(dropout_rate * 0.7),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(True), nn.Dropout2d(dropout_rate)
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn_input_size = 512 
        self.rnn1 = ImprovedBidirectionalLSTM(self.rnn_input_size, n_hidden // 2, n_hidden // 2, num_layers=2, dropout=dropout_rate)
        self.rnn2 = ImprovedBidirectionalLSTM(n_hidden // 2, n_hidden // 2, n_classes, num_layers=1, dropout=dropout_rate)
        
    def forward(self, input_tensor):
        conv = self.cnn(input_tensor)
        conv = self.adaptive_pool(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        
        output = self.rnn1(conv)
        output = self.rnn2(output)
        return output

class ANPRProcessor:
    """Main ANPR processing class, adapted from main.py for batch processing."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.yolo_model = None
        self.crnn_model = None
        self.char_list = None
        self.yolo_loaded = False
        self.crnn_loaded = False
        self.vehicle_plate_association_enabled = True
        
    def load_models(self):
        """Load YOLO and CRNN models"""
        try:
            # Load YOLO model
            if YOLO_AVAILABLE and Path(Config.YOLO_MODEL_PATH).exists():
                logger.info(f"Loading YOLO model from: {Config.YOLO_MODEL_PATH}")
                self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
                self.yolo_model.to('cpu')
                logger.info("YOLO model loaded successfully (CPU mode)")
                self.yolo_loaded = True
            else:
                logger.warning("YOLO model not available")
                self.yolo_loaded = False
                
            self._load_crnn_model()
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _load_crnn_model(self):
        """Load CRNN model for OCR"""
        try:
            model_path = Config.CRNN_MODEL_PATH
            if not Path(model_path).exists():
                model_path = Config.CRNN_MODEL_PATH_ALT
                
            if not Path(model_path).exists():
                raise FileNotFoundError(f"CRNN model not found at either path")
                
            logger.info(f"Loading CRNN model from: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'char_set' in checkpoint:
                self.char_list = checkpoint['char_set']
            else:
                logger.warning("Character set not found, using fallback")
                self.char_list = ['[blank]'] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            
            model_config = checkpoint.get('model_config', {})
            img_height = model_config.get('img_height', Config.OCR_IMG_HEIGHT)
            n_classes = model_config.get('n_classes', len(self.char_list))
            n_hidden = model_config.get('n_hidden', 256)
            
            first_layer_weight = list(checkpoint['model_state_dict'].values())[0]
            if len(first_layer_weight.shape) == 4:
                input_channels = first_layer_weight.shape[1]
            else:
                input_channels = 3
            
            self.crnn_model = CustomCRNN(img_height, n_classes, n_hidden, input_channels=input_channels)
            self.crnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.crnn_model.to(self.device)
            self.crnn_model.eval()
            
            logger.info(f"CRNN model loaded successfully! Character set size: {len(self.char_list)}")
            self.crnn_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading CRNN model: {e}")
            self.crnn_loaded = False
            raise
    
    def detect_vehicles_and_plates(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """Detect vehicles and license plates in frame using YOLO"""
        if self.yolo_model is None:
            return [], []
            
        try:
            with torch.no_grad():
                results = self.yolo_model(frame, conf=0.1, iou=Config.NMS_THRESHOLD, verbose=False, device='cpu')
            
            vehicles = []
            plates = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else -1
                        
                        area = (x2 - x1) * (y2 - y1)
                        
                        if cls == 0 or cls == 1:
                            if conf >= 0.15:
                                vehicles.append((x1, y1, x2, y2))
                        elif cls == 2:
                            if conf >= 0.1 and area >= 200:
                                plates.append((x1, y1, x2, y2))
            
            if self.vehicle_plate_association_enabled:
                filtered_plates = self._filter_plates_near_vehicles(plates, vehicles, frame.shape)
            else:
                filtered_plates = plates
            
            return vehicles, filtered_plates
            
        except Exception as e:
            logger.error(f"Error in plate detection: {e}")
            return [], []
    
    def _filter_plates_near_vehicles(self, plates: List[Tuple[int, int, int, int]], 
                                   vehicles: List[Tuple[int, int, int, int]], 
                                   frame_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        if not vehicles or not plates:
            return []
        
        filtered_plates = []
        
        for plate_bbox in plates:
            px1, py1, px2, py2 = plate_bbox
            is_associated = False
            for vehicle_bbox in vehicles:
                vx1, vy1, vx2, vy2 = vehicle_bbox
                if (vx1 <= px1 <= vx2 and vy1 <= py1 <= vy2 and
                    vx1 <= px2 <= vx2 and vy1 <= py2 <= vy2):
                    is_associated = True
                    break
            
            if is_associated:
                filtered_plates.append(plate_bbox)
        
        return filtered_plates

    def preprocess_plate_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess plate image for CRNN model"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            h, w = binary.shape
            scale = Config.OCR_IMG_HEIGHT / h
            new_width = int(w * scale)
            
            if new_width > Config.OCR_IMG_WIDTH:
                scale = Config.OCR_IMG_WIDTH / w
                new_height = int(h * scale)
                new_width = Config.OCR_IMG_WIDTH
            else:
                new_height = Config.OCR_IMG_HEIGHT
            
            resized = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            result = np.ones((Config.OCR_IMG_HEIGHT, Config.OCR_IMG_WIDTH), dtype=np.uint8) * 255
            y_offset = (Config.OCR_IMG_HEIGHT - new_height) // 2
            x_offset = (Config.OCR_IMG_WIDTH - new_width) // 2
            result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            pil_image = Image.fromarray(result)
            
            if hasattr(self, 'crnn_model') and self.crnn_model is not None:
                expected_channels = self.crnn_model.input_channels
            else:
                expected_channels = 1
                
            if expected_channels == 3:
                pil_image = pil_image.convert('RGB')
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            
            tensor_image = transform(pil_image).unsqueeze(0)
            return tensor_image
            
        except Exception as e:
            logger.error(f"Error preprocessing plate image: {e}")
            return None
    
    def decode_ctc_predictions(self, outputs) -> Tuple[List[str], List[float]]:
        """Decodes CTC predictions using beam search if available, otherwise greedy decoding."""
        if BEAM_SEARCH_AVAILABLE and outputs is not None:
            try:
                probs = torch.softmax(outputs, dim=2).permute(1, 0, 2).cpu().numpy()
                decoded_texts, confidences = [], []
                
                for p in probs:
                    beam = beam_search(p, vocabulary=self.char_list, beam_size=10)
                    if beam:
                        decoded_texts.append(beam[0][0])
                        confidences.append(beam[0][1])
                    else:
                        decoded_texts.append("")
                        confidences.append(0.0)
                return decoded_texts, confidences
            except Exception as e:
                logger.warning(f"Beam search decoding failed: {e}. Falling back to greedy decoding.")

        preds_idx = torch.argmax(outputs, dim=2).transpose(0, 1).cpu().numpy()
        decoded_texts, confidences = [], []
        probs = torch.softmax(outputs, dim=2).transpose(0, 1).cpu().detach().numpy()
        for i in range(preds_idx.shape[0]):
            batch_preds, batch_probs = preds_idx[i], probs[i]
            text, char_confidence, last_char_idx = [], [], 0
            for t, char_idx in enumerate(batch_preds):
                if char_idx != 0 and char_idx != last_char_idx:
                    if char_idx < len(self.char_list):
                        text.append(self.char_list[char_idx])
                        char_confidence.append(batch_probs[t, char_idx])
                last_char_idx = char_idx
            decoded_texts.append("".join(text))
            confidences.append(np.mean(char_confidence) if char_confidence else 0.0)
        return decoded_texts, confidences
    
    def recognize_plate_text(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize text from plate image using CRNN"""
        if self.crnn_model is None or self.char_list is None:
            return None, 0.0
            
        try:
            if not self._is_good_plate_image(plate_image):
                return None, 0.0
            
            tensor_image = self.preprocess_plate_image(plate_image)
            if tensor_image is None:
                return None, 0.0
            
            tensor_image = tensor_image.to(self.device)
            
            with torch.no_grad():
                outputs = self.crnn_model(tensor_image)
                decoded_texts, confidences = self.decode_ctc_predictions(outputs)
            
            if decoded_texts and confidences:
                text = decoded_texts[0].upper().replace(" ", "")
                confidence = confidences[0]
                
                if self._validate_plate_text(text) and confidence >= Config.MIN_OCR_CONFIDENCE:
                    return text, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error recognizing plate text: {e}")
            return None, 0.0

    def _is_good_plate_image(self, image: np.ndarray) -> bool:
        """Check if the plate image has sufficient quality for OCR"""
        try:
            h, w = image.shape[:2]
            if h < 25 or w < 60: return False
            aspect_ratio = w / h
            if aspect_ratio < 2.0 or aspect_ratio > 6.0: return False
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            contrast = np.std(gray)
            if contrast < 15: return False
            mean_intensity = np.mean(gray)
            if mean_intensity < 30 or mean_intensity > 225: return False
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50: return False
            return True
        except Exception:
            return False
    
    def _validate_plate_text(self, text: str) -> bool:
        """Enhanced validation for license plate text"""
        if not text or len(text) < 6:  # Minimum 6 characters for valid plate
            logger.debug(f"Plate validation failed: too short ({len(text) if text else 0} chars)")
            return False
        
        # Check if text contains reasonable characters for a license plate
        if not re.match(r'^[A-Z0-9]{6,12}$', text):
            logger.debug(f"Plate validation failed: invalid characters in '{text}'")
            return False
        
        # Reject obvious garbage patterns
        if self._is_garbage_text(text):
            logger.debug(f"Plate validation failed: garbage pattern '{text}'")
            return False
        
        # Check for reasonable license plate patterns
        # Indian plates: typically start with 2 letters, followed by 2 digits, then 1-2 letters, then 4 digits
        # Examples: KA31BR4210, TS15EX0371, GJ05SX1535
        indian_pattern = re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$', text)
        if indian_pattern:
            logger.debug(f"Plate validation passed (Indian pattern): '{text}'")
            return True
        
        # International patterns - be more strict
        # Pattern 1: ABC1234 (3 letters + 4 numbers)
        if re.match(r'^[A-Z]{3}[0-9]{4}$', text):
            logger.debug(f"Plate validation passed (international pattern 1): '{text}'")
            return True
        
        # Pattern 2: AB12CD34 (2 letters + 2 numbers + 2 letters + 2 numbers)
        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{2}$', text):
            logger.debug(f"Plate validation passed (international pattern 2): '{text}'")
            return True
        
        # Pattern 3: 123ABC (3 numbers + 3 letters)
        if re.match(r'^[0-9]{3}[A-Z]{3}$', text):
            logger.debug(f"Plate validation passed (international pattern 3): '{text}'")
            return True
            
        logger.debug(f"Plate validation failed: doesn't match expected patterns '{text}'")
        return False

    def _is_garbage_text(self, text: str) -> bool:
        """Check if text appears to be garbage/random characters"""
        if not text:
            return True
        
        # Check for too many repeated characters
        for char in set(text):
            if text.count(char) > len(text) * 0.6:  # More than 60% same character
                return True
        
        # Check for common OCR garbage patterns
        garbage_patterns = [
            r'[FHD]{3,}',  # Too many F, H, D characters (common OCR errors)
            r'[089]{5,}',  # Too many similar-looking numbers
            r'^[ILOQ]{2,}', # Starting with confusing characters
            r'[XVW]{3,}',  # Too many wide characters
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for reasonable distribution of letters vs numbers
        letters = sum(1 for c in text if c.isalpha())
        numbers = sum(1 for c in text if c.isdigit())
        
        # Reject if it's all letters or all numbers (except very specific cases)
        if letters == 0 or numbers == 0:
            # Allow some exceptions for valid patterns
            if not (re.match(r'^[A-Z]{6,8}$', text) or re.match(r'^[0-9]{6,8}$', text)):
                return False
        
        return False

def process_images(input_dir: str, output_dir: str):
    """
    Processes all images in the input directory, detects plates, performs OCR,
    and saves the cropped plates to the output directory.
    """
    if not YOLO_AVAILABLE:
        logger.error("YOLO (ultralytics) is not installed. Please install it to run this script.")
        return

    anpr_processor = ANPRProcessor()
    if not anpr_processor.load_models():
        logger.error("Failed to load models. Exiting.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load existing plates from the raw_ocr directory to avoid duplicates
    raw_ocr_dir = Path(r"D:\Work\Projects\ANPR\data\raw_ocr")
    existing_plates = set()
    if raw_ocr_dir.exists():
        for f in raw_ocr_dir.iterdir():
            if f.is_file():
                # Add filename without extension to the set
                existing_plates.add(f.stem)
    logger.info(f"Loaded {len(existing_plates)} existing plates from {raw_ocr_dir}")
    
    logger.info(f"Starting batch processing from '{input_dir}' to '{output_dir}'")
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).rglob(ext))

    if not image_paths:
        logger.warning(f"No images found in {input_dir}")
        return

    total_images = len(image_paths)
    processed_count = 0
    plates_found = 0

    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing [{i+1}/{total_images}] {image_path.name}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning(f"Could not read image: {image_path.name}. Skipping.")
            continue
        
        _, plate_boxes = anpr_processor.detect_vehicles_and_plates(frame)

        if not plate_boxes:
            logger.info(f"  -> No plates detected in {image_path.name}")
            continue

        for j, bbox in enumerate(plate_boxes):
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            plate_region = frame[y1:y2, x1:x2]

            if plate_region.size == 0:
                continue

            text, ocr_confidence = anpr_processor.recognize_plate_text(plate_region)

            if text:
                sanitized_text = re.sub(r'[\W_]+', '', text)
                
                # Check if plate already exists in the raw_ocr directory
                if sanitized_text in existing_plates:
                    logger.info(f"  -> Plate '{text}' already exists in raw_ocr. Skipping.")
                    continue
                
                # Determine state-wise subdirectory
                state_code = sanitized_text[:2]
                target_output_dir = output_path / state_code
                
                # If state directory doesn't exist, fall back to main output directory
                if not (target_output_dir.exists() and target_output_dir.is_dir()):
                    target_output_dir = output_path
                
                # New file naming logic with incrementing counter for duplicates up to _3
                base_filename = f"{sanitized_text}"
                extension = ".jpg"
                save_path = target_output_dir / f"{base_filename}{extension}"
                
                counter = 1
                # The loop will find the next available slot (_1, _2, _3)
                while save_path.exists() and counter <= 3:
                    new_filename = f"{base_filename}_{counter}{extension}"
                    save_path = target_output_dir / new_filename
                    counter += 1

                # If after the loop, the path STILL exists, it means we've reached max duplicates
                if save_path.exists():
                    logger.info(f"  -> Skipping plate '{text}', max duplicates reached for this plate.")
                    continue
                
                plates_found += 1
                cv2.imwrite(str(save_path), plate_region)
                logger.info(f"  -> Saved new plate '{text}' to {save_path.relative_to(output_path)} (Conf: {ocr_confidence:.2f})")
            else:
                logger.info(f"  -> Plate detected but OCR failed.")
        
        processed_count += 1
        
    logger.info("="*30)
    logger.info("Batch processing complete.")
    logger.info(f"Total images processed: {processed_count}/{total_images}")
    logger.info(f"Total license plates found and saved: {plates_found}")
    logger.info("="*30)


def main():
    """Main entry point for the batch processing script."""
    # Using raw strings for Windows paths
    input_directory = r"D:\Work\Projects\ANPR\data\annotation"
    output_directory = r"D:\Work\Projects\ANPR\data\anpr_annotation"
    
    if not Path(input_directory).exists():
        logger.error(f"Input directory does not exist: {input_directory}")
        sys.exit(1)
        
    process_images(input_directory, output_directory)

if __name__ == "__main__":
    main() 