#!/usr/bin/env python3
"""
OCR Test Script for Screenshots
Tests the CRNN model on images from Screenshots folder
Uses the same preprocessing pipeline as main.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Model path
    CRNN_MODEL_PATH = r"D:\Work\Projects\ANPR\models\ocr\crnn_v7\best_crnn_model_epoch125_acc0.9010.pth"
    
    # Input and output directories
    INPUT_DIR = r"C:\Users\Jagadeesh\Pictures\Screenshots"
    OUTPUT_DIR = r"C:\Users\Jagadeesh\Pictures\Screenshots\output"
    
    # OCR parameters
    OCR_IMG_HEIGHT = 64
    OCR_IMG_WIDTH = 256
    MIN_OCR_CONFIDENCE = 0.1  # Low threshold for testing
    
    # Character set file
    CHAR_SET_FILE = r"D:\Work\Projects\ANPR\data\combined_training_data_ocr\all_chars.txt"

class ImprovedBidirectionalLSTM(nn.Module):
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
    """Custom CRNN model matching train_custom_crnn.py exactly"""
    def __init__(self, img_height, n_classes, n_hidden=256):
        super(CustomCRNN, self).__init__()
        
        # CNN part exactly as in training script
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Dropout2d(0.3 * 0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(0.3 * 0.5),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(0.3 * 0.7),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(True), nn.Dropout2d(0.3)
        )
        
        # RNN part exactly as in training script
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn_input_size = 512 
        self.rnn1 = ImprovedBidirectionalLSTM(self.rnn_input_size, n_hidden // 2, n_hidden // 2, num_layers=2, dropout=0.3)
        self.rnn2 = ImprovedBidirectionalLSTM(n_hidden // 2, n_hidden // 2, n_classes, num_layers=1, dropout=0.3)
        
    def forward(self, input_tensor):
        conv = self.cnn(input_tensor)
        conv = self.adaptive_pool(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        
        output = self.rnn1(conv)
        output = self.rnn2(output)
        return output

class OCRProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.crnn_model = None
        self.char_list = None
        
        # Create output directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
    def load_character_set(self):
        """Load character set from file"""
        try:
            if Path(Config.CHAR_SET_FILE).exists():
                with open(Config.CHAR_SET_FILE, 'r', encoding='utf-8') as f:
                    all_characters = [line.strip() for line in f if line.strip()]
            else:
                # Fallback character set
                all_characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                logger.warning("Character set file not found, using fallback")
            
            # Ensure [blank] is the first character
            if '[blank]' not in all_characters:
                self.char_list = ['[blank]'] + all_characters
            else:
                all_characters.remove('[blank]')
                self.char_list = ['[blank]'] + all_characters
                
            logger.info(f"Character set loaded: {len(self.char_list)} characters")
            logger.info(f"Characters: {''.join(self.char_list)}")
            return True
        except Exception as e:
            logger.error(f"Error loading character set: {e}")
            return False
    
    def load_model(self):
        """Load CRNN model"""
        try:
            if not Path(Config.CRNN_MODEL_PATH).exists():
                logger.error(f"Model file not found: {Config.CRNN_MODEL_PATH}")
                return False
            
            logger.info(f"Loading CRNN model from: {Config.CRNN_MODEL_PATH}")
            
            # Load checkpoint
            checkpoint = torch.load(Config.CRNN_MODEL_PATH, map_location=self.device)
            
            # Get character set from model if available
            if 'char_set' in checkpoint:
                self.char_list = checkpoint['char_set']
                logger.info(f"Character set loaded from model: {len(self.char_list)} characters")
            else:
                if not self.load_character_set():
                    return False
            
            # Get model config
            model_config = checkpoint.get('model_config', {})
            img_height = model_config.get('img_height', Config.OCR_IMG_HEIGHT)
            n_classes = model_config.get('n_classes', len(self.char_list))
            n_hidden = model_config.get('n_hidden', 256)
            
            logger.info(f"Model config: img_height={img_height}, n_classes={n_classes}, n_hidden={n_hidden}")
            
            # Initialize model
            self.crnn_model = CustomCRNN(img_height, n_classes, n_hidden)
            
            # Load state dict
            model_dict = self.crnn_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            if len(pretrained_dict) != len(checkpoint['model_state_dict']):
                logger.warning(f"Loaded {len(pretrained_dict)}/{len(checkpoint['model_state_dict'])} layers")
            
            model_dict.update(pretrained_dict)
            self.crnn_model.load_state_dict(model_dict, strict=False)
            
            self.crnn_model.to(self.device)
            self.crnn_model.eval()
            
            logger.info("CRNN model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CRNN model: {e}")
            return False
    
    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing matching training script exactly"""
        try:
            img_array = np.array(image)
            
            # Ensure it's RGB
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                pass  # Already RGB
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE to LAB color space
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img_array = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            # Apply bilateral filter and sharpening
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_array = cv2.filter2D(img_array, -1, kernel)
            
            return Image.fromarray(img_array)
        except Exception as e:
            logger.warning(f"Failed to apply OCR preprocessing: {e}")
            return image
    
    def preprocess_image(self, image: np.ndarray):
        """Preprocess image for CRNN model"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')

            # Apply preprocessing
            pil_image = self.preprocess_for_ocr(pil_image)

            # Apply transforms
            transform = transforms.Compose([
                transforms.Resize((Config.OCR_IMG_HEIGHT, Config.OCR_IMG_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            tensor_image = transform(pil_image).unsqueeze(0)
            return tensor_image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def decode_ctc_predictions(self, outputs):
        """Decode CTC predictions using exact training script logic"""
        try:
            preds_idx = torch.argmax(outputs, dim=2)
            preds_idx = preds_idx.transpose(0, 1).cpu().numpy()
            
            decoded_texts = []
            confidences = []

            probs = torch.softmax(outputs, dim=2).transpose(0,1).cpu().detach().numpy()

            for i in range(preds_idx.shape[0]):
                batch_preds = preds_idx[i]
                batch_probs = probs[i]
                
                text = []
                char_confidence = []
                last_char_idx = 0 
                for t in range(len(batch_preds)):
                    char_idx = batch_preds[t]
                    if char_idx != 0 and char_idx != last_char_idx:
                        if char_idx < len(self.char_list):
                            text.append(self.char_list[char_idx])
                            char_confidence.append(batch_probs[t, char_idx])
                    last_char_idx = char_idx
                
                decoded_texts.append("".join(text))
                avg_conf = np.mean(char_confidence) if char_confidence else 0.0
                confidences.append(avg_conf)
                
            return decoded_texts, confidences
        except Exception as e:
            logger.error(f"Error in CTC decoding: {e}")
            return [], []
    
    def recognize_text(self, image: np.ndarray):
        """Recognize text from image"""
        if self.crnn_model is None or self.char_list is None:
            logger.error("Model or character set not loaded")
            return None, 0.0
            
        try:
            # Preprocess image
            tensor_image = self.preprocess_image(image)
            if tensor_image is None:
                return None, 0.0
            
            # Move to device
            tensor_image = tensor_image.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.crnn_model(tensor_image)
                decoded_texts, confidences = self.decode_ctc_predictions(outputs)
            
            if decoded_texts and confidences:
                # Keep original text with spaces, just convert to uppercase
                text = decoded_texts[0].upper()
                confidence = confidences[0]
                
                # Format license plate text properly (add spaces in standard positions)
                formatted_text = self.format_license_plate_text(text)
                return formatted_text, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error recognizing text: {e}")
            return None, 0.0
    
    def format_license_plate_text(self, text: str) -> str:
        """Format license plate text with proper spacing"""
        try:
            # Remove existing spaces for processing
            clean_text = text.replace(" ", "").replace("-", "")
            
            if not clean_text:
                return text
            
            # Indian license plate patterns with proper spacing
            # Pattern 1: AA00AA0000 -> AA 00 AA 0000 (e.g., UP 10 B 3633)
            if len(clean_text) >= 8 and clean_text[:2].isalpha() and clean_text[2:4].isdigit():
                if len(clean_text) == 10:  # Full format: UP10B3633 -> UP 10 B 3633
                    return f"{clean_text[:2]} {clean_text[2:4]} {clean_text[4:5]} {clean_text[5:]}"
                elif len(clean_text) == 9:  # Format: UP10AB123 -> UP 10 AB 123
                    return f"{clean_text[:2]} {clean_text[2:4]} {clean_text[4:6]} {clean_text[6:]}"
                elif len(clean_text) == 8:  # Format: UP10A123 -> UP 10 A 123
                    return f"{clean_text[:2]} {clean_text[2:4]} {clean_text[4:5]} {clean_text[5:]}"
            
            # Pattern 2: 00AAAA0000 -> 00 AA AA 0000 (BH series)
            if len(clean_text) >= 8 and clean_text[:2].isdigit() and "BH" in clean_text:
                return f"{clean_text[:2]} BH {clean_text[4:8]} {clean_text[8:]}"
            
            # Pattern 3: Other formats - add basic spacing
            if len(clean_text) >= 6:
                # Try to identify state code (first 2 letters)
                if clean_text[:2].isalpha():
                    # Basic format: AB1234 -> AB 1234
                    if len(clean_text) == 6:
                        return f"{clean_text[:2]} {clean_text[2:]}"
                    # Extended format: AB12CD34 -> AB 12 CD 34
                    elif len(clean_text) == 8:
                        return f"{clean_text[:2]} {clean_text[2:4]} {clean_text[4:6]} {clean_text[6:]}"
            
            # If no pattern matches, return with minimal spacing
            return text
            
        except Exception as e:
            logger.warning(f"Error formatting license plate text: {e}")
            return text
    
    def save_annotated_image(self, image, filename, text, confidence):
        """Save original image with recognized text as filename"""
        try:
            extension = Path(filename).suffix
            
            # Create filename with just the recognized text
            if text and confidence > Config.MIN_OCR_CONFIDENCE:
                # Remove all spaces and special characters for clean filename
                clean_text = text.replace(" ", "").replace("-", "").replace("_", "")
                clean_text = "".join(c for c in clean_text if c.isalnum())
                new_filename = f"{clean_text}{extension}"
            else:
                # If no text detected, use original filename with NO_TEXT prefix
                base_name = Path(filename).stem
                new_filename = f"NO_TEXT_{base_name}{extension}"
            
            # Handle duplicate filenames by adding a counter
            output_path = Path(Config.OUTPUT_DIR) / new_filename
            counter = 1
            original_new_filename = new_filename
            while output_path.exists():
                name_part = Path(original_new_filename).stem
                new_filename = f"{name_part}_{counter}{extension}"
                output_path = Path(Config.OUTPUT_DIR) / new_filename
                counter += 1
            
            # Save image with new filename
            cv2.imwrite(str(output_path), image)
            
            # Store label information
            label_info = {
                'original_filename': filename,
                'saved_filename': new_filename,
                'text': text if text and confidence > Config.MIN_OCR_CONFIDENCE else "",
                'confidence': confidence
            }
            
            logger.info(f"Saved: {filename} → {new_filename}")
            
            # Return label info to be collected
            return label_info
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None
    
    def save_labels_file(self, all_labels, timestamp):
        """Save all labels in a single structured file"""
        try:
            # Save labels.txt in standard format
            labels_path = Path(Config.OUTPUT_DIR) / "labels.txt"
            with open(labels_path, 'w', encoding='utf-8') as f:
                for label_info in all_labels:
                    if label_info['text']:  # Only write if text was detected
                        f.write(f"{label_info['saved_filename']}\t{label_info['text']}\n")
            
            # Save labels.csv for easy viewing
            labels_csv_path = Path(Config.OUTPUT_DIR) / "labels.csv"
            with open(labels_csv_path, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=['original_filename', 'saved_filename', 'text', 'confidence'])
                writer.writeheader()
                for label_info in all_labels:
                    if label_info['text']:  # Only write if text was detected
                        writer.writerow({
                            'original_filename': label_info['original_filename'],
                            'saved_filename': label_info['saved_filename'],
                            'text': label_info['text'],
                            'confidence': label_info['confidence']
                        })
            
            # Save labels.json for programmatic access
            labels_json_path = Path(Config.OUTPUT_DIR) / "labels.json"
            with open(labels_json_path, 'w', encoding='utf-8') as f:
                import json
                valid_labels = [label for label in all_labels if label['text']]
                json.dump(valid_labels, f, indent=2, ensure_ascii=False)
            
            # Save filename mapping for reference
            mapping_path = Path(Config.OUTPUT_DIR) / "filename_mapping.txt"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                f.write("FILENAME MAPPING\n")
                f.write("=" * 50 + "\n")
                f.write("Original → Saved (License Plate Text)\n")
                f.write("-" * 50 + "\n")
                for label_info in all_labels:
                    status = "✓" if label_info['text'] else "✗"
                    f.write(f"{status} {label_info['original_filename']} → {label_info['saved_filename']}\n")
                    if label_info['text']:
                        f.write(f"    License Plate: {label_info['text']} (conf: {label_info['confidence']:.3f})\n")
                    f.write("\n")
            
            logger.info(f"Labels saved:")
            logger.info(f"  Text format: {labels_path}")
            logger.info(f"  CSV format: {labels_csv_path}")
            logger.info(f"  JSON format: {labels_json_path}")
            logger.info(f"  Filename mapping: {mapping_path}")
            
            return len([label for label in all_labels if label['text']])
            
        except Exception as e:
            logger.error(f"Error saving labels file: {e}")
            return 0
    
    def process_images(self):
        """Process all images in the input directory"""
        if not self.load_model():
            logger.error("Failed to load model")
            return
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(Config.INPUT_DIR).glob(f"*{ext}"))
            image_files.extend(Path(Config.INPUT_DIR).glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No image files found in {Config.INPUT_DIR}")
            return
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        results = []
        all_labels = []
        
        for img_path in image_files:
            try:
                logger.info(f"Processing: {img_path.name}")
                
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Could not load image: {img_path}")
                    continue
                
                # Recognize text
                start_time = time.time()
                text, confidence = self.recognize_text(image)
                processing_time = time.time() - start_time
                
                # Create result
                result = {
                    'filename': img_path.name,
                    'text': text if text else "NO_TEXT_DETECTED",
                    'confidence': confidence,
                    'processing_time': processing_time
                }
                results.append(result)
                
                logger.info(f"Result: '{text}' (conf: {confidence:.3f}, time: {processing_time:.2f}s)")
                
                # Save image and collect label info
                label_info = self.save_annotated_image(image, img_path.name, text, confidence)
                if label_info:
                    all_labels.append(label_info)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({
                    'filename': img_path.name,
                    'text': "ERROR",
                    'confidence': 0.0,
                    'processing_time': 0.0
                })
        
        # Save all results and labels
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        labeled_count = self.save_labels_file(all_labels, timestamp)
        self.save_detailed_results(results)
        
        logger.info(f"Processing complete!")
        logger.info(f"Images saved: {len(all_labels)}")
        logger.info(f"Labels created: {labeled_count}")
        logger.info(f"Results saved to: {Config.OUTPUT_DIR}")
    
    def save_detailed_results(self, results):
        """Save detailed results in multiple formats"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save JSON results for easy programmatic access
            import json
            json_path = Path(Config.OUTPUT_DIR) / f"processing_results_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save text summary
            summary_path = Path(Config.OUTPUT_DIR) / f"processing_summary_{timestamp}.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("OCR PROCESSING SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                total_files = len(results)
                successful_ocr = sum(1 for r in results if r['text'] not in ["NO_TEXT_DETECTED", "ERROR"])
                high_confidence = sum(1 for r in results if r['confidence'] > Config.MIN_OCR_CONFIDENCE)
                avg_processing_time = sum(r['processing_time'] for r in results) / len(results) if results else 0
                
                f.write(f"Total files processed: {total_files}\n")
                f.write(f"Successful OCR detections: {successful_ocr}\n")
                f.write(f"High confidence detections: {high_confidence}\n")
                f.write(f"Success rate: {(successful_ocr/total_files)*100:.1f}%\n")
                f.write(f"High confidence rate: {(high_confidence/total_files)*100:.1f}%\n")
                f.write(f"Average processing time: {avg_processing_time:.3f}s\n\n")
                
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 30 + "\n")
                for result in results:
                    status = "✓" if result['confidence'] > Config.MIN_OCR_CONFIDENCE else "⚠"
                    if result['text'] in ["NO_TEXT_DETECTED", "ERROR"]:
                        status = "✗"
                    f.write(f"{status} {result['filename']:<30} | {result['text']:<20} | {result['confidence']:.3f}\n")
            
            logger.info(f"Processing results saved:")
            logger.info(f"  JSON: {json_path}")
            logger.info(f"  Summary: {summary_path}")
            
            # Print summary to console
            total_files = len(results)
            successful_ocr = sum(1 for r in results if r['text'] not in ["NO_TEXT_DETECTED", "ERROR"])
            high_confidence = sum(1 for r in results if r['confidence'] > Config.MIN_OCR_CONFIDENCE)
            
            logger.info(f"Summary: {total_files} files processed, {successful_ocr} with text detected, {high_confidence} with high confidence")
            
        except Exception as e:
            logger.error(f"Error saving detailed results: {e}")
    
    def save_results_csv(self, results):
        """Save results to CSV file (legacy method for compatibility)"""
        self.save_detailed_results(results)

def main():
    """Main function"""
    logger.info("=== OCR Screenshot Test ===")
    logger.info(f"Input directory: {Config.INPUT_DIR}")
    logger.info(f"Output directory: {Config.OUTPUT_DIR}")
    logger.info(f"Model: {Config.CRNN_MODEL_PATH}")
    
    # Check if input directory exists
    if not Path(Config.INPUT_DIR).exists():
        logger.error(f"Input directory does not exist: {Config.INPUT_DIR}")
        return
    
    # Create OCR processor and run
    processor = OCRProcessor()
    processor.process_images()

if __name__ == "__main__":
    main() 