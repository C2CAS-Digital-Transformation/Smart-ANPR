#!/usr/bin/env python3
"""
Debug script to test ANPR detection on a specific problematic image
"""
import cv2
import numpy as np
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append('src')

from main import ANPRProcessor, Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_image_detection(image_path):
    """Test ANPR detection on a specific image"""
    print(f"🔍 Testing ANPR detection on: {image_path}")
    
    # Load image
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"✅ Loaded image: {frame.shape}")
    
    # Initialize ANPR processor
    processor = ANPRProcessor()
    print("🔄 Loading models...")
    
    if not processor.load_models():
        print("❌ Failed to load models")
        return
    
    print("✅ Models loaded successfully")
    
    # Test YOLO detection with ultra-low confidence
    print("\n🔍 === YOLO DETECTION TEST ===")
    vehicles, plates = processor.detect_vehicles_and_plates(frame)
    
    print(f"📊 YOLO Results: {len(vehicles)} vehicles, {len(plates)} plates")
    
    # Draw and save results
    debug_frame = frame.copy()
    
    # Draw vehicles
    for i, (x1, y1, x2, y2) in enumerate(vehicles):
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(debug_frame, f"Vehicle {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw plates
    for i, (x1, y1, x2, y2) in enumerate(plates):
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Plate {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save debug frame
    debug_output_path = "debug_detection_result.jpg"
    cv2.imwrite(debug_output_path, debug_frame)
    print(f"💾 Saved debug result to: {debug_output_path}")
    
    # Test OCR on detected plates
    if plates:
        print("\n🔍 === OCR TEST ===")
        for i, (x1, y1, x2, y2) in enumerate(plates):
            print(f"\n📋 Testing plate {i+1}: bbox=({x1},{y1},{x2},{y2})")
            
            # Extract plate region
            plate_region = frame[y1:y2, x1:x2]
            
            if plate_region.size > 0:
                # Save plate region
                plate_path = f"debug_plate_{i+1}.jpg"
                cv2.imwrite(plate_path, plate_region)
                print(f"💾 Saved plate region to: {plate_path}")
                
                # Test OCR
                text, confidence = processor.recognize_plate_text(plate_region)
                
                if text:
                    print(f"✅ OCR Result: '{text}' (confidence: {confidence:.3f})")
                else:
                    print(f"❌ OCR Failed: No text recognized")
            else:
                print(f"❌ Empty plate region")
    else:
        print("\n❌ No plates detected for OCR testing")
        
        # Manual region extraction for testing
        print("\n🔧 === MANUAL REGION TEST ===")
        h, w = frame.shape[:2]
        
        # Try some common plate regions
        test_regions = [
            # Lower portion of image (typical plate location)
            (w//4, h//2, 3*w//4, 7*h//8),
            # Center region
            (w//3, h//3, 2*w//3, 2*h//3),
            # Bottom third
            (0, 2*h//3, w, h),
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(test_regions):
            print(f"\n🔧 Testing manual region {i+1}: ({x1},{y1},{x2},{y2})")
            test_region = frame[y1:y2, x1:x2]
            
            if test_region.size > 0:
                manual_path = f"debug_manual_region_{i+1}.jpg"
                cv2.imwrite(manual_path, test_region)
                print(f"💾 Saved manual region to: {manual_path}")
                
                # Test OCR on manual region
                text, confidence = processor.recognize_plate_text(test_region)
                if text:
                    print(f"✅ Manual OCR Result: '{text}' (confidence: {confidence:.3f})")
                else:
                    print(f"❌ Manual OCR Failed")

if __name__ == "__main__":
    # Test with a specific image - you can change this path
    test_image_path = "debug_failed_ocr_20250616_210622_647.jpg"
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    
    test_image_detection(test_image_path) 