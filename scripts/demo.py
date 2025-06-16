#!/usr/bin/env python3
"""
ANPR Demo Script
Simple demonstration of the ANPR system functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from utils.config import Paths, ensure_directories

def demo_camera():
    """Demo using webcam input"""
    print("🎥 Starting camera demo...")
    print("📱 This is a simplified demo. For full GUI, run: python src/main.py")
    
    # Ensure directories exist
    ensure_directories()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("📸 Camera opened successfully!")
    print("👁️  Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Simple frame display
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('ANPR Demo - Camera Feed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save frame
            save_path = Paths.OUTPUT_DIR / f"demo_frame_{frame_count}.jpg"
            cv2.imwrite(str(save_path), frame)
            print(f"💾 Frame saved: {save_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Demo completed!")

def demo_image(image_path):
    """Demo using single image"""
    print(f"🖼️  Processing image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Ensure directories exist
    ensure_directories()
    
    # Load and display image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return
    
    print("✅ Image loaded successfully!")
    
    # Simple image display
    cv2.putText(image, "ANPR Demo - Image Processing", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, "Press any key to close", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('ANPR Demo - Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save processed image
    save_path = Paths.OUTPUT_DIR / f"demo_processed_{Path(image_path).name}"
    cv2.imwrite(str(save_path), image)
    print(f"💾 Processed image saved: {save_path}")

def main():
    """Main demo function"""
    print("🚗 ANPR System Demo")
    print("==================")
    print()
    print("Available demos:")
    print("1. Camera demo")
    print("2. Image demo")
    print("3. Exit")
    print()
    
    while True:
        choice = input("Select demo (1-3): ").strip()
        
        if choice == "1":
            demo_camera()
            break
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            demo_image(image_path)
            break
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 