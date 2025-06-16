#!/usr/bin/env python3
"""
ANPR Model Evaluation Script
===========================
Comprehensive evaluation of trained YOLO models.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ANPRModelEvaluator:
    def __init__(self, model_path, data_yaml_path):
        self.model = YOLO(model_path)
        self.data_yaml_path = data_yaml_path
        # ANPR-specific classes only - no general object detection classes
        self.classes = ['Car', 'Motorcycle', 'Number_Plate']
        self.anpr_only = True  # Flag to indicate ANPR-specific model
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        logger.info("🔍 Starting model evaluation...")
        
        # Validation metrics
        val_results = self.model.val(data=self.data_yaml_path)
        
        logger.info("📊 Validation Results:")
        logger.info(f"   mAP@0.5: {val_results.box.map50:.3f}")
        logger.info(f"   mAP@0.5:0.95: {val_results.box.map:.3f}")
        
        # ANPR-specific per-class metrics (vehicles and plates only)
        for i, class_name in enumerate(self.classes):
            if i < len(val_results.box.maps):
                logger.info(f"   {class_name} mAP@0.5: {val_results.box.maps[i]:.3f}")
        
        # ANPR performance analysis
        if len(val_results.box.maps) >= 3:
            vehicle_map = (val_results.box.maps[0] + val_results.box.maps[1]) / 2  # Average of Car + Motorcycle
            plate_map = val_results.box.maps[2]  # Number Plate
            logger.info(f"   Vehicle Detection (Avg): {vehicle_map:.3f}")
            logger.info(f"   Number Plate Detection: {plate_map:.3f}")
            
            if plate_map < 0.8:
                logger.warning("   ⚠️ Number plate detection below optimal threshold (0.8)")
            if vehicle_map < 0.85:
                logger.warning("   ⚠️ Vehicle detection below optimal threshold (0.85)")
        
        return val_results
    
    def test_inference_speed(self, test_image_path=None):
        """Test model inference speed"""
        if test_image_path is None:
            # Create dummy image for speed test
            test_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        else:
            test_image = cv2.imread(test_image_path)
            test_image = cv2.resize(test_image, (416, 416))
        
        # Warmup
        for _ in range(5):
            _ = self.model(test_image, verbose=False)
        
        # Speed test
        import time
        times = []
        for _ in range(20):
            start = time.time()
            _ = self.model(test_image, verbose=False)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        logger.info(f"⚡ Inference Speed:")
        logger.info(f"   Average time: {avg_time*1000:.1f} ms")
        logger.info(f"   FPS: {fps:.1f}")
        
        return avg_time, fps
    
    def visualize_predictions(self, image_path, save_path=None):
        """Visualize model predictions"""
        results = self.model(image_path)
        
        # Plot results
        for i, result in enumerate(results):
            if save_path:
                output_path = f"{save_path}_prediction_{i}.jpg"
                result.save(output_path)
                logger.info(f"💾 Saved prediction: {output_path}")
        
        return results

def main():
    # Configuration
    model_path = "anpr_lightweight/yolo11n_anpr/weights/best.pt"
    data_yaml_path = r"D:\Work\Projects\ANPR_3\yolo_training_output\data.yaml"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Please train the model first using train_anpr_lightweight.py")
        return
    
    # Initialize evaluator
    evaluator = ANPRModelEvaluator(model_path, data_yaml_path)
    
    # Run evaluation
    val_results = evaluator.evaluate_model()
    
    # Test inference speed
    avg_time, fps = evaluator.test_inference_speed()
    
    # Test on sample images (if available)
    test_images_dir = Path(r"data\Model_Training_YOLOV11\Dataset\test\images")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpg"))[:5]  # Test on first 5 images
        
        for i, img_path in enumerate(test_images):
            logger.info(f"🖼️ Testing on: {img_path.name}")
            evaluator.visualize_predictions(str(img_path), f"test_result_{i}")
    
    logger.info("✅ Evaluation completed!")

if __name__ == "__main__":
    main()
