#!/usr/bin/env python3
"""
Lightweight YOLO Training Script for ANPR
==========================================
Auto-generated training script optimized for your dataset.

Dataset: 29,618 instances
Classes: Car (10,539), 
         Motorcycle (2,481), 
         Number_Plate (16,598)
Imbalance Ratio: 6.7:1
"""

import os
import sys
import torch
from ultralytics import YOLO
from pathlib import Path
import logging

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import Paths, ModelConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Training configuration
    config = {
        'data': str(Paths.YOLO_CONFIG_PATH),
        'model': 'yolo11n.pt',
        'epochs': ModelConfig.TRAIN_EPOCHS,
        'imgsz': ModelConfig.YOLO_INPUT_SIZE,
        'batch': ModelConfig.TRAIN_BATCH_SIZE,
        'workers': 8,
        'patience': ModelConfig.TRAIN_PATIENCE,
        'save_period': 10,
        'project': str(Paths.DETECTION_MODELS),
        'name': 'yolo11n_anpr',
        'exist_ok': True,
        'verbose': True,
        'device': device,
        'amp': True,
        'half': False,
        'plots': True,
        'save_json': True,
        
        # Optimized hyperparameters
        'lr0': 0.008,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.87,
        'dfl': 1.5,
        
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.05,
        'close_mosaic': 10,
    }
    
    try:
        # Initialize model
        logger.info("🚀 Loading YOLO model...")
        model = YOLO(config['model'])
        
        # Start training
        logger.info("🏋️ Starting training...")
        logger.info(f"   Model: {config['model']}")
        logger.info(f"   Image size: {config['imgsz']}")
        logger.info(f"   Batch size: {config['batch']}")
        logger.info(f"   Epochs: {config['epochs']}")
        
        results = model.train(**config)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Results saved to: {results.save_dir}")
        
        # Validation
        logger.info("🔍 Running validation...")
        val_results = model.val(data=config['data'])
        
        logger.info("✅ Training and validation completed!")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        
        # Try with smaller batch size
        logger.info("🔄 Retrying with smaller batch size...")
        config['batch'] = max(8, config['batch'] // 2)
        config['name'] = 'yolo11n_anpr_small_batch'
        
        try:
            model = YOLO(config['model'])
            results = model.train(**config)
            logger.info("✅ Training successful with reduced batch size!")
            return results
        except Exception as e2:
            logger.error(f"❌ Training failed even with smaller batch: {e2}")
            return None

if __name__ == "__main__":
    main()
