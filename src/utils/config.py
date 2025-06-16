"""
Configuration settings for ANPR system.
All file paths and parameters are centralized here for easy management.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directory structure
class Paths:
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_RAW = DATA_DIR / "raw"
    DATA_PROCESSED = DATA_DIR / "processed"
    DATA_ANNOTATIONS = DATA_DIR / "annotations"
    
    # Model directories
    MODELS_DIR = PROJECT_ROOT / "models"
    DETECTION_MODELS = MODELS_DIR / "detection"
    OCR_MODELS = MODELS_DIR / "ocr"
    
    # YOLO model paths
    YOLO_WEIGHTS_DIR = DETECTION_MODELS / "yolo11n_anpr"
    YOLO_MODEL_PATH = YOLO_WEIGHTS_DIR / "weights" / "best.pt"
    YOLO_MODEL_PATH_ALT = DETECTION_MODELS / "yolo11n_anpr_small_batch" / "weights" / "best.pt"
    YOLO_LAST_MODEL = YOLO_WEIGHTS_DIR / "weights" / "last.pt"
    YOLO_CONFIG_PATH = PROJECT_ROOT / "config" / "data.yaml"
    
    # CRNN model paths
    CRNN_WEIGHTS_DIR = OCR_MODELS / "crnn_v1"
    CRNN_MODEL_PATH = CRNN_WEIGHTS_DIR / "best_multiline_crnn_epoch292_acc0.9304.pth"
    CRNN_MODEL_PATH_ALT = CRNN_WEIGHTS_DIR / "checkpoint_epoch_310_acc0.923.pth"
    CRNN_CHARS_FILE = DATA_DIR / "processed" / "all_chars.txt"
    
    # Output directories
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    DETECTED_PLATES_DIR = OUTPUT_DIR / "detected_plates"
    LOGS_DIR = OUTPUT_DIR / "logs"
    RESULTS_DIR = OUTPUT_DIR / "results"
    
    # Config directory
    CONFIG_DIR = PROJECT_ROOT / "config"
    DEPLOYMENT_CONFIG = CONFIG_DIR / "deployment_config.json"
    
    # Scripts directory
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"

class ModelConfig:
    # YOLO Detection Configuration
    YOLO_CONFIDENCE_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_INPUT_SIZE = 416
    YOLO_MAX_DETECTIONS = 100
    
    # CRNN OCR Configuration
    OCR_IMG_HEIGHT = 64
    OCR_IMG_WIDTH = 256
    OCR_MIN_CONFIDENCE = 0.3
    OCR_HIDDEN_SIZE = 256
    OCR_DROPOUT_RATE = 0.3
    
    # Training Configuration
    TRAIN_EPOCHS = 200
    TRAIN_BATCH_SIZE = 32
    TRAIN_LEARNING_RATE = 0.008
    TRAIN_PATIENCE = 75

class CameraConfig:
    DEFAULT_CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30

class ProcessingConfig:
    # Detection parameters
    MIN_PLATE_AREA = 400
    PLATE_CONFIDENCE_THRESHOLD = 0.2
    
    # Duplicate prevention
    DUPLICATE_TIME_WINDOW = 5.0  # seconds
    SIMILARITY_THRESHOLD = 0.8
    UNIQUE_DETECTION_WINDOW = 10.0  # seconds
    
    # Parking detection
    PARKING_TIME_THRESHOLD = 3.0  # seconds

def get_available_crnn_model():
    """Get the best available CRNN model path."""
    if Paths.CRNN_MODEL_PATH.exists():
        return Paths.CRNN_MODEL_PATH
    elif Paths.CRNN_MODEL_PATH_ALT.exists():
        return Paths.CRNN_MODEL_PATH_ALT
    else:
        return None

def get_available_yolo_model():
    """Get the best available YOLO model path."""
    if Paths.YOLO_MODEL_PATH.exists():
        return Paths.YOLO_MODEL_PATH
    elif Paths.YOLO_MODEL_PATH_ALT.exists():
        return Paths.YOLO_MODEL_PATH_ALT
    elif Paths.YOLO_LAST_MODEL.exists():
        return Paths.YOLO_LAST_MODEL
    else:
        return None

def validate_model_paths():
    """Validate that required model files exist."""
    models_status = {
        'yolo_primary': Paths.YOLO_MODEL_PATH.exists(),
        'yolo_alt': Paths.YOLO_MODEL_PATH_ALT.exists(),
        'yolo_last': Paths.YOLO_LAST_MODEL.exists(),
        'crnn_primary': Paths.CRNN_MODEL_PATH.exists(),
        'crnn_alt': Paths.CRNN_MODEL_PATH_ALT.exists(),
        'chars_file': Paths.CRNN_CHARS_FILE.exists()
    }
    
    return models_status

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        Paths.DATA_DIR, Paths.DATA_RAW, Paths.DATA_PROCESSED, Paths.DATA_ANNOTATIONS,
        Paths.MODELS_DIR, Paths.DETECTION_MODELS, Paths.OCR_MODELS,
        Paths.YOLO_WEIGHTS_DIR, Paths.CRNN_WEIGHTS_DIR,
        Paths.OUTPUT_DIR, Paths.DETECTED_PLATES_DIR, Paths.LOGS_DIR, Paths.RESULTS_DIR,
        Paths.CONFIG_DIR, Paths.SCRIPTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Directory structure created at: {PROJECT_ROOT}")

if __name__ == "__main__":
    ensure_directories()
    
    # Validate model paths
    print("\n🔍 Checking model files...")
    models_status = validate_model_paths()
    
    for model_name, exists in models_status.items():
        status = "✅ Found" if exists else "❌ Missing"
        print(f"  {model_name}: {status}")
    
    # Show available models
    available_yolo = get_available_yolo_model()
    if available_yolo:
        print(f"\n🎯 Using YOLO model: {available_yolo.name}")
    else:
        print("\n⚠️  No YOLO model found!")
    
    available_crnn = get_available_crnn_model()
    if available_crnn:
        print(f"🤖 Using CRNN model: {available_crnn.name}")
    else:
        print("⚠️  No CRNN model found!")
    
    print("\nConfiguration loaded successfully!") 