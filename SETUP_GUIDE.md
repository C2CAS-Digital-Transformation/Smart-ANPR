# ANPR System - Quick Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- Webcam or video files for testing
- (Optional) CUDA-capable GPU for better performance

## 🚀 Quick Start (Fresh Installation)

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd ANPR
```

### Step 2: Install the Package

**Option A: Using setup.py (Recommended for Development)**
```bash
pip install -e .
```
This installs the package in "editable" mode with all dependencies.

**Option B: Using requirements.txt (Simpler)**
```bash
pip install -r requirements.txt
```

**Optional Extras:**
```bash
# For better OCR performance (optional)
pip install -e .[fast]

# For development tools (optional)
pip install -e .[dev]
```

### Step 3: Verify Essential Files

After cloning, verify that these files exist:

#### ✅ **Production Models** (in `models/application_runner/`)
- `best.pt` (~6MB) - YOLO11n ANPR detection model
- `best_crnn_model_epoch125_acc0.9010.pth` (~4MB) - CRNN OCR v7 model

#### ✅ **Configuration Files** (in `config/`)
- `data.yaml` - YOLO data configuration
- `deployment_config.json` - Deployment settings
- `custom_hyp_imbalanced.yaml` - Training hyperparameters

#### ✅ **Data Files**
- `data/processed/all_chars.txt` - OCR character set
- `data/Input/*.mp4` - Demo video files (optional)

#### ✅ **Source Code** (in `src/`)
- `main.py` - Main application
- All Python files in subdirectories

### Step 4: Verify Setup

Run this verification script:

```bash
python -c "
from pathlib import Path
import sys

print('🔍 Verifying ANPR Setup...\n')

# Check models
models_ok = True
model_files = [
    'models/application_runner/best.pt',
    'models/application_runner/best_crnn_model_epoch125_acc0.9010.pth'
]
for model in model_files:
    exists = Path(model).exists()
    status = '✅' if exists else '❌'
    print(f'{status} {model}')
    if not exists:
        models_ok = False

# Check data
data_file = 'data/processed/all_chars.txt'
data_ok = Path(data_file).exists()
status = '✅' if data_ok else '❌'
print(f'{status} {data_file}')

# Check config
config_ok = Path('config/data.yaml').exists()
status = '✅' if config_ok else '❌'
print(f'{status} config/data.yaml')

print()
if models_ok and data_ok and config_ok:
    print('✅ Setup verification PASSED! Ready to run.')
    sys.exit(0)
else:
    print('❌ Setup verification FAILED! Some files are missing.')
    print('   Make sure you cloned with: git clone --depth 1 <url>')
    sys.exit(1)
"
```

### Step 5: Run the Application

```bash
# Method 1: Direct execution
python src/main.py

# Method 2: If installed with setup.py
anpr
```

The application will:
1. Automatically load models from `models/application_runner/`
2. Create `outputs/` and `logs/` directories if they don't exist
3. Launch the PyQt5 GUI interface

## 📁 Repository Structure (What's Tracked)

### ✅ **Tracked Files** (Included in Git)

```
ANPR/
├── README.md                              # Documentation
├── SETUP_GUIDE.md                         # This file
├── requirements.txt                       # Python dependencies
├── setup.py                              # Package setup
├── .gitignore                            # Git ignore rules
│
├── models/
│   └── application_runner/               # ⭐ Production models (~10MB total)
│       ├── best.pt                       # YOLO11n detection
│       └── best_crnn_model_epoch125_acc0.9010.pth  # CRNN v7 OCR
│
├── config/                               # Configuration files
│   ├── data.yaml
│   ├── deployment_config.json
│   └── custom_hyp_imbalanced.yaml
│
├── data/
│   ├── Input/                            # Demo videos (optional)
│   │   └── *.mp4
│   └── processed/
│       └── all_chars.txt                 # OCR character set
│
├── src/                                  # Source code
│   ├── main.py                           # Main application
│   ├── models/                           # Model loaders
│   ├── training/                         # Training scripts
│   └── utils/                            # Utilities
│
└── scripts/                              # Utility scripts
    ├── evaluate.py
    └── demo.py
```

### ❌ **Ignored Files** (Not in Git)

```
ANPR/
├── models/
│   ├── detection/                        # ❌ Large training checkpoints (~500MB+)
│   └── ocr/                             # ❌ Large training checkpoints (~500MB+)
│
├── data/                                 # ❌ Most data files
│   ├── raw/                             # Raw input data
│   ├── annotation/                       # Training annotations
│   ├── scraped_images/                   # Downloaded images
│   └── ...
│
├── outputs/                              # ❌ Generated at runtime
├── logs/                                 # ❌ Generated at runtime
├── Trash/                                # ❌ Temporary files
└── __pycache__/                         # ❌ Python cache
```

## 🔧 Configuration

### Model Paths (Automatic)

The system uses **automatic path detection**:

```python
# In src/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "application_runner" / "best.pt"
CRNN_MODEL_PATH = PROJECT_ROOT / "models" / "application_runner" / "best_crnn_model_epoch125_acc0.9010.pth"
```

**No manual path configuration needed!** Works on:
- ✅ Windows
- ✅ Linux
- ✅ macOS

### Settings in GUI

All settings can be adjusted through the GUI:
- Confidence thresholds
- Camera resolution
- Detection modes (Real-time, Parking, Zone-based)
- Frame processing rate

## 🎯 Usage Modes

### 1. **Camera Mode**
- Live detection from webcam
- Adjustable resolution and FPS
- Real-time processing

### 2. **Video File Mode**
- Process pre-recorded videos
- Playback speed control (0.5x - 6x)
- Seek forward/backward
- Zone-based detection

### 3. **Detection Modes**
- **Real-time Detection**: Process all detected plates
- **Parking Vehicle Detection**: Detect stationary vehicles
- **Zone-based Detection**: Define entry/exit zones

## 📊 Model Information

### YOLO11n ANPR Detection
- **File**: `best.pt` (~6MB)
- **Performance**: 90.3% mAP@0.5
- **Classes**: Car, Motorcycle, Number_Plate
- **Speed**: 80-120 FPS (RTX 3060)

### CRNN v7 OCR
- **File**: `best_crnn_model_epoch125_acc0.9010.pth` (~4MB)
- **Accuracy**: 90.10%
- **Character Set**: 36 characters (0-9, A-Z, blank)
- **Input**: 256x64 RGB images

## 🐛 Troubleshooting

### Issue: Models not found
**Solution**: Verify files exist in `models/application_runner/`

### Issue: Camera not detected
**Solution**: Click "🔍 Detect Cameras" button in GUI

### Issue: Low detection accuracy
**Solution**: Adjust confidence thresholds in Settings panel

### Issue: Import errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📝 Output Files

Generated in `outputs/` directory:
- `car_PLATENUMBER_timestamp_conf.jpg` - Detected vehicles with plates
- `debug_*.jpg` - Debug frames
- `detection_log.jsonl` - JSON log of all detections

## 🔄 Updating the Repository

After pulling updates:

```bash
git pull origin main
pip install -r requirements.txt  # Update dependencies if changed
python src/main.py               # Run the application
```

## 💾 Repository Size

**Total Repository Size**: ~15-20 MB
- Production models: ~10 MB
- Source code: ~1 MB
- Configuration: <1 MB
- Demo videos: ~5 MB (optional)

**Note**: Training checkpoints are NOT included (would add ~1GB+)

## 🎓 For Development

If you want to train models or access full training history:

1. Download full model checkpoints separately
2. Place in `models/detection/` and `models/ocr/`
3. Use training scripts in `src/training/`

## 📧 Support

For issues or questions:
- Check the main `README.md`
- Review this setup guide
- Check the troubleshooting section

---

**Happy License Plate Recognition! 🚗📸**

