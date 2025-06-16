#!/usr/bin/env python3
"""
ANPR Environment Setup Script
Initializes the project environment and checks requirements
"""

import sys
import os
import subprocess
from pathlib import Path
import platform

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import ensure_directories, Paths

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    major, minor = sys.version_info[:2]
    
    if major < 3 or (major == 3 and minor < 8):
        print(f"❌ Python {major}.{minor} is not supported. Please use Python 3.8+")
        return False
    else:
        print(f"✅ Python {major}.{minor} is compatible")
        return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    try:
        ensure_directories()
        print("✅ Directory structure created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return False

def main():
    """Main setup function"""
    print("🚗 ANPR System Setup")
    print("====================")
    print()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    print("\n🎉 Basic setup completed!")
    print("\n🚀 Next steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Place your trained models in the models/ directory")
    print("3. Run: python src/main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 