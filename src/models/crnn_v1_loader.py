#!/usr/bin/env python3
"""
This script defines the CustomCRNN_V1 model architecture and provides a loader
function for the crnn_v1 models, which use a CTC loss and no attention.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CustomCRNN_V1(nn.Module):
    """Custom CRNN model for license plate recognition (V1 architecture)."""
    def __init__(self, img_height, n_classes, n_hidden=256, input_channels=1, dropout_rate=0.4):
        super(CustomCRNN_V1, self).__init__()
        self.input_channels = input_channels
        self.n_hidden = n_hidden

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )

        # RNN (Encoder)
        self.rnn = nn.LSTM(512, n_hidden, bidirectional=True, num_layers=2, dropout=dropout_rate)
        
        # Fully connected layer for CTC output
        self.fc = nn.Linear(n_hidden * 2, n_classes)

    def forward(self, input_tensor):
        # CNN forward pass
        conv = self.cnn(input_tensor)
        
        # Reshape for RNN
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # (w, b, c)

        # RNN forward pass
        rnn_output, _ = self.rnn(conv)

        # Fully connected layer
        output = self.fc(rnn_output)
        
        # Log softmax for CTC Loss
        return nn.functional.log_softmax(output, dim=2)

def load_crnn_v1_model(model_path: str, char_set: list):
    """
    Loads the CRNN v1 model (CTC-based).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"CRNN v1 model file not found at: {model_path}")
        raise FileNotFoundError(f"CRNN v1 model file not found at: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"✅ CRNN v1 checkpoint loaded from {model_path}")

        # --- Get model configuration ---
        model_config = checkpoint.get('model_config', {})
        if not model_config:
             logger.warning("⚠️ No 'model_config' found in checkpoint. Using default parameters.")
        
        n_classes = len(char_set)
        
        params = {
            'img_height': model_config.get('img_h', 32),
            'n_classes': n_classes,
            'n_hidden': model_config.get('n_hidden', 256),
            'dropout_rate': model_config.get('dropout', 0.4),
        }

        # --- Initialize Model ---
        model = CustomCRNN_V1(**params)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✅ CRNN v1 model (CTC) created and weights loaded.")
        logger.info(f"   - Classes: {n_classes}")
        logger.info(f"   - Config: {params}")

        return model, char_set

    except Exception as e:
        logger.error(f"❌ Failed to load CRNN v1 model: {e}", exc_info=True)
        raise 