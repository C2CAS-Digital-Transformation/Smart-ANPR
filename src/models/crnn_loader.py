#!/usr/bin/env python3
"""
This script defines the CustomCRNN_V4 model architecture and provides a loader function
for the crnn_v4 models, which do not use an attention mechanism.
"""

import torch
import torch.nn as nn
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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

class CustomCRNN_V4(nn.Module):
    """Custom CRNN model for license plate recognition - compatible with crnn_v4 architecture"""
    def __init__(self, img_height, n_classes, n_hidden=512, dropout_rate=0.5, input_channels=1):
        super(CustomCRNN_V4, self).__init__()
        self.input_channels = input_channels
        
        NUM_VERTICAL_ROWS = 3
        
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
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((NUM_VERTICAL_ROWS, None))
        self.rnn_input_size = 512 * NUM_VERTICAL_ROWS
        self.rnn1 = ImprovedBidirectionalLSTM(self.rnn_input_size, n_hidden, n_hidden, num_layers=2, dropout=dropout_rate)
        self.rnn2 = ImprovedBidirectionalLSTM(n_hidden, n_hidden, n_classes, num_layers=1, dropout=dropout_rate)
        
    def forward(self, input_tensor):
        conv = self.cnn(input_tensor)
        conv = self.adaptive_pool(conv)
        conv = conv.permute(3, 0, 1, 2)
        conv = conv.reshape(conv.size(0), conv.size(1), -1)
        
        encoder_outputs = self.rnn1(conv)
        ctc_output = self.rnn2(encoder_outputs)

        return ctc_output

def load_crnn_v4_model(model_path, device):
    """Load CRNN model for OCR"""
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"CRNN model not found at: {model_path}")
            
        logger.info(f"Loading CRNN v4 model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'char_set' in checkpoint:
            char_list = checkpoint['char_set']
        else:
            logger.warning("Character set not found in checkpoint, using fallback")
            char_list = ['[blank]'] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        
        model_config = checkpoint.get('model_config', {})
        img_height = model_config.get('img_height', 48)
        n_classes = len(char_list)
        n_hidden = 512
        input_channels = 1
        dropout_rate = 0.5
        
        model = CustomCRNN_V4(
            img_height=img_height,
            n_classes=n_classes,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            input_channels=input_channels
        )
        
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        
        new_state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        unloaded_keys = [k for k in model_dict if k not in new_state_dict]
        if unloaded_keys:
            logger.warning(f"{len(unloaded_keys)} keys in the model were NOT found in the checkpoint: {unloaded_keys[:5]}")

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=False)
        model.to(device)
        model.eval()
        
        logger.info(f"CRNN v4 model loaded successfully! Character set size: {len(char_list)}")
        return model, char_list
        
    except Exception as e:
        logger.error(f"Error loading CRNN v4 model: {e}", exc_info=True)
        return None, None 