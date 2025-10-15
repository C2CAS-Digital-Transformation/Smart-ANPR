#!/usr/bin/env python3
"""
This script defines the CustomCRNN_V3 model architecture, including the
attention mechanism, and provides a loader function for the CRNN v3 model.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Attention-based Decoder (for multi-line plates) ---
class Attention(nn.Module):
    """Attention network for the decoder"""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_attention = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attention = nn.Linear(decoder_dim, attention_dim)
        self.full_attention = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_attention(encoder_out)
        att2 = self.decoder_attention(decoder_hidden)
        # Ensure dimensions are compatible for broadcasting
        # att2.unsqueeze(1) -> [batch_size, 1, attention_dim]
        # att1 -> [batch_size, num_pixels, attention_dim]
        combined_att = self.relu(att1 + att2.unsqueeze(1))
        alpha = self.full_attention(combined_att).squeeze(2)
        alpha = self.softmax(alpha)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class AttentionDecoder(nn.Module):
    """Decoder with attention mechanism"""
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout=0.5):
        super(AttentionDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = [c - 1 for c in caption_lengths]

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # Use embeddings from the correct time step
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas

class CustomCRNN_V3(nn.Module):
    """CRNN model for license plate recognition with an attention mechanism (V3)."""
    def __init__(self, img_height, n_classes, n_hidden=256, dropout_rate=0.4, 
                 input_channels=1, embed_dim=128, attention_dim=256, decoder_hidden_dim=256):
        super(CustomCRNN_V3, self).__init__()
        self.input_channels = input_channels
        self.n_hidden = n_hidden
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # RNN (Encoder)
        self.rnn = nn.LSTM(512, n_hidden, bidirectional=True, num_layers=2, dropout=dropout_rate)

        # Decoder with Attention
        self.decoder = AttentionDecoder(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_hidden_dim,
            vocab_size=n_classes,
            encoder_dim=n_hidden * 2,
            dropout=dropout_rate
        )

    def forward(self, input_tensor, targets=None, teacher_forcing_ratio=0.5):
        # CNN forward pass
        conv = self.cnn(input_tensor)
        
        # Reshape for RNN
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)  # Remove height dimension, assuming it's 1
        conv = conv.permute(2, 0, 1)  # (w, b, c)

        # RNN forward pass
        encoder_outputs, _ = self.rnn(conv) # encoder_outputs: (w, b, n_hidden * 2)

        # Permute for decoder
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # (b, w, n_hidden * 2)

        # If training, use teacher forcing
        if self.training and targets is not None:
            # Create caption lengths (batch_size,)
            caption_lengths = [len(t) for t in targets]
            
            # Pass to decoder
            predictions, _, _, _ = self.decoder(encoder_outputs, targets, caption_lengths)
            return predictions
        
        # If evaluating, predict greedily
        else:
            batch_size = encoder_outputs.size(0)
            max_len = 20 # Max length for a license plate
            
            # Start with <start> token
            start_tokens = torch.ones(batch_size, dtype=torch.long).to(input_tensor.device) * 1 # Assuming 1 is <start>
            embeddings = self.decoder.embedding(start_tokens)

            # Initialize hidden state
            h, c = self.decoder.init_hidden_state(encoder_outputs)
            
            predictions = torch.zeros(batch_size, max_len, self.decoder.vocab_size).to(input_tensor.device)
            
            for t in range(max_len):
                attention_weighted_encoding, _ = self.decoder.attention(encoder_outputs, h)
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding
                
                h, c = self.decoder.decode_step(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c)
                )
                
                preds = self.decoder.fc(h)
                predictions[:, t, :] = preds
                
                # Get the next input from the current prediction
                _, next_word_idx = torch.max(preds, dim=1)
                embeddings = self.decoder.embedding(next_word_idx)
                
            return predictions

def load_crnn_v3_model(model_path: str, char_set: list):
    """
    Loads the CRNN v3 model with attention.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"CRNN v3 model file not found at: {model_path}")
        raise FileNotFoundError(f"CRNN v3 model file not found at: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"✅ CRNN v3 checkpoint loaded from {model_path}")

        # --- Get model configuration ---
        model_config = checkpoint.get('model_config', {})
        if not model_config:
             logger.warning("⚠️ No 'model_config' found in checkpoint. Using default parameters.")
        
        n_classes = len(char_set)
        img_height = model_config.get('img_h', 32)
        
        # These params are based on the original CustomCRNN with attention
        params = {
            'img_height': img_height,
            'n_classes': n_classes,
            'n_hidden': model_config.get('n_hidden', 256),
            'embed_dim': model_config.get('embed_dim', 128),
            'attention_dim': model_config.get('attention_dim', 256),
            'decoder_hidden_dim': model_config.get('decoder_hidden_dim', 256),
            'dropout_rate': model_config.get('dropout_rate', 0.4),
        }

        # --- Initialize Model ---
        model = CustomCRNN_V3(**params)
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✅ CRNN v3 model with attention created and weights loaded.")
        logger.info(f"   - Classes: {n_classes}")
        logger.info(f"   - Config: {params}")

        return model, char_set

    except Exception as e:
        logger.error(f"❌ Failed to load CRNN v3 model: {e}", exc_info=True)
        raise 