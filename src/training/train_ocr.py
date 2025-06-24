import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
from pathlib import Path
import numpy as np
import time
import logging
import cv2
import random
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import re
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from IPython.display import clear_output
import threading
from queue import Queue
from multiprocessing import freeze_support
# import argparse  # removed; not used
# import optuna  # handled via optional import below
# from training.train_ocr import train_crnn  # self import removed

# Add beam search decoding
try:
    from fast_ctc_decode import beam_search
    BEAM_SEARCH_AVAILABLE = True
except ImportError:
    BEAM_SEARCH_AVAILABLE = False
    logging.warning("fast_ctc_decode not available. Using greedy decoding.")

# Set matplotlib backend for better compatibility
plt.switch_backend('TkAgg')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Centralized Path Configuration ---
class Paths:
    PROJECT_ROOT = Path(r"D:\Work\Projects\ANPR")
    DATA_PROCESSED = PROJECT_ROOT / "data" / "combined_training_data_ocr"
    
    # OCR Model Paths
    MODEL_DIR = PROJECT_ROOT / "models" / "ocr"
    CRNN_V1_MODEL = MODEL_DIR / "crnn_v1" / "best_multiline_crnn_epoch292_acc0.9304.pth"
    CRNN_V2_DIR = MODEL_DIR / "crnn_v2"
    CRNN_V3_DIR = MODEL_DIR / "crnn_v3"  # New directory for v3 model
    
    # Character set file
    CRNN_CHARS_FILE = DATA_PROCESSED / "all_chars.txt"

# Add the parent directory to the path to import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
# from utils.config import Paths, ModelConfig # Commenting out to use local Paths
from training.data_processing import STATE_CODES

# Configuration for ANPR Project
DATA_DIR = Paths.DATA_PROCESSED
ALL_CHARS_FILE_PATH = Paths.CRNN_CHARS_FILE

TRAIN_CSV_PATH = DATA_DIR / "train_data" / "labels.csv"
VAL_CSV_PATH = DATA_DIR / "val_data" / "labels.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_data"
VAL_IMG_DIR = DATA_DIR / "val_data"

# New model save location for v3
MODEL_SAVE_PATH = Paths.CRNN_V3_DIR
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# --- Tuned Hyperparameters for CRNN v3 ---
# Goal: Improve generalization and reduce character confusion seen in v2.
# Adjusted for 6GB VRAM on RTX 3050.
BATCH_SIZE = 32          # Kept the same for memory stability
LEARNING_RATE = 8e-4     # Reduced LR for better stability
EPOCHS = 500             # Reduced epochs, early stopping should trigger anyway
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
HIDDEN_SIZE = 512

# --- Enhanced Regularization for CRNN v3 ---
DROPOUT_RATE = 0.5       # Increased dropout for better generalization
WEIGHT_DECAY = 1e-3      # Increased weight decay to combat overfitting
LABEL_SMOOTHING = 0.15   # Increased label smoothing
GRAD_CLIP = 0.5

# === Pretraining control ===
LOAD_PRETRAINED = False  # Train from scratch by default

# === Visualizer & HPO flags ===
ENABLE_VISUALIZER = True  # Set False for headless tuning runs
USE_PROXY_DATA = False    # When True, train on a small random subset for HPO
PROXY_FRACTION = 0.1      # 10 % of training samples when proxy mode is on

# Check Optuna availability for pruning support
try:
    import optuna  # noqa: F401 – optional dependency
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Holder so training loop can access current Optuna Trial (set externally)
optuna_trial = None

# Loss weights
ATTENTION_LOSS_WEIGHT = 0.4 # Reduced attention weight to focus more on the new Focal CTC loss
TEACHER_FORCING_RATIO = 0.6 # Probability of using teacher forcing

# Early stopping for >95% target
EARLY_STOPPING_PATIENCE = 140
MIN_DELTA = 0.001
TARGET_ACCURACY = 0.95

# Constants for model architecture
NUM_VERTICAL_ROWS = 3  # Keep 3 rows for multi-line support

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# -----------------------------------------------------------
#  Dummy visualizer – all methods are no-ops
#  Used when ENABLE_VISUALIZER == False (e.g. Optuna runs)
# -----------------------------------------------------------
class DummyVisualizer:
    def __getattr__(self, _):
        def _noop(*args, **kwargs):
            return None
        return _noop

class TrainingVisualizer:
    def __init__(self, target_accuracy=0.95, num_examples=6):
        self.target_accuracy = target_accuracy
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.char_accuracies = []
        self.learning_rates = []
        self.epochs = []
        self.batch_losses = []
        self.batch_numbers = []
        self.num_examples = num_examples
        
        # Create figure with subplots for metrics and examples
        self.fig = plt.figure(figsize=(20, 16))
        self.fig.suptitle('ANPR OCR Training Progress - Real Time (CRNN v3)', fontsize=16, fontweight='bold')
        
        # Gridspec for layout - added a row for the title and gave more space to examples
        gs = self.fig.add_gridspec(4, 6, height_ratios=[1, 1, 0.1, 1.2])
        
        # Main plot axes (spanning multiple columns)
        self.axes = {
            'loss': self.fig.add_subplot(gs[0, 0:2]),
            'acc': self.fig.add_subplot(gs[0, 2:4]),
            'char_acc': self.fig.add_subplot(gs[0, 4:6]),
            'lr': self.fig.add_subplot(gs[1, 0:2]),
            'batch_loss': self.fig.add_subplot(gs[1, 2:4]),
            'stats': self.fig.add_subplot(gs[1, 4:6]),
        }
        
        # Add a dedicated, invisible axis for the title text
        self.title_ax = self.fig.add_subplot(gs[2, :])
        
        # Axes for recognition examples
        self.example_axes = [self.fig.add_subplot(gs[3, i]) for i in range(self.num_examples)]
        
        # Configure subplots
        self.setup_plots()
        
        # Enable interactive mode
        plt.ion()
        self.fig.show()
        
        # Thread-safe data queue
        self.data_queue = Queue()
        
    def setup_plots(self):
        """Setup all subplot configurations"""
        # Plot 1: Training & Validation Loss
        self.axes['loss'].set_title('Training & Validation Loss', fontweight='bold')
        self.axes['loss'].set_xlabel('Epoch')
        self.axes['loss'].set_ylabel('Loss')
        self.axes['loss'].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy Progress
        self.axes['acc'].set_title('Validation Accuracy Progress', fontweight='bold')
        self.axes['acc'].set_xlabel('Epoch')
        self.axes['acc'].set_ylabel('Accuracy')
        self.axes['acc'].grid(True, alpha=0.3)
        self.axes['acc'].axhline(y=self.target_accuracy, color='red', linestyle='--', 
                               label=f'Target ({self.target_accuracy*100:.0f}%)', linewidth=2)
        self.axes['acc'].legend()
        
        # Plot 3: Character vs Exact Accuracy
        self.axes['char_acc'].set_title('Character vs Exact Match Accuracy', fontweight='bold')
        self.axes['char_acc'].set_xlabel('Epoch')
        self.axes['char_acc'].set_ylabel('Accuracy')
        self.axes['char_acc'].grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate Schedule
        self.axes['lr'].set_title('Learning Rate Schedule', fontweight='bold')
        self.axes['lr'].set_xlabel('Epoch')
        self.axes['lr'].set_ylabel('Learning Rate')
        self.axes['lr'].grid(True, alpha=0.3)
        
        # Plot 5: Batch Loss (Real-time)
        self.axes['batch_loss'].set_title('Batch Loss (Real-time)', fontweight='bold')
        self.axes['batch_loss'].set_xlabel('Batch Number')
        self.axes['batch_loss'].set_ylabel('Loss')
        self.axes['batch_loss'].grid(True, alpha=0.3)
        
        # Plot 6: Training Statistics
        self.axes['stats'].set_title('Training Statistics', fontweight='bold')
        self.axes['stats'].axis('off')
        
        # Plot 7: Live Recognition Examples title in its own dedicated axis
        self.title_ax.clear()
        self.title_ax.axis('off')
        self.title_ax.text(0.5, 0.5, 'Live Recognition Examples', ha='center', va='center', fontsize=14, fontweight='bold')
        
        for ax in self.example_axes:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Use simpler, more robust layout adjustment
            
    def is_minimized(self):
        """Check if the matplotlib window is minimized to prevent blocking."""
        try:
            # This works for the TkAgg backend used in the script.
            # It checks if the window state is 'iconic' (minimized).
            return self.fig.canvas.manager.window.state() == 'iconic'
        except Exception:
            # Fallback for other backends or if window is destroyed.
            # If we can't determine state, assume it's not minimized to be safe.
            return False
            
    def update_batch_loss(self, batch_num, loss):
        """Update batch loss in real-time"""
        self.batch_numbers.append(batch_num)
        self.batch_losses.append(loss)
        
        # Keep only last 500 batches for performance
        if len(self.batch_losses) > 500:
            self.batch_numbers = self.batch_numbers[-500:]
            self.batch_losses = self.batch_losses[-500:]
            
    def update_epoch_data(self, epoch, train_loss, val_loss, val_acc, char_acc, lr):
        """Update epoch-level data"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.char_accuracies.append(char_acc)
        self.learning_rates.append(lr)
        
    def update_plots(self):
        """Update all plots with current data"""
        if self.is_minimized():
            # Skip rendering if the window is minimized to avoid blocking and save resources
            return
            
        try:
            # Clear all main axes
            for key, ax in self.axes.items():
                ax.clear()
            
            self.setup_plots()
            
            if len(self.epochs) > 0:
                # Plot 1: Training & Validation Loss
                self.axes['loss'].plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
                self.axes['loss'].plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
                self.axes['loss'].legend()
                
                # Plot 2: Accuracy Progress
                self.axes['acc'].plot(self.epochs, [acc*100 for acc in self.val_accuracies], 
                                   'g-', label='Validation Accuracy', linewidth=3)
                self.axes['acc'].axhline(y=self.target_accuracy*100, color='red', linestyle='--', 
                                      label=f'Target ({self.target_accuracy*100:.0f}%)', linewidth=2)
                self.axes['acc'].set_ylim(0, 100)
                self.axes['acc'].legend()
                
                # Add progress indicator
                if self.val_accuracies:
                    current_acc = self.val_accuracies[-1] * 100
                    if current_acc >= self.target_accuracy * 100:
                        self.axes['acc'].text(0.5, 0.95, '🎯 TARGET ACHIEVED!', 
                                           transform=self.axes['acc'].transAxes, 
                                           ha='center', va='top', fontsize=12, 
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
                    else:
                        gap = self.target_accuracy * 100 - current_acc
                        self.axes['acc'].text(0.5, 0.95, f'Gap to target: {gap:.1f}%', 
                                           transform=self.axes['acc'].transAxes, 
                                           ha='center', va='top', fontsize=10, 
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
                
                # Plot 3: Character vs Exact Accuracy
                self.axes['char_acc'].plot(self.epochs, [acc*100 for acc in self.val_accuracies], 
                                   'g-', label='Exact Match', linewidth=2)
                self.axes['char_acc'].plot(self.epochs, [acc*100 for acc in self.char_accuracies], 
                                   'b-', label='Character Level', linewidth=2)
                self.axes['char_acc'].set_ylim(0, 100)
                self.axes['char_acc'].legend()
                
                # Plot 4: Learning Rate Schedule
                self.axes['lr'].plot(self.epochs, self.learning_rates, 'purple', linewidth=2)
                self.axes['lr'].set_yscale('log')
                
            # Plot 5: Batch Loss (Real-time)
            if len(self.batch_losses) > 0:
                self.axes['batch_loss'].plot(self.batch_numbers, self.batch_losses, 'orange', alpha=0.7, linewidth=1)
                if len(self.batch_losses) > 10:
                    # Add moving average
                    window = min(50, len(self.batch_losses) // 4)
                    moving_avg = []
                    for i in range(len(self.batch_losses)):
                        start_idx = max(0, i - window + 1)
                        moving_avg.append(np.mean(self.batch_losses[start_idx:i+1]))
                    self.axes['batch_loss'].plot(self.batch_numbers, moving_avg, 'red', linewidth=2, label=f'Moving Avg ({window})')
                    self.axes['batch_loss'].legend()
            
            # Plot 6: Training Statistics
            self.axes['stats'].axis('off')
            stats_text = "Training Statistics\n" + "="*25 + "\n"
            
            if len(self.epochs) > 0:
                current_epoch = self.epochs[-1]
                current_acc = self.val_accuracies[-1] * 100
                current_char_acc = self.char_accuracies[-1] * 100
                current_lr = self.learning_rates[-1]
                
                stats_text += f"Current Epoch: {current_epoch}\n"
                stats_text += f"Validation Accuracy: {current_acc:.2f}%\n"
                stats_text += f"Character Accuracy: {current_char_acc:.2f}%\n"
                stats_text += f"Learning Rate: {current_lr:.2e}\n\n"
                
                # Best performance
                best_acc = max(self.val_accuracies) * 100
                best_epoch = self.epochs[self.val_accuracies.index(max(self.val_accuracies))]
                stats_text += f"Best Accuracy: {best_acc:.2f}%\n"
                stats_text += f"Best Epoch: {best_epoch}\n\n"
                
                # Progress indicators
                if current_acc >= 95:
                    stats_text += "🎯 TARGET ACHIEVED!\n"
                    stats_text += "🏆 Model Ready for Production\n"
                elif current_acc >= 90:
                    stats_text += "🚀 Excellent Progress!\n"
                    stats_text += f"📈 {95-current_acc:.1f}% to target\n"
                elif current_acc >= 80:
                    stats_text += "📈 Good Progress\n"
                    stats_text += f"⏳ {95-current_acc:.1f}% to target\n"
                else:
                    stats_text += "🔄 Training in Progress\n"
                    stats_text += f"⏳ {95-current_acc:.1f}% to target\n"
                    
            else:
                stats_text += "Waiting for training data..."
            
            self.axes['stats'].text(0.05, 0.95, stats_text, transform=self.axes['stats'].transAxes, 
                               va='top', ha='left', fontsize=11, 
                               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Use simpler, more robust layout adjustment
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # allow GUI event loop to process
            
        except Exception as e:
            logger.warning(f"Error updating plots: {e}")

    def update_recognition_examples(self, model, dataloader, char_list, device):
        """Update the recognition examples plot with live predictions."""
        model.eval()
        
        images, true_texts, pred_texts, confs = [], [], [], []
        
        # Ensure we get a consistent set of examples
        if not hasattr(self, 'fixed_examples'):
            self.fixed_examples = []
            for batch in dataloader:
                if len(self.fixed_examples) >= self.num_examples:
                    break
                self.fixed_examples.append(batch)
        
        with torch.no_grad():
            for batch in self.fixed_examples:
                imgs_batch, targets_batch, target_lengths_batch = batch
                imgs_batch = imgs_batch.to(device)
                
                outputs, _ = model(imgs_batch)
                decoded_texts, confidences = decode_ctc_predictions(outputs, char_list)
                
                for i in range(len(decoded_texts)):
                    if len(images) < self.num_examples:
                        img_tensor = imgs_batch[i].cpu()
                        img_tensor = img_tensor * 0.5 + 0.5
                        images.append(img_tensor.permute(1, 2, 0).numpy())
                        
                        target_indices = targets_batch[i][:target_lengths_batch[i]].cpu().tolist()
                        true_text = "".join([char_list[idx] for idx in target_indices if idx != 0])
                        true_texts.append(true_text)
                        pred_texts.append(decoded_texts[i])
                        confs.append(confidences[i])

        for i in range(self.num_examples):
            ax = self.example_axes[i]
            ax.clear()
            ax.axis('off')
            if i < len(images):
                ax.imshow(images[i], cmap='gray')
                is_correct = (true_texts[i] == pred_texts[i])
                color = 'green' if is_correct else 'red'
                title = f"True: {true_texts[i]}\nPred: {pred_texts[i]} ({confs[i]:.2f})"
                ax.set_title(title, color=color, fontsize=10, pad=3)

    def save_plots(self, save_path):
        """Save the current plots"""
        try:
            self.fig.savefig(save_path / 'training_progress.png', dpi=300, bbox_inches='tight')
            logger.info(f"Training plots saved to: {save_path / 'training_progress.png'}")
        except Exception as e:
            logger.warning(f"Error saving plots: {e}")
            
    def close(self):
        """Close the visualization"""
        plt.close(self.fig)

if device.type == 'cuda':
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"Active CUDA Device: GPU {torch.cuda.current_device()}")
    logger.info(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    logger.warning("CUDA is not available. Training will run on CPU, which will be very slow.")
    logger.warning("Please ensure you have a CUDA-enabled GPU and the correct version of PyTorch with CUDA support installed.")

def get_character_set_from_file(file_path):
    """Load character set from all_chars.txt and prepend [blank] token."""
    if not file_path.exists():
        logger.error(f"Character set file not found: {file_path}")
        # Fallback to a generic character set if file not found
        generic_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        logger.warning(f"Using generic character set: {generic_chars}")
        all_characters = list(generic_chars)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_characters = [line.strip() for line in f if line.strip()]
    
    # Ensure [blank] is the first character (index 0) for CTC loss
    if '[blank]' not in all_characters:
        char_set = ['[blank]'] + all_characters
    else: # If already present, ensure it's at the beginning
        all_characters.remove('[blank]')
        char_set = ['[blank]'] + all_characters
        
    logger.info(f"Character set loaded from {file_path if file_path.exists() else 'fallback'}: {''.join(char_set)}")
    logger.info(f"Number of classes (including blank): {len(char_set)}")
    return char_set

class CustomOCRDataset(Dataset):
    def __init__(self, csv_path, img_dir, char_set_list, transform=None, max_samples=None, is_training=True):
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}. Please ensure data_processing.py has run successfully.")
            self.df = pd.DataFrame(columns=['filename', 'words']) # Empty dataframe

        if max_samples:
            self.df = self.df.head(max_samples)
        
        self.img_dir = Path(img_dir)
        self.char_set = char_set_list # Use the list directly
        self.transform = transform
        self.is_training = is_training
        
        # Create character mappings
        self.char2idx = {char: idx for idx, char in enumerate(self.char_set)}
        self.idx2char = {idx: char for idx, char in enumerate(self.char_set)}
        
        # Filter invalid data
        self.df = self.df.dropna(subset=['filename', 'words'])
        self.df['words'] = self.df['words'].astype(str)
        
        logger.info(f"Dataset loaded: {len(self.df)} samples from {csv_path}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['filename']
        
        try:
            # Load image as grayscale as it's already preprocessed
            image = Image.open(img_path).convert('L')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Using blank image.")
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
        
        if self.transform:
            image = self.transform(image)
        
        text = str(row['words']).strip()
        # Clean text - remove common artifacts
        text = text.replace('_', '').replace(' ', '').upper()
        
        # Filter text based on character set
        filtered_text = ''.join([char for char in text if char in self.char2idx])
        
        target = [self.char2idx.get(char, self.char2idx.get('[blank]', 0)) for char in filtered_text]
        if not target: # Handle empty filtered_text
            target = [self.char2idx.get('[blank]', 0)]

        return image, torch.tensor(target, dtype=torch.long), len(target)

# --- Attention-based Decoder (for multi-task learning) ---
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Parameter(torch.rand(decoder_hidden_dim))
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch_size, dec_hid_dim]
        # encoder_outputs = [seq_len, batch_size, enc_hid_dim]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.repeat(src_len, 1, 1) # [src_len, batch_size, dec_hid_dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [src_len, batch_size, dec_hid_dim]

        energy = energy.permute(0, 2, 1) # [src_len, dec_hid_dim, batch_size]
        v = self.v.repeat(batch_size, 1).unsqueeze(1) # [batch_size, 1, dec_hid_dim]
        energy = torch.bmm(v, energy.permute(2, 1, 0)).squeeze(1) # [batch_size, src_len]
        
        return F.softmax(energy, dim=1)

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        # Store hidden dimension so it can be accessed externally for zero-init
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(encoder_hidden_dim + emb_dim, decoder_hidden_dim)
        self.fc_out = nn.Linear(encoder_hidden_dim + decoder_hidden_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch_size]
        # hidden = [1, batch_size, dec_hid_dim]
        # encoder_outputs = [src_len, batch_size, enc_hid_dim]
        input = input.unsqueeze(0) # [1, batch_size]
        
        embedded = self.dropout(self.embedding(input)) # [1, batch_size, emb_dim]
        
        a = self.attention(hidden, encoder_outputs).unsqueeze(1) # [batch_size, 1, src_len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch_size, src_len, enc_hid_dim]
        
        weighted = torch.bmm(a, encoder_outputs) # [batch_size, 1, enc_hid_dim]
        weighted = weighted.permute(1, 0, 2) # [1, batch_size, enc_hid_dim]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden.squeeze(0)

class ImprovedBidirectionalLSTM(nn.Module):
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

class CustomCRNN(nn.Module):
    def __init__(self, img_height, n_classes, n_hidden=256):
        super(CustomCRNN, self).__init__()
        self.cnn = nn.Sequential(
            # Using 1 input channel for grayscale images
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Dropout2d(DROPOUT_RATE * 0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(DROPOUT_RATE * 0.5),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(DROPOUT_RATE * 0.7),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(True), nn.Dropout2d(DROPOUT_RATE)
        )
        
        # Keep multiple rows for multi-line support
        self.adaptive_pool = nn.AdaptiveAvgPool2d((NUM_VERTICAL_ROWS, None))
        self.rnn_input_size = 512 * NUM_VERTICAL_ROWS  # Multiply by number of rows
        self.rnn1 = ImprovedBidirectionalLSTM(self.rnn_input_size, n_hidden, n_hidden, num_layers=2, dropout=DROPOUT_RATE)
        self.rnn2 = ImprovedBidirectionalLSTM(n_hidden, n_hidden, n_classes, num_layers=1, dropout=DROPOUT_RATE)
        
        # Attention Decoder for multi-task learning
        self.attention_decoder = AttentionDecoder(
            output_dim=n_classes,
            emb_dim=256,
            encoder_hidden_dim=n_hidden, # from rnn1
            decoder_hidden_dim=n_hidden,
            dropout=DROPOUT_RATE
        )
        
    def forward(self, input_tensor, targets=None, teacher_forcing_ratio=0.5):
        conv = self.cnn(input_tensor)
        conv = self.adaptive_pool(conv)
        # Reshape to combine channels and height dimensions
        conv = conv.permute(3, 0, 1, 2)  # (W, B, C, H)
        conv = conv.reshape(conv.size(0), conv.size(1), -1)  # (W, B, C*H)
        
        encoder_outputs = self.rnn1(conv)
        ctc_output = self.rnn2(encoder_outputs)

        attention_output = None
        if self.training and targets is not None:
            batch_size = targets.shape[0]
            target_len = targets.shape[1]
            
            # Tensor to store decoder outputs
            attention_outputs = torch.zeros(target_len, batch_size, self.attention_decoder.output_dim).to(device)

            # Initial hidden state for the decoder (zeros)
            decoder_hidden = torch.zeros(batch_size, self.attention_decoder.decoder_hidden_dim).to(device)
            
            # First input to the decoder is the <sos> token (using [blank] index 0)
            decoder_input = torch.zeros(batch_size, dtype=torch.long).to(device)

            for t in range(target_len):
                decoder_output, decoder_hidden = self.attention_decoder(decoder_input, decoder_hidden.unsqueeze(0), encoder_outputs)
                attention_outputs[t] = decoder_output
                
                # Decide if we are going to use teacher forcing
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = decoder_output.argmax(1)
                
                # If teacher forcing, use actual next token; otherwise, use predicted token
                decoder_input = targets[:, t] if teacher_force else top1
            
            attention_output = attention_outputs.permute(1, 0, 2) # [batch_size, target_len, output_dim]

        return ctc_output, attention_output
        output = self.rnn2(output)
        return output

class FocalCTCLoss(nn.Module):
    """
    Focal CTC Loss implementation.
    Reduces loss for well-classified examples, focusing on hard-to-classify ones.
    This makes training more challenging and encourages better generalization.
    """
    def __init__(self, blank_index, alpha=1.0, gamma=1.0, reduction='mean'):
        super(FocalCTCLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Use 'none' reduction to get per-example losses
        self.ctc_loss = nn.CTCLoss(blank=blank_index, reduction='none', zero_infinity=True)
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: Log-probabilities of predictions (T, N, C)
        targets: Ground truth labels (N, S) or (sum(target_lengths))
        input_lengths: Lengths of predictions (N)
        target_lengths: Lengths of ground truth labels (N)
        """
        # Calculate standard CTC loss for each item in the batch
        per_example_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # The probability of the correct sequence is p = exp(-ctc_loss)
        p = torch.exp(-per_example_loss)
        
        # Calculate the Focal component: alpha * (1-p)^gamma
        focal_term = self.alpha * torch.pow(1 - p, self.gamma)
        
        # Modulate the CTC loss by the Focal component
        focal_ctc_loss = focal_term * per_example_loss
        
        if self.reduction == 'mean':
            return focal_ctc_loss.mean()
        elif self.reduction == 'sum':
            return focal_ctc_loss.sum()
        else:
            return focal_ctc_loss

def custom_collate_fn(batch):
    images, targets, lengths = zip(*batch)
    images = torch.stack(images, 0)
    
    valid_targets_info = [(target, length) for target, length in zip(targets, lengths) if length > 0]

    if not valid_targets_info:
        # Handle case where all targets in the batch are empty or invalid
        # Create a dummy batch: images, single blank target, length 1 for each item in batch
        dummy_target_val = 0 # Assuming 0 is the [blank] token index
        padded_targets = torch.full((len(images), 1), dummy_target_val, dtype=torch.long)
        target_lengths = torch.ones(len(images), dtype=torch.long)
        return images, padded_targets, target_lengths

    max_len = max(length for _, length in valid_targets_info)
    padded_targets_list = []
    actual_lengths_list = []

    for target, length in zip(targets, lengths):
        if length > 0:
            padded_target = torch.full((max_len,), 0, dtype=torch.long) # Pad with blank
            padded_target[:length] = target
            padded_targets_list.append(padded_target)
            actual_lengths_list.append(length)
        else: # Handle individual empty targets if any slipped through (should be rare)
            padded_targets_list.append(torch.full((max_len,), 0, dtype=torch.long))
            actual_lengths_list.append(1) # Min length 1 (for blank)

    targets_tensor = torch.stack(padded_targets_list, 0)
    target_lengths_tensor = torch.tensor(actual_lengths_list, dtype=torch.long)
    
    return images, targets_tensor, target_lengths_tensor


def train_epoch_step(model, images, targets, target_lengths, ctc_criterion, attention_criterion, optimizer, scaler=None):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    if scaler: # Mixed precision
        with torch.cuda.amp.autocast():
            ctc_outputs, attention_outputs = model(images, targets, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            
            # 1. CTC Loss
            log_probs = F.log_softmax(ctc_outputs, dim=2)
            input_lengths = torch.full(size=(ctc_outputs.size(1),), fill_value=ctc_outputs.size(0), dtype=torch.long, device=device)
            ctc_loss_val = ctc_criterion(log_probs, targets, input_lengths, target_lengths)
            
            # 2. Attention Loss
            attention_loss_val = 0
            if attention_outputs is not None:
                # Input: (N, C), Target: (N) where N = B * L
                attn_out_flat = attention_outputs.contiguous().view(-1, attention_outputs.shape[-1])
                targets_flat = targets.contiguous().view(-1)
                attention_loss_val = attention_criterion(attn_out_flat, targets_flat)

            # 3. Combined Loss
            combined_loss = (1 - ATTENTION_LOSS_WEIGHT) * ctc_loss_val + ATTENTION_LOSS_WEIGHT * attention_loss_val
        
        if torch.isnan(combined_loss) or torch.isinf(combined_loss) or combined_loss.item() > 100:
            logger.warning(f"Skipping batch due to unstable loss: {combined_loss.item()}")
            return None, None, None
        
        scaler.scale(combined_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else: # No mixed precision
        ctc_outputs, attention_outputs = model(images, targets, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        
        # 1. CTC Loss
        log_probs = F.log_softmax(ctc_outputs, dim=2)
        input_lengths = torch.full(size=(ctc_outputs.size(1),), fill_value=ctc_outputs.size(0), dtype=torch.long, device=device)
        ctc_loss_val = ctc_criterion(log_probs, targets, input_lengths, target_lengths)
        
        # 2. Attention Loss
        attention_loss_val = 0
        if attention_outputs is not None:
            attn_out_flat = attention_outputs.contiguous().view(-1, attention_outputs.shape[-1])
            targets_flat = targets.contiguous().view(-1)
            attention_loss_val = attention_criterion(attn_out_flat, targets_flat)
            
        # 3. Combined Loss
        combined_loss = (1 - ATTENTION_LOSS_WEIGHT) * ctc_loss_val + ATTENTION_LOSS_WEIGHT * attention_loss_val

        if torch.isnan(combined_loss) or torch.isinf(combined_loss) or combined_loss.item() > 100:
            logger.warning(f"Skipping batch due to unstable loss: {combined_loss.item()}")
            return None, None, None

        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        
    attn_loss_item = attention_loss_val.item() if isinstance(attention_loss_val, torch.Tensor) else attention_loss_val
    return combined_loss.item(), ctc_loss_val.item(), attn_loss_item


def decode_ctc_predictions(outputs, char_list, beam_size=10):
    """Decodes CTC predictions using beam search if available, otherwise greedy decoding."""
    if BEAM_SEARCH_AVAILABLE and outputs is not None:
        # Convert to probabilities and numpy array
        probs = torch.softmax(outputs, dim=2).permute(1, 0, 2).cpu().numpy()  # (B, T, C)
        
        decoded_texts = []
        confidences = []
        
        for p in probs:
            # Use beam search
            beam = beam_search(p, vocabulary=char_list, beam_size=beam_size)
            if beam:
                best_text = beam[0][0]
                confidence = beam[0][1]
                decoded_texts.append(best_text)
                confidences.append(confidence)
            else:
                decoded_texts.append("")
                confidences.append(0.0)
        
        return decoded_texts, confidences
    else:
        # Fallback to greedy decoding
        preds_idx = torch.argmax(outputs, dim=2)
        preds_idx = preds_idx.transpose(0, 1).cpu().numpy()
        
        decoded_texts = []
        confidences = []
        probs = torch.softmax(outputs, dim=2).transpose(0,1).cpu().detach().numpy()

        for i in range(preds_idx.shape[0]):
            batch_preds = preds_idx[i]
            batch_probs = probs[i]
            
            text = []
            char_confidence = []
            last_char_idx = 0 
            for t in range(len(batch_preds)):
                char_idx = batch_preds[t]
                if char_idx != 0 and char_idx != last_char_idx:
                    if char_idx < len(char_list):
                        text.append(char_list[char_idx])
                        char_confidence.append(batch_probs[t, char_idx])
                last_char_idx = char_idx
            
            decoded_texts.append("".join(text))
            avg_conf = np.mean(char_confidence) if char_confidence else 0.0
            confidences.append(avg_conf)
        
        return decoded_texts, confidences

def validate_model(model, dataloader, ctc_criterion, char_list, epoch_num):
    model.eval()
    total_ctc_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_confidences = []
    
    char_level_correct = 0
    char_level_total = 0

    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
            images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
            
            outputs, _ = model(images) # Unpack tuple, we only need CTC output for validation
            log_probs = F.log_softmax(outputs, dim=2)
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            
            ctc_loss_val = ctc_criterion(log_probs, targets, input_lengths, target_lengths)
            if not (torch.isnan(ctc_loss_val) or torch.isinf(ctc_loss_val)):
                total_ctc_loss += ctc_loss_val.item()

            # Decode predictions
            decoded_texts, confidences = decode_ctc_predictions(outputs, char_list)
            all_confidences.extend(confidences)

            for i in range(len(decoded_texts)):
                pred_text = decoded_texts[i]
                # Reconstruct target text for comparison
                target_label_indices = targets[i][:target_lengths[i]].cpu().tolist()
                true_text = "".join([char_list[idx] for idx in target_label_indices if idx != 0 and idx < len(char_list)])

                if pred_text == true_text:
                    correct_predictions += 1

                # Character-level accuracy
                for p_char, t_char in zip(pred_text, true_text):
                    if p_char == t_char:
                        char_level_correct += 1
                char_level_total += max(len(pred_text), len(true_text)) # Levenshtein-like denominator for char acc

                if batch_idx == 0 and i < 10: # Log more examples
                    logger.info(f"Epoch {epoch_num} Val Example: Pred='{pred_text}' (Conf: {confidences[i]:.2f}), True='{true_text}'")
            
            total_samples += images.size(0)
            if total_samples > 5000 and epoch_num < 10 : # Limit validation samples for faster early epochs
                break


    avg_val_loss = total_ctc_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    char_accuracy = char_level_correct / char_level_total if char_level_total > 0 else 0
    
    logger.info(f"Validation: Loss={avg_val_loss:.4f}, Exact Acc={accuracy:.4f}, Char Acc={char_accuracy:.4f}, Avg Conf={avg_confidence:.4f}")
    return avg_val_loss, accuracy, char_accuracy


def main():
    logger.info("=== Enhanced Custom CRNN Model Training for ANPR_3 v3 ===")
    
    # Initialize training visualizer (or dummy if disabled)
    visualizer = TrainingVisualizer(target_accuracy=TARGET_ACCURACY) if ENABLE_VISUALIZER else DummyVisualizer()
    if ENABLE_VISUALIZER:
        logger.info("🎨 Training visualizer initialized - Real-time plots will be displayed")
    else:
        logger.info("Visualizer disabled – running in headless mode.")
    
    char_list = get_character_set_from_file(ALL_CHARS_FILE_PATH)
    n_classes = len(char_list)
    
    if n_classes <= 1: # Should have at least blank + one character
        logger.error("Character set is too small or empty. Exiting.")
        return

    # --- Enhanced Data Augmentation for CRNN v3 ---
    # Goal: Simulate real-world conditions like lighting changes, motion blur, and perspective shifts.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomChoice([
            transforms.RandomAffine(degrees=7, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=8, fill=128),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.9, fill=128),
        ]),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.2),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.5)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = CustomOCRDataset(TRAIN_CSV_PATH, TRAIN_IMG_DIR, char_list, train_transform, is_training=True)
    val_dataset = CustomOCRDataset(VAL_CSV_PATH, VAL_IMG_DIR, char_list, val_transform, is_training=False)
    
    # ------------------------------------------------------------------
    # Optional: Use a smaller proxy subset for quick hyper-parameter tuning
    # ------------------------------------------------------------------
    if USE_PROXY_DATA and len(train_dataset) > 0:
        proxy_size = max(1, int(PROXY_FRACTION * len(train_dataset)))
        proxy_indices = np.random.choice(len(train_dataset), proxy_size, replace=False)
        train_dataset = Subset(train_dataset, proxy_indices)
        logger.info(f"Proxy mode ON – using {len(train_dataset)}/{int(1/PROXY_FRACTION)} of training data ({PROXY_FRACTION*100:.1f}%).")

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Please check data paths and CSV files.")
        return

    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty. Validation steps will be skipped, and the model will not be saved based on validation accuracy.")

    # ---- Balanced sampling so each state code is equally represented per epoch ----
    try:
        # Compute frequency of each state code (first two chars of the label)
        state_counts = Counter([w[:2] for w in train_dataset.df['words'] if len(w) >= 2])
        # Assign inverse-frequency weight to each sample
        sample_weights = [1.0 / state_counts[w[:2]] if len(w) >= 2 else 1.0 for w in train_dataset.df['words']]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        logger.info("Using WeightedRandomSampler for balanced state-code distribution in each epoch")
    except Exception as e:
        logger.warning(f"Could not create weighted sampler, falling back to random shuffle: {e}")
        sampler = None
    
    # Enhanced data loaders with better performance
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        shuffle=False if sampler is not None else True, 
        collate_fn=custom_collate_fn, 
        num_workers=0,  # Set to 0 to prevent multiprocessing issues on Windows
        pin_memory=True, 
        persistent_workers=False, # Not needed when num_workers is 0
        drop_last=True  # For more stable training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=0, # Set to 0 to prevent multiprocessing issues on Windows
        pin_memory=True, 
        persistent_workers=False # Not needed when num_workers is 0
    )
    
    # ------------------------------------------------------------------
    #  🔎  Quick per-state validation subset (1 plate per state + BH)
    # ------------------------------------------------------------------
    # Build a light-weight dataset (no augmentations) from the training CSV so
    # we can pick exactly one representative plate for every state code. This
    # allows us to track how the model behaves across the whole geographic
    # distribution each epoch without running a full validation pass.
    state_dataset_full = CustomOCRDataset(
        TRAIN_CSV_PATH, TRAIN_IMG_DIR, char_list, val_transform, is_training=False
    )

    # Map each state code -> first index encountered
    state_sample_indices = {}
    for idx, row in state_dataset_full.df.iterrows():
        label_text = str(row['words']).strip().upper()
        state_code = 'BH' if re.match(r'^\d{2}BH', label_text) else label_text[:2]
        if state_code not in state_sample_indices:
            state_sample_indices[state_code] = idx
        # Break early if we already have all states (incl. BH)
        if len(state_sample_indices) == len(STATE_CODES) + 1:  # +1 for BH
            break

    # Create subset dataloader (batch_size=1 for detailed logging)
    state_subset = Subset(state_dataset_full, list(state_sample_indices.values()))
    state_val_loader = DataLoader(
        state_subset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0  # tiny loader – no need for workers
    )

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {n_classes}")

    # Enhanced model with better architecture and initialization
    model = CustomCRNN(IMAGE_HEIGHT, n_classes, HIDDEN_SIZE).to(device)
    
    # Better weight initialization to prevent mode collapse
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    model.apply(init_weights)
    logger.info("Applied better weight initialization to prevent mode collapse")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # --- Loss function ---
    # Using standard CTCLoss for better numerical stability.
    # FocalCTCLoss can be re-introduced later if needed for hard-example mining.
    logger.info("Using standard nn.CTCLoss for stability.")
    ctc_criterion = nn.CTCLoss(blank=char_list.index('[blank]'), reduction='mean', zero_infinity=True).to(device)
    char_criterion = nn.CrossEntropyLoss(ignore_index=char_list.index('[blank]'), label_smoothing=LABEL_SMOOTHING).to(device)

    # Enhanced optimizer with better parameters
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),  # Adjusted beta2 for better convergence
        eps=1e-8
    )
    
    # OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=EPOCHS * len(train_loader),
                           pct_start=0.15, anneal_strategy='cos', div_factor=20, final_div_factor=1000)

    # Load pretrained model if it exists
    if LOAD_PRETRAINED:
        pretrained_model_path = Paths.CRNN_V2_DIR / "best_multiline_crnn_epoch51_acc0.9966.pth"
        if os.path.exists(pretrained_model_path):
            logger.info(f"Loading pretrained weights from best v2 model: {pretrained_model_path}")
            try:
                checkpoint = torch.load(pretrained_model_path, map_location=device)
                
                # Filter out mismatched keys (e.g., final layer if n_classes changed)
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                   if k in model_dict and model_dict[k].shape == v.shape}
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False) # Use strict=False to ignore non-matching keys
                logger.info(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} keys from pretrained model.")

            except Exception as e:
                logger.error(f"Could not load pretrained model: {e}. Starting from scratch.")
                model.apply(init_weights)
        else:
            logger.info("Pretrained flag is True but no v2 model found. Initializing weights from scratch.")
            model.apply(init_weights)
    else:
        logger.info("Training from scratch: NOT loading any pretrained weights.")
        model.apply(init_weights)

    # Enable Mixed Precision Training (AMP) for better performance on RTX GPUs
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and torch.cuda.is_available() else None
    if scaler:
        logger.info("Using Mixed Precision Training (AMP).")
    else:
        logger.info("Mixed Precision Training (AMP) is DISABLED for stability.")

    # Enhanced early stopping
    best_val_accuracy = 0
    best_char_accuracy = 0
    patience_counter = 0
    epochs_without_improvement = 0
    
    # Training history for analysis
    train_losses = []
    val_accuracies = []
    char_accuracies = []

    logger.info("Starting enhanced training loop...")
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        # Training phase
        running_train_loss = 0.0
        running_ctc_loss = 0.0
        running_char_loss = 0.0
        batches_processed = 0

        model.train()
        try:
            for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
                try:
                    images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
                    
                    loss, ctc_l, attn_l = train_epoch_step(
                        model, images, targets, target_lengths, 
                        ctc_criterion, char_criterion, optimizer, scaler
                    )
                    
                    if loss is not None:
                        running_train_loss += loss
                        running_ctc_loss += ctc_l or 0
                        running_char_loss += attn_l or 0
                        batches_processed += 1
                        
                        # Update batch loss visualization in real-time
                        global_batch_num = (epoch - 1) * len(train_loader) + batch_idx
                        visualizer.update_batch_loss(global_batch_num, loss)
                        
                        # Update plots every 10 batches for real-time feedback
                        if batch_idx % 10 == 0:
                            visualizer.update_plots()

                    scheduler.step() # Step OneCycleLR scheduler per batch

                    # Detailed logging every 100 batches for better monitoring
                    if batch_idx % 100 == 0 and batches_processed > 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        logger.info(f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | "
                                    f"Avg Loss: {running_train_loss/batches_processed:.4f} | "
                                    f"Avg CTC: {running_ctc_loss/batches_processed:.4f} | "
                                    f"Avg Attn: {running_char_loss/batches_processed:.4f} | "
                                    f"LR: {current_lr:.2e}")
                except Exception as batch_error:
                    logger.warning(f"Error in batch {batch_idx}: {batch_error}. Skipping batch.")
                    continue
        except Exception as train_error:
            logger.error(f"Training error in epoch {epoch}: {train_error}")
            # Don't break, continue to next epoch
        
        avg_epoch_train_loss = running_train_loss / batches_processed if batches_processed > 0 else 0
        train_losses.append(avg_epoch_train_loss)
        
        # Validation phase - check if val_loader is not empty to prevent crash
        if len(val_loader) > 0:
            try:
                val_loss, val_accuracy, val_char_accuracy = validate_model(
                    model, val_loader, ctc_criterion, char_list, epoch
                )
            except Exception as e:
                logger.error(f"Validation failed: {e}. Using training loss and 0 accuracy.")
                val_loss, val_accuracy, val_char_accuracy = avg_epoch_train_loss, 0.0, 0.0
        else:
            # If no validation data, use training loss and assume 0 accuracy
            val_loss, val_accuracy, val_char_accuracy = avg_epoch_train_loss, 0.0, 0.0

        val_accuracies.append(val_accuracy)
        char_accuracies.append(val_char_accuracy)

        # ---------------- Optuna pruning ----------------
        if OPTUNA_AVAILABLE and globals().get('optuna_trial', None) is not None:
            trial = globals()['optuna_trial']
            try:
                trial.report(val_accuracy, epoch)
                if trial.should_prune():
                    import optuna
                    logger.info("Optuna decided to prune at epoch %d" % epoch)
                    raise optuna.exceptions.TrialPruned()
            except Exception as _e:
                # If pruning exception bubbles up, higher-level caller should catch
                raise

        avg_ctc_loss_epoch = running_ctc_loss / batches_processed if batches_processed > 0 else 0
        avg_attn_loss_epoch = running_char_loss / batches_processed if batches_processed > 0 else 0

        # Quick per-state subset validation
        try:
            state_loss, state_exact_acc, state_char_acc = validate_model(
                model, state_val_loader, ctc_criterion, char_list, epoch
            )
        except Exception as e:
            logger.warning(f"State-subset validation failed: {e}")
            state_loss, state_exact_acc, state_char_acc = 0.0, 0.0, 0.0

        # Update visualizer with epoch data
        current_lr = optimizer.param_groups[0]['lr']
        visualizer.update_epoch_data(epoch, avg_epoch_train_loss, val_loss, val_accuracy, val_char_accuracy, current_lr)
        
        # Update recognition examples plot before drawing all updates
        visualizer.update_recognition_examples(model, state_val_loader, char_list, device)
        
        # Now, draw all updates to the plots
        visualizer.update_plots()
        
        epoch_duration = time.time() - epoch_start_time
        
        logger.info(f"Epoch {epoch}/{EPOCHS} Summary:")
        logger.info(f"  Train Loss: {avg_epoch_train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info(f"  Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        logger.info(f"  Char Accuracy: {val_char_accuracy:.4f} ({val_char_accuracy*100:.2f}%)")
        logger.info(f"  Avg Train CTC Loss: {avg_ctc_loss_epoch:.4f} | Avg Train Attn Loss: {avg_attn_loss_epoch:.4f}")
        logger.info(f"  State-Subset Accuracy: {state_exact_acc:.4f} ({state_exact_acc*100:.2f}%), Char Acc: {state_char_acc:.4f}")
        logger.info(f"  Duration: {epoch_duration:.2f}s")
        logger.info(f"  Best so far: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
        
        # Check if we've reached target accuracy
        if val_accuracy >= TARGET_ACCURACY:
            logger.info(f"🎯 TARGET ACCURACY ACHIEVED! Validation accuracy: {val_accuracy*100:.2f}%")

        # Enhanced model saving logic with >95% target focus
        improvement = False
        significant_improvement = False
        
        if val_accuracy > best_val_accuracy + MIN_DELTA:
            best_val_accuracy = val_accuracy
            best_char_accuracy = val_char_accuracy
            patience_counter = 0
            epochs_without_improvement = 0
            improvement = True
            
            # Check for significant improvement (>1% jump)
            if val_accuracy > best_val_accuracy + 0.01:
                significant_improvement = True
            
            # Save best model
            model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'char_accuracy': val_char_accuracy,
                'char_set': char_list,
                'model_config': {
                    'img_height': IMAGE_HEIGHT, 
                    'img_width': IMAGE_WIDTH, 
                    'n_classes': n_classes, 
                    'n_hidden': HIDDEN_SIZE,
                    'dropout_rate': DROPOUT_RATE
                },
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'char_accuracies': char_accuracies,
                'hyperparameters': {
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'epochs': EPOCHS,
                    'dropout_rate': DROPOUT_RATE,
                    'weight_decay': WEIGHT_DECAY,
                    'label_smoothing': LABEL_SMOOTHING
                }
            }
            
            save_path = MODEL_SAVE_PATH / f'best_multiline_crnn_epoch{epoch}_acc{val_accuracy:.4f}.pth'
            torch.save(model_checkpoint, save_path)
            
            # Enhanced success messaging
            if val_accuracy >= 0.95:
                logger.info(f"🎯 TARGET ACHIEVED! Validation accuracy: {val_accuracy*100:.2f}%")
                logger.info(f"✓ NEW BEST MODEL SAVED! Path: {save_path}")
            elif val_accuracy >= 0.90:
                logger.info(f"🚀 EXCELLENT PROGRESS! Accuracy: {val_accuracy*100:.2f}% (Target: 95%)")
                logger.info(f"✓ NEW BEST MODEL SAVED! Path: {save_path}")
            else:
                logger.info(f"✓ NEW BEST MODEL SAVED! Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            
            # Save milestone models
            if val_accuracy >= 0.95:
                milestone_path = MODEL_SAVE_PATH / f'milestone_95plus_acc{val_accuracy:.4f}_epoch{epoch}.pth'
                torch.save(model_checkpoint, milestone_path)
                logger.info(f"🏆 MILESTONE MODEL SAVED: {milestone_path}")
                
        else:
            patience_counter += 1
            epochs_without_improvement += 1
            
            # Enhanced progress monitoring
            accuracy_gap = 0.95 - val_accuracy
            if accuracy_gap > 0:
                logger.info(f"Progress: {val_accuracy*100:.2f}% | Gap to target: {accuracy_gap*100:.2f}% | "
                           f"No improvement: {epochs_without_improvement} epochs (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
            else:
                logger.info(f"Maintaining high performance: {val_accuracy*100:.2f}% | "
                           f"No improvement: {epochs_without_improvement} epochs")

        # Progressive early stopping - more patient as we get closer to target
        effective_patience = EARLY_STOPPING_PATIENCE
        if best_val_accuracy >= 0.92:  # Very close to target
            effective_patience = EARLY_STOPPING_PATIENCE + 20
        elif best_val_accuracy >= 0.90:  # Close to target
            effective_patience = EARLY_STOPPING_PATIENCE + 10
        elif best_val_accuracy >= 0.80:  # Good progress
            effective_patience = EARLY_STOPPING_PATIENCE + 5
            
        # Simpler early stopping like the successful version
        max_patience = 20 # Number of epochs to wait for improvement before stopping
        if patience_counter >= effective_patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            logger.info(f"Best validation accuracy achieved: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
            break
            
        # Enhanced checkpointing for high-performance models
        if epoch % 25 == 0 or (val_accuracy >= 0.90 and epoch % 10 == 0):
            checkpoint_path = MODEL_SAVE_PATH / f'checkpoint_epoch_{epoch}_acc{val_accuracy:.3f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'char_accuracy': val_char_accuracy,
                'char_set': char_list
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Gradient norm logging
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        logger.info(f"Gradient Norm: {total_norm:.4f}")

    # Enhanced training completion summary
    logger.info(f"\n{'='*80}")
    logger.info("🎓 ENHANCED MULTI-LINE CRNN TRAINING COMPLETED!")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Final Results:")
    logger.info(f"   🎯 Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    logger.info(f"   📝 Best Character Accuracy: {best_char_accuracy:.4f} ({best_char_accuracy*100:.2f}%)")
    logger.info(f"   📈 Total Training Epochs: {epoch}")
    logger.info(f"   ✅ Target (>95%) Achieved: {'🎉 YES!' if best_val_accuracy >= 0.95 else '⚠️  NO'}")
    
    if best_val_accuracy >= 0.95:
        logger.info(f"   🏆 SUCCESS: Model ready for production use!")
        logger.info(f"   📁 Model saved in: {MODEL_SAVE_PATH}")
    else:
        logger.info(f"   📊 Performance Gap: {(0.95 - best_val_accuracy)*100:.2f}% to target")
        logger.info(f"   💡 Recommendation: Consider extended training or hyperparameter tuning")
    
    # Enhanced model artifacts saving
    char_set_path = MODEL_SAVE_PATH / 'char_set.txt'
    with open(char_set_path, 'w', encoding='utf-8') as f:
        for char_item in char_list:
            f.write(f"{char_item}\n")
    logger.info(f"Character set saved to: {char_set_path}")
    
    # Enhanced training history with performance metrics
    history_path = MODEL_SAVE_PATH / 'training_history.json'
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'char_accuracies': char_accuracies,
        'best_val_accuracy': best_val_accuracy,
        'best_char_accuracy': best_char_accuracy,
        'total_epochs': epoch,
        'target_achieved': best_val_accuracy >= 0.95,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'dropout_rate': DROPOUT_RATE,
            'weight_decay': WEIGHT_DECAY,
            'label_smoothing': LABEL_SMOOTHING,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE
        },
        'dataset_info': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'num_classes': n_classes,
            'image_size': f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}"
        }
    }
    
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Enhanced training history saved to: {history_path}")
    
    # Performance summary file
    summary_path = MODEL_SAVE_PATH / 'performance_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Enhanced Multi-Line CRNN Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)\n")
        f.write(f"Best Character Accuracy: {best_char_accuracy:.4f} ({best_char_accuracy*100:.2f}%)\n")
        f.write(f"Target (>95%) Achieved: {'YES' if best_val_accuracy >= 0.95 else 'NO'}\n")
        f.write(f"Total Training Epochs: {epoch}\n")
        f.write(f"Training Dataset: {len(train_dataset)} samples\n")
        f.write(f"Validation Dataset: {len(val_dataset)} samples\n")
        f.write(f"Multi-line plates detected: 2,018 real plates\n")
        f.write(f"Model Architecture: Enhanced Multi-Line CRNN with Attention\n")
    
    logger.info(f"Performance summary saved to: {summary_path}")
    
    # Save final training plots
    visualizer.save_plots(MODEL_SAVE_PATH)
    
    # Keep plots open for a few seconds to view final results
    logger.info("Training complete! Keeping visualization open for 10 seconds...")
    time.sleep(10)
    
    # Close visualizer
    visualizer.close()
    logger.info("Training visualization closed.")

    # Return best validation accuracy so external HPO scripts can use it
    return best_val_accuracy

# -----------------------------------------------------------
#  External API for hyper-parameter optimisation               
# -----------------------------------------------------------

def apply_hyperparameters(hparams: dict):
    """Inject values from *hparams* into this module's globals."""
    global ENABLE_VISUALIZER, USE_PROXY_DATA, PROXY_FRACTION, LOAD_PRETRAINED
    for key, value in hparams.items():
        if key in globals():
            globals()[key] = value


def train_crnn(hparams: dict | None = None, trial: "optuna.trial.Trial | None" = None):
    """Convenience wrapper so external scripts (Optuna, Ray-Tune, etc.)
    can start a training run with an arbitrary hyper-parameter dict.

    Parameters
    ----------
    hparams : dict
        Keys matching any global in this module will overwrite that global
        before training starts.  Example::

            hparams = {"HIDDEN_SIZE": 384, "LEARNING_RATE": 1e-3}
    trial : optuna.trial.Trial, optional
        When supplied, the training loop will report validation accuracy
        each epoch and will honour trial pruning requests.

    Returns
    -------
    float
        Best validation accuracy achieved during the run (higher is better).
    """
    if hparams is None:
        hparams = {}

    if trial is not None:
        globals()["optuna_trial"] = trial
        # Disable visualiser for HPO – saves a lot of overhead
        hparams.setdefault("ENABLE_VISUALIZER", False)
        # Proxy mode is extremely useful for fast HPO
        hparams.setdefault("USE_PROXY_DATA", True)

    apply_hyperparameters(hparams)

    best_acc = main()

    # Clean-up so subsequent calls start with a blank slate
    globals()["optuna_trial"] = None
    return best_acc


if __name__ == '__main__':
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        plt.close('all')  # Close all matplotlib windows
    except Exception as e:
        logger.error(f"Training failed: {e}")
        plt.close('all')  # Close all matplotlib windows
        raise 

def objective(trial):
    hp = {
        "HIDDEN_SIZE": trial.suggest_categorical("hidden", [256,384,512]),
        "DROPOUT_RATE": trial.suggest_float("drop", 0.2, 0.6),
        "LEARNING_RATE": trial.suggest_float("lr", 5e-4, 3e-3, log=True),
        "USE_PROXY_DATA": True,      # 10 % subset
        "ENABLE_VISUALIZER": False,  # head-less
        "EPOCHS": 80                 # short runs
    }
    return train_crnn(hp, trial)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_params, study.best_value)
