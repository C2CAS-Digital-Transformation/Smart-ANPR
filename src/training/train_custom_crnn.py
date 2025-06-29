import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration for ANPR Project
BASE_PROJECT_DIR = Path(r"D:\Work\Projects\ANPR")
DATA_DIR = BASE_PROJECT_DIR / "data" / "combined_training_data_ocr"
CLEANED_OCR_DATA_DIR = DATA_DIR # For all_chars.txt, assuming it's in the root of the processed data dir

TRAIN_CSV_PATH = DATA_DIR / "train_data" / "labels.csv"
VAL_CSV_PATH = DATA_DIR / "val_data" / "labels.csv"
TRAIN_IMG_DIR = DATA_DIR / "train_data"
VAL_IMG_DIR = DATA_DIR / "val_data"
ALL_CHARS_FILE_PATH = CLEANED_OCR_DATA_DIR / "all_chars.txt"


MODEL_DIR = BASE_PROJECT_DIR / "models" / "ocr"
CRNN_V2_DIR = MODEL_DIR / "crnn_v7"

MODEL_SAVE_PATH = CRNN_V2_DIR
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Enhanced Training parameters (can be tuned)
BATCH_SIZE = 16 
LEARNING_RATE = 1e-3
EPOCHS = 350
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256 # Adjust if your license plates have a very different aspect ratio
HIDDEN_SIZE = 256 

# Enhanced regularization parameters
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 0.5

# Character-level loss weighting
CHAR_LOSS_WEIGHT = 0.3
CTC_LOSS_WEIGHT = 0.7

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

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
            self.df = pd.DataFrame(columns=['filename', 'words', 'plate_type']) # Empty dataframe

        if max_samples:
            self.df = self.df.head(max_samples)
        
        self.img_dir = Path(img_dir)
        self.char_set = char_set_list # Use the list directly
        self.transform = transform
        self.is_training = is_training
        
        # Create character mappings
        self.char2idx = {char: idx for idx, char in enumerate(self.char_set)}
        self.idx2char = {idx: char for idx, char in enumerate(self.char_set)}
        
        # Filter invalid data and handle missing plate_type for robustness
        self.df = self.df.dropna(subset=['filename', 'words'])
        self.df['words'] = self.df['words'].astype(str)
        if 'plate_type' not in self.df.columns:
            self.df['plate_type'] = 'unknown'
        self.df['plate_type'] = self.df['plate_type'].fillna('unknown')
        
        logger.info(f"Dataset loaded: {len(self.df)} samples from {csv_path}")
        
    def __len__(self):
        return len(self.df)
    
    def preprocess_image(self, image):
        """Enhanced image preprocessing for license plates"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3: # Color image
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img_array = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        elif len(img_array.shape) == 2: # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_array = clahe.apply(img_array)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB) # Convert to RGB for consistency

        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_array = cv2.filter2D(img_array, -1, kernel)
        
        return Image.fromarray(img_array)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['filename']
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.is_training: # Apply more aggressive preprocessing for training
                image = self.preprocess_image(image)
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Using blank image.")
            image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        text = str(row['words']).strip() # Changed 'label' to 'words'
        filtered_text = ''.join([char for char in text if char in self.char2idx]) # Filter based on char2idx keys
        
        target = [self.char2idx.get(char, self.char2idx.get('[blank]', 0)) for char in filtered_text] # Default to blank if char not found
        if not target: # Handle empty filtered_text
            target = [self.char2idx.get('[blank]', 0)]

        plate_type = row.get('plate_type', 'unknown')
        return image, torch.tensor(target, dtype=torch.long), len(target), plate_type

class ImprovedBidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(ImprovedBidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=False, # CRNN typically has sequence dim first for RNN
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size * 2, output_size) # *2 for bidirectional
        self.dropout_layer = nn.Dropout(dropout) # Renamed to avoid conflict

    def forward(self, input_tensor):
        recurrent, _ = self.rnn(input_tensor)
        output = self.dropout_layer(recurrent)
        output = self.linear(output)
        return output

class CustomCRNN(nn.Module):
    def __init__(self, img_height, n_classes, n_hidden=256):
        super(CustomCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Dropout2d(DROPOUT_RATE * 0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 64x32x128

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(DROPOUT_RATE * 0.5),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 128x16x64

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(DROPOUT_RATE * 0.7),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 256x8x64

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 512x4x64
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(True), nn.Dropout2d(DROPOUT_RATE) # 512x3x63
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None)) # Output H will be 1
        
        # Calculate RNN input size based on CNN output
        # After CNN: (batch, 512, H_out, W_out)
        # H_out from adaptive_pool is 1.
        # So, features per time step for RNN is 512 * 1 = 512
        self.rnn_input_size = 512 
        self.rnn1 = ImprovedBidirectionalLSTM(self.rnn_input_size, n_hidden // 2, n_hidden // 2, num_layers=2, dropout=DROPOUT_RATE)
        self.rnn2 = ImprovedBidirectionalLSTM(n_hidden // 2, n_hidden // 2, n_classes, num_layers=1, dropout=DROPOUT_RATE)
        
    def forward(self, input_tensor):
        conv = self.cnn(input_tensor)
        conv = self.adaptive_pool(conv)
        conv = conv.squeeze(2) # Remove height dimension (H=1): (batch, channels, width)
        conv = conv.permute(2, 0, 1) # (width, batch, channels) for RNN
        
        output = self.rnn1(conv)
        output = self.rnn2(output) # Output shape: (seq_len, batch, num_classes)
        return output

def custom_collate_fn(batch):
    images, targets, lengths, plate_types = zip(*batch)
    images = torch.stack(images, 0)
    
    valid_targets_info = [(target, length) for target, length in zip(targets, lengths) if length > 0]

    if not valid_targets_info:
        # Handle case where all targets in the batch are empty or invalid
        # Create a dummy batch: images, single blank target, length 1 for each item in batch
        dummy_target_val = 0 # Assuming 0 is the [blank] token index
        padded_targets = torch.full((len(images), 1), dummy_target_val, dtype=torch.long)
        target_lengths = torch.ones(len(images), dtype=torch.long)
        return images, padded_targets, target_lengths, plate_types

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
    
    return images, targets_tensor, target_lengths_tensor, plate_types


def train_epoch_step(model, images, targets, target_lengths, ctc_criterion, char_criterion, optimizer, scaler=None):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    if scaler: # Mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(images) # (seq_len, batch, num_classes)
            log_probs = F.log_softmax(outputs, dim=2) # CTC loss expects log_softmax
            
            # CTC Loss
            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            ctc_loss_val = ctc_criterion(log_probs, targets, input_lengths, target_lengths)
            
            # Character-level auxiliary loss
            char_loss_val = 0
            if outputs.size(0) > 1 and CHAR_LOSS_WEIGHT > 0: # Ensure sequence length for char loss
                for i in range(targets.size(0)): # Iterate over batch
                    if target_lengths[i] > 0:
                        seq_len_eff = min(outputs.size(0), target_lengths[i])
                        if seq_len_eff > 0: # Ensure there's something to compare
                            # output_slice: (effective_seq_len, num_classes)
                            # target_slice: (effective_seq_len)
                            output_slice = outputs[:seq_len_eff, i, :] 
                            target_slice = targets[i, :seq_len_eff]
                            char_loss_val += char_criterion(output_slice, target_slice)
                char_loss_val = char_loss_val / targets.size(0) if targets.size(0) > 0 else 0

            combined_loss = CTC_LOSS_WEIGHT * ctc_loss_val + CHAR_LOSS_WEIGHT * char_loss_val
        
        if torch.isnan(combined_loss) or torch.isinf(combined_loss) or combined_loss.item() > 50: # Skip unstable gradients
            logger.warning(f"Skipping batch due to unstable loss: {combined_loss.item()}")
            return None, None, None

        scaler.scale(combined_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else: # No mixed precision
        outputs = model(images)
        log_probs = F.log_softmax(outputs, dim=2)
        input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
        ctc_loss_val = ctc_criterion(log_probs, targets, input_lengths, target_lengths)
        
        char_loss_val = 0
        if outputs.size(0) > 1 and CHAR_LOSS_WEIGHT > 0:
            for i in range(targets.size(0)):
                if target_lengths[i] > 0:
                    seq_len_eff = min(outputs.size(0), target_lengths[i])
                    if seq_len_eff > 0:
                        output_slice = outputs[:seq_len_eff, i, :]
                        target_slice = targets[i, :seq_len_eff]
                        char_loss_val += char_criterion(output_slice, target_slice)
            char_loss_val = char_loss_val / targets.size(0) if targets.size(0) > 0 else 0
            
        combined_loss = CTC_LOSS_WEIGHT * ctc_loss_val + CHAR_LOSS_WEIGHT * char_loss_val

        if torch.isnan(combined_loss) or torch.isinf(combined_loss) or combined_loss.item() > 50:
            logger.warning(f"Skipping batch due to unstable loss: {combined_loss.item()}")
            return None, None, None

        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        
    return combined_loss.item(), ctc_loss_val.item(), (char_loss_val.item() if isinstance(char_loss_val, torch.Tensor) else char_loss_val)


def decode_ctc_predictions(outputs, char_list):
    """Decodes CTC predictions. outputs shape: (seq_len, batch, num_classes)"""
    preds_idx = torch.argmax(outputs, dim=2)  # (seq_len, batch)
    preds_idx = preds_idx.transpose(0, 1).cpu().numpy()  # (batch, seq_len)
    
    decoded_texts = []
    confidences = []

    probs = torch.softmax(outputs, dim=2).transpose(0,1).cpu().detach().numpy() # (batch, seq_len, num_classes)

    for i in range(preds_idx.shape[0]): # Iterate over batch
        batch_preds = preds_idx[i]
        batch_probs = probs[i]
        
        text = []
        char_confidence = []
        last_char_idx = 0 
        for t in range(len(batch_preds)):
            char_idx = batch_preds[t]
            if char_idx != 0 and char_idx != last_char_idx : # Not blank and not repeated
                if char_idx < len(char_list): # Check index bounds
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
    logged_examples_by_type = set()

    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths, plate_types) in enumerate(dataloader):
            images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
            
            outputs = model(images) # (seq_len, batch, num_classes)
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
                plate_type = plate_types[i]
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

                # Log one example from each plate type
                if plate_type not in logged_examples_by_type:
                    logger.info(f"Epoch {epoch_num} Val Example [{plate_type.upper()}]: Pred='{pred_text}' (Conf: {confidences[i]:.2f}), True='{true_text}'")
                    logged_examples_by_type.add(plate_type)
            
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
    logger.info("=== Custom CRNN Model Training for ANPR_3 ===")
    
    char_list = get_character_set_from_file(ALL_CHARS_FILE_PATH)
    n_classes = len(char_list)
    
    if n_classes <= 1: # Should have at least blank + one character
        logger.error("Character set is too small or empty. Exiting.")
        return

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=3),
        transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize for RGB
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize for RGB
    ])
    
    train_dataset = CustomOCRDataset(TRAIN_CSV_PATH, TRAIN_IMG_DIR, char_list, train_transform)
    val_dataset = CustomOCRDataset(VAL_CSV_PATH, VAL_IMG_DIR, char_list, val_transform)
    
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Please check data paths and CSV files.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True, persistent_workers=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=2, pin_memory=True, persistent_workers=True if device.type == 'cuda' else False)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {n_classes}")

    model = CustomCRNN(IMAGE_HEIGHT, n_classes, HIDDEN_SIZE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    ctc_criterion = nn.CTCLoss(blank=char_list.index('[blank]'), reduction='mean', zero_infinity=True).to(device)
    char_criterion = nn.CrossEntropyLoss(ignore_index=char_list.index('[blank]'), label_smoothing=LABEL_SMOOTHING).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=EPOCHS * len(train_loader),
                           pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and torch.cuda.is_available() else None
    if scaler:
        logger.info("Using Mixed Precision Training (AMP).")

    best_val_accuracy = 0
    patience_counter = 0
    max_patience = 20 # Number of epochs to wait for improvement before stopping

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        
        running_train_loss = 0.0
        running_ctc_loss = 0.0
        running_char_loss = 0.0
        batches_processed = 0

        for batch_idx, (images, targets, target_lengths, _) in enumerate(train_loader):
            images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
            
            loss, ctc_l, char_l = train_epoch_step(model, images, targets, target_lengths, ctc_criterion, char_criterion, optimizer, scaler)
            
            if loss is not None:
                running_train_loss += loss
                running_ctc_loss += ctc_l or 0
                running_char_loss += char_l or 0
                batches_processed += 1

            scheduler.step() # Step OneCycleLR scheduler per batch

            if batch_idx % 100 == 0 and batches_processed > 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | "
                            f"Avg Loss: {running_train_loss/batches_processed:.4f} | "
                            f"Avg CTC: {running_ctc_loss/batches_processed:.4f} | "
                            f"Avg Char: {running_char_loss/batches_processed:.4f} | "
                            f"LR: {current_lr:.2e}")
        
        avg_epoch_train_loss = running_train_loss / batches_processed if batches_processed > 0 else 0
        logger.info(f"Epoch {epoch} Training Avg Loss: {avg_epoch_train_loss:.4f}")
        
        val_loss, val_accuracy, val_char_accuracy = validate_model(model, val_loader, ctc_criterion, char_list, epoch)
        
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch} completed in {epoch_duration:.2f}s. Val Acc: {val_accuracy:.4f}, Val Char Acc: {val_char_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_accuracy,
                'char_set': char_list,
                'model_config': {'img_height': IMAGE_HEIGHT, 'img_width': IMAGE_WIDTH, 'n_classes': n_classes, 'n_hidden': HIDDEN_SIZE}
            }
            torch.save(model_checkpoint, MODEL_SAVE_PATH / f'best_crnn_model_epoch{epoch}_acc{val_accuracy:.4f}.pth', _use_new_zipfile_serialization=False)
            logger.info(f"✓ New best model saved! Accuracy: {val_accuracy:.4f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement in val accuracy for {patience_counter} epochs.")

        if patience_counter >= max_patience:
            logger.info(f"Early stopping triggered after {max_patience} epochs without improvement.")
            break
            
        if (epoch) % 10 == 0 : # Save checkpoint every 10 epochs
             torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'char_set': char_list
            }, MODEL_SAVE_PATH / f'checkpoint_epoch_{epoch}.pth', _use_new_zipfile_serialization=False)
             logger.info(f"Checkpoint saved for epoch {epoch}")


    logger.info(f"Training finished. Best Validation Accuracy: {best_val_accuracy:.4f}")
    char_set_path = MODEL_SAVE_PATH / 'char_set.txt'
    with open(char_set_path, 'w', encoding='utf-8') as f:
        for char_item in char_list: # Iterate over items in char_list
            f.write(f"{char_item}\\n") # Write each item
    logger.info(f"Character set saved to: {char_set_path}")

if __name__ == '__main__':
    main() 