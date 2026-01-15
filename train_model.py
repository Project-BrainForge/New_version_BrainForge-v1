import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import warnings
import scipy.io
import os
import json
import csv
import logging
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Try to import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# ============================================
# CNN + Transformer Hybrid Model
# ============================================

class CNNTransformerHybrid(nn.Module):
    """
    Hybrid CNN-Transformer for EEG to Source Activity Mapping
    CNN extracts local temporal features, Transformer captures long-range dependencies
    """
    def __init__(self, input_dim=75, output_dim=994, 
                 cnn_channels=[75, 128, 256],  # Channel progression for CNN
                 kernel_sizes=[3, 5, 3],       # Kernel sizes for each CNN layer
                 d_model=256,                  # Transformer dimension
                 nhead=8,                      # Number of attention heads
                 num_transformer_layers=4,     # Number of transformer layers
                 dropout=0.1,
                 use_residual=True):           # Use residual connections
        super().__init__()
        
        self.use_residual = use_residual
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # ============================================
        # 1. CNN Feature Extractor (Local Temporal Patterns)
        # ============================================
        self.cnn_layers = nn.ModuleList()
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels[1:], kernel_sizes)):
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, 
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.cnn_layers.append(conv_layer)
            in_channels = out_channels
        
        # Projection to transformer dimension if needed
        if in_channels != d_model:
            self.cnn_projection = nn.Linear(in_channels, d_model)
            self.has_cnn_projection = True
        else:
            self.has_cnn_projection = False
        
        # ============================================
        # 2. Positional Encoding
        # ============================================
        self.positional_encoding = LearnablePositionalEncoding(
            d_model=d_model, 
            max_len=500, 
            dropout=dropout
        )
        
        # ============================================
        # 3. Transformer Encoder (Long-range Dependencies)
        # ============================================
        transformer_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_layers,
            num_layers=num_transformer_layers
        )
        
        # ============================================
        # 4. Multi-scale Feature Fusion
        # ============================================
        # Optional: Direct skip connection from input to output
        if self.use_residual:
            if input_dim != d_model:
                self.residual_projection = nn.Linear(input_dim, d_model)
            else:
                self.residual_projection = None
        
        # ============================================
        # 5. Output Projection (Dimensionality Expansion)
        # ============================================
        # Multi-layer projection to handle 75 -> 994 expansion
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len=500, input_dim=75)
        Returns:
            output: (batch_size, seq_len=500, output_dim=994)
        """
        batch_size, seq_len, _ = x.shape
        
        # Store original for residual connection
        if self.use_residual:
            x_residual = x
        
        # ============================================
        # CNN Feature Extraction
        # ============================================
        # Convert to channel-first for CNN: (batch, channels, seq_len)
        x_cnn = x.transpose(1, 2)  # (batch, 75, 500)
        
        # Apply CNN layers
        for conv_layer in self.cnn_layers:
            x_cnn = conv_layer(x_cnn)
        
        # Convert back to sequence-first: (batch, seq_len, features)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, 500, cnn_output_channels)
        
        # Project to transformer dimension if needed
        if self.has_cnn_projection:
            x_cnn = self.cnn_projection(x_cnn)  # (batch, 500, d_model)
        
        # ============================================
        # Add Residual Connection (Optional)
        # ============================================
        if self.use_residual:
            if self.residual_projection is not None:
                x_residual = self.residual_projection(x_residual)
            # Element-wise addition
            x_combined = x_cnn + x_residual
        else:
            x_combined = x_cnn
        
        # ============================================
        # Positional Encoding
        # ============================================
        x_combined = self.positional_encoding(x_combined)
        
        # ============================================
        # Transformer for Long-range Dependencies
        # ============================================
        # Create attention mask (optional, for causal masking if needed)
        attention_mask = None  # Use full attention
        
        # Apply transformer
        transformer_out = self.transformer_encoder(
            x_combined, 
            mask=attention_mask
        )
        
        # ============================================
        # Output Projection
        # ============================================
        output = self.output_projection(transformer_out)
        
        return output


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding instead of fixed sinusoidal
    """
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.position_emb, std=0.02)
    
    def forward(self, x):
        x = x + self.position_emb[:, :x.size(1), :]
        return self.dropout(x)


# ============================================
# Logging Utilities
# ============================================

def setup_logging(log_dir, experiment_name=None):
    """
    Set up comprehensive logging system
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name for this experiment (optional)
    
    Returns:
        logger: Logger instance
        log_dir: Path to log directory
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment subdirectory with timestamp
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    exp_log_dir = log_dir / experiment_name
    exp_log_dir.mkdir(exist_ok=True)
    
    # Set up file logger
    log_file = exp_log_dir / 'training.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log directory: {exp_log_dir}")
    
    # Set up TensorBoard if available
    tb_writer = None
    if TENSORBOARD_AVAILABLE:
        tb_log_dir = exp_log_dir / 'tensorboard'
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info(f"TensorBoard logging enabled. Run: tensorboard --logdir {tb_log_dir}")
    
    return logger, exp_log_dir, tb_writer


def save_training_history(log_dir, train_losses, val_losses, metrics_history=None):
    """
    Save training history to JSON and CSV files
    
    Args:
        log_dir: Directory to save history files
        train_losses: List of training losses
        val_losses: List of validation losses
        metrics_history: Optional dict of additional metrics
    """
    log_dir = Path(log_dir)
    
    # Save as JSON
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_epochs': len(train_losses),
        'best_train_loss': min(train_losses) if train_losses else None,
        'best_val_loss': min(val_losses) if val_losses else None,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None
    }
    
    if metrics_history:
        history.update(metrics_history)
    
    json_file = log_dir / 'training_history.json'
    with open(json_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save as CSV
    csv_file = log_dir / 'training_history.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            writer.writerow([i+1, train_loss, val_loss])
    
    logging.info(f"Training history saved to {json_file} and {csv_file}")


def save_config(log_dir, config):
    """
    Save training configuration to JSON file
    
    Args:
        log_dir: Directory to save config
        config: Configuration dictionary
    """
    log_dir = Path(log_dir)
    config_file = log_dir / 'config.json'
    
    # Convert Path objects to strings for JSON serialization
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, Path):
            config_serializable[key] = str(value)
        else:
            config_serializable[key] = value
    
    with open(config_file, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    
    logging.info(f"Configuration saved to {config_file}")


# ============================================
# Dataset and Training Utilities
# ============================================

class EEGSourceDataset(Dataset):
    def __init__(self, eeg_data, source_data, transform=None):
        """
        Args:
            eeg_data: (n_samples, 500, 75)
            source_data: (n_samples, 500, 994)
        """
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.source_data = torch.FloatTensor(source_data)
        self.transform = transform
        
        # Validate dimensions
        assert len(self.eeg_data.shape) == 3, "EEG data must be 3D"
        assert len(self.source_data.shape) == 3, "Source data must be 3D"
        assert self.eeg_data.shape[0] == self.source_data.shape[0], "Sample count mismatch"
        assert self.eeg_data.shape[1] == 500, "Time points must be 500"
        assert self.source_data.shape[1] == 500, "Time points must be 500"
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        source = self.source_data[idx]
        
        if self.transform:
            eeg = self.transform(eeg)
        
        return eeg, source


class TemporalAugmentation:
    """Augmentation for temporal data"""
    def __init__(self, noise_std=0.01, dropout_p=0.1, time_warp_scale=0.1):
        self.noise_std = noise_std
        self.dropout_p = dropout_p
        self.time_warp_scale = time_warp_scale
    
    def __call__(self, x):
        # Add Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        
        # Random time dropout (mask random time steps)
        if self.dropout_p > 0 and torch.rand(1) < 0.5:
            mask = torch.rand(x.shape[0]) > self.dropout_p
            x = x * mask.unsqueeze(1).float()
        
        return x


# ============================================
# Data Loading Functions
# ============================================

def load_mat_files(data_dir):
    """
    Load all MAT files from the directory and extract EEG and source data
    
    Args:
        data_dir: Path to directory containing MAT files
    
    Returns:
        eeg_data: numpy array of shape (n_samples, 500, 75)
        source_data: numpy array of shape (n_samples, 500, 994)
    """
    data_dir = Path(data_dir)
    mat_files = sorted([f for f in data_dir.glob('*.mat') if 'sample_' in f.name])
    
    print(f"Found {len(mat_files)} sample MAT files")
    
    eeg_list = []
    source_list = []
    skipped = 0
    
    print("Loading MAT files...")
    for mat_file in tqdm(mat_files):
        try:
            data = scipy.io.loadmat(str(mat_file))
            
            # Check if required keys exist
            if 'eeg_data' not in data or 'source_data' not in data:
                skipped += 1
                continue
            
            # Extract eeg_data and source_data
            eeg = data['eeg_data']  # Shape: (500, 75)
            source = data['source_data']  # Shape: (500, 994)
            
            # Ensure correct shape and type
            eeg = np.array(eeg, dtype=np.float32)
            source = np.array(source, dtype=np.float32)
            
            # Verify shapes
            if eeg.shape != (500, 75):
                skipped += 1
                continue
            if source.shape != (500, 994):
                skipped += 1
                continue
            
            eeg_list.append(eeg)
            source_list.append(source)
            
        except Exception as e:
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"Skipped {skipped} files (missing keys or wrong shape)")
    
    # Stack into arrays
    eeg_data = np.stack(eeg_list, axis=0)  # (n_samples, 500, 75)
    source_data = np.stack(source_list, axis=0)  # (n_samples, 500, 994)
    
    print(f"\nLoaded data shapes:")
    print(f"EEG data: {eeg_data.shape}")
    print(f"Source data: {source_data.shape}")
    
    return eeg_data, source_data


def train_hybrid_model(eeg_data, source_data, config=None):
    """
    Train the CNN-Transformer hybrid model
    """
    if config is None:
        config = {
            'model_type': 'hybrid',  # 'hybrid' or 'time_distributed'
            'batch_size': 16,
            'epochs': 150,
            'learning_rate': 3e-4,
            'weight_decay': 1e-5,
            'dropout': 0.15,
            'gradient_clip': 1.0,
            'patience': 20,
            'use_augmentation': True,
            'checkpoint_dir': 'checkpoints',  # Directory to save checkpoints
            'checkpoint_interval': 10  # Save checkpoint every N epochs
        }
    
    # Prepare data
    dataset = EEGSourceDataset(eeg_data, source_data)
    
    # Split train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Apply augmentation to training data
    if config.get('use_augmentation', True):
        # Create a wrapper dataset that applies augmentation
        class AugmentedDataset(Dataset):
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                eeg, source = self.base_dataset[idx]
                if self.transform:
                    eeg = self.transform(eeg)
                return eeg, source
        
        transform = TemporalAugmentation(noise_std=0.01, dropout_p=0.1)
        train_dataset = AugmentedDataset(train_dataset, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model
    model = CNNTransformerHybrid(
        input_dim=75,
        output_dim=994,
        cnn_channels=[75, 128, 256, 256],
        kernel_sizes=[3, 5, 3],
        d_model=256,
        nhead=8,
        num_transformer_layers=4,
        dropout=config['dropout'],
        use_residual=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with temporal smoothness regularization
    class TemporalMSELoss(nn.Module):
        def __init__(self, alpha=0.1):
            super().__init__()
            self.mse = nn.MSELoss()
            self.alpha = alpha  # Weight for temporal smoothness
        
        def forward(self, pred, target):
            mse_loss = self.mse(pred, target)
            
            # Temporal smoothness loss
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            smooth_loss = self.mse(pred_diff, target_diff)
            
            return mse_loss + self.alpha * smooth_loss
    
    criterion = TemporalMSELoss(alpha=0.1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    # Set up logging
    log_dir = config.get('log_dir', 'logs')
    experiment_name = config.get('experiment_name', None)
    logger, exp_log_dir, tb_writer = setup_logging(log_dir, experiment_name)
    
    # Save configuration
    save_config(exp_log_dir, config)
    
    # Create checkpoint directory (inside log directory)
    checkpoint_dir = exp_log_dir / config.get('checkpoint_dir', 'checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_interval = config.get('checkpoint_interval', 10)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    learning_rates = []
    
    logger.info("=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}/")
    logger.info(f"Checkpoint interval: Every {checkpoint_interval} epochs")
    logger.info(f"Log directory: {exp_log_dir}")
    logger.info("=" * 60)
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        for batch_eeg, batch_source in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]', leave=False):
            batch_eeg = batch_eeg.to(device)
            batch_source = batch_source.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_eeg)
            loss = criterion(predictions, batch_source)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        train_losses.append(train_loss)
        
        # Log to TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch_eeg, batch_source in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Val]', leave=False):
                batch_eeg = batch_eeg.to(device)
                batch_source = batch_source.to(device)
                
                predictions = model(batch_eeg)
                loss = criterion(predictions, batch_source)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        val_losses.append(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Log to TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
            tb_writer.add_scalar('Learning_Rate', current_lr, epoch)
            tb_writer.add_scalars('Loss/Comparison', {
                'Train': train_loss,
                'Validation': val_loss
            }, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_path = checkpoint_dir / 'best_hybrid_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'config': config
            }, best_model_path)
            logger.info(f'✓ Best model saved! (val_loss: {best_val_loss:.6f})')
            if tb_writer is not None:
                tb_writer.add_scalar('Best/Validation_Loss', best_val_loss, epoch)
        else:
            patience_counter += 1
        
        # Save regular checkpoint
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == config['epochs']:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'config': config
            }, checkpoint_path)
            logger.info(f'✓ Checkpoint saved: {checkpoint_path}')
        
        # Log progress
        logger.info(f'Epoch [{epoch+1:03d}/{config["epochs"]}] | '
                   f'Train Loss: {train_loss:.6f} | '
                   f'Val Loss: {val_loss:.6f} | '
                   f'Best Val Loss: {best_val_loss:.6f} | '
                   f'LR: {current_lr:.2e} | '
                   f'Patience: {patience_counter}/{config["patience"]}')
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            # Save final checkpoint before stopping
            final_checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}_final.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'config': config,
                'early_stopped': True
            }, final_checkpoint_path)
            logger.info(f'✓ Final checkpoint saved: {final_checkpoint_path}')
            break
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{len(train_losses):03d}_final.pth'
    torch.save({
        'epoch': len(train_losses) - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'config': config,
        'early_stopped': False
    }, final_checkpoint_path)
    logger.info(f'✓ Final checkpoint saved: {final_checkpoint_path}')
    
    # Save training history
    save_training_history(exp_log_dir, train_losses, val_losses, {
        'learning_rates': learning_rates
    })
    
    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()
    
    # Load best model
    best_model_path = checkpoint_dir / 'best_hybrid_model.pth'
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best model: {best_model_path}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final train loss: {train_losses[-1]:.6f}")
    logger.info(f"Final val loss: {val_losses[-1]:.6f}")
    logger.info(f"Total epochs: {len(train_losses)}")
    logger.info(f"All logs saved to: {exp_log_dir}")
    logger.info("=" * 60)
    
    return model, train_losses, val_losses


# ============================================
# Resume Training from Checkpoint
# ============================================

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load a checkpoint and resume training state
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        optimizer: Optimizer instance (optional)
        scheduler: Scheduler instance (optional)
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
    
    return checkpoint


# ============================================
# Inference and Evaluation
# ============================================

def predict_source_activity(model, eeg_input, device='auto'):
    """
    Predict source activity from EEG input
    """
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Add batch dimension if needed
        if len(eeg_input.shape) == 2:
            eeg_input = eeg_input.unsqueeze(0)
        
        eeg_input = eeg_input.to(device)
        predictions = model(eeg_input)
    
    return predictions.cpu().numpy()


def evaluate_temporal_metrics(predictions, targets):
    """
    Evaluate temporal consistency metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = {}
    
    # 1. Mean Squared Error
    metrics['mse'] = np.mean((predictions - targets) ** 2)
    
    # 2. Temporal Correlation
    pred_flat = predictions.reshape(-1, predictions.shape[-1])
    target_flat = targets.reshape(-1, targets.shape[-1])
    
    # Compute correlation for each source
    correlations = []
    for i in range(predictions.shape[-1]):
        corr = np.corrcoef(pred_flat[:, i], target_flat[:, i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    metrics['mean_correlation'] = np.mean(correlations) if correlations else 0
    metrics['median_correlation'] = np.median(correlations) if correlations else 0
    
    # 3. Temporal Smoothness (Mean of absolute second derivative)
    pred_diff2 = np.diff(predictions, n=2, axis=1)
    target_diff2 = np.diff(targets, n=2, axis=1)
    
    metrics['pred_smoothness'] = np.mean(np.abs(pred_diff2))
    metrics['target_smoothness'] = np.mean(np.abs(target_diff2))
    metrics['smoothness_ratio'] = metrics['pred_smoothness'] / metrics['target_smoothness'] if metrics['target_smoothness'] > 0 else 0
    
    # 4. Phase Consistency (using Hilbert transform)
    try:
        from scipy.signal import hilbert
        pred_analytic = hilbert(predictions, axis=1)
        target_analytic = hilbert(targets, axis=1)
        pred_phase = np.angle(pred_analytic)
        target_phase = np.angle(target_analytic)
        phase_diff = np.abs(pred_phase - target_phase) % (2*np.pi)
        phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
        metrics['mean_phase_error'] = np.mean(phase_diff)
    except:
        metrics['mean_phase_error'] = None
    
    return metrics


# ============================================
# Command Line Arguments
# ============================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CNN-Transformer Hybrid Model for EEG Source Activity Mapping')
    
    # Model configuration
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (auto-generated if None)')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, default='labeled_spikes_data/labeled_spikes_data', help='Path to data directory')
    parser.add_argument('--use_augmentation', type=bool, default=True, help='Use data augmentation')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=256, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers')
    
    return parser.parse_args()


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Build config from arguments
    best_config = {
        'model_type': 'hybrid',
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'gradient_clip': args.gradient_clip,
        'patience': args.patience,
        'use_augmentation': args.use_augmentation,
        'checkpoint_dir': args.checkpoint_dir,
        'checkpoint_interval': args.checkpoint_interval,
        'log_dir': args.log_dir,
        'experiment_name': args.experiment_name
    }
    
    # Load data from MAT files
    data_dir = Path(args.data_dir)
    print("=" * 60)
    print("Loading Dataset from MAT Files")
    print("=" * 60)
    
    eeg_data, source_data = load_mat_files(data_dir)
    
    print("\n" + "=" * 60)
    print("Training CNN-Transformer Hybrid Model")
    print("=" * 60)
    
    # Train the model
    model, train_losses, val_losses = train_hybrid_model(eeg_data, source_data, config=best_config)
    
    # Test inference on a few samples
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_indices = np.random.choice(len(eeg_data), min(5, len(eeg_data)), replace=False)
    test_eeg = torch.FloatTensor(eeg_data[test_indices])
    test_source = source_data[test_indices]
    
    predictions = predict_source_activity(model, test_eeg, device=device)
    
    print(f"Input shape: {test_eeg.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # Evaluate metrics
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    metrics = evaluate_temporal_metrics(predictions, test_source)
    for key, value in metrics.items():
        if value is not None:
            print(f"{key}: {value:.4f}")
    
    # Log evaluation metrics
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Evaluation Metrics")
    logger.info("=" * 60)
    for key, value in metrics.items():
        if value is not None:
            logger.info(f"{key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"All logs and checkpoints saved to: logs/")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")

