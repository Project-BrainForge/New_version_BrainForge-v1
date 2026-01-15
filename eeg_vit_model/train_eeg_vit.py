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
import sys

# Add parent directory to path for shared utilities
sys.path.append(str(Path(__file__).parent.parent))
from train_model import (
    setup_logging, save_training_history, save_config,
    EEGSourceDataset, TemporalAugmentation, load_mat_files,
    evaluate_temporal_metrics, predict_source_activity
)

warnings.filterwarnings('ignore')

# Try to import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# ============================================
# EEG Vision Transformer (EEG-ViT)
# ============================================

class EEGVisionTransformer(nn.Module):
    """
    Vision Transformer adapted for EEG signal processing
    Treats each time point as a patch with all EEG channels
    """
    def __init__(self, 
                 input_dim=75,           # EEG channels
                 output_dim=994,         # Source activity channels
                 time_points=500,        # Fixed time dimension
                 patch_size=5,           # Number of time points per patch
                 d_model=256,            # Embedding dimension
                 depth=8,                # Number of transformer blocks
                 heads=8,                # Number of attention heads
                 mlp_ratio=4,            # MLP expansion ratio
                 dropout=0.1,
                 emb_dropout=0.1,
                 use_cls_token=True,     # Use CLS token for global features
                 use_sin_pos_emb=True,   # Use sinusoidal positional encoding
                 layer_scale=True):      # Layer scale for stability
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_points = time_points
        self.patch_size = patch_size
        self.use_cls_token = use_cls_token
        
        # Calculate number of patches
        self.num_patches = time_points // patch_size
        assert time_points % patch_size == 0, "Time points must be divisible by patch size"
        
        # Patch dimensions
        patch_dim = input_dim * patch_size
        
        # ============================================
        # 1. Patch Embedding
        # ============================================
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ============================================
        # 2. CLS Token
        # ============================================
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            num_tokens = self.num_patches + 1
        else:
            num_tokens = self.num_patches
        
        # ============================================
        # 3. Positional Encoding
        # ============================================
        if use_sin_pos_emb:
            self.pos_embedding = SinusoidalPositionalEncoding(
                d_model, max_len=num_tokens, dropout=emb_dropout
            )
        else:
            self.pos_embedding = LearnablePositionalEmbedding(
                num_tokens, d_model, dropout=emb_dropout
            )
        
        # ============================================
        # 4. Transformer Encoder Blocks
        # ============================================
        self.transformer = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                heads=heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                layer_scale=layer_scale
            ) for _ in range(depth)
        ])
        
        # ============================================
        # 5. Layer Normalization
        # ============================================
        self.norm = nn.LayerNorm(d_model)
        
        # ============================================
        # 6. Output Head (Time-distributed)
        # ============================================
        # We need to reconstruct full time series from patches
        self.output_head = nn.ModuleList([
            # Expand patch features
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            
            # Process expanded features
            nn.Sequential(
                nn.Linear(d_model * 2, d_model * 4),
                nn.LayerNorm(d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            
            # Project to output dimension per patch
            nn.Sequential(
                nn.Linear(d_model * 4, output_dim * patch_size),
                nn.LayerNorm(output_dim * patch_size)
            )
        ])
        
        # Optional: Time smoothing layer
        self.temporal_smoothing = nn.Conv1d(
            output_dim, output_dim, 
            kernel_size=3, padding=1, groups=output_dim
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embedding[0].weight)
        nn.init.zeros_(self.patch_embedding[0].bias)
        
        # Initialize CLS token
        if self.use_cls_token:
            nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize output layers
        for layer in self.output_head:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, time_points=500, input_dim=75)
        Returns:
            output: (batch_size, time_points=500, output_dim=994)
        """
        batch_size = x.shape[0]
        
        # ============================================
        # 1. Create Patches
        # ============================================
        # Reshape to patches: (batch, num_patches, patch_size * input_dim)
        patches = x.view(
            batch_size, 
            self.num_patches, 
            self.patch_size * self.input_dim
        )
        
        # ============================================
        # 2. Patch Embedding
        # ============================================
        patch_embeddings = self.patch_embedding(patches)  # (batch, num_patches, d_model)
        
        # ============================================
        # 3. Add CLS Token
        # ============================================
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        else:
            embeddings = patch_embeddings
        
        # ============================================
        # 4. Add Positional Embedding
        # ============================================
        embeddings = self.pos_embedding(embeddings)
        
        # ============================================
        # 5. Apply Transformer Blocks
        # ============================================
        for transformer_block in self.transformer:
            embeddings = transformer_block(embeddings)
        
        # ============================================
        # 6. Layer Normalization
        # ============================================
        embeddings = self.norm(embeddings)
        
        # ============================================
        # 7. Prepare for Output
        # ============================================
        if self.use_cls_token:
            # Use only patch tokens, discard CLS token
            patch_outputs = embeddings[:, 1:, :]  # (batch, num_patches, d_model)
        else:
            patch_outputs = embeddings
        
        # ============================================
        # 8. Process Through Output Head
        # ============================================
        # Expand features
        x_out = patch_outputs
        for layer in self.output_head:
            x_out = layer(x_out)
        
        # Reshape to full time series
        # x_out: (batch, num_patches, output_dim * patch_size)
        output = x_out.view(
            batch_size, 
            self.num_patches * self.patch_size, 
            self.output_dim
        )  # (batch, time_points, output_dim)
        
        # Ensure correct time dimension (500)
        if output.shape[1] != self.time_points:
            output = F.interpolate(
                output.transpose(1, 2), 
                size=self.time_points, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        # ============================================
        # 9. Optional Temporal Smoothing
        # ============================================
        if self.temporal_smoothing is not None:
            output = self.temporal_smoothing(output.transpose(1, 2)).transpose(1, 2)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with optional layer scale"""
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1, layer_scale=True):
        super().__init__()
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(dim)
        
        # MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.mlp_norm = nn.LayerNorm(dim)
        
        # Layer scale (from CaiT)
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(torch.ones(dim))
            self.gamma2 = nn.Parameter(torch.ones(dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention
        attn_input = self.attention_norm(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        if self.layer_scale:
            x = x + self.dropout(self.gamma1 * attn_output)
        else:
            x = x + self.dropout(attn_output)
        
        # MLP
        mlp_input = self.mlp_norm(x)
        mlp_output = self.mlp(mlp_input)
        if self.layer_scale:
            x = x + self.dropout(self.gamma2 * mlp_output)
        else:
            x = x + self.dropout(mlp_output)
        
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embeddings"""
    def __init__(self, num_tokens, d_model, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, d_model))
        self.dropout = nn.Dropout(p=dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        return x + self.dropout(self.pos_embed)


# ============================================
# Alternative: EEG ViT with Channel Attention
# ============================================

class EEGViTWithChannelAttention(nn.Module):
    """
    Enhanced ViT with channel-wise attention before patch creation
    """
    def __init__(self, input_dim=75, output_dim=994, time_points=500,
                 patch_size=5, d_model=256, depth=6, heads=8, 
                 channel_reduction=16, dropout=0.1):
        super().__init__()
        
        # ============================================
        # 1. Channel Attention Module
        # ============================================
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_dim, input_dim // channel_reduction),
            nn.ReLU(),
            nn.Linear(input_dim // channel_reduction, input_dim),
            nn.Sigmoid()
        )
        
        # ============================================
        # 2. Channel-wise Projection
        # ============================================
        self.proj_dim = d_model // 2
        self.channel_projection = nn.Linear(input_dim, self.proj_dim)
        
        # ============================================
        # 3. Patch Creation and Embedding
        # ============================================
        self.num_patches = time_points // patch_size
        patch_dim = self.proj_dim * patch_size
        
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # ============================================
        # 4. Positional Encoding
        # ============================================
        self.pos_embedding = LearnablePositionalEmbedding(
            self.num_patches + 1, d_model, dropout
        )
        
        # ============================================
        # 5. Transformer
        # ============================================
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=d_model, heads=heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # ============================================
        # 6. Output Reconstruction
        # ============================================
        # Multi-scale output reconstruction
        self.output_reconstruction = nn.ModuleList([
            # Decoder block 1
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            # Decoder block 2
            nn.Sequential(
                nn.Linear(d_model * 2, d_model * 4),
                nn.LayerNorm(d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            # Project to output
            nn.Sequential(
                nn.Linear(d_model * 4, output_dim * patch_size),
                nn.Tanh()  # Optional activation
            )
        ])
        
        # Temporal upsampling if needed
        if time_points != self.num_patches * patch_size:
            self.upsample = nn.Sequential(
                nn.Upsample(size=time_points, mode='linear', align_corners=False),
                nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
            )
        else:
            self.upsample = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for block in self.output_reconstruction:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        
        # ============================================
        # 1. Channel Attention
        # ============================================
        # Compute channel weights
        x_transposed = x.transpose(1, 2)  # (batch, channels, time)
        channel_weights = self.channel_attention(x_transposed)
        channel_weights = channel_weights.unsqueeze(-1)  # (batch, channels, 1)
        
        # Apply channel attention
        x_weighted = x_transposed * channel_weights
        x_weighted = x_weighted.transpose(1, 2)  # (batch, time, channels)
        
        # ============================================
        # 2. Channel Projection
        # ============================================
        x_proj = self.channel_projection(x_weighted)  # (batch, 500, d_model//2)
        
        # ============================================
        # 3. Create Patches
        # ============================================
        patches = x_proj.view(
            batch_size,
            self.num_patches,
            self.patch_size * self.proj_dim
        )
        
        # ============================================
        # 4. Patch Embedding
        # ============================================
        patch_embeddings = self.patch_embedding(patches)
        
        # ============================================
        # 5. Add CLS Token and Position
        # ============================================
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        embeddings = self.pos_embedding(embeddings)
        
        # ============================================
        # 6. Transformer Blocks
        # ============================================
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        embeddings = self.norm(embeddings)
        
        # ============================================
        # 7. Output Reconstruction (use patch tokens)
        # ============================================
        patch_outputs = embeddings[:, 1:, :]  # Remove CLS token
        
        # Process through decoder
        x_out = patch_outputs
        for block in self.output_reconstruction:
            x_out = block(x_out)
        
        # Reshape to full time series
        output = x_out.view(
            batch_size,
            self.num_patches * self.patch_size,
            self.output_dim
        )
        
        # Upsample if needed
        if self.upsample is not None:
            output = self.upsample(output.transpose(1, 2)).transpose(1, 2)
        
        return output


# ============================================
# Hybrid: CNN + ViT
# ============================================

class CNNViTHybrid(nn.Module):
    """
    CNN for local feature extraction + ViT for global context
    """
    def __init__(self, input_dim=75, output_dim=994, time_points=500,
                 cnn_channels=[64, 128, 256], vit_dim=256, 
                 patch_size=5, depth=4, heads=8, dropout=0.1):
        super().__init__()
        
        # ============================================
        # 1. CNN Feature Extractor
        # ============================================
        self.cnn = nn.Sequential(
            # Conv block 1
            nn.Conv1d(input_dim, cnn_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Conv block 2
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Conv block 3
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ============================================
        # 2. Project to ViT dimension
        # ============================================
        self.vit_dim = vit_dim
        self.cnn_projection = nn.Linear(cnn_channels[2], vit_dim)
        
        # ============================================
        # 3. ViT for global context
        # ============================================
        self.num_patches = time_points // patch_size
        patch_dim = vit_dim * patch_size
        
        self.patch_embedding = nn.Linear(patch_dim, vit_dim)
        self.pos_embedding = LearnablePositionalEmbedding(
            self.num_patches + 1, vit_dim, dropout
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=vit_dim, heads=heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(vit_dim)
        
        # ============================================
        # 4. Output Head
        # ============================================
        self.output_head = nn.Sequential(
            nn.Linear(vit_dim, vit_dim * 2),
            nn.LayerNorm(vit_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(vit_dim * 2, output_dim * patch_size),
            nn.LayerNorm(output_dim * patch_size)
        )
        
        # Reshape layer
        self.reshape = nn.Sequential(
            nn.Linear(output_dim * patch_size, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # CNN feature extraction
        x_cnn = x.transpose(1, 2)  # (batch, channels, time)
        x_cnn = self.cnn(x_cnn)  # (batch, cnn_channels[-1], time)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, time, cnn_channels[-1])
        
        # Project to ViT dimension
        x_proj = self.cnn_projection(x_cnn)  # (batch, 500, vit_dim)
        
        # Create patches
        patches = x_proj.view(
            batch_size,
            self.num_patches,
            self.patch_size * self.vit_dim
        )
        
        # Patch embedding
        patch_embeddings = self.patch_embedding(patches)
        
        # Add CLS token and position
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        embeddings = self.pos_embedding(embeddings)
        
        # Transformer
        for block in self.transformer_blocks:
            embeddings = block(embeddings)
        
        embeddings = self.norm(embeddings)
        
        # Use patch tokens for output
        patch_outputs = embeddings[:, 1:, :]
        
        # Output head
        x_out = self.output_head(patch_outputs)
        
        # Reshape to full time series
        output = x_out.view(
            batch_size,
            self.num_patches * self.patch_size,
            self.output_dim
        )
        
        # Final projection
        output = self.reshape(output)
        
        return output


# ============================================
# Training and Evaluation
# ============================================

def train_eeg_vit(eeg_data, source_data, config=None):
    """
    Train EEG Vision Transformer with comprehensive logging and checkpointing
    """
    if config is None:
        config = {
            'model_type': 'eeg_vit',  # 'eeg_vit', 'vit_channel', 'cnn_vit'
            'batch_size': 32,
            'epochs': 200,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'dropout': 0.1,
            'patch_size': 5,
            'd_model': 256,
            'depth': 8,
            'heads': 8,
            'warmup_epochs': 10,
            'use_mixup': False,
            'mixup_alpha': 0.2,
            'gradient_clip': 1.0,
            'patience': 25,
            'use_augmentation': True,
            'checkpoint_dir': 'checkpoints',
            'checkpoint_interval': 10,
            'log_dir': 'logs',
            'experiment_name': None
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
    
    # Create model
    if config['model_type'] == 'eeg_vit':
        model = EEGVisionTransformer(
            input_dim=75,
            output_dim=994,
            time_points=500,
            patch_size=config['patch_size'],
            d_model=config['d_model'],
            depth=config['depth'],
            heads=config['heads'],
            dropout=config['dropout']
        )
    elif config['model_type'] == 'vit_channel':
        model = EEGViTWithChannelAttention(
            input_dim=75,
            output_dim=994,
            time_points=500,
            patch_size=config['patch_size'],
            d_model=config['d_model'],
            depth=config['depth'],
            heads=config['heads'],
            dropout=config['dropout']
        )
    elif config['model_type'] == 'cnn_vit':
        model = CNNViTHybrid(
            input_dim=75,
            output_dim=994,
            time_points=500,
            patch_size=config['patch_size'],
            vit_dim=config['d_model'],
            depth=config['depth'],
            heads=config['heads'],
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss with frequency domain regularization
    class SpectralLoss(nn.Module):
        def __init__(self, alpha=0.1):
            super().__init__()
            self.mse = nn.MSELoss()
            self.alpha = alpha
        
        def forward(self, pred, target):
            # Time domain loss
            time_loss = self.mse(pred, target)
            
            # Frequency domain loss (encourages similar spectral properties)
            pred_fft = torch.fft.rfft(pred, dim=1)
            target_fft = torch.fft.rfft(target, dim=1)
            
            # Magnitude loss
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
            freq_loss = self.mse(pred_mag, target_mag)
            
            return time_loss + self.alpha * freq_loss
    
    criterion = SpectralLoss(alpha=0.1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    # Scheduler with warmup
    def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = warmup_cosine_scheduler(
        optimizer, config['warmup_epochs'], config['epochs']
    )
    
    # Set up logging
    log_dir = config.get('log_dir', 'logs')
    experiment_name = config.get('experiment_name', None)
    logger, exp_log_dir, tb_writer = setup_logging(log_dir, experiment_name)
    
    # Save configuration
    save_config(exp_log_dir, config)
    
    # Create checkpoint directory
    checkpoint_dir = exp_log_dir / config.get('checkpoint_dir', 'checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_interval = config.get('checkpoint_interval', 10)
    
    # Mixup augmentation
    def mixup_data(x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y, lam
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    learning_rates = []
    
    logger.info("=" * 60)
    logger.info("Starting Training - EEG Vision Transformer")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Model type: {config['model_type']}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Patch size: {config['patch_size']}")
    logger.info(f"Model depth: {config['depth']}")
    logger.info(f"Attention heads: {config['heads']}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}/")
    logger.info(f"Checkpoint interval: Every {checkpoint_interval} epochs")
    logger.info(f"Log directory: {exp_log_dir}")
    logger.info("=" * 60)
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_idx, (eeg, source) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]', leave=False)):
            eeg, source = eeg.to(device), source.to(device)
            
            # Apply mixup if enabled
            if config.get('use_mixup', False):
                eeg, source, lam = mixup_data(eeg, source, config['mixup_alpha'])
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(eeg)
            loss = criterion(predictions, source)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        train_losses.append(train_loss)
        
        # Log to TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for eeg, source in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Val]', leave=False):
                eeg, source = eeg.to(device), source.to(device)
                predictions = model(eeg)
                loss = criterion(predictions, source)
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
            best_model_path = checkpoint_dir / f'best_{config["model_type"]}_model.pth'
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
    best_model_path = checkpoint_dir / f'best_{config["model_type"]}_model.pth'
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
# Main Execution
# ============================================

# ============================================
# Command Line Arguments
# ============================================

def parse_arguments():
    """Parse command line arguments for EEG-ViT training"""
    parser = argparse.ArgumentParser(description='Train EEG Vision Transformer for Source Activity Mapping')
    
    # Model type
    parser.add_argument('--model_type', type=str, default='eeg_vit',
                       choices=['eeg_vit', 'vit_channel', 'cnn_vit'],
                       help='Type of model to train (eeg_vit, vit_channel, cnn_vit)')
    
    # Model architecture
    parser.add_argument('--patch_size', type=int, default=5, help='Number of time points per patch')
    parser.add_argument('--d_model', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=8, help='Number of transformer blocks')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    
    # Data augmentation
    parser.add_argument('--use_augmentation', type=bool, default=True, help='Use data augmentation')
    parser.add_argument('--use_mixup', type=bool, default=False, help='Use mixup augmentation')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (auto-generated if None)')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, default='labeled_spikes_data/labeled_spikes_data', help='Path to data directory')
    
    return parser.parse_args()


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Build config from arguments
    optimal_config = {
        'model_type': args.model_type,
        'patch_size': args.patch_size,
        'd_model': args.d_model,
        'depth': args.depth,
        'heads': args.heads,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'gradient_clip': args.gradient_clip,
        'patience': args.patience,
        'use_augmentation': args.use_augmentation,
        'use_mixup': args.use_mixup,
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
    print("Training EEG Vision Transformer")
    print("=" * 60)
    
    # Train the model
    model, train_losses, val_losses = train_eeg_vit(eeg_data, source_data, config=optimal_config)
    
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

