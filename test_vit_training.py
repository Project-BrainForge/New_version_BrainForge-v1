#!/usr/bin/env python
"""Test Vision Transformer model training"""

import numpy as np
import torch
from train_model import train_model

# Create minimal test data
n_samples = 5
eeg_data = np.random.randn(n_samples, 500, 75).astype(np.float32)
source_data = np.random.randn(n_samples, 500, 994).astype(np.float32)

print('Testing Vision Transformer model training...')
print(f'EEG shape: {eeg_data.shape}')
print(f'Source shape: {source_data.shape}')

# Configure for ViT
config = {
    'model_type': 'vit',
    'patch_size_channels': 5,
    'patch_size_time': 10,
    'd_model': 128,
    'nhead': 4,
    'num_transformer_layers': 2,
    'dropout': 0.1,
    'batch_size': 2,
    'epochs': 2,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'patience': 5,
    'use_augmentation': False,
    'log_dir': 'test_logs',
    'experiment_name': 'test_vit',
    'checkpoint_dir': 'checkpoints',
    'checkpoint_interval': 1
}

try:
    model, train_losses, val_losses = train_model(eeg_data, source_data, config=config)
    print('✓ Training completed successfully!')
    print(f'Train losses: {train_losses}')
    print(f'Val losses: {val_losses}')
except Exception as e:
    print(f'✗ Training failed: {e}')
    import traceback
    traceback.print_exc()
