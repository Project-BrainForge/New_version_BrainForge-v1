#!/usr/bin/env python
"""Test loading checkpoint and making predictions"""

import torch
import numpy as np
from train_model import VisionTransformerESI, load_checkpoint, predict_source_activity, load_mat_files
from pathlib import Path

# Load data
print("Loading data...")
eeg_data, source_data = load_mat_files(Path('labeled_spikes_data'))

# Create model
print("Creating model...")
model = VisionTransformerESI(
    n_channels=75,
    n_sources=994,
    time_steps=500,
    patch_size_channels=5,
    patch_size_time=10,
    d_model=256,
    nhead=8,
    num_transformer_layers=4,
    dropout=0.1
)

# Load checkpoint
checkpoint_path = 'logs/quick_test/checkpoints/best_vit_model.pth'
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = load_checkpoint(checkpoint_path, model)

print(f"Model loaded from epoch {checkpoint.get('epoch')}")
print(f"Best val loss from checkpoint: {checkpoint.get('best_val_loss'):.6f}")

# Test prediction
print("\nTesting prediction...")
test_eeg = torch.FloatTensor(eeg_data[:3]).transpose(1, 2)  # (3, 75, 500)
predictions = predict_source_activity(model, test_eeg, device='cpu', model_type='vit')

print(f"Predictions shape: {predictions.shape}")
print(f"Expected shape: (3, 500, 994)")
assert predictions.shape == (3, 500, 994), f"Shape mismatch: {predictions.shape}"

print("✓ All tests passed!")
