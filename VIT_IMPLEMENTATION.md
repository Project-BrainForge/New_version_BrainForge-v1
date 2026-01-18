# Vision Transformer ESI Implementation

## Overview
Replaced the CNN-Transformer hybrid model with a pure Vision Transformer (ViT) for EEG-to-source inverse problem. The model treats EEG data as spatiotemporal patches, applies dual learnable positional embeddings for channels and time, and uses a TransformerEncoder to model global dependencies.

## Model Architecture

**VisionTransformerESI** class:
- **Input:** (batch, channels=75, time=500)
- **Output:** (batch, time=500, n_sources=994)

### Key Components:
1. **Patch Embedding:** Flattens spatiotemporal patches and projects to d_model dimension
   - Configurable patch sizes: `patch_size_channels` (default 5) and `patch_size_time` (default 10)
   - Total patches: (15 × 50) = 750 tokens

2. **Dual Positional Embeddings:**
   - Separate learnable embeddings for channel positions (0-14) and time positions (0-49)
   - Combined additively before TransformerEncoder

3. **TransformerEncoder:**
   - Configurable attention heads (nhead=8, default)
   - Configurable number of layers (num_transformer_layers=4, default)
   - GELU activation, batch_first=True

4. **Output Projection:**
   - Multi-layer expansion to source space
   - Aggregates over spatial patches, repeats over time, trims to original sequence length

## Usage

### Command Line
```bash
# Train ViT with default settings
python train_model.py \
  --data_dir labeled_spikes_data/labeled_spikes_data \
  --model_type vit

# Train with custom patch sizes and hyperparameters
python train_model.py \
  --data_dir labeled_spikes_data/labeled_spikes_data \
  --model_type vit \
  --patch_size_channels 3 \
  --patch_size_time 5 \
  --d_model 256 \
  --nhead 8 \
  --num_transformer_layers 4 \
  --batch_size 32 \
  --epochs 200 \
  --learning_rate 2e-4
```

### Python API
```python
from train_model import train_model, VisionTransformerESI
import numpy as np

# Load your EEG and source data
eeg_data = np.load('eeg_data.npy')        # (n_samples, 500, 75)
source_data = np.load('source_data.npy')  # (n_samples, 500, 994)

# Configure training
config = {
    'model_type': 'vit',
    'patch_size_channels': 5,
    'patch_size_time': 10,
    'd_model': 256,
    'nhead': 8,
    'num_transformer_layers': 4,
    'dropout': 0.1,
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 2e-4,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'patience': 25,
    'use_augmentation': True,
    'log_dir': 'logs',
    'experiment_name': 'my_experiment',
    'checkpoint_dir': 'checkpoints',
    'checkpoint_interval': 10
}

# Train model
model, train_losses, val_losses = train_model(eeg_data, source_data, config=config)

# Make predictions
from train_model import predict_source_activity
predictions = predict_source_activity(model, eeg_input, device='cuda', model_type='vit')
```

## Key Changes from Hybrid Model

1. **Removed CNN layers:** Direct patch embedding instead of Conv1d layers
2. **Patch-based tokenization:** Treats (channels × time) as 2D grid of patches
3. **Dual positional encodings:** Separate embeddings for spatial (channels) and temporal (time) axes
4. **No hard-coded dimensions:** All key dimensions are parameters (n_channels, n_sources, time_steps, patch sizes)
5. **Data transpose:** EEGSourceDataset now handles (500, 75) → (75, 500) transpose for ViT input

## Training & Evaluation

- **Loss function:** TemporalMSELoss with smoothness regularization (α=0.1)
- **Optimizer:** AdamW with configurable learning rate and weight decay
- **Scheduler:** CosineAnnealingWarmRestarts for learning rate scheduling
- **Logging:** TensorBoard support, checkpoint saving, training history
- **Device:** Automatic GPU/CPU detection

## Output Shapes

| Stage | Shape |
|-------|-------|
| Input (raw EEG) | (batch, 500, 75) |
| After transpose for ViT | (batch, 75, 500) |
| Patches (before embedding) | (batch, 15, 50, 50) |
| Flattened patches | (batch, 750, d_model) |
| After TransformerEncoder | (batch, 750, d_model) |
| Output projection | (batch, 750, 994) |
| Aggregated & repeated | (batch, 500, 994) |

## Files Modified

- **train_model.py:** 
  - Added VisionTransformerESI class
  - Updated train_model() function to support both 'vit' and 'hybrid' model types
  - Added CLI argument parsing with argparse
  - Updated EEGSourceDataset to handle data transpose based on model type
  - Updated predict_source_activity() and main execution block
  - Preserved all existing training infrastructure (optimizer, scheduler, logging)

## Default Hyperparameters

```python
patch_size_channels = 5     # Divide 75 channels into ~15 patches
patch_size_time = 10        # Divide 500 time points into ~50 patches
d_model = 256               # Embedding dimension
nhead = 8                   # Attention heads
num_transformer_layers = 4  # Depth of transformer
dropout = 0.1               # Regularization
batch_size = 32             # Training batch size
learning_rate = 2e-4        # AdamW learning rate
epochs = 200                # Maximum training epochs
patience = 25               # Early stopping patience
```

## Testing

Test ViT forward pass:
```bash
python -c "
import torch
from train_model import VisionTransformerESI

model = VisionTransformerESI(n_channels=75, n_sources=994, time_steps=500)
x = torch.randn(2, 75, 500)
output = model(x)
assert output.shape == (2, 500, 994)
print('✓ ViT test passed!')
"
```

Train on synthetic data:
```bash
python test_vit_training.py
```
