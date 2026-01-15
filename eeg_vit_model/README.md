# EEG Vision Transformer (EEG-ViT) Model

This folder contains the EEG Vision Transformer implementation for mapping EEG signals (75 channels) to source activity (994 sources).

## Features

- **Vision Transformer Architecture**: Adapts ViT for temporal EEG signal processing
- **Multiple Model Variants**:
  - `eeg_vit`: Standard EEG Vision Transformer
  - `vit_channel`: ViT with channel attention
  - `cnn_vit`: Hybrid CNN + ViT model
- **Spectral Loss**: Combines time-domain and frequency-domain losses
- **Comprehensive Logging**: File logs, TensorBoard, and training history
- **Checkpoint Saving**: Regular checkpoints and best model saving

## Installation

Install dependencies (same as main project):

```bash
pip install -r ../requirements.txt
```

## Usage

### Training the Model

Run the training script from the project root:

```bash
cd eeg_vit_model
python train_eeg_vit.py
```

Or from the project root:

```bash
python eeg_vit_model/train_eeg_vit.py
```

The script will:
1. Load all MAT files from `labeled_spikes_data/labeled_spikes_data/`
2. Split data into train/validation sets (80/20)
3. Train the EEG Vision Transformer model
4. Save checkpoints regularly and the best model
5. Evaluate on test samples and print metrics

### Configuration

The default configuration uses optimal settings:

```python
optimal_config = {
    'model_type': 'eeg_vit',      # 'eeg_vit', 'vit_channel', 'cnn_vit'
    'patch_size': 5,              # 100 patches of 5 time points each
    'd_model': 256,               # Embedding dimension
    'depth': 8,                   # 8 transformer blocks
    'heads': 8,                   # 8 attention heads
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 200,
    'dropout': 0.1,
    'weight_decay': 1e-5,
    'warmup_epochs': 10,
    'gradient_clip': 1.0,
    'patience': 25,
    'use_augmentation': True,
    'checkpoint_dir': 'checkpoints',
    'checkpoint_interval': 10,
    'log_dir': 'logs',
    'experiment_name': None       # Auto-generate with timestamp
}
```

### Model Variants

1. **EEG-ViT** (`eeg_vit`): Standard Vision Transformer
   - Patch-based processing
   - CLS token for global features
   - Learnable or sinusoidal positional encoding

2. **ViT with Channel Attention** (`vit_channel`):
   - Channel-wise attention before patch creation
   - Enhanced feature extraction

3. **CNN-ViT Hybrid** (`cnn_vit`):
   - CNN for local feature extraction
   - ViT for global context

## Model Architecture

### EEG Vision Transformer

1. **Patch Creation**: Divides 500 time points into patches (default: 100 patches of 5 time points)
2. **Patch Embedding**: Linear projection to embedding dimension
3. **Positional Encoding**: Learnable or sinusoidal positional embeddings
4. **Transformer Blocks**: Multi-head self-attention + MLP
5. **Output Head**: Multi-layer projection to reconstruct full time series

### Loss Function

Uses **Spectral Loss** combining:
- Time-domain MSE loss
- Frequency-domain magnitude loss (encourages similar spectral properties)

## Output Files

All outputs are saved to `logs/experiment_YYYYMMDD_HHMMSS/`:

- **`training.log`**: Complete training log
- **`config.json`**: Training configuration
- **`training_history.json`**: Full training history (JSON)
- **`training_history.csv`**: Training history (CSV)
- **`tensorboard/`**: TensorBoard logs
- **`checkpoints/`**: Model checkpoints
  - `best_eeg_vit_model.pth`: Best model
  - `checkpoint_epoch_XXX.pth`: Regular checkpoints
  - `checkpoint_epoch_XXX_final.pth`: Final checkpoint

## Using the Trained Model

```python
from eeg_vit_model.train_eeg_vit import EEGVisionTransformer, predict_source_activity
import torch

# Initialize model
model = EEGVisionTransformer(
    input_dim=75,
    output_dim=994,
    time_points=500,
    patch_size=5,
    d_model=256,
    depth=8,
    heads=8,
    dropout=0.1
)

# Load checkpoint
checkpoint = torch.load('logs/experiment_XXX/checkpoints/best_eeg_vit_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
eeg_input = torch.FloatTensor(eeg_data)  # Shape: (batch, 500, 75)
predictions = predict_source_activity(model, eeg_input)
# predictions shape: (batch, 500, 994)
```

## Requirements

Same as main project:
- Python 3.7+
- PyTorch 1.12+
- NumPy 1.21+
- SciPy 1.7+
- tqdm 4.62+
- tensorboard 2.8+ (optional)

## Notes

- The model expects input sequences of length 500 time steps
- Input dimension: 75 (EEG channels)
- Output dimension: 994 (source locations)
- Patch size must divide evenly into 500 (default: 5 → 100 patches)
- Training uses warmup cosine scheduler
- Early stopping based on validation loss plateau

