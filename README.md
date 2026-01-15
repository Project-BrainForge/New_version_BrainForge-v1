# CNN-Transformer Hybrid Model for EEG to Source Activity Mapping

This implementation provides a hybrid CNN-Transformer architecture for mapping EEG signals (75 channels) to source activity (994 sources) with temporal consistency.

## Features

- **Hybrid Architecture**: Combines CNN for local temporal feature extraction and Transformer for long-range dependencies
- **Temporal Smoothness**: Includes temporal smoothness regularization in the loss function
- **Data Augmentation**: Supports temporal data augmentation (noise injection, time dropout)
- **Comprehensive Evaluation**: Includes temporal correlation, smoothness, and phase consistency metrics

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The code expects MAT files in the following structure:
```
labeled_spikes_data/
  labeled_spikes_data/
    sample_00000.mat
    sample_00001.mat
    ...
```

Each MAT file should contain:
- `eeg_data`: numpy array of shape (500, 75) - EEG signals
- `source_data`: numpy array of shape (500, 994) - Source activity (ground truth)

## Usage

### Training the Model

Run the training script:

```bash
python train_model.py
```

The script will:
1. Load all MAT files from `labeled_spikes_data/labeled_spikes_data/`
2. Split data into train/validation sets (80/20)
3. Train the CNN-Transformer hybrid model
4. Save checkpoints regularly (every N epochs) and the best model
5. Evaluate on test samples and print metrics

### Configuration

You can modify the configuration in the `train_model.py` file:

```python
best_config = {
    'model_type': 'hybrid',
    'batch_size': 32,          # Adjust based on GPU memory
    'epochs': 200,
    'learning_rate': 2e-4,
    'weight_decay': 1e-5,
    'dropout': 0.2,            # Higher dropout for regularization
    'gradient_clip': 1.0,
    'patience': 25,            # Early stopping patience
    'use_augmentation': True,  # Enable data augmentation
    'checkpoint_dir': 'checkpoints',  # Directory to save checkpoints
    'checkpoint_interval': 10,  # Save checkpoint every N epochs
    'log_dir': 'logs',  # Directory to save logs
    'experiment_name': None  # None = auto-generate with timestamp
}
```

### Testing Data Loading

To verify data loading works correctly:

```bash
python test_data_loading.py
```

### Using the Trained Model

```python
from train_model import CNNTransformerHybrid, predict_source_activity, load_checkpoint
import torch

# Initialize model (same architecture as training)
model = CNNTransformerHybrid(
    input_dim=75,
    output_dim=994,
    cnn_channels=[75, 128, 256, 256],
    kernel_sizes=[3, 5, 3],
    d_model=256,
    nhead=8,
    num_transformer_layers=4,
    dropout=0.2,
    use_residual=True
)

# Load the trained model from checkpoint
checkpoint = load_checkpoint('checkpoints/best_hybrid_model.pth', model)

# Or load from a specific checkpoint
# checkpoint = load_checkpoint('checkpoints/checkpoint_epoch_100.pth', model)

# Predict source activity from EEG
eeg_input = torch.FloatTensor(eeg_data)  # Shape: (batch, 500, 75)
predictions = predict_source_activity(model, eeg_input)
# predictions shape: (batch, 500, 994)
```

## Model Architecture

### CNN-Transformer Hybrid

1. **CNN Feature Extractor**: Extracts local temporal patterns using 1D convolutions
   - Channel progression: 75 → 128 → 256 → 256
   - Kernel sizes: 3, 5, 3

2. **Positional Encoding**: Learnable positional encodings for temporal information

3. **Transformer Encoder**: Captures long-range dependencies
   - 4 transformer layers
   - 8 attention heads
   - GELU activation

4. **Output Projection**: Multi-layer projection to expand from 256 to 994 dimensions

### Loss Function

The model uses a temporal MSE loss with smoothness regularization:
- MSE loss between predictions and targets
- Temporal smoothness loss (penalizes rapid changes between time steps)

## Evaluation Metrics

The evaluation includes:
- **MSE**: Mean squared error
- **Mean/Median Correlation**: Temporal correlation between predictions and targets
- **Temporal Smoothness**: Measures smoothness of predictions vs targets
- **Phase Consistency**: Phase alignment using Hilbert transform

## Logging

The training script provides comprehensive logging:

### Log Files

All logs are saved to `logs/experiment_YYYYMMDD_HHMMSS/` directory:

- **`training.log`**: Complete training log with all events, metrics, and messages
- **`config.json`**: Training configuration saved as JSON
- **`training_history.json`**: Full training history (losses, learning rates) as JSON
- **`training_history.csv`**: Training history in CSV format for easy analysis
- **`tensorboard/`**: TensorBoard logs for visualization (if TensorBoard is installed)

### Logged Information

The logger captures:
- Training start/end times
- Model architecture details (parameters, layers)
- Dataset information (train/val split, batch size)
- Every epoch: train loss, validation loss, learning rate, patience counter
- Checkpoint saves (best model, regular checkpoints, final checkpoint)
- Early stopping events
- Evaluation metrics
- All print statements are also logged to file

### TensorBoard Visualization

If TensorBoard is installed, you can visualize training progress:

```bash
# Install TensorBoard
pip install tensorboard

# View logs
tensorboard --logdir logs/experiment_YYYYMMDD_HHMMSS/tensorboard
```

TensorBoard shows:
- Training and validation loss curves
- Learning rate schedule
- Loss comparison graphs

## Checkpoints

The training script automatically saves checkpoints:

- **Best Model**: `logs/experiment_XXX/checkpoints/best_hybrid_model.pth` - Model with best validation loss
- **Regular Checkpoints**: `logs/experiment_XXX/checkpoints/checkpoint_epoch_XXX.pth` - Saved every N epochs (default: 10)
- **Final Checkpoint**: `logs/experiment_XXX/checkpoints/checkpoint_epoch_XXX_final.pth` - Saved at the end of training

Each checkpoint contains:
- Model state dictionary
- Optimizer state dictionary
- Scheduler state dictionary
- Training and validation losses (full history)
- Learning rates history
- Configuration
- Epoch number
- Best validation loss

### Resuming Training from Checkpoint

```python
from train_model import CNNTransformerHybrid, load_checkpoint, train_hybrid_model
import torch

# Initialize model (same architecture as training)
model = CNNTransformerHybrid(
    input_dim=75,
    output_dim=994,
    cnn_channels=[75, 128, 256, 256],
    kernel_sizes=[3, 5, 3],
    d_model=256,
    nhead=8,
    num_transformer_layers=4,
    dropout=0.2,
    use_residual=True
)

# Load checkpoint
checkpoint = load_checkpoint('checkpoints/checkpoint_epoch_050.pth', model)

# Resume training from checkpoint epoch
start_epoch = checkpoint['epoch'] + 1
# ... continue training from start_epoch
```

## Requirements

- Python 3.7+
- PyTorch 1.12+
- NumPy 1.21+
- SciPy 1.7+
- tqdm 4.62+
- tensorboard 2.8+ (optional, for visualization)

## Notes

- The model expects input sequences of length 500 time steps
- Input dimension: 75 (EEG channels)
- Output dimension: 994 (source locations)
- Training uses early stopping based on validation loss
- Data augmentation is applied only to training data
- All training events are logged to both console and log files
- Each experiment gets its own timestamped directory in `logs/`
- Training history is saved in both JSON and CSV formats for easy analysis

