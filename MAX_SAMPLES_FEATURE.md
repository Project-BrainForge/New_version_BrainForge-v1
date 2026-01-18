# Data Limiting Feature - max_samples Parameter

## Overview
The `--max_samples` parameter allows you to limit the number of data files loaded from your dataset. This is useful for:
- Quick testing with subset of data
- Memory-constrained environments
- Gradual dataset exploration
- Development/debugging

## Usage

### Training with Limited Data

Load only first 20 samples:
```bash
python train_model.py \
  --model_type vit \
  --data_dir labeled_spikes_data \
  --max_samples 20 \
  --epochs 200
```

Load only first 10 samples for quick testing:
```bash
python train_model.py \
  --model_type vit \
  --data_dir labeled_spikes_data \
  --max_samples 10 \
  --epochs 5 \
  --batch_size 2
```

Load all samples (default, max_samples=None):
```bash
python train_model.py \
  --model_type vit \
  --data_dir labeled_spikes_data
```

### Inference with Limited Data

Process first 5 samples:
```bash
python inference_vit.py \
  --checkpoint logs/exp3/checkpoints/best_vit_model.pth \
  --eeg_data labeled_spikes_data \
  --max_samples 5 \
  --output predictions_5samples.mat
```

Process all samples:
```bash
python inference_vit.py \
  --checkpoint logs/exp3/checkpoints/best_vit_model.pth \
  --eeg_data labeled_spikes_data \
  --output predictions_all.mat
```

### Python API

Training:
```python
from train_model import load_mat_files, train_model

# Load only first 50 samples
eeg_data, source_data = load_mat_files('labeled_spikes_data', max_samples=50)

config = {
    'model_type': 'vit',
    'batch_size': 32,
    'epochs': 200,
    'learning_rate': 2e-4,
    # ... other config params
}

model, train_losses, val_losses = train_model(eeg_data, source_data, config=config)
```

Inference:
```python
from inference_vit import load_eeg_data, run_inference

# Load first 10 samples for inference
eeg_data = load_eeg_data('labeled_spikes_data', max_samples=10)

predictions = run_inference(
    checkpoint_path='logs/exp3/checkpoints/best_vit_model.pth',
    eeg_data=eeg_data,
    output_path='predictions.mat',
    batch_size=4
)
```

## Examples

### Scenario 1: Quick Testing
Test model training with minimal data before scaling up:
```bash
python train_model.py \
  --model_type vit \
  --data_dir labeled_spikes_data \
  --max_samples 5 \
  --epochs 3 \
  --batch_size 2
```

### Scenario 2: Gradual Exploration
Train on increasing amounts of data:
```bash
# Test 1: 10 samples
python train_model.py --max_samples 10 --epochs 10 --experiment_name test_10
# Test 2: 50 samples
python train_model.py --max_samples 50 --epochs 10 --experiment_name test_50
# Test 3: All samples
python train_model.py --epochs 200 --experiment_name final
```

### Scenario 3: Limited Inference
Make predictions on subset of data for validation:
```bash
python inference_vit.py \
  --checkpoint logs/final/checkpoints/best_vit_model.pth \
  --eeg_data labeled_spikes_data \
  --max_samples 10 \
  --output validation_predictions.mat
```

## How It Works

1. **Loading**: When `max_samples` is specified, only the first N sorted MAT files are loaded
2. **Files are sorted alphabetically** by filename for consistency
3. **Remaining logic unchanged**: All training, evaluation, and inference pipelines work the same way
4. **Default behavior**: When `max_samples=None` (default), all available files are loaded

## Tips

- Use with `--epochs` parameter to control training duration for subset tests
- Combine with `--batch_size` to optimize for memory usage
- Use `--max_samples` in inference to validate on a portion of test data
- Check loaded data shape in output to confirm correct number of samples was loaded

Example output showing data limiting:
```
Found 21 sample MAT files
Loading first 10 samples (max_samples=10)
Loading MAT files...
100%|██████████| 10/10 [00:00<00:00, 126.23it/s]
...
Loaded data shapes:
EEG data: (10, 500, 75)
Source data: (10, 500, 994)
```
