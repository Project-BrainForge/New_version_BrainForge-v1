# Data Limiting Feature - Sample Range Loading

## Overview
Control which samples to load from your dataset using flexible range-based parameters:
- **`--max_samples N`**: Load first N samples (simple, takes precedence)
- **`--start_sample S --end_sample E`**: Load samples from index S to E (exclusive end, like Python slicing)

Useful for:
- Quick testing with subset of data
- Memory-constrained environments
- Gradual dataset exploration
- **Range-based selection for specific sample ranges** (e.g., samples 20-50)

## Usage

### Training with Limited Data

**Option 1: Load first N samples**
```bash
python train_model.py \
  --model_type vit \
  --data_dir labeled_spikes_data \
  --max_samples 20 \
  --epochs 200
```

**Option 2: Load a range of samples (samples 20-50)**
```bash
python train_model.py \
  --model_type vit \
  --data_dir labeled_spikes_data \
  --start_sample 20 \
  --end_sample 50 \
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

Load all samples (default, no limiting arguments):
```bash
python train_model.py \
  --model_type vit \
  --data_dir labeled_spikes_data
```

### Inference with Limited Data

**Option 1: Process first N samples**
```bash
python inference_vit.py \
  --checkpoint logs/exp3/checkpoints/best_vit_model.pth \
  --eeg_data labeled_spikes_data \
  --max_samples 5 \
  --output predictions_5samples.mat
```

**Option 2: Process a range of samples (samples 10-20)**
```bash
python inference_vit.py \
  --checkpoint logs/exp3/checkpoints/best_vit_model.pth \
  --eeg_data labeled_spikes_data \
  --start_sample 10 \
  --end_sample 20 \
  --output predictions_10to20.mat
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

# Or load samples 20-50
eeg_data, source_data = load_mat_files('labeled_spikes_data', start_sample=20, end_sample=50)

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

# Or load samples 10-20
eeg_data = load_eeg_data('labeled_spikes_data', start_sample=10, end_sample=20)

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
# Test 1: First 10 samples
python train_model.py --max_samples 10 --epochs 10 --experiment_name test_10

# Test 2: First 50 samples
python train_model.py --max_samples 50 --epochs 10 --experiment_name test_50

# Test 3: All samples
python train_model.py --epochs 200 --experiment_name final
```

### Scenario 3: Range-based Testing
Test on specific sample ranges without loading all data:
```bash
# Test on samples 0-15
python train_model.py --start_sample 0 --end_sample 15 --epochs 10 --experiment_name range_0_15

# Test on samples 15-21 (remaining samples)
python train_model.py --start_sample 15 --end_sample 21 --epochs 10 --experiment_name range_15_21
```

### Scenario 4: Limited Inference
Make predictions on subset of data for validation:
```bash
python inference_vit.py \
  --checkpoint logs/final/checkpoints/best_vit_model.pth \
  --eeg_data labeled_spikes_data \
  --start_sample 0 \
  --end_sample 10 \
  --output validation_predictions_0_10.mat
```

### Scenario 5: Cross-Validation Simulation
Simulate k-fold cross-validation with range-based loading (with 21 samples):
```bash
# Fold 1: Train on samples 0-15, validate on 15-21
python train_model.py --start_sample 0 --end_sample 15 --epochs 50 --experiment_name fold_1

# Fold 2: Train on samples 5-20, validate on others
python train_model.py --start_sample 5 --end_sample 20 --epochs 50 --experiment_name fold_2
```

## How It Works

### Parameter Priority
1. **`--max_samples`** (highest priority): If specified, loads first N samples only
   - Example: `--max_samples 10` loads samples [0:10]
2. **`--start_sample` / `--end_sample`**: If max_samples not set, uses these range parameters
   - Example: `--start_sample 20 --end_sample 50` loads samples [20:50] (exclusive end, like Python)
   - If only start_sample given: loads from that index to end
   - If only end_sample given: loads from 0 to that index

### Implementation Details
- Files are sorted alphabetically by filename for consistency
- Indexing is 0-based: first sample is index 0
- End index is exclusive (Python slice notation)
- Remaining logic (training, evaluation, inference) works identically for all subset sizes
- Default behavior (no parameters): loads all available files

### Example Index Mapping
With 21 total files (sample_00000 to sample_00999):
```
--start_sample 5 --end_sample 15  →  loads 10 files (indices 5,6,7,8,9,10,11,12,13,14)
--max_samples 10                  →  loads 10 files (indices 0,1,2,3,4,5,6,7,8,9)
--start_sample 15 --end_sample 21 →  loads 6 files  (indices 15,16,17,18,19,20)
--start_sample 20                 →  loads 1 file   (index 20)
```

## Tips

- Use with `--epochs` parameter to control training duration for subset tests
- Combine with `--batch_size` to optimize for memory usage
- Range-based loading lets you simulate cross-validation without data duplication
- Use `--start_sample` / `--end_sample` to test on non-contiguous portions or specific ranges
- Check console output to confirm correct number of samples was loaded

Example output showing range-based loading:
```
Found 21 sample MAT files
Loading samples [5:15] (got 10 samples from 21)
Loading MAT files...
100%|██████████| 10/10 [00:00<00:00, 126.23it/s]
...
Loaded data shapes:
EEG data: (10, 500, 75)
Source data: (10, 500, 994)
```

### Comparison: max_samples vs start_sample/end_sample

| Use Case | Command | Loads |
|----------|---------|-------|
| First N samples | `--max_samples 20` | Samples 0-19 |
| Specific range | `--start_sample 20 --end_sample 50` | Samples 20-49 |
| From index to end | `--start_sample 15` | Samples 15-20 |
| Remaining samples | `--start_sample 15 --end_sample 21` | Samples 15-20 |
| All samples | (no arguments) | All 21 samples |
