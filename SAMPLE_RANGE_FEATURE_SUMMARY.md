# Sample Range Loading - Feature Summary

## What Changed
Added flexible sample selection to both training and inference pipelines:
- **`--start_sample S`**: Start index (inclusive, default 0)
- **`--end_sample E`**: End index (exclusive, Python slice style)
- **`--max_samples N`**: Load first N samples (takes precedence if specified)

## Key Benefits
✓ Load any contiguous range of samples without loading all data
✓ Perfect for testing specific sample ranges (e.g., "from 20 to 50")
✓ Supports Python-style slicing [start:end)
✓ Backward compatible - existing code works unchanged
✓ Works identically in both `train_model.py` and `inference_vit.py`

## Implementation Details

### Files Modified
1. **train_model.py**
   - Added CLI arguments: `--start_sample`, `--end_sample`
   - Updated `load_mat_files()` signature: added `start_sample`, `end_sample` parameters
   - Updated `load_mat_files()` logic to apply range after max_samples check
   - Updated main execution: passes new parameters to load_mat_files()

2. **inference_vit.py**
   - Added CLI arguments: `--start_sample`, `--end_sample`
   - Updated `load_eeg_data()` signature: added `start_sample`, `end_sample` parameters
   - Updated `load_eeg_data()` to pass range to `load_mat_files()`

3. **MAX_SAMPLES_FEATURE.md** (documentation updated)

### Priority & Behavior
```
if max_samples specified:
    load first max_samples files
else if start_sample != 0 OR end_sample != None:
    load mat_files[start_sample:end_sample]
else:
    load all files
```

## Usage Examples

### Training
```bash
# Load samples 5-15 (10 samples)
python train_model.py --data_dir labeled_spikes_data --start_sample 5 --end_sample 15 --epochs 100

# Load samples 0-10 (first 10)
python train_model.py --data_dir labeled_spikes_data --end_sample 10 --epochs 100

# Load samples 15-end (last 6 with 21 total)
python train_model.py --data_dir labeled_spikes_data --start_sample 15 --epochs 100
```

### Inference
```bash
# Predict on samples 10-20
python inference_vit.py \
  --checkpoint logs/exp3/checkpoints/best_vit_model.pth \
  --eeg_data labeled_spikes_data \
  --start_sample 10 \
  --end_sample 20 \
  --output predictions.mat
```

## Test Results

✅ **Test 1**: `--start_sample 3 --end_sample 8`
- Found: 21 samples → Loaded: 5 samples [3:8] ✓

✅ **Test 2**: `--start_sample 5 --end_sample 15`
- Found: 21 samples → Loaded: 10 samples [5:15] ✓
- Data split: 8 train, 2 validation ✓

✅ **Test 3**: `--max_samples 12`
- Takes precedence: Loads first 12 samples ✓

✅ **Test 4**: Inference with `--start_sample 10 --end_sample 20`
- Loaded: 10 samples
- Output shape: (10, 500, 994) ✓

## Python API Support
```python
# Training
eeg_data, source_data = load_mat_files(
    'labeled_spikes_data',
    start_sample=20,
    end_sample=50
)

# Inference
eeg_data = load_eeg_data(
    'labeled_spikes_data',
    start_sample=10,
    end_sample=20
)
```

## Backward Compatibility
- ✅ All existing commands work unchanged
- ✅ Default behavior (no parameters): loads all files
- ✅ `--max_samples` behavior unchanged
- ✅ No breaking changes to API

## Notes
- Files are sorted alphabetically for consistent indexing
- End index is exclusive (Python convention)
- With 21 sample files:
  - `[0:21]` = all samples
  - `[5:15]` = samples 5 through 14 (10 samples)
  - `[15:]` = samples 15 through 20 (6 samples)
