# Quick Start: Source Localization

## What Changed?

✅ **ViT Model:** Now preserves per-channel information (→ better channel-to-source mapping)  
✅ **Inference:** Added thresholding + peak detection for precise source identification  
✅ **Output:** Can now get sparse (source_id, time, activation) predictions  

---

## 5-Minute Quick Start

### Standard Dense Inference (Old Way)
```bash
python inference_vit.py \
    --checkpoint logs/exp3/checkpoints/best_vit_model.pth \
    --eeg_data labeled_spikes_data/ \
    --output results/predictions.mat
# Output: predictions.mat (500 × 994 matrix)
```

### With Source Localization (New Way)
```bash
python inference_vit.py \
    --checkpoint logs/exp3/checkpoints/best_vit_model.pth \
    --eeg_data labeled_spikes_data/ \
    --output results/predictions \
    --enable_localization \
    --percentile 90 \
    --detect_peaks
# Output: predictions_sparse_predictions.mat + statistics
```

---

## Key Functions

### 1. Thresholding
```python
from inference_vit import apply_threshold

thresholded, mask = apply_threshold(
    predictions,
    threshold_method='relative',
    percentile=90
)
# Keeps only top 10% most active sources per timepoint
```

### 2. Peak Detection
```python
from inference_vit import detect_peaks

peaks = detect_peaks(
    predictions,
    distance=5,  # Min spacing between peaks
    prominence=None
)
# Returns: [(sample_id, source_id, time, activation), ...]
```

### 3. Sparse Output
```python
from inference_vit import get_sparse_predictions

sparse, stats = get_sparse_predictions(
    predictions,
    threshold_method='relative',
    percentile=90,
    detect_peaks_flag=True,
    distance=5
)

print(f"Found {stats['n_active_sources']} active sources")
print(f"Sparsity: {stats['sparsity']*100:.1f}%")
```

### 4. Full Pipeline
```python
from inference_vit import run_inference_with_localization

sparse_preds, stats = run_inference_with_localization(
    checkpoint_path='best_model.pth',
    eeg_data=eeg_array,  # (n_samples, 500, 75)
    output_path='results/predictions',
    threshold_method='relative',
    percentile=90,
    detect_peaks_flag=True
)
```

---

## What You Get

### Sparse Format (Recommended)
```
✓ List of activated sources: [{source_id, time, activation}, ...]
✓ 90%+ smaller file size
✓ Clear source identification
✓ Peak prominence scores
✓ Temporal event information
```

### Dense Format (Optional)
```
✓ Full (500 × 994) matrix
✓ Low-activation sources zeroed out
✓ Direct comparison with raw predictions
```

---

## Choosing Thresholds

| Goal | Percentile | Peak Detection |
|------|-----------|----------------|
| **Tight localization** | 95 | Yes |
| **Balanced** | 90 | Yes |
| **Broad identification** | 70 | No |
| **Continuous activity** | 80 | No |

---

## Expected Improvements

✅ **Fewer false positive sources** (cleaner predictions)  
✅ **Better temporal precision** (peak times vs. continuous)  
✅ **Reduced file sizes** (sparse storage)  
✅ **More interpretable results** (clear source locations)  
✅ **Better channel attribution** (after retraining)

---

## Next: Retrain the Model

The new ViT preserves channel information during training. Retrain for best results:

```bash
python train_model.py \
    --model_type vit \
    --eeg_data labeled_spikes_data/ \
    --output logs/improved_vit/ \
    --epochs 100
```

Training the model with the improved architecture will allow it to learn precise channel-to-source mappings.

---

## Full Documentation

See [SOURCE_LOCALIZATION_GUIDE.md](SOURCE_LOCALIZATION_GUIDE.md) for:
- Detailed parameter explanations
- Advanced usage patterns
- Troubleshooting tips
- Performance considerations
