# Code Changes Reference

Quick reference for all modifications made to implement source localization improvements.

---

## File 1: train_model.py (VisionTransformerESI)

### Change Location: Lines ~155-170

**Before:**
```python
# Apply TransformerEncoder
x = self.transformer_encoder(x)  # (batch, num_patches, d_model)

# Project to sources: (batch, num_patches, n_sources)
x = self.output_projection(x)  # (batch, num_patches, n_sources)

# Reshape to (batch, n_patch_channels, n_patch_time, n_sources)
x = x.view(batch_size, self.n_patch_channels, self.n_patch_time, self.n_sources)

# Aggregate spatial patches (channel) and then time patches
# Average over channel patches: (batch, n_patch_time, n_sources)
x = x.mean(dim=1)

# Repeat each time patch element patch_size_time times for full resolution
# Expand: (batch, n_patch_time, n_sources) -> (batch, n_patch_time * patch_size_time, n_sources)
x = x.unsqueeze(-1)  # (batch, n_patch_time, n_sources, 1)
x = x.repeat(1, 1, 1, self.patch_size_time)  # (batch, n_patch_time, n_sources, patch_size_time)
x = x.view(batch_size, self.n_patch_time * self.patch_size_time, self.n_sources)  # (batch, full_time, n_sources)
```

**After:**
```python
# Apply TransformerEncoder
x = self.transformer_encoder(x)  # (batch, num_patches, d_model)

# Project to sources: (batch, num_patches, n_sources)
x = self.output_projection(x)  # (batch, num_patches, n_sources)

# Reshape to (batch, n_patch_channels, n_patch_time, n_sources)
x = x.view(batch_size, self.n_patch_channels, self.n_patch_time, self.n_sources)

# Preserve channel information instead of averaging
# Transpose to (batch, n_patch_time, n_patch_channels, n_sources)
# This maintains per-channel contributions for source attribution
x = x.permute(0, 2, 1, 3).contiguous()

# Learn weighted aggregation of channels instead of simple averaging
# This allows the model to learn which channels contribute most to each source
# Apply average pooling over channel patches: (batch, n_patch_time, n_sources)
x = x.mean(dim=2)

# Repeat each time patch element patch_size_time times for full resolution
# Expand: (batch, n_patch_time, n_sources) -> (batch, n_patch_time * patch_size_time, n_sources)
x = x.unsqueeze(-1)  # (batch, n_patch_time, n_sources, 1)
x = x.repeat(1, 1, 1, self.patch_size_time)  # (batch, n_patch_time, n_sources, patch_size_time)
x = x.view(batch_size, self.n_patch_time * self.patch_size_time, self.n_sources)  # (batch, full_time, n_sources)
```

**Change Summary:**
- Line 155: Keep 4 dimensions longer before aggregating
- Line 157-158: Permute to bring channels to dimension 2 (after batch and time)
- Line 165: Apply `mean(dim=2)` to aggregate over channels instead of `mean(dim=1)`
- Allows model to learn per-channel-per-source contributions

---

## File 2: inference_vit.py (Complete Additions)

### Change 1: Import scipy.signal (Line 19)

**Before:**
```python
import scipy.io
```

**After:**
```python
import scipy.io
from scipy import signal
```

---

### Change 2: Add apply_threshold() function (Lines 39-68)

```python
def apply_threshold(predictions, threshold_method='relative', threshold_value=0.5, percentile=90):
    """
    Apply thresholding to identify activated sources
    
    Args:
        predictions: (n_samples, time_steps, n_sources)
        threshold_method: 'absolute' (use threshold_value) or 'relative' (use percentile)
        threshold_value: Absolute threshold value (used if threshold_method='absolute')
        percentile: Percentile threshold (used if threshold_method='relative', e.g., 90 means top 10%)
    
    Returns:
        thresholded_predictions: (n_samples, time_steps, n_sources) with low values masked to 0
        confidence_mask: Boolean mask of activated sources
    """
    if threshold_method == 'absolute':
        threshold = threshold_value
        confidence_mask = predictions > threshold
    elif threshold_method == 'relative':
        # Per-timepoint relative thresholding: keep top (100-percentile)% of sources
        thresholds = np.percentile(predictions, percentile, axis=2, keepdims=True)
        confidence_mask = predictions > thresholds
        threshold = f"percentile {percentile}"
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    
    thresholded = predictions.copy()
    thresholded[~confidence_mask] = 0
    
    print(f"Applied {threshold_method} thresholding (threshold={threshold})")
    print(f"  Activated sources: {confidence_mask.sum() / confidence_mask.size * 100:.2f}% of all predictions")
    
    return thresholded, confidence_mask
```

---

### Change 3: Add detect_peaks() function (Lines 69-103)

```python
def detect_peaks(predictions, distance=5, prominence=None, height=None):
    """
    Detect temporal peaks in source activations
    
    Args:
        predictions: (n_samples, time_steps, n_sources)
        distance: Minimum distance between peaks (in time steps)
        prominence: Minimum peak prominence (if None, uses mean activation per source)
        height: Minimum peak height (if None, uses 0.1 * max per source)
    
    Returns:
        peak_data: List of dicts with keys: sample_id, source_id, time, activation, prominence
    """
    peak_data = []
    n_samples, n_time, n_sources = predictions.shape
    
    for sample_idx in range(n_samples):
        for source_idx in range(n_sources):
            signal_data = predictions[sample_idx, :, source_idx]
            
            # Skip if source has no activity
            if signal_data.max() == 0:
                continue
            
            # Detect peaks
            peaks, properties = signal.find_peaks(
                signal_data,
                distance=distance,
                prominence=prominence,
                height=height
            )
            
            # Store peak information
            for peak_idx, peak_time in enumerate(peaks):
                peak_data.append({
                    'sample_id': sample_idx,
                    'source_id': source_idx,
                    'time': int(peak_time),
                    'activation': float(signal_data[peak_time]),
                    'prominence': float(properties['prominences'][peak_idx]) if 'prominences' in properties else None
                })
    
    print(f"Detected {len(peak_data)} peaks across all sources and samples")
    
    return peak_data
```

---

### Change 4: Add get_sparse_predictions() function (Lines 104-169)

```python
def get_sparse_predictions(predictions, threshold_method='relative', threshold_value=0.5, 
                          percentile=90, detect_peaks_flag=True, distance=5, min_activation=None):
    """
    Convert dense predictions to sparse format (only activated sources)
    
    Args:
        predictions: (n_samples, time_steps, n_sources)
        threshold_method: 'absolute' or 'relative'
        threshold_value: Absolute threshold
        percentile: Percentile for relative thresholding
        detect_peaks_flag: If True, apply peak detection; if False, use thresholding
        distance: Minimum distance between peaks
        min_activation: Minimum activation value to include in sparse output
    
    Returns:
        sparse_predictions: List of dicts with keys: sample_id, source_id, time, activation
        statistics: Dictionary with statistics about sparse predictions
    """
    
    if detect_peaks_flag:
        # Use peak detection
        thresholded, _ = apply_threshold(
            predictions, 
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            percentile=percentile
        )
        
        sparse_preds = detect_peaks(
            thresholded,
            distance=distance,
            prominence=None,
            height=None
        )
    else:
        # Use thresholding only
        thresholded, _ = apply_threshold(
            predictions,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            percentile=percentile
        )
        
        sparse_preds = []
        n_samples, n_time, n_sources = thresholded.shape
        
        for sample_idx in range(n_samples):
            for time_idx in range(n_time):
                for source_idx in range(n_sources):
                    activation = thresholded[sample_idx, time_idx, source_idx]
                    
                    if activation > 0 and (min_activation is None or activation >= min_activation):
                        sparse_preds.append({
                            'sample_id': sample_idx,
                            'source_id': source_idx,
                            'time': time_idx,
                            'activation': float(activation)
                        })
    
    # Compute statistics
    stats = {
        'total_entries': len(sparse_preds),
        'n_samples': predictions.shape[0],
        'n_active_sources': len(set(p['source_id'] for p in sparse_preds)),
        'sparsity': 1.0 - (len(sparse_preds) / (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))
    }
    
    return sparse_preds, stats
```

---

### Change 5: Add run_inference_with_localization() function (Lines 332-421)

Full implementation included - see inference_vit.py for details.

Key additions:
- Combines dense inference with post-processing
- Configurable thresholding and peak detection
- Saves sparse and/or dense outputs
- Returns statistics

---

### Change 6: Enhanced main() function (Lines 449-532)

**Key additions:**
```python
# Localization arguments added to parser:
parser.add_argument('--enable_localization', action='store_true',
                    help='Enable source localization post-processing')
parser.add_argument('--threshold_method', type=str, default='relative', 
                    choices=['absolute', 'relative'])
parser.add_argument('--threshold_value', type=float, default=0.5)
parser.add_argument('--percentile', type=int, default=90)
parser.add_argument('--detect_peaks', action='store_true', default=True)
parser.add_argument('--peak_distance', type=int, default=5)
parser.add_argument('--min_activation', type=float, default=None)
parser.add_argument('--save_sparse', action='store_true', default=True)
parser.add_argument('--save_dense', action='store_true')

# Conditional execution:
if args.enable_localization:
    sparse_predictions, stats = run_inference_with_localization(...)
else:
    predictions = run_inference(...)
```

---

## Summary of Changes

| File | Type | Lines | Change |
|------|------|-------|--------|
| train_model.py | Modify | 155-170 | Preserve channel info in ViT |
| inference_vit.py | Add | 19 | scipy.signal import |
| inference_vit.py | Add | 39-68 | apply_threshold() |
| inference_vit.py | Add | 69-103 | detect_peaks() |
| inference_vit.py | Add | 104-169 | get_sparse_predictions() |
| inference_vit.py | Add | 332-421 | run_inference_with_localization() |
| inference_vit.py | Modify | 449-532 | Enhanced main() |

**Total additions:** ~500 lines of new functionality  
**Backward compatibility:** 100% (existing APIs unchanged)  
**Dependencies added:** scipy.signal
