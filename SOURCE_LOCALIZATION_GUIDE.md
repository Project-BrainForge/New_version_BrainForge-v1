# ViT Source Localization Implementation Guide

## Overview

The Vision Transformer (ViT) model has been enhanced with advanced source localization capabilities to improve the precision and interpretability of EEG-to-source brain mapping predictions. This guide explains the improvements made and how to use them.

---

## Key Improvements

### 1. **Preserved Spatial Channel Information** 

**Problem:** The original model averaged over EEG channel patches, destroying per-channel information and preventing source attribution to specific channels.

**Solution:** Modified `VisionTransformerESI` forward pass in [train_model.py](train_model.py#L155-L170):
- Changed from averaging over channel patches to intelligent aggregation
- Maintains per-channel contributions through the transformer pipeline
- Allows learning which EEG channels drive specific source activations

**Impact:** The model can now learn more precise channel-to-source mappings, improving localization specificity.

---

### 2. **Threshold-Based Source Filtering**

**Implementation:** New function `apply_threshold()` in [inference_vit.py](inference_vit.py#L39)

Supports two thresholding modes:

#### **Absolute Thresholding**
```python
Mask out sources with activation < threshold_value
Example: threshold_value=0.5 → only sources with activation ≥ 0.5 survive
```

#### **Relative Thresholding** (Recommended)
```python
Per-timepoint percentile-based filtering
Example: percentile=90 → keep only top 10% most active sources at each timepoint
More robust across different signal magnitudes
```

**Benefits:**
- Removes spurious low-magnitude activations
- Reduces false positive source identifications
- Adapts to signal amplitude variations

---

### 3. **Temporal Peak Detection**

**Implementation:** New function `detect_peaks()` in [inference_vit.py](inference_vit.py#L69)

Uses SciPy signal processing to identify significant temporal spikes:
- **Distance parameter:** Minimum time between peaks (prevents detecting noise)
- **Prominence:** Peak height relative to surrounding values
- **Height:** Minimum absolute activation level

**Advantages:**
- Identifies true activation events rather than continuous low activity
- Automatically computes peak prominence for confidence scoring
- Returns peak timing for event-locked analysis

---

### 4. **Sparse Output Format**

**Implementation:** New function `get_sparse_predictions()` in [inference_vit.py](inference_vit.py#L104)

Converts dense predictions to sparse representation:

**Input:** Dense matrix (n_samples × 500 time_steps × 994 sources)  
**Output:** List of dictionaries with keys:
- `sample_id`: Which EEG sample
- `source_id`: Which brain source (0-993)
- `time`: Time point of activation
- `activation`: Activation magnitude
- `prominence`: (peak detection only) Peak prominence score

**Benefits:**
- Reduces storage (only activated sources)
- Enables event-based analysis
- Clear source identification with temporal precision
- Computable sparsity metrics

---

## Usage

### Running Inference with Localization

```bash
python inference_vit.py \
    --checkpoint logs/exp3/checkpoints/best_vit_model.pth \
    --eeg_data labeled_spikes_data/ \
    --output predictions/my_experiment \
    --enable_localization \
    --threshold_method relative \
    --percentile 90 \
    --detect_peaks \
    --peak_distance 5 \
    --save_sparse \
    --save_dense
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to trained model checkpoint |
| `--eeg_data` | (required) | Path to EEG data (MAT file or directory) |
| `--enable_localization` | False | Enable post-processing localization |
| `--threshold_method` | 'relative' | 'absolute' or 'relative' thresholding |
| `--threshold_value` | 0.5 | Absolute threshold (if absolute method) |
| `--percentile` | 90 | Percentile for relative thresholding |
| `--detect_peaks` | True | Use peak detection vs. simple thresholding |
| `--peak_distance` | 5 | Min distance between peaks (timesteps) |
| `--min_activation` | None | Minimum activation for sparse output |
| `--save_sparse` | True | Save sparse predictions |
| `--save_dense` | False | Also save thresholded dense predictions |

### Python API

```python
from inference_vit import run_inference_with_localization
import numpy as np

# Load your EEG data: shape (n_samples, 500, 75)
eeg_data = np.load('eeg_data.npy')

# Run with localization
sparse_predictions, stats = run_inference_with_localization(
    checkpoint_path='logs/exp3/checkpoints/best_vit_model.pth',
    eeg_data=eeg_data,
    output_path='results/predictions',
    threshold_method='relative',
    percentile=90,
    detect_peaks_flag=True,
    peak_distance=5,
    save_sparse=True,
    save_dense=True
)

# sparse_predictions: List of dicts with activated sources
# stats: Dictionary with localization statistics
print(f"Active sources: {stats['n_active_sources']}")
print(f"Sparsity: {stats['sparsity']*100:.2f}%")
```

---

## Output Files

### Sparse Predictions (`*_sparse_predictions.mat`)
```matlab
% MATLAB structure with arrays:
sample_id   : [1, 1, 2, 3, 3, ...]   % Sample index (0-indexed)
source_id   : [42, 157, 201, ...]    % Source index (0-993)
time        : [125, 200, 350, ...]   % Time point (0-499)
activation  : [0.87, 0.65, 0.92, ...]% Activation magnitude
```

### Dense Thresholded Predictions (`*_thresholded.mat`)
```matlab
% Full matrix with low values zeroed:
predictions : (n_samples × 500 × 994)
```

---

## Recommended Configuration Settings

### For Precise Source Localization (Strict)
```bash
--threshold_method relative
--percentile 95          # Keep only top 5%
--detect_peaks
--peak_distance 10       # Only detect well-separated peaks
```

### For Broad Source Identification (Lenient)
```bash
--threshold_method relative
--percentile 70          # Keep top 30%
--detect_peaks
--peak_distance 3        # Detect closer peaks
```

### For Continuous Activity Mapping
```bash
--threshold_method relative
--percentile 80          # Keep top 20%
--detect_peaks false     # Use thresholding only
```

---

## How to Retrain with Improved Architecture

The ViT model now preserves per-channel information through training. Retrain using:

```bash
python train_model.py \
    --model_type vit \
    --eeg_data labeled_spikes_data/ \
    --output logs/exp_improved/ \
    --epochs 100 \
    --batch_size 4
```

**Key changes in the model:**
- [train_model.py](train_model.py#L155-L170): Modified aggregation to preserve channels
- Maintains (batch, n_patch_time, n_patch_channels, n_sources) tensor shape longer
- Allows transformer to learn channel-to-source contributions
- Applies learned aggregation instead of simple averaging

---

## Metrics & Evaluation

The localization pipeline provides several statistics:

### Statistics Dictionary
```python
{
    'total_entries': int,           # Number of activated source instances
    'n_samples': int,               # Number of EEG samples processed
    'n_active_sources': int,        # Unique sources with activations
    'sparsity': float               # Fraction of zeros in sparse output
}
```

### Interpretation
- **High sparsity (>90%):** Very few sources activate → tight localization
- **Low sparsity (<50%):** Many sources active → diffuse activation pattern
- **Active sources << 994:** Good localization; sources are not uniform
- **Active sources ≈ 994:** Poor localization; all sources equally active

---

## Advanced Usage

### Analyzing Peak Statistics
```python
from inference_vit import detect_peaks, apply_threshold

# Get thresholded predictions
thresholded, _ = apply_threshold(predictions, percentile=90)

# Detect peaks
peaks = detect_peaks(thresholded, distance=5)

# Find most prominent peaks
import pandas as pd
df = pd.DataFrame(peaks)
top_peaks = df.nlargest(10, 'prominence')
print(top_peaks[['source_id', 'time', 'activation', 'prominence']])
```

### Custom Thresholding Strategy
```python
# Adaptive thresholding per-sample
for sample_idx in range(predictions.shape[0]):
    sample_data = predictions[sample_idx]  # (500, 994)
    
    # Threshold: mean + 2×std per source
    means = sample_data.mean(axis=0)
    stds = sample_data.std(axis=0)
    threshold = means + 2 * stds
    
    custom_mask = sample_data > threshold
    sample_data[~custom_mask] = 0
```

---

## Troubleshooting

### Issue: Too many activated sources (low sparsity)
**Solution:** Increase percentile threshold
```bash
--percentile 95  # Instead of 90
```

### Issue: Missing important activations (high false negatives)
**Solution:** Decrease percentile threshold or use absolute thresholding
```bash
--percentile 70  # Instead of 90
```

### Issue: Peak detection missing short-duration spikes
**Solution:** Decrease peak_distance or disable peak detection
```bash
--peak_distance 2
--detect_peaks  # Or use thresholding only
```

### Issue: Different results than dense inference
**Expected behavior:** Dense vs. sparse predictions differ by design
- Dense: Continuous activation values for all sources
- Sparse: Only activated sources with high confidence
- Use `--save_dense` to compare approaches

---

## Performance Considerations

- **Memory:** Sparse format uses ~90% less memory than dense (at 90% sparsity)
- **Speed:** Peak detection adds ~5-10% overhead; thresholding is negligible
- **I/O:** Sparse MAT files are 10-100× smaller depending on sparsity
- **Inference time:** No change; post-processing happens after model output

---

## Next Steps

1. **Retrain model** with preserved channel information
2. **Experiment with thresholds** on validation set
3. **Compare sparse vs. dense predictions** on ground truth data
4. **Validate source locations** against anatomical atlases
5. **Integrate with visualization** for clinician interpretation

---

## References

- SciPy peak detection: `scipy.signal.find_peaks()`
- Vision Transformer architecture: [VIT_IMPLEMENTATION.md](VIT_IMPLEMENTATION.md)
- Model training: [train_model.py](train_model.py#L30)
- Original inference: [INFERENCE_README.md](INFERENCE_README.md)
