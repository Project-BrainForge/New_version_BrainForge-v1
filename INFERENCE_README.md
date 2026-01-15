# Inference Script for EEG to Source Activity Models

This script allows you to run inference on trained models using the real labeled dataset.

## Usage

### Basic Usage

```bash
python run_inference.py --checkpoint <path_to_checkpoint> --model_type <model_type>
```

### Examples

#### 1. Run inference with CNN-Transformer Hybrid model:

```bash
python run_inference.py \
    --checkpoint logs/experiment_20240101_120000/checkpoints/best_hybrid_model.pth \
    --model_type hybrid \
    --input_dir labeled_spikes_data/labeled_spikes_data \
    --output_dir inference_results/hybrid \
    --evaluate
```

#### 2. Run inference with EEG-ViT model:

```bash
python run_inference.py \
    --checkpoint logs/experiment_20240101_120000/checkpoints/best_eeg_vit_model.pth \
    --model_type eeg_vit \
    --input_dir labeled_spikes_data/labeled_spikes_data \
    --output_dir inference_results/eeg_vit \
    --evaluate
```

#### 3. Run inference on specific subject/folder:

```bash
python run_inference.py \
    --checkpoint logs/experiment_XXX/checkpoints/best_hybrid_model.pth \
    --model_type hybrid \
    --input_dir labeled_spikes_data/labeled_spikes_data \
    --subject VEP \
    --output_dir inference_results/VEP \
    --evaluate
```

## Arguments

### Required Arguments

- `--checkpoint`: Path to model checkpoint file (`.pth` file)

### Optional Arguments

- `--model_type`: Type of model to use
  - `hybrid` (default): CNN-Transformer hybrid model
  - `eeg_vit`: Standard EEG Vision Transformer
  - `vit_channel`: ViT with channel attention
  - `cnn_vit`: CNN + ViT hybrid

- `--device`: Device to use (`auto`, `cpu`, `cuda`)
  - Default: `auto` (uses GPU if available)

- `--input_dir`: Directory containing input MAT files
  - Default: `labeled_spikes_data/labeled_spikes_data`

- `--output_dir`: Directory to save output predictions
  - Default: `inference_results`

- `--subject`: Specific subject/folder to process (optional)
  - If specified, processes only files in `input_dir/subject/`

- `--batch_size`: Batch size for inference
  - Default: 32

- `--save_individual`: Save individual prediction files for each input
  - Default: False (only saves aggregated results)

- `--evaluate`: Evaluate predictions against ground truth
  - Default: False
  - Requires `source_data` in input MAT files

## Output Files

The script generates the following output files in the specified `--output_dir`:

1. **`all_predictions_<model_type>.mat`**: Aggregated predictions for all processed files
   - Contains: `predictions`, `targets` (if available), `file_names`, `model_type`, `checkpoint`

2. **`evaluation_metrics_<model_type>.mat`**: Evaluation metrics (if `--evaluate` is used)
   - Contains: MSE, correlation, smoothness, phase error, etc.

3. **`pred_<original_filename>.mat`**: Individual prediction files (if `--save_individual` is used)
   - Contains: `prediction`, `eeg_data`, `source_data`, `file_name`

## Input File Format

The script expects MAT files with the following structure:

```matlab
eeg_data: (500, 75) - EEG signals
source_data: (500, 994) - Source activity (ground truth, optional for evaluation)
```

Files should be named with pattern: `sample_*.mat`

## Model Checkpoint Format

Checkpoints should contain:
- `model_state_dict`: Model weights
- `config`: Model configuration dictionary
- `best_val_loss`: Best validation loss (optional)
- `epoch`: Training epoch (optional)

## Examples with Different Models

### CNN-Transformer Hybrid

```bash
python run_inference.py \
    --checkpoint logs/experiment_XXX/checkpoints/best_hybrid_model.pth \
    --model_type hybrid \
    --batch_size 16 \
    --evaluate
```

### EEG Vision Transformer

```bash
python run_inference.py \
    --checkpoint logs/experiment_XXX/checkpoints/best_eeg_vit_model.pth \
    --model_type eeg_vit \
    --batch_size 32 \
    --evaluate
```

### ViT with Channel Attention

```bash
python run_inference.py \
    --checkpoint logs/experiment_XXX/checkpoints/best_vit_channel_model.pth \
    --model_type vit_channel \
    --batch_size 32 \
    --evaluate
```

### CNN-ViT Hybrid

```bash
python run_inference.py \
    --checkpoint logs/experiment_XXX/checkpoints/best_cnn_vit_model.pth \
    --model_type cnn_vit \
    --batch_size 32 \
    --evaluate
```

## Evaluation Metrics

When `--evaluate` is used, the script computes:

- **MSE**: Mean Squared Error
- **Mean Correlation**: Average temporal correlation across all sources
- **Median Correlation**: Median temporal correlation
- **Temporal Smoothness**: Measures smoothness of predictions vs targets
- **Phase Error**: Phase alignment error (if scipy.signal available)

## Notes

- The script automatically detects the model architecture from the checkpoint
- Batch processing is used for efficient inference
- Predictions are saved in the same format as input (MAT files)
- If ground truth is available, evaluation metrics are computed
- GPU is used automatically if available (unless `--device cpu` is specified)

## Troubleshooting

1. **Checkpoint not found**: Ensure the checkpoint path is correct and the file exists
2. **Model type mismatch**: Make sure `--model_type` matches the checkpoint
3. **CUDA out of memory**: Reduce `--batch_size`
4. **No files found**: Check that `--input_dir` contains MAT files with `sample_` prefix

