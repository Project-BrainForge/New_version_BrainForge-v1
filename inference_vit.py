#!/usr/bin/env python
"""
Inference script for Vision Transformer ESI model
Loads trained model checkpoint and predicts source activity from EEG data
Includes localization post-processing: thresholding and peak detection
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from train_model import (
    VisionTransformerESI, 
    load_checkpoint, 
    predict_source_activity,
    load_mat_files,
    evaluate_temporal_metrics
)
import scipy.io
from scipy import signal


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


def load_eeg_data(data_path, max_samples=None, start_sample=0, end_sample=None):
    """
    Load EEG data from MAT file or directory
    
    Args:
        data_path: Path to MAT file or directory containing MAT files
        max_samples: Maximum number of samples to load (None = load all)
        start_sample: Start index for range (inclusive). Default: 0
        end_sample: End index for range (exclusive). Default: None (loads to end)
    
    Returns:
        eeg_data: numpy array of shape (n_samples, 500, 75) or (500, 75) for single file
    """
    data_path = Path(data_path)
    
    if data_path.is_file() and data_path.suffix == '.mat':
        # Single MAT file
        print(f"Loading single EEG file: {data_path}")
        data = scipy.io.loadmat(str(data_path))
        
        if 'eeg_data' not in data:
            raise ValueError(f"MAT file does not contain 'eeg_data' key. Available keys: {list(data.keys())}")
        
        eeg = np.array(data['eeg_data'], dtype=np.float32)  # (500, 75)
        if eeg.ndim == 2:
            eeg = eeg[np.newaxis, :, :]  # (1, 500, 75)
        
        return eeg
    
    elif data_path.is_dir():
        # Directory of MAT files
        print(f"Loading EEG files from directory: {data_path}")
        eeg_data, _ = load_mat_files(
            data_path,
            max_samples=max_samples,
            start_sample=start_sample,
            end_sample=end_sample
        )
        return eeg_data
    
    else:
        raise FileNotFoundError(f"Path not found or not a MAT file: {data_path}")


def run_inference(
    checkpoint_path,
    eeg_data,
    output_path=None,
    device='auto',
    batch_size=4
):
    """
    Run inference on EEG data using trained ViT model
    
    Args:
        checkpoint_path: Path to checkpoint file (best_vit_model.pth)
        eeg_data: EEG data array of shape (n_samples, 500, 75)
        output_path: Path to save predictions (optional)
        device: 'auto', 'cpu', or 'cuda'
        batch_size: Batch size for inference
    
    Returns:
        predictions: numpy array of shape (n_samples, 500, 994)
    """
    
    # Auto-detect device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nUsing device: {device}")
    
    # Create model
    print("Creating Vision Transformer model...")
    model = VisionTransformerESI(
        n_channels=75,
        n_sources=994,
        time_steps=500,
        patch_size_channels=5,
        patch_size_time=10,
        d_model=256,
        nhead=8,
        num_transformer_layers=4,
        dropout=0.1
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Convert EEG to ViT format: (n_samples, 500, 75) -> (n_samples, 75, 500)
    n_samples = eeg_data.shape[0]
    eeg_vit = torch.FloatTensor(eeg_data).transpose(1, 2)  # (n_samples, 75, 500)
    
    print(f"\nEEG input shape: {eeg_vit.shape}")
    
    # Batch inference
    all_predictions = []
    n_batches = int(np.ceil(n_samples / batch_size))
    
    print(f"Running inference on {n_samples} samples ({n_batches} batches)...")
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_eeg = eeg_vit[start_idx:end_idx].to(device)
            batch_predictions = model(batch_eeg)  # (batch_size, 500, 994)
            
            all_predictions.append(batch_predictions.cpu().numpy())
            
            if (batch_idx + 1) % max(1, n_batches // 5) == 0:
                print(f"  Processed {end_idx}/{n_samples} samples")
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)  # (n_samples, 500, 994)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"Predictions mean: {predictions.mean():.6f}")
    print(f"Predictions std: {predictions.std():.6f}")
    
    # Save predictions if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving predictions to: {output_path}")
        
        if output_path.suffix == '.mat':
            scipy.io.savemat(str(output_path), {'predictions': predictions})
        elif output_path.suffix == '.npy':
            np.save(str(output_path), predictions)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        print(f"✓ Predictions saved!")
    
    return predictions


def run_inference_with_localization(
    checkpoint_path,
    eeg_data,
    output_path=None,
    device='auto',
    batch_size=4,
    threshold_method='relative',
    threshold_value=0.5,
    percentile=90,
    detect_peaks_flag=True,
    peak_distance=5,
    min_activation=None,
    save_sparse=True,
    save_dense=False
):
    """
    Run inference with source localization post-processing
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        eeg_data: EEG data array (n_samples, 500, 75)
        output_path: Base path for output files (without extension)
        device: 'auto', 'cpu', or 'cuda'
        batch_size: Batch size for inference
        threshold_method: 'absolute' or 'relative'
        threshold_value: Absolute threshold value
        percentile: Percentile for relative thresholding
        detect_peaks_flag: If True, use peak detection; else use thresholding
        peak_distance: Minimum distance between peaks
        min_activation: Minimum activation value for sparse output
        save_sparse: If True, save sparse predictions
        save_dense: If True, save dense thresholded predictions
    
    Returns:
        sparse_predictions: Sparse source predictions
        statistics: Dictionary with localization statistics
    """
    
    # Get dense predictions
    print("\n" + "="*60)
    print("STEP 1: Running dense inference")
    print("="*60)
    predictions = run_inference(
        checkpoint_path=checkpoint_path,
        eeg_data=eeg_data,
        output_path=None,  # Don't save dense yet
        device=device,
        batch_size=batch_size
    )
    
    # Apply localization post-processing
    print("\n" + "="*60)
    print("STEP 2: Applying source localization post-processing")
    print("="*60)
    
    sparse_predictions, stats = get_sparse_predictions(
        predictions,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        percentile=percentile,
        detect_peaks_flag=detect_peaks_flag,
        distance=peak_distance,
        min_activation=min_activation
    )
    
    print("\nLocalization Statistics:")
    print("-" * 50)
    print(f"Total activated regions: {stats['total_entries']}")
    print(f"Number of samples: {stats['n_samples']}")
    print(f"Number of active sources: {stats['n_active_sources']}")
    print(f"Sparsity: {stats['sparsity']*100:.2f}% (zeros after localization)")
    
    # Save outputs
    if output_path:
        output_base = Path(output_path).stem
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_sparse:
            # Save sparse predictions as MAT file
            sparse_mat_path = output_dir / f"{output_base}_sparse_predictions.mat"
            sparse_dict = {
                'sample_id': np.array([p['sample_id'] for p in sparse_predictions]),
                'source_id': np.array([p['source_id'] for p in sparse_predictions]),
                'time': np.array([p['time'] for p in sparse_predictions]),
                'activation': np.array([p['activation'] for p in sparse_predictions])
            }
            scipy.io.savemat(str(sparse_mat_path), sparse_dict)
            print(f"\n✓ Sparse predictions saved: {sparse_mat_path}")
        
        if save_dense:
            # Save thresholded dense predictions
            thresholded, _ = apply_threshold(
                predictions,
                threshold_method=threshold_method,
                threshold_value=threshold_value,
                percentile=percentile
            )
            dense_mat_path = output_dir / f"{output_base}_thresholded.mat"
            scipy.io.savemat(str(dense_mat_path), {'predictions': thresholded})
            print(f"✓ Thresholded dense predictions saved: {dense_mat_path}")
    
    return sparse_predictions, stats


def evaluate_with_ground_truth(predictions, source_data):
    """
    Evaluate predictions against ground truth source data
    
    Args:
        predictions: Model predictions (n_samples, 500, 994)
        source_data: Ground truth source data (n_samples, 500, 994)
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("\nEvaluating predictions against ground truth...")
    
    # Reshape for metric computation
    pred_flat = predictions.reshape(-1, 994)  # (n_samples * 500, 994)
    source_flat = source_data.reshape(-1, 994)
    
    metrics = evaluate_temporal_metrics(pred_flat, source_flat)
    
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"{key:.<40} {value:.6f}")
            else:
                print(f"{key:.<40} {value}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Inference script for Vision Transformer ESI model with source localization'
    )
    
    # Model and data arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (e.g., logs/exp3/checkpoints/best_vit_model.pth)')
    parser.add_argument('--eeg_data', type=str, required=True,
                        help='Path to EEG data (MAT file or directory with MAT files)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to load (None = load all)')
    parser.add_argument('--start_sample', type=int, default=0,
                        help='Start index for sample range (inclusive). Default: 0')
    parser.add_argument('--end_sample', type=int, default=None,
                        help='End index for sample range (exclusive). Default: None (loads to end)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (e.g., predictions.mat or predictions.npy)')
    parser.add_argument('--source_data', type=str, default=None,
                        help='Path to ground truth source data (optional, for evaluation)')
    
    # Inference arguments
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    
    # Source localization arguments
    parser.add_argument('--enable_localization', action='store_true',
                        help='Enable source localization post-processing (thresholding + peak detection)')
    parser.add_argument('--threshold_method', type=str, default='relative', choices=['absolute', 'relative'],
                        help='Thresholding method: absolute (fixed value) or relative (percentile)')
    parser.add_argument('--threshold_value', type=float, default=0.5,
                        help='Absolute threshold value (used if threshold_method=absolute)')
    parser.add_argument('--percentile', type=int, default=90,
                        help='Percentile threshold for relative thresholding (e.g., 90 keeps top 10%)')
    parser.add_argument('--detect_peaks', action='store_true', default=True,
                        help='Use temporal peak detection instead of simple thresholding')
    parser.add_argument('--peak_distance', type=int, default=5,
                        help='Minimum distance between detected peaks (in time steps)')
    parser.add_argument('--min_activation', type=float, default=None,
                        help='Minimum activation value for sparse output')
    parser.add_argument('--save_sparse', action='store_true', default=True,
                        help='Save sparse predictions (source_id, time, activation)')
    parser.add_argument('--save_dense', action='store_true',
                        help='Also save thresholded dense predictions')
    
    args = parser.parse_args()
    
    # Load EEG data
    print("=" * 60)
    print("Vision Transformer ESI - Inference with Source Localization")
    print("=" * 60)
    
    eeg_data = load_eeg_data(
        args.eeg_data,
        max_samples=args.max_samples,
        start_sample=args.start_sample,
        end_sample=args.end_sample
    )
    print(f"EEG data shape: {eeg_data.shape}")
    
    # Run inference with or without localization
    if args.enable_localization:
        print("\n✓ Source localization enabled")
        sparse_predictions, stats = run_inference_with_localization(
            checkpoint_path=args.checkpoint,
            eeg_data=eeg_data,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
            threshold_method=args.threshold_method,
            threshold_value=args.threshold_value,
            percentile=args.percentile,
            detect_peaks_flag=args.detect_peaks,
            peak_distance=args.peak_distance,
            min_activation=args.min_activation,
            save_sparse=args.save_sparse,
            save_dense=args.save_dense
        )
        predictions = sparse_predictions
    else:
        print("\n✓ Running standard dense inference (no localization)")
        predictions = run_inference(
            checkpoint_path=args.checkpoint,
            eeg_data=eeg_data,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size
        )
    
    # Evaluate against ground truth if provided
    if args.source_data and not args.enable_localization:
        print("\n" + "=" * 60)
        source_data = load_eeg_data(args.source_data)
        if source_data.shape[-1] == 75:
            # This is actually EEG, try to load it differently
            print("Note: source_data appears to be EEG format, skipping evaluation")
        else:
            evaluate_with_ground_truth(predictions, source_data)
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    
    return predictions


if __name__ == "__main__":
    main()
