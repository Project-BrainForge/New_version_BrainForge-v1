#!/usr/bin/env python
"""
Inference script for Vision Transformer ESI model
Loads trained model checkpoint and predicts source activity from EEG data
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


def load_eeg_data(data_path, max_samples=None):
    """
    Load EEG data from MAT file or directory
    
    Args:
        data_path: Path to MAT file or directory containing MAT files
        max_samples: Maximum number of samples to load (None = load all)
    
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
        eeg_data, _ = load_mat_files(data_path, max_samples=max_samples)
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
        description='Inference script for Vision Transformer ESI model'
    )
    
    # Model and data arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (e.g., logs/exp3/checkpoints/best_vit_model.pth)')
    parser.add_argument('--eeg_data', type=str, required=True,
                        help='Path to EEG data (MAT file or directory with MAT files)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to load (None = load all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions (e.g., predictions.mat or predictions.npy)')
    parser.add_argument('--source_data', type=str, default=None,
                        help='Path to ground truth source data (optional, for evaluation)')
    
    # Inference arguments
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for inference')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Load EEG data
    print("=" * 60)
    print("Vision Transformer ESI - Inference")
    print("=" * 60)
    
    eeg_data = load_eeg_data(args.eeg_data, max_samples=args.max_samples)
    print(f"EEG data shape: {eeg_data.shape}")
    
    # Run inference
    predictions = run_inference(
        checkpoint_path=args.checkpoint,
        eeg_data=eeg_data,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Evaluate against ground truth if provided
    if args.source_data:
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
