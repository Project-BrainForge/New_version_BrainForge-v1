import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
from tqdm import tqdm
import sys

# Import models
from train_model import (
    CNNTransformerHybrid, 
    load_mat_files,
    predict_source_activity,
    evaluate_temporal_metrics
)

# Import EEG-ViT models
sys.path.append(str(Path(__file__).parent / 'eeg_vit_model'))
try:
    from train_eeg_vit import (
        EEGVisionTransformer,
        EEGViTWithChannelAttention,
        CNNViTHybrid
    )
    EEG_VIT_AVAILABLE = True
except ImportError:
    EEG_VIT_AVAILABLE = False
    print("Warning: EEG-ViT models not available")


def load_model_checkpoint(checkpoint_path, model_type='hybrid', device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Type of model ('hybrid', 'eeg_vit', 'vit_channel', 'cnn_vit')
        device: Device to load model on
    
    Returns:
        model: Loaded model
        checkpoint: Checkpoint dictionary
    """
    print(f"=> Loading checkpoint: {checkpoint_path}")
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    # Get model configuration from checkpoint
    config = checkpoint.get('config', {})
    
    # Initialize model based on type
    if model_type == 'hybrid' or model_type == 'cnn_transformer':
        model = CNNTransformerHybrid(
            input_dim=75,
            output_dim=994,
            cnn_channels=config.get('cnn_channels', [75, 128, 256, 256]),
            kernel_sizes=config.get('kernel_sizes', [3, 5, 3]),
            d_model=config.get('d_model', 256),
            nhead=config.get('nhead', 8),
            num_transformer_layers=config.get('num_transformer_layers', 4),
            dropout=config.get('dropout', 0.2),
            use_residual=True
        )
    
    elif model_type == 'eeg_vit':
        if not EEG_VIT_AVAILABLE:
            raise ImportError("EEG-ViT models not available")
        model = EEGVisionTransformer(
            input_dim=75,
            output_dim=994,
            time_points=500,
            patch_size=config.get('patch_size', 5),
            d_model=config.get('d_model', 256),
            depth=config.get('depth', 8),
            heads=config.get('heads', 8),
            dropout=config.get('dropout', 0.1)
        )
    
    elif model_type == 'vit_channel':
        if not EEG_VIT_AVAILABLE:
            raise ImportError("EEG-ViT models not available")
        model = EEGViTWithChannelAttention(
            input_dim=75,
            output_dim=994,
            time_points=500,
            patch_size=config.get('patch_size', 5),
            d_model=config.get('d_model', 256),
            depth=config.get('depth', 6),
            heads=config.get('heads', 8),
            dropout=config.get('dropout', 0.1)
        )
    
    elif model_type == 'cnn_vit':
        if not EEG_VIT_AVAILABLE:
            raise ImportError("EEG-ViT models not available")
        model = CNNViTHybrid(
            input_dim=75,
            output_dim=994,
            time_points=500,
            patch_size=config.get('patch_size', 5),
            vit_dim=config.get('d_model', 256),
            depth=config.get('depth', 4),
            heads=config.get('heads', 8),
            dropout=config.get('dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"=> Loaded checkpoint: {checkpoint_path}")
    print(f"=> Model type: {model_type}")
    print(f"=> Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"=> Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"=> Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, checkpoint


def process_mat_files(input_dir, output_dir, model, device, batch_size=32):
    """
    Process all MAT files in input directory and save predictions
    
    Args:
        input_dir: Directory containing input MAT files
        output_dir: Directory to save output MAT files
        model: Trained model
        device: Device to run inference on
        batch_size: Batch size for inference
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all MAT files
    mat_files = sorted([f for f in input_dir.glob('*.mat') if 'sample_' in f.name])
    
    if len(mat_files) == 0:
        print(f'WARNING: No MAT files found in {input_dir}')
        return
    
    print(f"Found {len(mat_files)} MAT files to process")
    
    all_predictions = []
    all_targets = []
    file_names = []
    
    # Process files in batches
    for i in tqdm(range(0, len(mat_files), batch_size), desc="Processing batches"):
        batch_files = mat_files[i:i+batch_size]
        batch_eeg = []
        batch_source = []
        batch_names = []
        
        for mat_file in batch_files:
            try:
                data = loadmat(str(mat_file))
                
                if 'eeg_data' not in data or 'source_data' not in data:
                    continue
                
                eeg = data['eeg_data']  # Shape: (500, 75)
                source = data['source_data']  # Shape: (500, 994)
                
                # Normalize if needed (optional)
                # eeg = eeg / np.max(np.abs(eeg))
                
                batch_eeg.append(eeg)
                batch_source.append(source)
                batch_names.append(mat_file.name)
                
            except Exception as e:
                print(f"Error loading {mat_file.name}: {e}")
                continue
        
        if len(batch_eeg) == 0:
            continue
        
        # Convert to tensors
        batch_eeg_tensor = torch.FloatTensor(np.array(batch_eeg)).to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(batch_eeg_tensor)
            predictions_np = predictions.cpu().numpy()
        
        # Store results
        all_predictions.extend(predictions_np)
        all_targets.extend(batch_source)
        file_names.extend(batch_names)
        
        # Save individual predictions
        for j, (pred, name) in enumerate(zip(predictions_np, batch_names)):
            output_file = output_dir / f'pred_{name}'
            savemat(str(output_file), {
                'prediction': pred,
                'eeg_data': batch_eeg[j],
                'source_data': batch_source[j],
                'file_name': name
            })
    
    return all_predictions, all_targets, file_names


def evaluate_predictions(predictions, targets):
    """
    Evaluate predictions against targets
    
    Args:
        predictions: List of prediction arrays
        targets: List of target arrays
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = evaluate_temporal_metrics(predictions, targets)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run Inference on EEG to Source Activity Models')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='hybrid',
                       choices=['hybrid', 'eeg_vit', 'vit_channel', 'cnn_vit'],
                       help='Type of model to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    # Data arguments
    parser.add_argument('--input_dir', type=str, 
                       default='labeled_spikes_data/labeled_spikes_data',
                       help='Directory containing input MAT files')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save output predictions')
    parser.add_argument('--subject', type=str, default=None,
                       help='Specific subject/folder to process (optional)')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual prediction files')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate predictions against ground truth')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("EEG to Source Activity Inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load model
    try:
        model, checkpoint = load_model_checkpoint(
            args.checkpoint, 
            model_type=args.model_type,
            device=device
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        return
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Prepare input directory
    input_dir = Path(args.input_dir)
    if args.subject:
        input_dir = input_dir / args.subject
    
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    print("\n" + "=" * 60)
    print("Processing MAT files...")
    print("=" * 60)
    
    process_start = time.time()
    
    try:
        predictions, targets, file_names = process_mat_files(
            input_dir,
            output_dir,
            model,
            device,
            batch_size=args.batch_size
        )
        
        print(f"\nProcessed {len(predictions)} files in {time.time() - process_start:.2f} seconds")
        
        # Save aggregated results
        if len(predictions) > 0:
            aggregated_file = output_dir / f'all_predictions_{args.model_type}.mat'
            savemat(str(aggregated_file), {
                'predictions': np.array(predictions),
                'targets': np.array(targets) if targets else None,
                'file_names': file_names,
                'model_type': args.model_type,
                'checkpoint': args.checkpoint
            })
            print(f"Saved aggregated results to: {aggregated_file}")
        
        # Evaluate if requested and targets are available
        if args.evaluate and targets and len(targets) > 0:
            print("\n" + "=" * 60)
            print("Evaluation Metrics")
            print("=" * 60)
            
            metrics = evaluate_predictions(predictions, targets)
            
            for key, value in metrics.items():
                if value is not None:
                    print(f"{key}: {value:.6f}")
            
            # Save metrics
            metrics_file = output_dir / f'evaluation_metrics_{args.model_type}.mat'
            savemat(str(metrics_file), metrics)
            print(f"\nSaved metrics to: {metrics_file}")
        
    except Exception as e:
        print(f"ERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

