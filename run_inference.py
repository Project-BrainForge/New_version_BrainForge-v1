import argparse
import os
import time
from pathlib import Path
import numpy as np
import glob
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


def process_mat_files(folder_name, model, device, model_id, checkpoint_name, normalize=True, output_dir='result/hybrid', data_pattern='sample_*_eeg.mat', batch_size=32, save_individual=False):
    """
    Process all MAT files in folder and save predictions
    
    Args:
        folder_name: Folder containing input MAT files
        model: Trained model
        device: Device to run inference on
        model_id: Model ID for output naming
        checkpoint_name: Checkpoint name for output naming
        normalize: Whether to normalize data by max absolute value
        output_dir: Directory to save output files
        data_pattern: File pattern to match (e.g., 'sample_*_eeg.mat' or 'data*.mat')
        batch_size: Batch size for processing
        save_individual: Save individual prediction files
    """
    folder_path = Path(folder_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching MAT files
    flist = glob.glob(str(folder_path / data_pattern))
    
    if len(flist) == 0:
        print(f'WARNING: NO FILES FOUND in {folder_name} matching pattern {data_pattern}')
        return None
    
    # Sort files based on natural number
    flist = sorted(flist)
    
    print(f"Found {len(flist)} files in {folder_name}")
    
    test_data = []
    file_names = []
    
    # Load and preprocess data
    for i in flist:
        try:
            # Try loading with 'eeg_data' key first (new format)
            data_dict = loadmat(i)
            if 'eeg_data' in data_dict:
                data = data_dict['eeg_data']  # Shape: (500, 75)
            elif 'data' in data_dict:
                data = data_dict['data']  # Shape: (500, 75) - old format
            else:
                print(f"  WARNING: {os.path.basename(i)} - no 'eeg_data' or 'data' key found")
                continue
            
            # Normalize by max absolute value
            if normalize:
                data = data / np.max(np.abs(data[:]))
            
            test_data.append(data)
            file_names.append(os.path.basename(i))
        except Exception as e:
            print(f"  Error loading {i}: {e}")
            continue
    
    if len(test_data) == 0:
        print(f'WARNING: No valid data loaded from {folder_name}')
        return None
    
    # Process in batches
    all_out = []
    
    for batch_idx in range(0, len(test_data), batch_size):
        batch_end = min(batch_idx + batch_size, len(test_data))
        batch_data = test_data[batch_idx:batch_end]
        batch_names = file_names[batch_idx:batch_end]
        
        # Convert to tensor: (batch, 500, 75)
        data_tensor = torch.from_numpy(np.array(batch_data)).to(device, torch.float)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            out = model(data_tensor)  # Output: (batch, 500, 994)
        
        # Get predictions
        batch_out = out.detach().cpu().numpy()
        all_out.extend(batch_out)
        
        # Save individual files if requested
        if save_individual:
            for j, (pred, name) in enumerate(zip(batch_out, batch_names)):
                output_file = output_path / f'pred_{name}'
                try:
                    savemat(str(output_file), {'prediction': pred})
                except Exception as e:
                    print(f"  Error saving {output_file}: {e}")
    
    # Save aggregated results
    all_out_array = np.array(all_out)
    output_file = output_path / f'all_predictions_{checkpoint_name}.mat'
    savemat(str(output_file), {'all_out': all_out_array, 'file_names': file_names})
    
    print(f'Saved {len(all_out)} predictions to: {output_file}')
    
    return all_out_array


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
    start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EEG to Source Activity Model Inference')
    
    # ============ Model Arguments ============
    parser.add_argument('--device', default='cpu', type=str, help='Device to run on (cpu or cuda)')
    parser.add_argument('--model_id', type=int, default=1, help='Model ID')
    parser.add_argument('--model_type', type=str, default='hybrid',
                       choices=['hybrid', 'eeg_vit', 'vit_channel', 'cnn_vit'],
                       help='Type of model to use')
    parser.add_argument('--resume', default='best', type=str, 
                       help='Checkpoint to resume (e.g., epoch_010 or best)')
    parser.add_argument('--checkpoint', type=str, default='',
                       help='Direct path to checkpoint file (overrides model_id/resume)')
    parser.add_argument('--checkpoint_path', type=str, default='',
                       help='Alternative path argument for checkpoint file')
    
    # ============ Data Arguments ============
    parser.add_argument('--input_dir', type=str, default='labeled_spikes_data/labeled_spikes_data',
                       help='Input directory containing MAT files')
    parser.add_argument('--output_dir', type=str, default='result/hybrid',
                       help='Output directory to save predictions')
    parser.add_argument('--subject_list', type=str, nargs='+', default=['VEP'],
                       help='List of subjects/folders to process (e.g., VEP or source/VEP)')
    parser.add_argument('--data_pattern', type=str, default='sample_*_eeg.mat',
                       help='File pattern to match (e.g., data*.mat or sample_*_eeg.mat)')
    
    # ============ Processing Arguments ============
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize data by max absolute value')
    parser.add_argument('--save_individual', action='store_true', default=False,
                       help='Save individual prediction files')
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='Evaluate predictions if ground truth available')
    
    # ============ Other Arguments ============
    parser.add_argument('--info', default='', type=str, help='Additional information about this model')
    
    args = parser.parse_args()
    
    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = torch.cuda.is_available() and args.device != 'cpu'
    device = torch.device(args.device if use_cuda else "cpu")
    
    print("=" * 60)
    print("EEG to Source Activity Inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model ID: {args.model_id}")
    print(f"Model Type: {args.model_type}")
    print("=" * 60)
    
    # =============================== LOAD MODEL =====================================================================================================
    if args.checkpoint:
        # Use --checkpoint argument (preferred)
        fn = args.checkpoint
        checkpoint_name = os.path.basename(fn).replace('.pth', '').replace('.tar', '')
    elif args.checkpoint_path:
        # Use checkpoint_path argument
        fn = args.checkpoint_path
        checkpoint_name = os.path.basename(fn).replace('.pth', '').replace('.tar', '')
    else:
        # Use model_id structure (like original)
        result_root = f'logs/experiment_*/checkpoints'  # Search in logs
        # Try to find checkpoint
        if args.resume:
            if args.resume == 'best':
                fn_pattern = f'logs/*/checkpoints/best_{args.model_type}_model.pth'
            else:
                fn_pattern = f'logs/*/checkpoints/checkpoint_{args.resume}.pth'
        else:
            fn_pattern = f'logs/*/checkpoints/best_{args.model_type}_model.pth'
        
        import glob as glob_module
        matches = glob_module.glob(fn_pattern)
        if len(matches) == 0:
            print(f"ERROR: No checkpoint found matching pattern: {fn_pattern}")
            return
        fn = matches[0]  # Use first match
        checkpoint_name = os.path.basename(fn).replace('.pth', '').replace('.tar', '')
    
    print(f"=> Load checkpoint: {fn}")
    
    if not os.path.isfile(fn):
        print(f"ERROR: no checkpoint found at {fn}")
        return
    
    # Load model
    try:
        model, checkpoint = load_model_checkpoint(
            fn,
            model_type=args.model_type,
            device=device
        )
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))
    print('Prepare time:', time.time() - start_time)
    
    # =============================== EVALUATION =====================================================================================================
    model.eval()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data pattern: {args.data_pattern}")
    print("=" * 60)
    
    # Create output directory
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process input directory
    input_dir_path = Path(args.input_dir)
    
    if not input_dir_path.exists():
        print(f'ERROR: Input directory {args.input_dir} does not exist')
        return
    
    process_start = time.time()
    
    # Process files in input directory
    all_out = process_mat_files(
        str(input_dir_path),
        model,
        device,
        args.model_id,
        checkpoint_name,
        normalize=args.normalize,
        output_dir=args.output_dir,
        data_pattern=args.data_pattern,
        batch_size=args.batch_size,
        save_individual=args.save_individual
    )
    
    if all_out is not None:
        print(f'Processed {args.input_dir} in {time.time() - process_start:.2f} seconds')
    
    print(f'\nTotal run time: {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    main()

