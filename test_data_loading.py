"""
Quick test script to verify data loading works correctly
"""
import sys
from pathlib import Path
from train_model import load_mat_files
import numpy as np

# Test data loading
data_dir = Path('labeled_spikes_data') / 'labeled_spikes_data'

print("Testing data loading...")
try:
    eeg_data, source_data = load_mat_files(data_dir)
    
    print("\n✓ Data loaded successfully!")
    print(f"  EEG data shape: {eeg_data.shape}")
    print(f"  Source data shape: {source_data.shape}")
    print(f"  EEG data dtype: {eeg_data.dtype}")
    print(f"  Source data dtype: {source_data.dtype}")
    print(f"  EEG data range: [{eeg_data.min():.4f}, {eeg_data.max():.4f}]")
    print(f"  Source data range: [{source_data.min():.4f}, {source_data.max():.4f}]")
    
    # Check for NaN or Inf
    if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
        print("  ⚠ Warning: EEG data contains NaN or Inf values")
    else:
        print("  ✓ EEG data is clean (no NaN/Inf)")
    
    if np.any(np.isnan(source_data)) or np.any(np.isinf(source_data)):
        print("  ⚠ Warning: Source data contains NaN or Inf values")
    else:
        print("  ✓ Source data is clean (no NaN/Inf)")
    
    print("\n✓ All checks passed! Ready to train.")
    
except Exception as e:
    print(f"\n✗ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

