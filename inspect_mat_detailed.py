#!/usr/bin/env python
"""Check detailed structure of MAT files"""

import scipy.io
import numpy as np
from pathlib import Path

data_dir = Path('labeled_spikes_data')
mat_files = sorted([f for f in data_dir.glob('*.mat') if 'sample_' in f.name])

if mat_files:
    print(f"Found {len(mat_files)} MAT files")
    print(f"\nInspecting first file: {mat_files[0].name}")
    
    data = scipy.io.loadmat(str(mat_files[0]))
    
    print(f"\nFull keys and details:")
    for key in data.keys():
        if not key.startswith('__'):
            value = data[key]
            print(f"\n{key}:")
            print(f"  shape: {value.shape}")
            print(f"  dtype: {value.dtype}")
            if value.size < 20:
                print(f"  value: {value}")
            else:
                print(f"  first few elements: {value.flat[:10]}")
    
    # Check if labels could be source data
    if 'labels' in data:
        labels = data['labels']
        print(f"\n\nLabels shape: {labels.shape}")
        print(f"Labels min: {labels.min()}, max: {labels.max()}")
        print(f"Labels non-zero count: {np.count_nonzero(labels)}")
else:
    print("No MAT files found")
