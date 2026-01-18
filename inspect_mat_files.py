#!/usr/bin/env python
"""Inspect MAT file structure to understand the data format"""

import scipy.io
from pathlib import Path

data_dir = Path('labeled_spikes_data')
mat_files = sorted([f for f in data_dir.glob('*.mat') if 'sample_' in f.name])

if mat_files:
    print(f"Found {len(mat_files)} MAT files")
    print(f"\nInspecting first file: {mat_files[0].name}")
    
    data = scipy.io.loadmat(str(mat_files[0]))
    
    print(f"\nKeys in MAT file:")
    for key in data.keys():
        if not key.startswith('__'):
            value = data[key]
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
else:
    print("No MAT files found")
