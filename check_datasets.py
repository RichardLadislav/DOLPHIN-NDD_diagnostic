# -*- coding: utf-8 -*-
"""
Quick integrity and sanity check for train-tf.pkl and test-tf.pkl
"""

import os
from joblib import load
import numpy as np

def check_file(path):
    print(f"\n--- Checking: {path} ---")
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    size_gb = os.path.getsize(path) / (1024 ** 3)
    print(f"File size: {size_gb:.2f} GB")

    try:
        # mmap_mode='r' prevents full loading into RAM
        data = load(path, mmap_mode='r')
        if not isinstance(data, dict):
            print("❌ Unexpected structure:", type(data))
            return

        num_writers = len(data)
        lengths = [len(v) for v in data.values() if hasattr(v, '__len__')]
        num_samples = sum(lengths)
        avg_len = np.mean(lengths) if lengths else 0
        example_key = next(iter(data.keys()))
        example_arr = data[example_key][0]

        print(f"✅ Successfully loaded.")
        print(f" Writers : {num_writers}")
        print(f" Samples : {num_samples}")
        print(f" Avg samples/writer : {avg_len:.1f}")
        print(f" Example shape : {np.asarray(example_arr).shape}")
        print(f" Example dtype : {np.asarray(example_arr).dtype}")
    except Exception as e:
        print("❌ Failed to load:", e)

for fname in ["./data/OLIWER/train-tf.pkl", "./data/OLIWER/test-tf.pkl"]:
    check_file(fname)
