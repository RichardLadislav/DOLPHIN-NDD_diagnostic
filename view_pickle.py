# -*- coding: utf-8 -*-
"""
Command-line viewer for handwriting .pkl / .joblib datasets.
Usage:
    python view_pickle.py --path ./data/LBD_CZ_002/LBD_CZ_002.pkl
"""

import argparse, pickle
from pathlib import Path
import numpy as np

try:
    from joblib import load as joblib_load
except ImportError:
    joblib_load = None


def load_any_pkl(path):
    """Load pickle or joblib file safely."""
    try:
        if joblib_load is not None:
            return joblib_load(path)
    except Exception:
        pass
    with open(path, "rb") as f:
        return pickle.load(f, encoding="iso-8859-1")


def summarize_dataset(data):
    """Print summary: number of writers, samples, and shape stats."""
    if not isinstance(data, dict):
        print(f"[!] Unexpected format: {type(data)}")
        return

    n_writers = len(data)
    n_samples = sum(len(v) for v in data.values())
    print(f"\n📂 Writers: {n_writers} | ✍️ Samples: {n_samples}")

    # gather lengths and feature dims
    lens, feats = [], []
    for w, samples in list(data.items())[:20]:  # inspect first 20 writers
        for arr in samples:
            if isinstance(arr, np.ndarray):
                lens.append(arr.shape[0])
                feats.append(arr.shape[1] if arr.ndim == 2 else 1)

    if lens and feats:
        print(f"⏱ Avg length: {np.mean(lens):.1f} ± {np.std(lens):.1f}")
        print(f"🧩 Feature dims: {set(feats)}")
        print(f"📏 Min/Max length: {min(lens)} / {max(lens)}")

    print("\nExample writer IDs:", list(data.keys())[:5])


def inspect_writer(data, writer_id):
    """Print details for a specific writer."""
    if writer_id not in data:
        print(f"[!] Writer '{writer_id}' not found.")
        return
    samples = data[writer_id]
    print(f"\nWriter {writer_id}: {len(samples)} samples")
    for i, arr in enumerate(samples[:5]):
        if isinstance(arr, np.ndarray):
            print(f"  Sample {i}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  Sample {i}: type={type(arr)}")


def main():
    parser = argparse.ArgumentParser(description="View summary of .pkl handwriting dataset.")
    parser.add_argument('--path', type=str, required=True, help="Path to .pkl or .joblib file.")
    parser.add_argument('--writer', type=str, default=None,
                        help="Optional writer ID to inspect in detail.")
    args = parser.parse_args()

    path = Path(args.path)
    assert path.exists(), f"File not found: {path}"

    print(f"\n🔍 Loading {path} ...")
    data = load_any_pkl(str(path))
    summarize_dataset(data)

    if args.writer:
        inspect_writer(data, args.writer)


if __name__ == '__main__':
    main()
