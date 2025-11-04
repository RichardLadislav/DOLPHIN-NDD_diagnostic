# -*- coding: utf-8 -*-
"""
Split handwriting dataset (e.g., LBD_CZ_002-tf.pkl) task-wise.
Each resulting file contains all subjects but only one task index.
"""

import pickle
import os
from pathlib import Path
from joblib import load, dump  # works with joblib or pickle files
from tqdm import tqdm

def load_any(path):
    """Try joblib first; fall back to pickle."""
    path = Path(path)
    try:
        return load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="iso-8859-1")

def save_pkl(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=5)

def split_taskwise(src_path: str, out_dir: str):
    """
    Parameters
    ----------
    src_path : str
        Path to the full LBD_CZ_002-tf.pkl file.
    out_dir : str
        Directory where separate task files will be saved.
    """
    src_path = Path(src_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] {src_path}")
    data = load_any(src_path)

    # determine maximum number of tasks per subject
    max_tasks = max(len(v) for v in data.values())
    print(f"Detected up to {max_tasks} tasks per subject.")

    # initialize per-task dicts
    task_dicts = [dict() for _ in range(max_tasks)]

    # iterate over subjects
    for subj, samples in tqdm(data.items(), desc="Splitting tasks", unit="subject"):
        for t_idx, arr in enumerate(samples):
            task_dicts[t_idx][subj] = [arr]

    # save per-task pickle
    for t_idx, task_data in enumerate(task_dicts):
        if len(task_data) == 0:
            continue
        out_path = out_dir / f"LBD_CZ_002-task{t_idx+1}.pkl"
        save_pkl(task_data, out_path)
        print(f"[SAVE] Task {t_idx+1}: {len(task_data)} subjects → {out_path}")

    print("✅ Done splitting dataset by task.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Split handwriting dataset (e.g., LBD_CZ_002-tf.pkl) task-wise")
    ap.add_argument("--src", type=str, required=True, help="Path to LBD_CZ_002-tf.pkl")
    ap.add_argument("--out", type=str, default="./data/LBD_CZ_002_tasks", help="Output directory")
    args = ap.parse_args()

    split_taskwise(args.src, args.out)
