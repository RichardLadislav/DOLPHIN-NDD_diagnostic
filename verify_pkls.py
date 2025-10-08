# -*- coding: utf-8 -*-
import os, sys, pickle, traceback
import numpy as np

def try_joblib(path):
    try:
        from joblib import load
        obj = load(path, mmap_mode=None)  # mmap ignored for compressed; OK
        print("  joblib.load: OK")
        return obj
    except Exception as e:
        print("  joblib.load: FAIL ->", repr(e))
        traceback.print_exc()
        return None

def try_pickle(path):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print("  pickle.load: OK")
        return obj
    except Exception as e:
        print("  pickle.load: FAIL ->", repr(e))
        traceback.print_exc()
        return None

def check_file(path):
    print(f"\n--- Checking: {path} ---")
    if not os.path.exists(path):
        print("  ❌ File not found")
        return
    size_gb = os.path.getsize(path)/(1024**3)
    print(f"  size ≈ {size_gb:.2f} GB")

    obj = try_joblib(path)
    if obj is None:
        obj = try_pickle(path)

    if obj is None:
        print("  ❌ Could not load with joblib or pickle (likely corrupted).")
        return

    if not isinstance(obj, dict):
        print(f"  ⚠️ Unexpected root type: {type(obj)} (expected dict)")
        return

    # light stats without loading everything
    keys = list(obj.keys())
    n_writers = len(keys)
    lengths = [len(obj[k]) for k in keys]
    n_samples = sum(lengths)
    example = obj[keys[0]][0]
    ex_shape = np.asarray(example).shape
    ex_dtype = np.asarray(example).dtype
    print(f"  ✅ Structure OK: writers={n_writers}, samples={n_samples}, avg/writer={np.mean(lengths):.1f}")
    print(f"  Example array: shape={ex_shape}, dtype={ex_dtype}")

if __name__ == "__main__":
    check_file(r".\data\OLIWER\train-tf.pkl")
    check_file(r".\data\OLIWER\test-tf.pkl")
