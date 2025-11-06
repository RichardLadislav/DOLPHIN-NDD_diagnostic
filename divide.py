# -*- coding:utf-8 -*-
"""
Memory-safe data divider + feature extractor for OLIWER.

- Keeps SINGLE big files per split:
  ./data/OLIWER/train.pkl
  ./data/OLIWER/test.pkl
  ./data/OLIWER/train-tf.pkl
  ./data/OLIWER/test-tf.pkl

- Uses in-place transforms + joblib.dump for low memory footprint.
- Adds progress bar (tqdm), atomic writes, and partial save on errors.
"""

import os
import gc
import glob
import json
import pickle
import argparse
import tempfile
import numpy as np
from tqdm import tqdm
from joblib import dump, load  # load is here if you later want mmap_mode='r'
from pathlib import Path
from utils import time_functions, clock


# ---------------------------
# Atomic single-file writer
# ---------------------------

def _safe_dump_single_file(obj, out_path, compress=('lz4', 3), protocol=5):
    """
    Atomic write of ONE file:
      1) dump to a temp file in the same directory
      2) os.replace() to final path

    Keeps a single .pkl file as required by the pipeline.
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", dir=out_dir)
    os.close(fd)
    try:
        dump(obj, tmp_path, compress=compress, protocol=protocol)
        os.replace(tmp_path, out_path)
    finally:
        # If anything went wrong before replace, clean up tmp
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------
# Merge + split
# ---------------------------

def divide_data(src_root='./data', tgt_root='./data/OLIWER', seed=123):
    """
    Merges OLHWDB2, DCOH-E, COUCH09 into OLIWER.pkl (single file),
    then creates train/test splits and writes train.pkl, test.pkl (single files).
    """
    oliwer_pkl = f'{tgt_root}/OLIWER.pkl'
    if not os.path.exists(oliwer_pkl):
        print('Merging data.')
        with open(f'{src_root}/OLHWDB2/OLHWDB2.pkl','rb') as f:
            olhwdb2 = pickle.load(f, encoding='iso-8859-1')
        with open(f'{src_root}/DCOH-E/DCOH-E.pkl','rb') as f:
            dcohe = pickle.load(f, encoding='iso-8859-1')
        with open(f'{src_root}/COUCH09/COUCH09.pkl','rb') as f:
            couch = pickle.load(f, encoding='iso-8859-1')

        olhwdb_cnt = dcohe_cnt = couch_cnt = 0
        olhwdb_writer_cnt = dcohe_writer_cnt = couch_writer_cnt = 0
        data = {}

        for k in olhwdb2:
            if len(olhwdb2[k]) <= 20:
                continue
            data[str(k)] = olhwdb2[k]
            olhwdb_cnt += len(olhwdb2[k]); olhwdb_writer_cnt += 1

        for k in couch:
            if len(couch[k]) <= 20:
                continue
            newk = f'couch{k}'
            data[newk] = couch[k]
            couch_cnt += len(couch[k]); couch_writer_cnt += 1

        for k in dcohe:
            if len(dcohe[k]) <= 20:
                continue
            data[k] = dcohe[k]
            dcohe_cnt += len(dcohe[k]); dcohe_writer_cnt += 1

        cnt = sum(len(v) for v in data.values())
        print('user:', len(data), 'sample:', cnt)
        print('dcohe samples:', dcohe_cnt, dcohe_writer_cnt)
        print('olhwdb2 samples:', olhwdb_cnt, olhwdb_writer_cnt)
        print('couch samples:', couch_cnt, couch_writer_cnt)

        os.makedirs(tgt_root, exist_ok=True)
        # single-file dump (joblib is robust for big numpy-heavy dicts)
        _safe_dump_single_file(data, oliwer_pkl, compress=('lz4', 3), protocol=5)
        del data, olhwdb2, dcohe, couch
        gc.collect()
    else:
        print('Loading existing data.')
        with open(oliwer_pkl, 'rb') as f:
            data = pickle.load(f, encoding='iso-8859-1')
        print('user:', len(data), 'sample:', np.sum([len(data[k]) for k in data.keys()]))

    split_path = f'{tgt_root}/split.json'
    if os.path.exists(split_path):
        with open(split_path, 'r', encoding='utf-8') as f:
            split = json.load(f)
        train_writers = split['train_writers']
        test_writers = split['test_writers']
        print('Loading existing splits.')
    else:
        # reproducible 80/20 split
        if 'data' not in locals():
            with open(oliwer_pkl, 'rb') as f:
                data = pickle.load(f, encoding='iso-8859-1')
        rng = np.random.default_rng(seed=seed)
        keys = list(data.keys())
        train_num = int(0.8 * len(keys))
        train_writers = list(rng.choice(keys, size=train_num, replace=False))
        test_writers = list(set(keys) - set(train_writers))
        split = {'train_writers': train_writers, 'test_writers': test_writers}
        os.makedirs(tgt_root, exist_ok=True)
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split, f, indent=4, ensure_ascii=False)
        print('Generating new splits.')

    # build small dicts and write single files
    if 'data' not in locals():
        with open(oliwer_pkl, 'rb') as f:
            data = pickle.load(f, encoding='iso-8859-1')

    train = {k: data[k] for k in train_writers}
    test  = {k: data[k] for k in test_writers}

    _safe_dump_single_file(train, f'{tgt_root}/train.pkl', compress=('lz4', 3), protocol=5)
    _safe_dump_single_file(test,  f'{tgt_root}/test.pkl',  compress=('lz4', 3), protocol=5)

    # free
    del data, train, test
    gc.collect()


# ---------------------------
# Feature extraction (ONE big file per split)
# ---------------------------

@clock
def extract_and_store(src_root='./data/OLIWER/train.pkl',
                      tgt_root='./data/OLIWER/train-tf.pkl',
                      compress=('lz4', 3),
                      progress=True,
                      gc_every=50):
    """
    - Loads source dict once.
    - Transforms in place: each sample -> time_functions(sample) -> float32 contiguous.
    - Writes ONE big file via atomic joblib dump.
    - Progress bar + partial save on error/interrupt.
    """
    with open(src_root, 'rb') as f:
        handwriting_info = pickle.load(f, encoding='iso-8859-1')

    keys = list(handwriting_info.keys())
    bar = tqdm(keys, desc=f"extract_and_store → {os.path.basename(tgt_root)}", unit="writer") if progress else keys

    processed_count = 0
    failed = []

    try:
        for k in bar:
            try:
                lst = handwriting_info[k]  # list of arrays
                # apply your feature transform
                for i in range(len(lst)):
                    lst[i] = time_functions(lst[i])
                # standardize dtype/layout to shrink file + speed I/O
                for i in range(len(lst)):
                    lst[i] = np.asarray(lst[i], dtype=np.float32, order='C')

                processed_count += 1
                if progress:
                    bar.set_postfix_str(f"ok={processed_count}, fail={len(failed)}", refresh=False)

                if gc_every and (processed_count % gc_every == 0):
                    gc.collect()

            except Exception as e:
                failed.append(k)
                if progress:
                    bar.set_postfix_str(f"ok={processed_count}, fail={len(failed)}", refresh=False)
                # keep going; we will save whatever we have

        # final single-file write (atomic)
        _safe_dump_single_file(handwriting_info, tgt_root, compress=compress, protocol=5)

    except KeyboardInterrupt:
        partial = tgt_root + ".partial"
        _safe_dump_single_file(handwriting_info, partial, compress=compress, protocol=5)
        print(f"\n[extract_and_store] Interrupted. Partial data saved to: {partial}")
        raise
    except Exception as e:
        partial = tgt_root + ".partial"
        try:
            _safe_dump_single_file(handwriting_info, partial, compress=compress, protocol=5)
            print(f"\n[extract_and_store] Error occurred. Partial data saved to: {partial}\nError: {e}")
        except Exception as e2:
            print(f"\n[extract_and_store] Error occurred and partial save failed: {e} / save_err={e2}")
        raise
    finally:
        if progress and hasattr(bar, "close"):
            bar.close()
        gc.collect()

    if failed:
        print(f"[extract_and_store] Completed with {len(failed)} failed writers: "
              f"{failed[:5]}{'...' if len(failed) > 5 else ''}")
    else:
        print("[extract_and_store] Completed successfully with no failed writers.")


# ---------------------------
# CLI
# ---------------------------

## PreLBD extraction adition 
def _robust_load_pkl(path):
    """Try joblib.load first (if compressed), else pickle.load."""
    try:
        from joblib import load as joblib_load
        return joblib_load(str(path))
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="iso-8859-1")

@clock
def extract_and_store_PRELBD(src_path="./data/LBD_CZ_002/LBD_CZ_002.pkl",
                             tgt_path="./data/LBD_CZ_002/LBD_CZ_002-tf.pkl",
                             compress=("lz4", 3),
                             progress=True,
                             gc_every=50,
                             p_col=None):
    """
    Build a dedicated PRELBD time-functions dataset:
      src_path: LBD_CZ_002.pkl (dict {writer_id: [np.ndarray(T,3 or more), ...]})
      tgt_path: LBD_CZ_002-tf.pkl (dict {writer_id: [np.ndarray(T,14), ...]})

    Notes:
    - If your LBD_CZ_002.pkl samples are already (T,3) = (x,y,p), leave p_col=None.
    - If they still contain extra columns and p is at a custom index, set p_col (e.g., p_col=6).
    - Writes ONE separate file for LBD_CZ_002 only; does NOT merge with OLIWER.
    """
    src_path = Path(src_path)
    tgt_path = Path(tgt_path)
    assert src_path.exists(), f"Missing source: {src_path}"

    handwriting_info = _robust_load_pkl(src_path)

    keys = list(handwriting_info.keys())
    bar = tqdm(keys, desc=f"LBD_CZ_002 → {tgt_path.name}", unit="writer") if progress else keys

    processed_count, failed = 0, []

    try:
        for k in bar:
            try:
                seq_list = handwriting_info[k]  # list of arrays
                for i in range(len(seq_list)):
                    arr = seq_list[i]

                    # If your stored arrays still have >3 cols, select (x,y,p) before TFs:
                    if p_col is not None and arr.shape[1] > 3:
                        arr = arr[:, [0, 1, p_col]].astype(np.float32, copy=False)

                    # Compute 14 time functions
                    tf = time_functions(arr)  # expects (T,3) -> returns (T,14)

                    # Standardize dtype/layout
                    seq_list[i] = np.asarray(tf, dtype=np.float32, order="C")

                processed_count += 1
                if progress:
                    bar.set_postfix_str(f"ok={processed_count}, fail={len(failed)}", refresh=False)

                if gc_every and (processed_count % gc_every == 0):
                    gc.collect()

            except Exception:
                failed.append(k)
                if progress:
                    bar.set_postfix_str(f"ok={processed_count}, fail={len(failed)}", refresh=False)
                # continue; we'll still save what we have

        # Save as a separate PRELBD-tf.pkl (atomic)
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        _safe_dump_single_file(handwriting_info, tgt_path.as_posix(), compress=compress, protocol=5)

    except KeyboardInterrupt:
        partial = tgt_path.as_posix() + ".partial"
        _safe_dump_single_file(handwriting_info, partial, compress=compress, protocol=5)
        print(f"\n[extract_and_store_PRELBD] Interrupted. Partial saved to: {partial}")
        raise
    except Exception as e:
        partial = tgt_path.as_posix() + ".partial"
        try:
            _safe_dump_single_file(handwriting_info, partial, compress=compress, protocol=5)
            print(f"\n[extract_and_store_PRELBD] Error. Partial saved to: {partial}\nError: {e}")
        except Exception as e2:
            print(f"\n[extract_and_store_PRELBD] Error and partial save failed: {e} / save_err={e2}")
        raise
    finally:
        if progress and hasattr(bar, "close"):
            bar.close()
        gc.collect()

    if failed:
        print(f"[extract_and_store_PRELBD] Completed with {len(failed)} failed writers: "
              f"{failed[:5]}{'...' if len(failed) > 5 else ''}")
    else:
        print("[extract_and_store_PRELBD] Completed successfully with no failed writers.")

@clock
def extract_and_store_PRELBD_prelbd( compress=("lz4", 3),
                              progress=True,
                              gc_every=50,
                              p_col=None):
    srcs = [
            './data/LBD_CZ_002_raw_tasks/1_1/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/3_1/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/3_2/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/3_3/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/3_4/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/3_5/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/9_1/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/10_1/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/15_1/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/16_1/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/17_1/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/18_1/LBD_CZ_002.pkl',
            './data/LBD_CZ_002_raw_tasks/19_1/LBD_CZ_002.pkl'
              ]
    for src_path in srcs:
        src = Path(src_path)
        # derive task name from folder, e.g. "3_2"
        task_name = src.parent.name
        # build target file path cleanly
        tgt_dir = Path(str(src.parent).replace("data", "data-tf"))
        tgt_dir.mkdir(parents=True, exist_ok=True)
        tgt_path = tgt_dir / f"{task_name}-tf.pkl"

        print(f"\n[Task {task_name}] Extracting → {tgt_path}")

        extract_and_store_PRELBD(
            src_path=str(src),
            tgt_path=str(tgt_path),
            compress=compress,
            progress=progress,
            gc_every=gc_every,
            p_col=p_col
        )
#    for src_path in srcs:
#        extract_and_store_PRELBD(
#            src_path=src_path,
#            tgt_path="./data/LBD_CZ_002/LBD_CZ_002-tf.pkl",
#            compress=compress,
#            progress=progress,
#            gc_every=gc_every,
#            p_col=p_col
#        )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--divide', action='store_true',
                        help="Merge datasets and create train/test splits (OLIWER).")
    parser.add_argument('--extract', action='store_true',
                        help="Run feature extraction on OLIWER train/test and write single big files.")
    parser.add_argument('--src_root', type=str, default='./data',
                        help="Root containing OLHWDB2, DCOH-E, COUCH09.")
    parser.add_argument('--tgt_root', type=str, default='./data/OLIWER',
                        help="Target root for OLIWER outputs.")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--no_compress', action='store_true',
                        help="Disable compression for fastest I/O.")
    parser.add_argument('--gc_every', type=int, default=50,
                        help="GC every N writers during extraction.")

    # --- NEW: standalone PRELBD TF extraction ---
    parser.add_argument('--extract_prelbd', action='store_true',
                        help="Extract time-functions for LBD_CZ_002 only (no merging with OLIWER).")
    parser.add_argument('--prelbd_src', type=str, default='./data/LBD_CZ_002/LBD_CZ_002.pkl',
                        help="Path to LBD_CZ_002.pkl (writer->list of (T,3 or more) arrays).")
    parser.add_argument('--prelbd_tgt', type=str, default='./data/LBD_CZ_002/LBD_CZ_002-tf.pkl',
                        help="Output path for LBD_CZ_002 time-functions pickle.")
    parser.add_argument('--p_col', type=int, default=None,
                        help="Pressure column index in LBD_CZ_002pkl if not already at index 2 (e.g., 6).")
    parser.add_argument('--extract_prelbd_task_wise', action='store_true',
                        help="Extract time-functions for LBD_CZ_002 tas-wise only (no merging with OLIWER).")
    args = parser.parse_args()

    compress = None if args.no_compress else ('lz4', 3)

    # ---- OLIWER pipeline (unchanged) ----
    if args.divide:
        divide_data(args.src_root, args.tgt_root, seed=args.seed)

    if args.extract:
        extract_and_store(f'{args.tgt_root}/train.pkl', f'{args.tgt_root}/train-tf.pkl',
                          compress=compress, progress=True, gc_every=args.gc_every)
        extract_and_store(f'{args.tgt_root}/test.pkl',  f'{args.tgt_root}/test-tf.pkl',
                          compress=compress, progress=True, gc_every=args.gc_every)

    # ---- NEW: PRELBD-only time-function extraction ----
    if args.extract_prelbd:
        extract_and_store_PRELBD(
            src_path=args.prelbd_src,
            tgt_path=args.prelbd_tgt,
            compress=compress,
            progress=True,
            gc_every=args.gc_every,
            p_col=args.p_col  # set to 6 if your PRELBD has p at column 6
        )
    if  args.extract_prelbd_task_wise:
        extract_and_store_PRELBD_prelbd(
            compress=compress,
            progress=True,
            gc_every=args.gc_every,
            p_col=args.p_col  # set to 6 if your PRELBD has p at column 6
        )

if __name__ == '__main__':
    main()

