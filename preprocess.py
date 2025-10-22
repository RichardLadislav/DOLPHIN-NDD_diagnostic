
# -*- coding: utf-8 -*-

import numpy as np
import os,pickle,argparse
from utils import centernorm_size,interpolate_torch
from pathlib import Path
from typing import List
from tqdm import tqdm
from joblib import dump as joblib_dump

    
def preprocess_DCOHE(src_root='./data-raw/DCOH-E'):
    writing = {}
    writers = os.listdir(src_root)
    for i,w in enumerate(writers):
        writing[w] = []
        for file in os.listdir(f'{src_root}/{w}'):
            info = []
            with open(f'{src_root}/{w}/{file}','r',encoding='utf-8') as f:
                lines = f.readlines()
                lines = lines[1:]
                lines = [l.strip() for l in lines]
            info = [list(map(lambda x:float(x),l.split()[:3])) for l in lines]
            info = np.array(info,np.float32)
            info = centernorm_size(info)
            if 'dcoh-e' in file:
                info = interpolate_torch(info,interp_ratio=2)
            writing[w].append(info)
    tgt_root = src_root.replace('data-raw','data')
    os.makedirs(tgt_root,exist_ok=True)
    with open(f'{tgt_root}/DCOH-E.pkl','wb') as f:
        pickle.dump(writing,f)

def preprocess_OLHWDB2(src_root='./data-raw/OLHWDB2',interp=4):
    writing = {}
    writers = os.listdir(src_root)
    for i,w in enumerate(writers):
        writing[i] = []
        for file in os.listdir(f'{src_root}/{w}'):
            info = []
            with open(f'{src_root}/{w}/{file}','r',encoding='utf-8') as f:
                lines = f.readlines()
                lines = lines[1:]
                lines = [l.strip() for l in lines]
            info = [list(map(lambda x:float(x),l.split()[:3])) for l in lines]
            info = np.array(info,np.float32)
            info = centernorm_size(info)
            if interp != None:
                info = interpolate_torch(info,interp_ratio=interp)
            writing[i].append(info)
    tgt_root = src_root.replace('data-raw','data')
    os.makedirs(tgt_root,exist_ok=True)
    with open(f'{tgt_root}/OLHWDB2.pkl','wb') as f:
        pickle.dump(writing,f)

def preprocess_COUCH(src_root='./data-raw/COUCH09',interp=4):
    writing = {}
    writers = os.listdir(src_root)
    for i,w in enumerate(writers):
        writing[i] = []
        for file in os.listdir(f'{src_root}/{w}'):
            with open(f'{src_root}/{w}/{file}','r',encoding='utf-8') as f:
                lines = f.readlines()
                lines = lines[1:]
                lines = [l.strip() for l in lines]
            info = [list(map(lambda x:float(x),l.split()[:3])) for l in lines]
            info = np.array(info,np.float32)
            info = centernorm_size(info)
            if interp != None:
                info = interpolate_torch(info,interp_ratio=interp)
            writing[i].append(info)
    tgt_root = src_root.replace('data-raw','data')
    os.makedirs(tgt_root,exist_ok=True)
    with open(f'{tgt_root}/COUCH09.pkl','wb') as f:
        pickle.dump(writing,f)
        
# Part where for preprocessing of preLBD database 

def _read_svc_keep_cols(path: Path, keep_cols=(0, 1, 6), encoding="utf-8") -> np.ndarray:
    """
    Robust reader for .svc (generic delimited text) that returns only selected columns.
    - Tries to auto-detect delimiter among [',', ';', '\t', ' ']
    - Skips the first line (header)
    - Ignores malformed/empty lines
    Returns np.ndarray (T, len(keep_cols)) in float32.
    """
    # read raw text
    with path.open("r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return np.zeros((0, len(keep_cols)), dtype=np.float32)

    # candidate delimiters, best-effort
    candidates = [",", ";", "\t", " "]
    header = lines[0].strip()
    body = [ln.strip() for ln in lines[1:] if ln.strip()]  # skip header

    # pick delimiter by highest split stability
    def split_with(d, s): 
        return [tok for tok in s.split(d) if tok != ""]

    # choose delimiter that yields most consistent column counts on a few sample lines
    delim = None
    best_score = -1
    for d in candidates:
        counts = []
        for s in body[:20]:
            toks = split_with(d, s)
            counts.append(len(toks))
        if counts:
            score = -np.std(counts)  # lower std = more consistent
        else:
            score = -1e9
        if score > best_score:
            best_score = score
            delim = d

    rows: List[List[float]] = []
    for s in body:
        toks = split_with(delim, s) if delim is not None else s.split()
        # guard for short lines
        if max(keep_cols, default=0) >= len(toks):
            continue
        try:
            vals = [float(toks[i]) for i in keep_cols]
            rows.append(vals)
        except Exception:
            continue

    if not rows:
        return np.zeros((0, len(keep_cols)), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)


def _save_dict(obj, out_path: Path, use_joblib: bool, compress=("lz4", 3), protocol=5):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if use_joblib and joblib_dump is not None:
        joblib_dump(obj, out_path.as_posix(), compress=compress, protocol=protocol)
    else:
        with out_path.open("wb") as f:
            pickle.dump(obj, f, protocol=protocol)


def preprocess_PRELBD(
    src_root: str = "./data-raw/LBD_CZ_002",
    tgt_root: str = "./data/LBD_CZ_002",
    interp: int | None = 4,
    use_joblib: bool = False,
    keep_cols=(0, 1, 6),
    min_samples_per_writer: int = 0,
):
    """
    Build a single pickle for PRELBD:
      { writer_id(str): [ np.ndarray(T,3)[x,y,p], ... ], ... }

    - Reads .svc files under writer subfolders in src_root
    - Keeps ONLY columns (0,1,6) as (x,y,p)
    - Applies centernorm_size + optional interpolate_torch
    - Writes LBD_CZ_002.pkl under tgt_root
    """
    src = Path(src_root)
    tgt = Path(tgt_root)
    assert src.exists(), f"Missing source root: {src}"

    writers = sorted([d for d in src.iterdir() if d.is_dir()])
    writing: dict[str, list[np.ndarray]] = {}

    for wdir in tqdm(writers, desc="Preprocessing LBD_CZ_002", unit="writer"):
        wkey = wdir.name  # use folder name as stable writer id (string)
        samples: list[np.ndarray] = []
        # accept .svc (and optionally .csv/.txt if present)
        files = sorted([p for p in wdir.iterdir() if p.is_file() and p.suffix.lower() in {".svc", ".csv", ".txt"}])

        for fp in files:
            arr = _read_svc_keep_cols(fp, keep_cols=keep_cols)
            if arr.shape[0] == 0:
                continue
            # normalize + optional interpolation
            arr = centernorm_size(arr)
            if interp is not None:
                arr = interpolate_torch(arr, interp_ratio=interp)
            samples.append(arr)

        if len(samples) > min_samples_per_writer:
            writing[wkey] = samples

    out_path = tgt / "LBD_CZ_002.pkl"
    _save_dict(writing, out_path, use_joblib=use_joblib)

    total = sum(len(v) for v in writing.values())
    print(f"[LBD_CZ_002] writers: {len(writing)} | samples: {total} | saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='couch',
        help='Processed dataset names: [olhwdb2, dcohe, couch, prelbd]'
    )
    parser.add_argument(
        '--interp',
        type=int,
        default=4,
        help='Interpolation ratio (set None to disable interpolation).'
    )
    parser.add_argument(
        '--joblib',
        action='store_true',
        help='Use joblib for compressed dumping (only for PRELBD).'
    )
    opt = parser.parse_args()

    ds = opt.dataset.lower()
    print(f"Start preprocessing {ds.upper()} dataset.")

    if ds == 'olhwdb2':
        preprocess_OLHWDB2(interp=opt.interp)
    elif ds == 'dcohe':
        preprocess_DCOHE()
    elif ds == 'couch':
        preprocess_COUCH(interp=opt.interp)
    elif ds == 'prelbd':
        preprocess_PRELBD(
            src_root='./data-raw/LBD_CZ_002',
            tgt_root='./data/LBD_CZ_002',
            interp=opt.interp,
            use_joblib=opt.joblib,
            keep_cols=(0,1,6)
        )
    else:
        raise ValueError(f"Unknown dataset: {ds}")

    print(f"End preprocessing {ds.upper()} dataset.")
