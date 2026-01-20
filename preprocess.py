
# -*- coding: utf-8 -*-

import numpy as np
import os,pickle,argparse
from utils import centernorm_size,interpolate_torch
from pathlib import Path
from typing import List
from tqdm import tqdm
from joblib import dump as joblib_dump
from typing import Sequence, Iterable
import json


    
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

def _read_json_keep_cols(fp: Path, keep_cols: Sequence[str] = ("x", "y", "pressure")) -> np.ndarray:
    """
    Reads your DYS json and returns np.ndarray of shape (T, len(keep_cols)).
    Expects:
      obj["data"] is a dict where each kept column is a list/array of length T
    """
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return np.empty((0, len(keep_cols)), dtype=np.float32)

    data = obj.get("data", None)
    if not isinstance(data, dict):
        return np.empty((0, len(keep_cols)), dtype=np.float32)

    # verify columns exist
    for c in keep_cols:
        if c not in data:
            return np.empty((0, len(keep_cols)), dtype=np.float32)

    # determine T as min length across selected columns (defensive)
    lengths = []
    cols = []
    for c in keep_cols:
        v = data.get(c, [])
        if not isinstance(v, list):
            # allow numpy arrays etc., but must be iterable
            try:
                v = list(v)
            except Exception:
                return np.empty((0, len(keep_cols)), dtype=np.float32)
        cols.append(v)
        lengths.append(len(v))

    T = min(lengths) if lengths else 0
    if T <= 0:
        return np.empty((0, len(keep_cols)), dtype=np.float32)

    # build (T, C)
    out = np.empty((T, len(keep_cols)), dtype=np.float32)
    for j, v in enumerate(cols):
        try:
            out[:, j] = np.asarray(v[:T], dtype=np.float32)
        except Exception:
            return np.empty((0, len(keep_cols)), dtype=np.float32)

    # drop rows with NaN/Inf (optional but usually sensible)
    mask = np.isfinite(out).all(axis=1)
    out = out[mask]

    return out

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

    def split_with(d, s):
        toks = s.split(d)
        if len(toks) == 1 and " " in s:  # fallback if delimiter failed
            toks = s.split()
        return [t for t in toks if t != ""]


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

        max_column_debug = max(keep_cols, default=0)  
        max_lentoks_debug=len(toks)
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

    for wdir in tqdm(writers, desc="Preprocessing DYS_CZ_004_raw_tasks", unit="writer"):
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

def preprocess_DYS(
    src_root: str = "./data-raw/DYS_CZ_004_raw_tasks",
    tgt_root: str = "./data/DYS_CZ_004_raw_tasks",
    interp: int | None = 4,
    use_joblib: bool = False,
    keep_cols: Sequence[str] = ("x", "y", "pressure"),
    min_samples_per_writer: int = 0,
    out_name: str = "DYS_CZ_004.pkl",
):
    """
    Builds a single pickle:
      { writer_id(str): [ np.ndarray(T,3)[x,y,p], ... ], ... }

    Assumes src_root contains writer subfolders, each containing JSON samples.
    """
    src = Path(src_root)
    tgt = Path(tgt_root)
    if not src.exists():
        raise FileNotFoundError(f"Missing source root: {src}")
    tgt.mkdir(parents=True, exist_ok=True)

    writers = sorted([d for d in src.iterdir() if d.is_dir()])
    writing: dict[str, list[np.ndarray]] = {}

    for wdir in tqdm(writers, desc="Preprocessing DYS_CZ_004", unit="writer"):
        wkey = wdir.name
        samples: list[np.ndarray] = []

        #files = sorted([p for p in wdir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
        files = sorted(wdir.rglob("*.json"))
        files = sorted([p for p in files if p.is_file() and p.suffix.lower() == ".json"])
        if wdir == writers[0]:
            print(f"[DEBUG] Example writer dir: {wdir}")
            print(f"[DEBUG] JSON files found: {len(files)}")
            if len(files) > 0:
                print(f"[DEBUG] First JSON: {files[0]}")
                import json
                with open(files[0], "r", encoding="utf-8") as f:
                    obj = json.load(f)
                print(f"[DEBUG] top-level keys: {list(obj.keys())}")
                data = obj.get("data", None)
                print(f"[DEBUG] type(data): {type(data)}")
                if isinstance(data, dict):
                    print(f"[DEBUG] data keys: {list(data.keys())[:20]}")
                    for k in keep_cols:
                        v = data.get(k, None)
                        print(f"[DEBUG] col '{k}': {'MISSING' if v is None else f'len={len(v)} type={type(v)}'}")
        # --- END DIAGNOSTIC ---
        for fp in files:
            arr = _read_json_keep_cols(fp, keep_cols=keep_cols)
            if arr.shape[0] == 0:
                continue

            arr = centernorm_size(arr)
            if interp is not None:
                arr = interpolate_torch(arr, interp_ratio=interp)

            samples.append(arr)

        if len(samples) > min_samples_per_writer:
            writing[wkey] = samples

    out_path = tgt / out_name
    _save_dict(writing, out_path, use_joblib=use_joblib)

    total = sum(len(v) for v in writing.values())
    print(f"[DYS_CZ_004] writers: {len(writing)} | samples: {total} | saved -> {out_path}")

def preprocess_DYS_task_wise(
    src_root: str = "./data-raw/DYS_CZ_004_raw_tasks",
    tgt_root: str = "./data/DYS_CZ_004_raw_tasks",
    tasks: Sequence[str] = ("Letters", "Loops", "Rainbow", "Saw", "SentenceCopy"),
    interp: int | None = 4,
    use_joblib: bool = False,
    keep_cols: Sequence[str] = ("x", "y", "pressure"),
    min_samples_per_writer: int = 0,
):
    """
    Builds one pickle per task:
      ./data/DYS_CZ_004_raw_tasks/<TaskName>/DYS_CZ_004_<TaskName>.pkl

    Assumes layout:
      src_root/<TaskName>/<WriterID>/*.json
    """
    src_base = Path(src_root)
    tgt_base = Path(tgt_root)

    for task in tasks:
        task_src = src_base / task
        task_tgt = tgt_base / task

        preprocess_DYS(
            src_root=str(task_src),
            tgt_root=str(task_tgt),
            interp=interp,
            use_joblib=use_joblib,
            keep_cols=keep_cols,
            min_samples_per_writer=min_samples_per_writer,
            out_name=f"DYS_CZ_004.pkl",
            #out_name=f"DYS_CZ_004_{task}.pkl",
        )
def preprocess_PRELBD_task_wise(
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
    srcs = ['./data-raw/LBD_CZ_002_raw_tasks/1_1',
            './data-raw/LBD_CZ_002_raw_tasks/3_1',
            './data-raw/LBD_CZ_002_raw_tasks/3_2',
            './data-raw/LBD_CZ_002_raw_tasks/3_3',
            './data-raw/LBD_CZ_002_raw_tasks/3_4',
            './data-raw/LBD_CZ_002_raw_tasks/3_5',
            './data-raw/LBD_CZ_002_raw_tasks/9_1',
            './data-raw/LBD_CZ_002_raw_tasks/10_1',
            './data-raw/LBD_CZ_002_raw_tasks/15_1',
            './data-raw/LBD_CZ_002_raw_tasks/16_1',
            './data-raw/LBD_CZ_002_raw_tasks/17_1',
            './data-raw/LBD_CZ_002_raw_tasks/18_1',
            './data-raw/LBD_CZ_002_raw_tasks/19_1'
              ]
    for src_root in srcs:
        preprocess_PRELBD(
            src_root=src_root,
            tgt_root=src_root.replace('data-raw','data'),    
            interp=interp,
            use_joblib=use_joblib,
            keep_cols=keep_cols,    
            min_samples_per_writer=min_samples_per_writer,
        )        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='prelbd_task_wise',
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
    elif ds == 'prelbd_task_wise':
        preprocess_PRELBD_task_wise(
            interp=opt.interp,
            use_joblib=opt.joblib,
            keep_cols=(0,1,6)
        )
    elif ds == 'dys_task_wise':
        preprocess_DYS_task_wise(
            interp=opt.interp,
            use_joblib=opt.joblib,
            keep_cols=["x","y","pressure"]
        )
    else:
        raise ValueError(f"Unknown dataset: {ds}")

    print(f"End preprocessing {ds.upper()} dataset.")
