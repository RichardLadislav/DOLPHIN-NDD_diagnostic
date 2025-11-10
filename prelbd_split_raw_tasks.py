#TODO: delete this while file more
# -*- coding: utf-8 -*-
"""
Split PRELBD raw SVC database into task-wise raw sub-databases.

- Input layout (example):
    data-raw/LBD_CZ_002/
        HC-33#1/
            HC-33#1_w.cz.fnusa.17_1.svc
            HC-33#1_w.cz.fnusa.17_2.svc
            ...
        pre-LBD-7#1/
            pre-LBD-7#1_w.cz.fnusa.17_1.svc
            ...

- Output:
    data/LBD_CZ_002_raw_tasks/
        LBD_CZ_002-task01.pkl     # dict {subject: [np.ndarray(T,3), ...], ...}
        LBD_CZ_002-task02.pkl
        ...
        summary.json / summary.csv

Defaults:
- keep_cols=(0,1,6) → x, y, p
- encoding='utf-8'
- auto delimiter detection (',', ';', '\t', ' ')
"""

from __future__ import annotations
import re, csv, json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm.auto import tqdm

# -----------------------------
# Parsing & I/O helpers
# -----------------------------

TASK_PATTERNS = [
    re.compile(r'[_-](\d{1,2})(?=\.[A-Za-z0-9]+$)'),  # trailing _12.svc or -12.svc
    re.compile(r'[_-]task[_-]?(\d{1,2})(?=\.[A-Za-z0-9]+$)', re.IGNORECASE),  # ..._task12.svc
]

def parse_task_id_from_name(name: str) -> Optional[int]:
    """Try to parse task id from filename with several patterns; return int or None."""
    for pat in TASK_PATTERNS:
        m = pat.search(name)
        if m:
            return int(m.group(1))
    # fallback: last number in the stem
    stem = Path(name).stem
    nums = re.findall(r'(\d+)', stem)
    if nums:
        return int(nums[-1])
    return None

def read_svc_keep_cols(path: Path, keep_cols=(0,1,6), encoding="utf-8") -> np.ndarray:
    """
    Robust SVC reader:
    - skip header line
    - auto-detect delimiter among [',',';','\\t',' ']
    - ignore malformed/short lines
    Returns float32 array shape (T, len(keep_cols)); empty (0, k) if nothing usable.
    """
    with path.open("r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    if not lines:
        return np.zeros((0, len(keep_cols)), dtype=np.float32)

    body = [ln.strip() for ln in lines[1:] if ln.strip()]  # drop header
    if not body:
        return np.zeros((0, len(keep_cols)), dtype=np.float32)

    def split_with(d, s): return [tok for tok in s.split(d) if tok != ""]

    delim = " "
    rows: List[List[float]] = []
    for s in body:
        toks = split_with(delim, s) if delim is not None else s.split()
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


def save_pickle(obj, path: Path, protocol=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def subject_label_from_folder(name: str) -> Optional[int]:
    """
    0 → HC (healthy), 1 → pre-LBD (patient); None if unknown naming.
    """
    low = name.lower()
    if low.startswith("hc"):
        return 0
    if low.startswith("pre-lbd"):
        return 1
    return None

# -----------------------------
# Core split
# -----------------------------

def split_prelbd_raw_by_task(
    src_root: str = "./data-raw/LBD_CZ_002",
    out_root: str = "./data/LBD_CZ_002_raw_tasks",
    keep_cols: Tuple[int,int,int] = (0,1,6),
    encoding: str = "utf-8",
) -> None:
    src = Path(src_root)
    out = Path(out_root)
    assert src.exists(), f"Missing source: {src}"

    # Collect: task_id -> {subject: [arrays]}
    task_buckets: Dict[int, Dict[str, List[np.ndarray]]] = {}
    per_subject_task_count: Dict[str, Dict[int, int]] = {}
    per_task_stats: Dict[int, Dict[str,int]] = {}  # {task: {"subjects": n, "HC": n0, "preLBD": n1, "files": nfiles}}

    subjects = [d for d in src.iterdir() if d.is_dir()]
    for subj_dir in tqdm(sorted(subjects), desc="Scanning subjects", unit="subj"):
        subj_name = subj_dir.name
        subj_label = subject_label_from_folder(subj_name)
        files = sorted(p for p in subj_dir.iterdir() if p.is_file() and p.suffix.lower() == ".svc")

        for f in files:
            t_id = parse_task_id_from_name(f.name)
            if t_id is None:
                # Skip if no task id is derivable
                continue
            arr = read_svc_keep_cols(f, keep_cols=keep_cols, encoding=encoding)
            if arr.shape[0] == 0:
                continue
            # insert
            bucket = task_buckets.setdefault(t_id, {})
            bucket.setdefault(subj_name, []).append(arr)
            # bookkeeping
            per_subject_task_count.setdefault(subj_name, {})
            per_subject_task_count[subj_name][t_id] = per_subject_task_count[subj_name].get(t_id, 0) + 1
            s = per_task_stats.setdefault(t_id, {"subjects": 0, "HC": 0, "preLBD": 0, "files": 0})
            s["files"] += 1
            if subj_label == 0:
                s["HC"] += 1
            elif subj_label == 1:
                s["preLBD"] += 1

    # finalize subject counts per task
    for t_id, bucket in task_buckets.items():
        per_task_stats.setdefault(t_id, {"subjects": 0, "HC": 0, "preLBD": 0, "files": 0})
        per_task_stats[t_id]["subjects"] = len(bucket)

    # Save per task
    out.mkdir(parents=True, exist_ok=True)
    discovered_tasks = sorted(task_buckets.keys())
    for t_id in discovered_tasks:
        data_t = task_buckets[t_id]
        out_path = out / f"LBD_CZ_002-task{t_id:02d}.pkl"
        save_pickle(data_t, out_path)
        print(f"[SAVE] task {t_id:02d}: {len(data_t)} subjects → {out_path}")

    # Write summaries
    summary_json = {
        "src_root": str(src.resolve()),
        "out_root": str(out.resolve()),
        "tasks_found": discovered_tasks,
        "per_task_stats": per_task_stats,
        "notes": "subjects counted per task; files = total samples kept. HC=0, preLBD=1.",
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["task_id", "subjects", "HC", "preLBD", "files"])
        for t_id in discovered_tasks:
            s = per_task_stats[t_id]
            wr.writerow([t_id, s["subjects"], s["HC"], s["preLBD"], s["files"]])

    print(f"\n✅ Done. Wrote {len(discovered_tasks)} task files under: {out}")
    print(f"   Summary: {out/'summary.json'}  &  {out/'summary.csv'}")


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Split PRELBD raw SVC database into task-wise raw sub-databases")
    ap.add_argument("--src_root", type=str, default="./data-raw/LBD_CZ_002", help="Root with subject folders and .svc files")
    ap.add_argument("--out_root", type=str, default="./data/LBD_CZ_002_raw_tasks", help="Where to write task pickles and summary")
    ap.add_argument("--keep_cols", type=str, default="0,1,6", help="Columns to keep as x,y,p (comma-separated indices)")
    ap.add_argument("--encoding", type=str, default="utf-8", help="Text encoding for .svc files")
    args = ap.parse_args()

    cols = tuple(int(x) for x in args.keep_cols.split(","))
    split_prelbd_raw_by_task(
        src_root=args.src_root,
        out_root=args.out_root,
        keep_cols=cols,
        encoding=args.encoding,
    )
