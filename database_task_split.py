# -*- coding: utf-8 -*-
"""
Split raw .svc data into task-wise raw databases (by copying files).
Creates subfolders: out_root/<task>/<subject>/<filename>.svc
Logs skipped or unrecognized files to 'unrecognized_tasks.log'.
"""

import re
import shutil
from pathlib import Path
import argparse
from collections import defaultdict
from typing import Optional

PRE_LBD_TASKS = [
    "1_1", "3_1", "3_2", "3_3", "3_4", "3_5", "9_1", "10_1", "15_1", "16_1", "17_1", "18_1", "19_1"
]

DYS_TASKS = ["Letters", "Loops", "Rainbow", "Saw", "SentenceCopy"]

DEFAULT_TASKS = DYS_TASKS

# Numeric token before extension, e.g. "_17_1.svc"
NUMERIC_TASK_RE = re.compile(r'(\d{1,2}_\d+)(?=\.[A-Za-z0-9]+$)')


def parse_task(name: str, tasks: list[str]) -> Optional[str]:
    """
    Extract task token from filename.

    - If tasks look numeric (e.g., '17_1'), use numeric regex at end of filename.
    - Otherwise, look for any task keyword as a token anywhere in filename.
    """
    # Heuristic: if all tasks are numeric-like, use numeric parser
    if all(NUMERIC_TASK_RE.fullmatch(t) for t in tasks):
        m = NUMERIC_TASK_RE.search(name)
        return m.group(1) if m else None

    # Keyword parser (case-insensitive), match whole-word-ish tokens
    base = Path(name).stem
    # Split on common separators to get tokens
    tokens = re.split(r'[^A-Za-z0-9]+', base.lower())
    task_map = {t.lower(): t for t in tasks}

    for tok in tokens:
        if tok in task_map:
            return task_map[tok]

    # Also support CamelCase token embedded (e.g., "...SentenceCopy...")
    lower_base = base.lower()
    for t_lower, t_orig in task_map.items():
        if t_lower in lower_base:
            return t_orig

    return None


def split_into_tasks(src_root: str, out_root: str, tasks: list[str], file_ext: str = ".json"):
    src = Path(src_root)
    out = Path(out_root)
    assert src.exists(), f"Missing source root: {src}"

    out.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    skipped_files = []  # store (subject, filename, reason)

    print(f"[SCAN] {src}")

    for subj_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
        subject = subj_dir.name
        for f in sorted(subj_dir.rglob(f"*{file_ext}")):
            task = parse_task(f.name, tasks)
            if task is None:
                skipped_files.append((subject, f.name, "no recognizable task token"))
                continue
            if task not in tasks:
                skipped_files.append((subject, f.name, f"unknown task '{task}'"))
                continue

            dest_dir = out / task / subject
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest_dir / f.name)
            stats[task] += 1

    print("\n=== Task Split Summary ===")
    for t in tasks:
        print(f"  {t:>12}: {stats[t]} files")
    print(f"  Skipped: {len(skipped_files)} files (logged below)\n")

    log_path = out / "unrecognized_tasks.log"
    if skipped_files:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# Unrecognized or skipped .svc files from {src_root}\n")
            log.write(f"# Total: {len(skipped_files)}\n\n")
            for subj, fname, reason in skipped_files:
                log.write(f"{subj}/{fname}  <-- {reason}\n")
        print(f"[LOG] {len(skipped_files)} unrecognized files → {log_path}")
    else:
        print("[LOG] All files matched tasks; no unrecognized entries.")

    print(f"[DONE] Output written to {out.resolve()}")


def main():
    ap = argparse.ArgumentParser(
        description="Split raw .svc data into task-wise raw databases (by copying)."
    )
    ap.add_argument("--src_root", type=str, default="./data-raw/DYS_CZ_004",
                    help="Path to raw root (e.g., ./data-raw/DYS_CZ_004)")
    ap.add_argument("--out_root", type=str, default="./data/DYS_CZ_004_raw_tasks",
                    help="Where to write task-wise raw folders")
    ap.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                    help="Comma-separated task list")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    split_into_tasks(args.src_root, args.out_root, tasks)


if __name__ == "__main__":
    main()

