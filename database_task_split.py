# -*- coding: utf-8 -*-
"""
Split raw PRELBD .svc data into 13 task-wise raw databases (by copying files).
Creates subfolders: out_root/<task>/<subject>/<filename>.svc
Logs skipped or unrecognized files to 'unrecognized_tasks.log'.
"""

import re
import shutil
from pathlib import Path
import argparse
from collections import defaultdict

# Default 13 tasks
DEFAULT_TASKS = [
    "1_1", "3_1", "3_2", "3_3", "3_4", "3_5", "9_1", "10_1", "15_1", "16_1", "17_1", "18_1", "19_1"
]

# Detect trailing "<num>_<num>" before extension, e.g. "_17_1.svc"
TASK_RE = re.compile(r'(\d{1,2}_\d+)(?=\.[A-Za-z0-9]+$)')

def parse_task(name: str) -> str | None:
    """Extract task token like '17_1' from filename ending with '_17_1.svc'."""
    m = TASK_RE.search(name)
    return m.group(1) if m else None


def split_into_tasks(src_root: str, out_root: str, tasks: list[str]):
    src = Path(src_root)
    out = Path(out_root)
    assert src.exists(), f"Missing source root: {src}"

    out.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    skipped_files = []  # store (subject, filename, reason)

    print(f"[SCAN] {src}")

    for subj_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
        subject = subj_dir.name
        for f in sorted(subj_dir.rglob("*.svc")):
            task = parse_task(f.name)
            if task is None:
                skipped_files.append((subject, f.name, "no task pattern"))
                continue
            if task not in tasks:
                skipped_files.append((subject, f.name, f"unknown task '{task}'"))
                continue

            # Create destination folder and copy
            dest_dir = out / task / subject
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest_dir / f.name)
            stats[task] += 1

    # Summary
    print("\n=== Task Split Summary ===")
    for t in tasks:
        print(f"  {t:>4}: {stats[t]} files")
    print(f"  Skipped: {len(skipped_files)} files (logged below)\n")

    # Write log
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
        description="Split raw PRELBD .svc data into 13 task-wise raw databases (by copying)."
    )
    #ap.add_argument("--src_root", type=str, required=True, help="Path to raw root (e.g., ./data-raw/LBD_CZ_002)")
    ap.add_argument("--src_root", type=str, default="./data-raw/LBD_CZ_002", help="Path to raw root (e.g., ./data-raw/LBD_CZ_002)")
    #ap.add_argument("--out_root", type=str, required=True, help="Where to write task-wise raw folders")
    ap.add_argument("--out_root", type=str, default="./data/LBD_CZ_002_raw_tasks", help="Where to write task-wise raw folders")
    ap.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                    help="Comma-separated task list (default = 13 PRELBD tasks)")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    split_into_tasks(args.src_root, args.out_root, tasks)


if __name__ == "__main__":
    main()
