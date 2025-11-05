# split_prelbd_raw_into_tasks.py
# Minimal splitter: copy .svc files into 13 task folders by parsing "..._<task>.svc" (e.g., "..._17_1.svc").
# Keeps raw files untouched. Creates out_root/<task>/<subject>/file.svc

import re, shutil
from pathlib import Path
import argparse
from collections import defaultdict

# Default 13 tasks (adjust if needed)
DEFAULT_TASKS = [
    "1_1", "3_1", "3_2", "3_3", "3_4", "3_5", "9_1", "15_1", "16_1", "17_1", "18_1", "19_1",
]

# Regex: capture the trailing "<number>_<number>" right before the extension
TASK_RE = re.compile(r'(\d{1,2}_\d+)(?=\.[A-Za-z0-9]+$)')

def parse_task(name: str) -> str | None:
    """
    Extract a task token like '17_1' from a filename ending with '_17_1.svc'.
    Returns None if no match.
    """
    m = TASK_RE.search(name)
    return m.group(1) if m else None

def split_into_tasks(src_root: str, out_root: str, tasks: list[str]):
    src = Path(src_root)
    out = Path(out_root)
    assert src.exists(), f"Missing source root: {src}"

    out.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    skipped = 0

    # iterate subjects (folders) and their .svc files
    for subj_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
        subject = subj_dir.name  # e.g., "HC-33#1", "pre-LBD-7#1"
        for f in sorted(subj_dir.rglob("*.svc")):
            task = parse_task(f.name)
            if task is None or task not in tasks:
                skipped += 1
                continue

            # target: out_root/<task>/<subject>/<filename>
            dest_dir = out / task / subject
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest_dir / f.name)
            stats[task] += 1

    # summary
    print("\n=== Done ===")
    print(f"Output root: {out.resolve()}")
    for t in tasks:
        print(f"  {t}: {stats[t]} files")
    print(f"Skipped (task not recognized/not in target set): {skipped} files")

def main():
    ap = argparse.ArgumentParser(description="Split raw PRELBD .svc data into 13 task-wise raw databases (by copy).")
    ap.add_argument("--src_root", type=str, required=True, help="Path to raw root (e.g., ./data-raw/LBD_CZ_002)")
    ap.add_argument("--out_root", type=str, required=True, help="Where to write task-wise raw folders")
    ap.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                    help="Comma-separated task list (e.g., '1_1,3_1,3_2,...').")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    split_into_tasks(args.src_root, args.out_root, tasks)

if __name__ == "__main__":
    main()
