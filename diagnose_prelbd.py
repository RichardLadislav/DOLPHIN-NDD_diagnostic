# diagnose_prelbd.py
# -*- coding: utf-8 -*-

import os, argparse, pickle, sys, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from joblib import load as joblib_load
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# --- repo-local imports (ensure your repo root is on PYTHONPATH) ---
from model import DOLPHIN

# -----------------------------
# Labeling: HC -> 0, pre-LBD -> 1
# -----------------------------
def label_from_key(k: str) -> int:
    k_low = k.lower()
    if k_low.startswith("hc"):
        return 0
    if k_low.startswith("pre-lbd") or k_low.startswith("pre_lbd") or k_low.startswith("prelbd"):
        return 1
    return -1  # unknown -> filtered later

# -----------------------------
# Dataset with diagnostic labels
# -----------------------------
class WritingDiagnostic(Dataset):
    """
    handwriting_info: {subject_id: [np.ndarray(T, F), ...], ...}
    labels are diagnosis: HC=0, pre-LBD=1 (derived from subject_id)
    """
    def __init__(self, handwriting_info: dict, transform=None):
        super().__init__()
        self.transform = transform
        self.features, self.lengths, self.labels, self.subject_ids = [], [], [], []
        for subj, arrs in handwriting_info.items():
            y = label_from_key(subj)
            if y < 0:  # skip unrecognized subjects
                continue
            for a in arrs:
                a = np.asarray(a)
                if a.ndim != 2 or a.shape[0] == 0:
                    continue
                self.features.append(a)
                self.lengths.append(a.shape[0])
                self.labels.append(y)
                self.subject_ids.append(subj)
        assert len(self.features) > 0, "No usable samples found."

        self.features = list(self.features)
        self.lengths = np.array(self.lengths, dtype=np.int64)
        self.labels  = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.lengths[idx]), int(self.labels[idx]), self.subject_ids[idx]

# Collate -> (xs[B,T,3], ys[B], lens[B]) selecting (x,y,p) columns
def collate_fn_dolphin(batch, cols=(0,1,6)):
    arrs, lens, ys, subs = zip(*batch)
    lens = np.asarray(lens, dtype=np.int64)
    B, Tm = len(arrs), int(lens.max())
    xs  = torch.zeros(B, Tm, len(cols), dtype=torch.float32)
    ys  = torch.tensor(ys, dtype=torch.long)
    lens_t = torch.tensor(lens, dtype=torch.int64)
    for i,(a,L) in enumerate(zip(arrs, lens)):
        a = torch.as_tensor(a, dtype=torch.float32)
        xs[i,:L,:] = a[:L, cols]
    return xs, ys, lens_t, subs

# -----------------------------
# IO helpers
# -----------------------------
def load_any_pkl(path: Path):
    try:
        return joblib_load(path.as_posix(), mmap_mode="r")
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f, encoding="iso-8859-1")

# -----------------------------
# Embedding extractor (frozen)
# -----------------------------
@torch.no_grad()
def extract_embeddings(model, loader, device="cpu"):
    model.eval().to(device)
    feats, labels, subjects = [], [], []
    for xs, ys, lens, subs in tqdm(loader, desc="Embedding", unit="batch"):
        xs, lens = xs.to(device), lens.to(device)
        y_vec, _, f3 = model(xs, lens)         # y_vec: (B,384), f3: (B,384)
        y_vec = F.normalize(y_vec, dim=1)
        f3    = F.normalize(f3, dim=1)
        emb   = torch.cat([y_vec, f3], dim=1).cpu().numpy()  # (B,768)
        feats.append(emb); labels.append(ys.numpy()); subjects.extend(subs)
    X = np.vstack(feats)
    y = np.concatenate(labels)
    X = normalize(X, norm="l2", axis=1)
    return X, y, np.array(subjects)

# -----------------------------
# Subject-level aggregation
# -----------------------------
def aggregate_by_subject(X, y, subjects, reducer="mean"):
    uniq = np.unique(subjects)
    Xg, yg, sg = [], [], []
    for s in uniq:
        idx = np.where(subjects == s)[0]
        if reducer == "median":
            Xs = np.median(X[idx], axis=0)
        else:
            Xs = np.mean(X[idx], axis=0)
        ys = int(np.round(np.mean(y[idx])))  # all same in our labeling; this is safe
        Xg.append(Xs); yg.append(ys); sg.append(s)
    return np.vstack(Xg), np.array(yg), np.array(sg)

# -----------------------------
# Visualization
# -----------------------------
def plot_tsne(X, y, save_path=None, perplexity=35, max_points=8000, title="t-SNE of embeddings"):
    N = len(X)
    idx = np.arange(N)
    if max_points and N > max_points:
        rng = np.random.default_rng(0); idx = rng.choice(N, size=max_points, replace=False)
    Z = TSNE(n_components=2, init="pca", learning_rate="auto",
             perplexity=min(perplexity, max(5, len(idx)//3)),
             n_iter=1500, random_state=0).fit_transform(X[idx])
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,7))
    plt.scatter(Z[:,0], Z[:,1], c=y[idx], s=10, alpha=0.9, cmap="coolwarm")
    plt.title(title); plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close()

def plot_umap(X, y, save_path=None, n_neighbors=15, min_dist=0.1, max_points=8000, title="UMAP of embeddings"):
    try:
        import umap
    except Exception:
        print("[WARN] umap-learn not installed; skipping UMAP.")
        return
    N = len(X)
    idx = np.arange(N)
    if max_points and N > max_points:
        rng = np.random.default_rng(0); idx = rng.choice(N, size=max_points, replace=False)
    Z = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=0).fit_transform(X[idx])
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,7))
    plt.scatter(Z[:,0], Z[:,1], c=y[idx], s=10, alpha=0.9, cmap="coolwarm")
    plt.title(title); plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2"); plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close()

# -----------------------------
# Simple diagnostic classifier
# -----------------------------
def evaluate_classifier(X, y, cv=5, C=1.0, seed=0):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_true, y_pred = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=500, n_jobs=None)
        clf.fit(X[tr], y[tr])
        y_hat = clf.predict(X[te])
        y_true.append(y[te]); y_pred.append(y_hat)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-9)  # recall for class 1
    spec = tn / (tn + fp + 1e-9)  # recall for class 0
    return bal_acc, cm, sens, spec

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="DOLPHIN → diagnostic embeddings → t-SNE/UMAP + LR CV")
    ap.add_argument("--pkl", type=str, required=True, help="Path to PRELBD-tf.pkl (joblib or pickle).")
    ap.add_argument("--cols", type=str, default="0,1,6", help="Indices for (x,y,p), e.g. '0,1,6'.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--aggregate", action="store_true", help="Average embeddings per subject before analysis.")
    ap.add_argument("--tsne", action="store_true", help="Make a t-SNE plot.")
    ap.add_argument("--umap", action="store_true", help="Make a UMAP plot (requires umap-learn).")
    ap.add_argument("--outdir", type=str, default="./outs_prelbd")
    ap.add_argument("--ckpt", type=str, default=None, help="Optional model checkpoint to load (strict=False).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pkl_path = Path(args.pkl); assert pkl_path.exists(), f"Missing: {pkl_path}"
    cols = tuple(int(x) for x in args.cols.split(","))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] {pkl_path}")
    data = load_any_pkl(pkl_path)

    # dataset & loader
    ds = WritingDiagnostic(data)
    print(f"Subjects (unique): {len(np.unique(ds.subject_ids))} | Samples: {len(ds)} | Label ratio (1's): {ds.labels.mean():.3f}")

    # small subset? (keep all by default)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=lambda b: collate_fn_dolphin(b, cols=cols))

    # model
    num_classes_dummy = 1000  # logits unused; we only use embeddings
    model = DOLPHIN(d_in=3, num_classes=num_classes_dummy)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=args.device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print("[Model] Loaded checkpoint with strict=False")
    model = model.to(args.device).eval()

    # embeddings
    X, y, subs = extract_embeddings(model, loader, device=args.device)
    if args.aggregate:
        X, y, subs = aggregate_by_subject(X, y, subs, reducer="mean")
        print(f"[Aggregate] Per subject: {X.shape[0]} vectors")

    # plots
    if args.tsne:
        plot_tsne(X, y, save_path=outdir / ("tsne_subjects.png" if args.aggregate else "tsne_samples.png"))
    if args.umap:
        plot_umap(X, y, save_path=outdir / ("umap_subjects.png" if args.aggregate else "umap_samples.png"))

    # classification
    bal_acc, cm, sens, spec = evaluate_classifier(X, y, cv=5, C=1.0, seed=0)
    tn, fp, fn, tp = cm.ravel()
    print("\n=== Diagnostic CV (LogReg, 5-fold, class_weight=balanced) ===")
    print(f"Balanced accuracy: {bal_acc*100:.2f}%")
    print(f"Sensitivity (TPR, class=1): {sens*100:.2f}%")
    print(f"Specificity (TNR, class=0): {spec*100:.2f}%")
    print("Confusion matrix [[TN FP][FN TP]]:\n", cm)

    # save metrics
    np.savez(outdir / "metrics.npz", bal_acc=bal_acc, cm=cm, sens=sens, spec=spec)

if __name__ == "__main__":
    main()
