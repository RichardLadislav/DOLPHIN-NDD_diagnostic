# diagnose_prelbd.py
# -*- coding: utf-8 -*-

import os, argparse, pickle, sys, math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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

# ==== DIAGNOSTIC REPORT TOOLKIT ====
import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
# --- repo-local imports (ensure your repo root is on PYTHONPATH) ---
from model import DOLPHIN

# ==== DIAGNOSTIC REPORT TOOLKIT ====
import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
# -----------------------------
# Labeling: HC -> 0, pre-LBD -> 1
# -----------------------------
# ---- JSON helper: convert numpy types to vanilla Python ----
def to_serializable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


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
#        emb   = torch.cat([y_vec, f3], dim=1).cpu().numpy()  # (B,768)
#        emb   = torch.cat([y_vec], dim=1).cpu().numpy()   (B,384)
        emb    torch.cat([f3], dim=1).cpu().numpy()   (B,768)
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
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    # 1) sanitize
    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # drop duplicate rows to avoid degenerate distances
    uniq_idx = np.unique(X, axis=0, return_index=True)[1]
    X = X[uniq_idx]
    y = y[uniq_idx]

    # 2) subsample if requested
    N = len(X)
    if N == 0:
        print("[t-SNE] No points to plot.")
        return
    if max_points and N > max_points:
        rng = np.random.default_rng(0)
        sel = rng.choice(N, size=max_points, replace=False)
        X = X[sel]; y = y[sel]; N = len(X)

    # 3) choose a safe perplexity
    # must be < N; rule of thumb: < N/3
    safe_perp = min(perplexity, max(5, (N // 3) - 1))
    if safe_perp >= N:
        safe_perp = max(5, N // 4) if N >= 8 else max(2, N // 2)
    if N <= 5:
        print(f"[t-SNE] Too few points (N={N}) for t-SNE; skipping.")
        return

    # 4) try a few settings gracefully
    tried = []
    for metric in ("euclidean", "cosine"):
        p = safe_perp
        while p >= 2:
            try:
                Z = TSNE(
                    n_components=2,
                    init="pca",
                    learning_rate="auto",
                    perplexity=p,
                    metric=metric,
                    max_iter=1500,
                    random_state=0,
                ).fit_transform(X)
                # success -> plot
                plt.figure(figsize=(7,7))
                plt.scatter(Z[:,0], Z[:,1], c=y, s=10, alpha=0.9, cmap="coolwarm")
                plt.title(f"{title} (perp={p}, metric={metric}, N={N})")
                plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.tight_layout()
                if save_path is not None:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(save_path, dpi=200)
                else:
                    plt.show()
                plt.close()
                return
            except Exception as e:
                tried.append((p, metric, str(e)))
                p = int(p * 0.6)  # shrink and retry

    # If we get here, all attempts failed
    print("[t-SNE] Failed after attempts:")
    for p, m, err in tried[-5:]:
        print(f"  - perp={p}, metric={m}: {err}")

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
# Report toolkit
# -----------------------------


def _ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def compute_separability_metrics(X, y):
    """Returns dict with silhouette, DB, CH for: high-D X and (optionally) 2D embeddings Z if provided later."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    res = {}
    # Guard for degenerate cases
    if len(np.unique(y)) < 2 or len(X) < 5:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    # Silhouette needs > 1 sample per class; skip if tiny classes
    try:
        res["silhouette"] = float(silhouette_score(X, y, metric="euclidean"))
    except Exception:
        res["silhouette"] = np.nan
    try:
        res["davies_bouldin"] = float(davies_bouldin_score(X, y))
    except Exception:
        res["davies_bouldin"] = np.nan
    try:
        res["calinski_harabasz"] = float(calinski_harabasz_score(X, y))
    except Exception:
        res["calinski_harabasz"] = np.nan
    return res

def evaluate_classifier_cv(X, y, cv=5, seed=0, C=1.0):
    """LogReg (balanced), Stratified K-fold. Returns dict with BA, sensitivity, specificity, AUC (if possible), CM."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    y_true, y_pred, y_prob = [], [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=500)
        clf.fit(X[tr], y[tr])
        y_hat = clf.predict(X[te])
        y_true.append(y[te]); y_pred.append(y_hat)
        # proba only if binary
        if len(np.unique(y)) == 2 and hasattr(clf, "predict_proba"):
            y_prob.append(clf.predict_proba(X[te])[:, 1])
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan)
    sens = tp / (tp + fn + 1e-9) if cm.size == 4 else np.nan
    spec = tn / (tn + fp + 1e-9) if cm.size == 4 else np.nan
    auc = np.nan
    if y_prob:
        try:
            auc = roc_auc_score(np.concatenate(y_true if isinstance(y_true, list) else [y_true]),
                                np.concatenate(y_prob))
        except Exception:
            pass
    return {
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "auc": float(auc),
        "confusion_matrix": cm
    }

def aggregate_by_subject_mean(X, y, subs):
    """One vector per subject (mean)."""
    subs = np.asarray(subs)
    Xg, yg, sg = [], [], []
    for s in np.unique(subs):
        idx = np.where(subs == s)[0]
        Xg.append(np.mean(X[idx], axis=0))
        yg.append(int(np.round(np.mean(y[idx]))))  # labels should be consistent per subject
        sg.append(s)
    return np.vstack(Xg), np.array(yg), np.array(sg)

def plot_dendrogram_and_heatmap(X, subs, save_prefix):
    Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)

    # 1) condensed distances for linkage
    Y = pdist(X, metric="cosine")        # 1-D condensed vector
    Z = linkage(Y, method="average")     # average linkage on condensed distances

    # 2) dendrogram
    plt.figure(figsize=(12, 5))
    dn = dendrogram(Z, labels=subs.tolist(), leaf_rotation=90, leaf_font_size=8, color_threshold=None)
    plt.title("Hierarchical Clustering (cosine)")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_dendrogram.png", dpi=200)
    plt.close()

    # 3) similarity heatmap, ordered by dendrogram leaves
    order = dn["leaves"]
    D = squareform(Y)                    # back to square matrix for visualization
    D_ord = D[np.ix_(order, order)]
    S = 1.0 - D_ord                      # cosine similarity

    subs_ord = [subs[i] for i in order]
    plt.figure(figsize=(7, 6))
    im = plt.imshow(S, aspect="auto", interpolation="nearest")
    plt.title("Cosine Similarity (ordered by dendrogram)")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    step = max(1, len(subs_ord)//20)
    plt.xticks(range(0, len(subs_ord), step), [subs_ord[i] for i in range(0, len(subs_ord), step)],
               rotation=90, fontsize=7)
    plt.yticks(range(0, len(subs_ord), step), [subs_ord[i] for i in range(0, len(subs_ord), step)],
               fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_similarity.png", dpi=200)
    plt.close()

def neighbor_overlap_score(Z2d, y, k=10):
    """
    Simple manifold separability sanity: for each point, fraction of k-NN that share the same label.
    Returns mean overlap in [0,1].
    """
    Z2d = np.asarray(Z2d, dtype=np.float64)
    y = np.asarray(y)
    if len(Z2d) < k+1:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(Z2d)
    idx = nn.kneighbors(return_distance=False)[:, 1:]  # drop self
    same = (y[idx] == y[:, None]).mean()
    return float(same)

def save_metrics_csv(path, metrics_dict):
    _ensure_dir(Path(path).parent)
    flat = {k:(v.tolist() if isinstance(v, np.ndarray) else v) for k,v in metrics_dict.items()}
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        for k,v in flat.items():
            wr.writerow([k, v])

def make_full_report(
    X, y, subs,
    outdir="./outs_prelbd",
    Z_tsne=None, Z_umap=None,
    subject_level=True,
    cv=5,
):
    """
    Generates: separability metrics, LR-CV metrics, dendrogram + similarity heatmap,
    optional neighbor-overlap metrics on 2D plots, and saves NPZ/CSV/PNGs.
    """
    outdir = Path(outdir); _ensure_dir(outdir)

    # 0) (Optional) aggregate per subject for clustering & metrics
    X_use, y_use, subs_use = (aggregate_by_subject_mean(X, y, subs) if subject_level else (X, y, subs))

    # 1) separability (high-D embeddings)
    sep = compute_separability_metrics(X_use, y_use)

    # 2) classifier CV (on the same representation)
    clf = evaluate_classifier_cv(X_use, y_use, cv=cv, seed=0, C=1.0)

    # 3) hierarchical clustering (on subject-level recommended)
    plot_dendrogram_and_heatmap(X_use, subs_use, save_prefix=str(outdir / ("hc_subjects" if subject_level else "hc_samples")))

    # 4) 2D neighbor-overlap (if 2D projections provided)
    proj = {}
    if Z_tsne is not None and len(Z_tsne) == len(X):
        Zt_use = aggregate_by_subject_mean(Z_tsne, y, subs)[0] if subject_level else Z_tsne
        proj["tsne_neighbor_overlap@10"] = neighbor_overlap_score(Zt_use, y_use, k=10)
        # save fig (optional here; you already plotted elsewhere)
    if Z_umap is not None and len(Z_umap) == len(X):
        Zu_use = aggregate_by_subject_mean(Z_umap, y, subs)[0] if subject_level else Z_umap
        proj["umap_neighbor_overlap@10"] = neighbor_overlap_score(Zu_use, y_use, k=10)

    # 5) persist all metrics
    results = {
        "subject_level": subject_level,
        "n_points": int(len(X_use)),
        "separability": sep,
        "classifier": {
            "balanced_accuracy": clf["balanced_accuracy"],
            "sensitivity": clf["sensitivity"],
            "specificity": clf["specificity"],
            "auc": clf["auc"],
            "confusion_matrix": clf["confusion_matrix"],
        },
        "projection_neighbors": proj,
    }
    np.savez(outdir / "diagnostic_metrics.npz", **{
        "subject_level": subject_level,
        "n_points": int(len(X_use)),
        "silhouette": sep["silhouette"],
        "davies_bouldin": sep["davies_bouldin"],
        "calinski_harabasz": sep["calinski_harabasz"],
        "balanced_accuracy": clf["balanced_accuracy"],
        "sensitivity": clf["sensitivity"],
        "specificity": clf["specificity"],
        "auc": clf["auc"],
        "confusion_matrix": clf["confusion_matrix"],
        **proj
    })
    save_metrics_csv(outdir / "diagnostic_metrics.csv", results)
    with open(outdir / "diagnostic_metrics.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(results), f, indent=2)
    print("\n[REPORT]")
    print(f"  Points: {results['n_points']} (subject_level={results['subject_level']})")
    print(f"  Silhouette:      {sep['silhouette']:.4f}")
    print(f"  Davies-Bouldin:  {sep['davies_bouldin']:.4f}  (lower is better)")
    print(f"  Calinski-Harab.: {sep['calinski_harabasz']:.2f} (higher is better)")
    print(f"  Balanced Acc.:   {clf['balanced_accuracy']*100:.2f}%")
    print(f"  Sensitivity:     {clf['sensitivity']*100:.2f}%")
    print(f"  Specificity:     {clf['specificity']*100:.2f}%")
    if "tsne_neighbor_overlap@10" in proj:
        print(f"  t-SNE neighbor overlap@10:  {proj['tsne_neighbor_overlap@10']:.3f}")
    if "umap_neighbor_overlap@10" in proj:
        print(f"  UMAP neighbor overlap@10:   {proj['umap_neighbor_overlap@10']:.3f}")
    print(f"  Saved: {outdir}/diagnostic_metrics.(npz|csv|json), dendrogram & heatmap PNGs")

# -----------------------------
# Embedding file store
# -----------------------------
def save_embeddings(X: np.ndarray,
                    y: np.ndarray,
                    subs: np.ndarray,
                    out_path: Path,
                    fmt: str = "npz",
                    header_prefix: str = "f"):
    """
    Save embeddings for downstream classifiers.

    Args:
        X: (N, D) embeddings
        y: (N,) integer labels (0/1)
        subs: (N,) subject ids (strings)
        out_path: full path incl. filename (extension ignored; 'fmt' is used)
        fmt: 'npz' or 'csv'
        header_prefix: column prefix for CSV feature columns
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt.lower() == "npz":
        # Fast, binary – perfect for sklearn: np.load(...); X=..., y=..., subs=...
        np.savez(out_path.with_suffix(".npz"), X=X, y=y, subs=subs)
        print(f"[SAVE] Embeddings (NPZ): {out_path.with_suffix('.npz')}")
        return

    if fmt.lower() == "csv":
        # Wide CSV: subject,label,f0,...,f{D-1}
        D = X.shape[1]
        headers = ["subject", "label"] + [f"{header_prefix}{i}" for i in range(D)]
        csv_path = out_path.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            import csv as _csv
            wr = _csv.writer(f)
            wr.writerow(headers)
            for s, yi, row in zip(subs, y, X):
                wr.writerow([s, int(yi)] + [float(v) for v in row.tolist()])
        print(f"[SAVE] Embeddings (CSV): {csv_path}")
        return

    raise ValueError(f"Unsupported fmt: {fmt} (use 'npz' or 'csv')")

# -----------------------------
# Main
# -----------------------------
def main():
    
    ap = argparse.ArgumentParser(description="DOLPHIN → diagnostic embeddings → t-SNE/UMAP + LR CV")
    ap.add_argument("--pkl", type=str, required=True, help="Path to PRELBD-tf.pkl (joblib or pickle).")
    ap.add_argument("--cols", type=str, default="0,1,6", help="Indices for (x,y,p), e.g. '0,1,6'.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--aggregate", action="store_true", help="Average embeddings per subject before plots (tsne/umap).")
    ap.add_argument("--report_subject_level", action="store_true", help="Generate the report on subject-level means instead of per-sample.")
    ap.add_argument("--tsne", action="store_true", help="Make a t-SNE plot.")
    ap.add_argument("--umap", action="store_true", help="Make a UMAP plot (requires umap-learn).")
    ap.add_argument("--outdir", type=str, default="./outs_prelbd")
    ap.add_argument("--ckpt", type=str, default=None, help="Optional model checkpoint to load (strict=False).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cv", type=int, default=5, help="Stratified K-fold for the diagnostic classifier in report.")
    
    ap.add_argument("--save_embeddings", action="store_true", help="Dump extracted embeddings to disk for downstream classifiers.")
    ap.add_argument("--emb_format", type=str, default="npz", help="Embeddings format: 'npz' (recommended) or 'csv'.")
    ap.add_argument("--emb_dir", type=str, default=None, help="Directory to save embeddings. Default: <outdir>/<task>/")
    ap.add_argument("--aggregate_export", action="store_true", help="If set, also export subject-level mean embeddings.")

    args = ap.parse_args()

    pkl_path = Path(args.pkl); assert pkl_path.exists(), f"Missing: {pkl_path}"
    cols = tuple(int(x) for x in args.cols.split(","))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] {pkl_path}")
    data = load_any_pkl(pkl_path)

    # dataset & loader
    ds = WritingDiagnostic(data)
    print(f"Subjects (unique): {len(np.unique(ds.subject_ids))} | Samples: {len(ds)} | Label ratio (1's): {ds.labels.mean():.3f}")

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=lambda b: collate_fn_dolphin(b, cols=cols))

    # model
    num_classes_dummy = 1000
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
        # this aggregation is ONLY for visualization; the report has its own subject_level switch
        X_vis, y_vis, subs_vis = aggregate_by_subject_mean(X, y, subs)
        print(f"[Aggregate] Visualization per subject: {X_vis.shape[0]} vectors")
    else:
        X_vis, y_vis, subs_vis = X, y, subs

    # plots
    if args.tsne:
        plot_tsne(X_vis, y_vis, save_path=outdir / ("tsne_subjects.png" if args.aggregate else "tsne_samples.png"))
    if args.umap:
        plot_umap(X_vis, y_vis, save_path=outdir / ("umap_subjects.png" if args.aggregate else "umap_samples.png"))

    # legacy quick classifier on the same representation you plotted
    bal_acc, cm, sens, spec = evaluate_classifier(X_vis, y_vis, cv=5, C=1.0, seed=0)
    tn, fp, fn, tp = cm.ravel()
    print("\n=== Diagnostic CV on plotted representation ===")
    print(f"Balanced accuracy: {bal_acc*100:.2f}%")
    print(f"Sensitivity (TPR, class=1): {sens*100:.2f}%")
    print(f"Specificity (TNR, class=0): {spec*100:.2f}%")
    print("Confusion matrix [[TN FP][FN TP]]:\n", cm)
    np.savez(outdir / "metrics.npz", bal_acc=bal_acc, cm=cm, sens=sens, spec=spec)

    # ===== NEW: full report (independent of plotting aggregation) =====
    make_full_report(
        X, y, subs,
        outdir=str(outdir),
        subject_level=bool(args.report_subject_level),
        cv=args.cv
    )

if __name__ == "__main__":
# call python diagnose_prelbd.py --pkl ./data/LBD_CZ_002/LBD_CZ_002-tf.pkl --batch 1 --report_subject_level --tsne --umap --outdir ./out_prelbd
    main()
