# viz.py
# -*- coding: utf-8 -*-
import os, math, json
import numpy as np
import matplotlib.pyplot as plt

# ---------- basic helpers ----------
def _mkdir(p): os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    _mkdir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ---------- CMC + mAP ----------
def plot_cmc(cmc_vec, save_path="output/cmc_curve.png", title="CMC (Cumulative Match Characteristic)"):
    """cmc_vec: 1D numpy/torch of length Ng (values in [0,1])"""
    y = np.array(cmc_vec, dtype=float)
    x = np.arange(1, len(y)+1)
    _mkdir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(6,4))
    plt.plot(x, y*100.0, linewidth=2)
    plt.xlabel("Rank K")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def bar_retrieval_scores(r1, r5, r10, mAP, save_path="output/retrieval_bars.png"):
    _mkdir(os.path.dirname(save_path) or ".")
    names = ["Rank@1", "Rank@5", "Rank@10", "mAP"]
    vals  = [r1, r5, r10, mAP*100.0]
    plt.figure(figsize=(6,4))
    plt.bar(names, vals)
    for i,v in enumerate(vals):
        plt.text(i, v+0.5, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 105)
    plt.ylabel("Percentage (%)")
    plt.title("Retrieval Summary")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------- t-SNE / UMAP of embeddings ----------
def plot_tsne(embeddings, labels, save_path="output/tsne.png", perplexity=40, max_points=5000):
    """embeddings: [N,D] numpy; labels: [N] numpy (writer ids)"""
    from sklearn.manifold import TSNE
    _mkdir(os.path.dirname(save_path) or ".")
    N = len(labels)
    if N > max_points:
        # sample to keep plot readable & fast
        idx = np.random.choice(N, size=max_points, replace=False)
        X = embeddings[idx]
        y = labels[idx]
    else:
        X = embeddings
        y = labels
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=0)
    Z = tsne.fit_transform(X)
    plt.figure(figsize=(6,6))
    sc = plt.scatter(Z[:,0], Z[:,1], c=y, s=5, cmap="tab20", alpha=0.8)
    plt.title("t-SNE of embeddings")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_umap(embeddings, labels, save_path="output/umap.png", n_neighbors=15, min_dist=0.1, max_points=10000):
    """requires: pip install umap-learn"""
    import umap
    _mkdir(os.path.dirname(save_path) or ".")
    N = len(labels)
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        X = embeddings[idx]; y = labels[idx]
    else:
        X = embeddings; y = labels
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
    Z = reducer.fit_transform(X)
    plt.figure(figsize=(6,6))
    plt.scatter(Z[:,0], Z[:,1], c=y, s=5, cmap="tab20", alpha=0.8)
    plt.title("UMAP of embeddings")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------- sequence length & feature histograms ----------
def plot_length_hist(lengths, save_path="output/length_hist.png", bins=50, title="Sequence length distribution"):
    _mkdir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(6,4))
    plt.hist(lengths, bins=bins)
    plt.xlabel("T (timesteps)"); plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_timeseries(sample_tf, save_path="output/timefunctions.png", max_cols=4, title="Time-function snapshot"):
    """
    sample_tf: [T, D] time-function matrix (e.g., from your dataset)
    Plots first min(D, 8) channels over time.
    """
    _mkdir(os.path.dirname(save_path) or ".")
    T, D = sample_tf.shape
    use = min(D, 8)
    rows = math.ceil(use / max_cols)
    plt.figure(figsize=(3*max_cols, 2.2*rows))
    for i in range(use):
        ax = plt.subplot(rows, max_cols, i+1)
        ax.plot(np.arange(T), sample_tf[:, i], linewidth=1)
        ax.set_title(f"ch {i}")
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------- quick 2D stroke plotter (if you still have raw x,y) ----------
def plot_xy_traj(xy, save_path="output/trajectory.png", title="Handwriting trajectory (x,y)"):
    """
    xy: [T,2] array of normalized coordinates (if available).
    If you only have time-functions, skip this.
    """
    _mkdir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(4,4))
    plt.plot(xy[:,0], -xy[:,1], linewidth=1)  # flip y for nicer orientation
    plt.axis("equal"); plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
