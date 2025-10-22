# viz.py
# -*- coding: utf-8 -*-
import os, math, json
import numpy as np
import matplotlib.pyplot as plt
# Tryout of 3D s-TNS plotting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


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

def save_topk_panel(query_vec, gallery_mat, gallery_labels, label_names=None, k=5, save_path="output/topk.png"):
    """Shows the top-k gallery items (just text + bars). Replace with trajectory plots if you have raw x,y."""
    sims = gallery_mat @ query_vec  # both L2-normalized
    idx = np.argsort(-sims)[:k]
    scores = sims[idx]
    labels = gallery_labels[idx]
    txt = [f"{i+1}. label={labels[i]}  score={scores[i]:.3f}" for i in range(k)]
    _mkdir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(6, 1 + 0.35*k))
    plt.barh(range(k), scores[::-1])
    plt.yticks(range(k), [t for t in txt[::-1]], fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================================================
# 3D t-SNE (temporal+frequency embedding) preview
# ================================================

def tsne_3d(features, labels, save_path=None, max_points=8000, perplexity=35, random_state=42):
    """
    features : (N, D) numpy or torch tensor (use all_features from your pipeline)
    labels   : (N,)  numpy or torch tensor of ints (use all_labels)
    """
    # to numpy
    if "torch" in str(type(features)):
        features = features.detach().cpu().numpy()
    if "torch" in str(type(labels)):
        labels = labels.detach().cpu().numpy()

    # optional subsample to keep it fast & legible
    N = len(features)
    if max_points and N > max_points:
        idx = np.random.RandomState(random_state).choice(N, size=max_points, replace=False)
        X = features[idx]
        y = labels[idx]
    else:
        X, y = features, labels

    # safety: t-SNE requires perplexity < n_samples
    perp = min(perplexity, max(5, len(X)//3))

    # cosine-friendly scaling
    Xn = normalize(X, norm="l2", axis=1)

    tsne = TSNE(
        n_components=3,
        init="pca",
        learning_rate="auto",
        perplexity=perp,
        metric="euclidean",
        n_iter=1500,
        random_state=random_state,
        verbose=0,
    )
    Z = tsne.fit_transform(Xn)  # (n, 3)

    # 3D scatter
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    # color by label; keep defaults simple
    sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=6, c=y, alpha=0.9)
    ax.set_title("3D t-SNE of embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    # light frame
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1,1,1,0.0))
        axis._axinfo["grid"]["linewidth"] = 0.3

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved 3D t-SNE to: {save_path}")
    plt.show()
    return Z, y