# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model_utils import db_augmentation, average_query_expansion


def _ensure_2d(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Ensure the feature array is 2D [N, D]."""
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return x.reshape(1, -1)
        return x
    # torch
    if x.dim() == 1:
        return x.view(1, -1)
    return x


def compute_mAP(rank_idx: np.ndarray, positive_idx: np.ndarray) -> Tuple[float, torch.Tensor]:
    """
    Compute AP and CMC for a single query given:
      - rank_idx: np.ndarray of shape [Ng] with indices of gallery sorted (best->worst)
      - positive_idx: np.ndarray of positions in the gallery that are true matches

    Returns:
      ap (float), cmc (torch.Tensor [Ng]) where cmc[k] = 1 from the first hit onward.
    """
    ap = 0.0
    cmc = torch.zeros((len(rank_idx)), dtype=torch.float32)

    if positive_idx.size == 0:
        cmc[0] = -1  # mark invalid
        return ap, cmc

    # mask over the ranked list telling where positives sit
    mask = np.in1d(rank_idx, positive_idx)
    rows_pos = np.argwhere(mask).flatten()  # rank positions (0-based) where positives occur
    cmc[rows_pos[0]:] = 1  # once the first correct appears, CMC is 1 onward

    # Trapezoidal AP between consecutive positive hits (smooths precision curve)
    len_pos = len(rows_pos)
    for i in range(len_pos):
        precision_now = (i + 1) / (rows_pos[i] + 1)
        precision_prev = (i / rows_pos[i]) if rows_pos[i] != 0 else 1.0
        ap += 0.5 * (precision_prev + precision_now)
    ap /= len_pos
    return ap, cmc


def evaluate_single(
    qf: torch.Tensor, ql: int, gf: torch.Tensor, gl: np.ndarray
) -> Tuple[float, torch.Tensor]:
    """
    Single-query evaluation (kept for compatibility / debugging).
    Assumes qf, gf are already L2-normalized for cosine similarity.

    Args:
      qf: torch.Tensor [D] or [1, D]
      ql: int (query label)
      gf: torch.Tensor [Ng, D]
      gl: numpy array [Ng] of gallery labels

    Returns:
      (ap, cmc) for this query
    """
    qf = _ensure_2d(qf).view(-1)  # [D]
    score = torch.matmul(gf, qf.view(-1, 1)).squeeze(1)  # [Ng]
    rank_idx = torch.argsort(score, descending=True).cpu().numpy()
    pos = np.argwhere(gl == ql).flatten()
    return compute_mAP(rank_idx, pos)


def compute_metrics(
    res: Dict[str, np.ndarray],
    logger,
    dba: bool,
    device: torch.device,
    verbose: bool = True,
    renorm_if_needed: bool = True,
) -> Tuple[float, float, float, float, float]:
    """
    Vectorized evaluation over all queries.

    Args:
      res:
        {
          'query_feature':  np.ndarray [Nq, D],
          'query_label':    np.ndarray [Nq],
          'gallery_feature':np.ndarray [Ng, D],
          'gallery_label':  np.ndarray [Ng],
        }
      logger: logger for timing/info
      dba: whether to apply DBA + AQE before scoring
      device: torch.device
      verbose: print summary logs
      renorm_if_needed: if True and dba==False, L2-normalize features defensively

    Returns:
      time_avg (s/query), mAP, Rank@1(%), Rank@5(%), Rank@10(%)
    """
    qf = res["query_feature"]
    ql = res["query_label"]
    gf = res["gallery_feature"]
    gl = res["gallery_label"]

    # Optional DBA + AQE (numpy space)
    if dba:
        t0 = time.time()
        qf, gf = db_augmentation(qf, gf, topk=10)
        qf, gf = average_query_expansion(qf, gf, topk=5)
        # renorm after DBA/AQE
        qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
        gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
        logger.info(f"DBA & AQE time: {time.time() - t0:.4f}s")
    elif renorm_if_needed:
        # Defensive: ensure cosine similarity downstream behaves as intended
        qf = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12)
        gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)

    # Move to torch (vectorized scoring)
    tq = torch.from_numpy(qf).to(device=device, dtype=torch.float32)  # [Nq, D]
    tg = torch.from_numpy(gf).to(device=device, dtype=torch.float32)  # [Ng, D]

    # Cosine similarity matrix (since features are L2-normalized)
    t0 = time.time()
    sims = torch.matmul(tq, tg.t())  # [Nq, Ng]
    time_total = time.time() - t0
    time_avg = float(time_total / max(1, tq.size(0)))

    # Rank indices for all queries at once (desc)
    rank_idx_all = torch.argsort(sims, dim=1, descending=True).cpu().numpy()

    # Compute AP & CMC per query on CPU
    gl_np = gl  # already numpy
    CMC = torch.zeros((tg.size(0)), dtype=torch.float32)
    ap_sum = 0.0
    valid_queries = 0

    unique_labels = np.unique(ql)
    # map label -> gallery positions for quick lookup
    label2pos = {lab: np.where(gl_np == lab)[0] for lab in unique_labels}

    for i in range(rank_idx_all.shape[0]):
        pos = label2pos.get(ql[i], np.array([], dtype=int))
        ap_i, cmc_i = compute_mAP(rank_idx_all[i], pos)
        if cmc_i[0] == -1:
            continue
        CMC += cmc_i
        ap_sum += ap_i
        valid_queries += 1

    if valid_queries == 0:
        if verbose:
            logger.info("[compute_metrics] No valid queries found (no positives in gallery).")
        return time_avg, 0.0, 0.0, 0.0, 0.0

    CMC /= valid_queries
    mAP = ap_sum / valid_queries

    r1 = float(CMC[0] * 100.0)
    r5 = float(CMC[4] * 100.0) if CMC.numel() > 4 else float(CMC[-1] * 100.0)
    r10 = float(CMC[9] * 100.0) if CMC.numel() > 9 else float(CMC[-1] * 100.0)

    if verbose:
        logger.info(
            f"[eval] Rank@1: {r1:.4f}%  Rank@5: {r5:.4f}%  Rank@10: {r10:.4f}%"
        )
        logger.info(f"[eval] mAP: {mAP * 100.0:.4f}%")
        logger.info(f"[eval] avg time/query: {time_avg:.6f}s")

    return time_avg, mAP, r1, r5, r10


if __name__ == "__main__":
    # Minimal smoke test stub:
    # This block expects you to construct a `res` dict programmatically and call compute_metrics.
    # Example (toy random, NOT a real eval):
    #
    #   import numpy as np, torch
    #   from this_file import compute_metrics
    #   class DummyLogger: 
    #       def info(self, s): print(s)
    #   logger = DummyLogger()
    #   Nq, Ng, D = 5, 20, 16
    #   # build labels: each query has at least one positive in gallery
    #   ql = np.arange(Nq)
    #   gl = np.concatenate([np.array([i, i]) for i in ql] + [np.arange(Ng - 2*Nq)])
    #   qf = np.random.randn(Nq, D).astype(np.float32)
    #   gf = np.random.randn(Ng, D).astype(np.float32)
    #   res = {'query_feature': qf, 'query_label': ql, 'gallery_feature': gf, 'gallery_label': gl}
    #   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #   compute_metrics(res, logger, dba=False, device=device, verbose=True)
    #
    # By default, we do nothing in __main__ to avoid accidental runs.
    pass
