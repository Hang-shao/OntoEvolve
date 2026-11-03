"""src.utils — small utilities shared by drac/metrics/methods.

Utilities are deliberately pure-python/numpy so unit tests do not need sklearn.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


# -----------------------------------------------------------------------------
# vector helpers
# -----------------------------------------------------------------------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; rows with zero norm are kept as zeros."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return x / n


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Cosine distance matrix in [0, 2]; relies on row-normalised input."""
    sim = np.clip(x @ x.T, -1.0, 1.0)
    return 1.0 - sim


# -----------------------------------------------------------------------------
# kNN / local-density estimates
# -----------------------------------------------------------------------------
def local_knn_distances(
    member_mask: np.ndarray,
    full_vectors: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return flattened kNN distances for every member of `member_mask`.

    For cluster members, look up their k nearest neighbours in the *whole*
    dataset (so cross-cluster pairs are observed too).  Skip self-distance.
    """
    members = np.where(member_mask)[0]
    if members.size == 0:
        return np.zeros(0, dtype=np.float64)
    k_eff = max(1, min(k, full_vectors.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(full_vectors)
    dists, _ = nn.kneighbors(full_vectors[members])
    return dists[:, 1:].flatten()


def compute_eps_local(
    member_mask: np.ndarray,
    full_vectors: np.ndarray,
    k: int,
    q: float,
) -> float:
    """epsilon_i = quantile_q(D_k(C_i))."""
    dists = local_knn_distances(member_mask, full_vectors, k)
    if dists.size == 0:
        return 0.0
    return float(np.quantile(dists, max(0.0, min(1.0, q))))


# -----------------------------------------------------------------------------
# within-cluster dispersion (coefficient of variation)
# -----------------------------------------------------------------------------
def compute_cv(
    member_indices: np.ndarray,
    full_vectors: np.ndarray,
    sample_pairs: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """CV of within-cluster pairwise L2 distances."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(member_indices)
    if n < 2:
        return 0.0
    pts = full_vectors[member_indices]
    if n <= 200:
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = d[iu]
    else:
        m = sample_pairs
        a = rng.integers(0, n, size=m)
        b = rng.integers(0, n, size=m)
        same = a == b
        b[same] = (b[same] + 1) % n
        d = np.linalg.norm(pts[a] - pts[b], axis=1)
        if d.size == 0:
            return 0.0
    m = float(np.mean(d))
    if m <= 1e-12:
        return 0.0
    return float(np.std(d) / m)


# -----------------------------------------------------------------------------
# binary search for tau (module C)
# -----------------------------------------------------------------------------
def binary_search_tau(
    cut_fn,
    target_k: int,
    tau_low: float,
    tau_high: float,
    *,
    tolerance_pct: float = 0.10,
    max_iter: int = 20,
    no_improve_tol: int = 3,
) -> Tuple[float, int]:
    """Robustised binary search to find tau giving K_pred close to target_k.

    Parameters
    ----------
    cut_fn : callable(float) -> int
        Given a candidate tau, return the number of concepts produced.
    target_k : int
        Desired number of concepts.
    tau_low, tau_high : float
        Search interval.

    Returns
    -------
    best_tau, best_k_pred
    """
    best_tau = (tau_low + tau_high) / 2
    best_k = cut_fn(best_tau)
    best_dev = abs(best_k - target_k) / max(1, target_k)
    no_improve = 0
    lo, hi = tau_low, tau_high
    for it in range(max_iter):
        mid = (lo + hi) / 2
        k_pred = cut_fn(mid)
        dev = abs(k_pred - target_k) / max(1, target_k)
        if dev < best_dev:
            best_dev = dev
            best_tau = mid
            best_k = k_pred
            no_improve = 0
        else:
            no_improve += 1
        if dev <= tolerance_pct:
            return mid, k_pred
        if no_improve >= no_improve_tol:
            return best_tau, best_k
        if k_pred < target_k * (1 - tolerance_pct):
            hi = mid  # too few clusters -> cut harder -> tau down
        elif k_pred > target_k * (1 + tolerance_pct):
            lo = mid  # too many clusters -> consolidate -> tau up
        else:
            return mid, k_pred
    return best_tau, best_k


# -----------------------------------------------------------------------------
# tree depth helper
# -----------------------------------------------------------------------------
def sibling_center_cosine_sim(
    left_idx: np.ndarray, right_idx: np.ndarray, vectors: np.ndarray
) -> float:
    """Cosine similarity between two cluster centers (mean of members)."""
    cl = vectors[left_idx].mean(axis=0)
    cr = vectors[right_idx].mean(axis=0)
    nl = np.linalg.norm(cl)
    nr = np.linalg.norm(cr)
    if nl < 1e-12 or nr < 1e-12:
        return 0.0
    return float(np.dot(cl, cr) / (nl * nr))
