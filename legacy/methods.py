"""src.methods — modules B (auto K) and C (dynamic cut) and the 5 methods.

Methods:

    B1  original DRAC + ground-truth K  + horizontal cut
    B2  original DRAC + auto K         + horizontal cut
    M1  module A        + ground-truth K  + horizontal cut
    M2  module A        + auto K         + horizontal cut
    M3  module A        + auto K         + dynamic (module C) cut

All methods return a `MethodResult` with: labels, k_pred, linkage, timings,
notes (e.g. scores_per_k for module B debug).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .drac import DRAC, DRACResult, horizontal_cut
from .metrics import (
    k_relative_error,
    safe_davies_bouldin,
    safe_nmi,
    safe_silhouette,
)
from .utils import binary_search_tau, compute_cv


# -----------------------------------------------------------------------------
# MethodResult + registry
# -----------------------------------------------------------------------------
@dataclass
class MethodResult:
    labels: np.ndarray                # final flat labels (n,)
    k_pred: int
    linkage: np.ndarray               # the dendrogram (n-1, 4)
    members: List[np.ndarray]         # cluster members per scipy id
    drac_mode: str
    timings: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, object] = field(default_factory=dict)


METHODS = ("B1", "B2", "M1", "M2", "M3")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _pairwise_distance_cos(vectors: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray) -> float:
    if len(a_idx) == 0 or len(b_idx) == 0:
        return 0.0
    ca = vectors[a_idx].mean(axis=0)
    cb = vectors[b_idx].mean(axis=0)
    na = float(np.linalg.norm(ca))
    nb = float(np.linalg.norm(cb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(ca, cb) / (na * nb))


def _active_internal_ids_at_layer(linkage: np.ndarray, K_l: int, n: int) -> List[int]:
    """scipy ids of internal nodes that are *still top-level* at layer K_l.

    Performs the first (n - K_l) merges with union-find and returns the ids
    of internal roots.
    """
    if K_l <= 1 or K_l >= n:
        return []
    parent = list(range(2 * n - 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_merges = min(n - K_l, len(linkage))
    for i in range(n_merges):
        a, b = int(linkage[i, 0]), int(linkage[i, 1])
        new_root = n + i
        ra, rb = find(a), find(b)
        parent[ra] = new_root
        parent[rb] = new_root

    roots = {find(i) for i in range(n)}
    return sorted(r for r in roots if r >= n)


def _target_cluster_size(vectors: np.ndarray) -> float:
    """s* = max(5, floor(n * (d_med / d_global)^2))."""
    n = vectors.shape[0]
    # nearest-neighbour distances
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(vectors)
    d, _ = nn.kneighbors(vectors)
    d_nn = d[:, 1]
    # average global distance: sample to keep it cheap for large n
    if n <= 1500:
        from scipy.spatial.distance import cdist
        all_d = cdist(vectors, vectors)
        iu = np.triu_indices(n, k=1)
        d_global = float(np.mean(all_d[iu]))
    else:
        rng = np.random.default_rng(0)
        m = 1500
        idx = rng.integers(0, n, size=(m, 2))
        d_global = float(
            np.linalg.norm(vectors[idx[:, 0]] - vectors[idx[:, 1]], axis=1).mean()
        )
    d_med = float(np.median(d_nn))
    ratio = d_med / max(d_global, 1e-12)
    return max(C.HP.s_star_min, math.floor(n * (ratio * ratio)))


# -----------------------------------------------------------------------------
# Module B — auto K
# -----------------------------------------------------------------------------
def module_b_select_k(
    linkage: np.ndarray,
    vectors: np.ndarray,
    drac_result: DRACResult,
    *,
    score_cache: Optional[Dict] = None,
    lambda1: float = C.HP.lambda1,
    lambda2: float = C.HP.lambda2,
    lambda3: float = C.HP.lambda3,
    w1: float = C.HP.w1,
    w2: float = C.HP.w2,
) -> Tuple[int, Dict]:
    """Module B: select K* via Q_total = λ1 Q_c + λ2 Q_h + λ3 Q_g.

    Returns: (K_star, info_dict)  -- info_dict has per-K scores for plotting.
    """
    n = len(vectors)
    s_star = _target_cluster_size(vectors)
    n_sqrt = math.sqrt(n)
    K_min = max(C.HP.k_min_K, int(math.floor(n_sqrt * C.HP.k_min_ratio)))
    K_max = min(n - 1, int(math.floor(n_sqrt * C.HP.k_max_ratio)))
    K_min = max(2, K_min)
    K_max = max(K_max, K_min + 1)
    candidates = list(range(K_min, K_max + 1))
    info: Dict = {"K_min": K_min, "K_max": K_max, "s_star": s_star, "rows": []}

    # cache SC/DBI per K (sometimes fcluster gives same labels as previous)
    sc_cache: Dict[int, float] = {}
    dbi_cache: Dict[int, float] = {}

    # precompute the euclidean distance matrix once (SC with precomputed)
    from scipy.spatial.distance import cdist
    D_full = cdist(vectors, vectors)

    best_k = K_min
    best_q = float("-inf")
    best_row = None
    for K_l in candidates:
        labels = horizontal_cut(linkage, K_l, n)
        if K_l not in sc_cache:
            sc_cache[K_l] = safe_silhouette(vectors, labels, dist_matrix=D_full)
        if K_l not in dbi_cache:
            dbi_cache[K_l] = safe_davies_bouldin(vectors, labels)
        sc = sc_cache[K_l]
        dbi = dbi_cache[K_l]
        # Q_cluster is well-defined only when both scores are computable
        if np.isnan(sc) or np.isnan(dbi):
            q_c = float("nan")
        else:
            # normalise both terms to [0,1] to avoid scale mismatch:
            # SC in [-1,1] -> (SC+1)/2 in [0,1]; DBI in [0,inf) -> 1/(1+DBI) in (0,1]
            q_c = w1 * ((sc + 1.0) / 2.0) + w2 * (1.0 / (1.0 + dbi))
        # Q_hierarchy
        internal_ids = _active_internal_ids_at_layer(linkage, K_l, n)
        if internal_ids:
            sims = []
            for cid in internal_ids:
                i_under = n + int(np.where(linkage[:, 0] == cid)[0][0]) if False else None
                # find the row whose new_root equals cid
                row_idx = cid - n  # since cid = n + i, i = cid - n
                # guard against cid that isn't actually a created root
                if row_idx < 0 or row_idx >= len(linkage):
                    continue
                a, b = int(linkage[row_idx, 0]), int(linkage[row_idx, 1])
                sims.append(_pairwise_distance_cos(vectors,
                                                  drac_result.members[a],
                                                  drac_result.members[b]))
            q_h = 1.0 - float(np.mean(sims))
        else:
            q_h = 0.0
        # Q_granularity
        avg_size = n / K_l
        q_g = float(np.exp(-((avg_size - s_star) / max(s_star, 1e-9)) ** 2))
        # Q_total (substitute nan by 0)
        def _x(x): return 0.0 if (x != x) else x

        q_total = lambda1 * _x(q_c) + lambda2 * _x(q_h) + lambda3 * _x(q_g)
        info["rows"].append({
            "K": K_l, "SC": sc, "DBI": dbi,
            "Q_cluster": _x(q_c), "Q_hierarchy": q_h, "Q_granularity": q_g,
            "Q_total": q_total,
        })
        if q_total > best_q:
            best_q = q_total
            best_k = K_l
            best_row = info["rows"][-1]
    info["best_row"] = best_row
    return best_k, info


# -----------------------------------------------------------------------------
# Module C — dynamic tree cut
# -----------------------------------------------------------------------------
def _cv_for_node(node_id: int, drac_result: DRACResult, vectors: np.ndarray, rng) -> float:
    members = drac_result.members[node_id]
    return compute_cv(members, vectors, sample_pairs=C.HP.cv_sample_size, rng=rng)


def module_c_dynamic_cut(
    drac_result: DRACResult,
    vectors: np.ndarray,
    K_star: int,
    *,
    tau_init: float = C.HP.tau_init,
    tau_low: float = C.HP.tau_low,
    tau_high: float = C.HP.tau_high,
    rng: Optional[np.random.Generator] = None,
    fallback_tol: float = C.HP.tolerance_pct,
    fixed_tau: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Top-down CV-based cut, tau binary-searched so concepts ~ K_star.

    If no tau gives K_pred close enough to
    K_star, fall back to a horizontal cut at K_star (closest layer).

    `fixed_tau` bypasses the binary search entirely (used by the tau
    sensitivity scan: cut at this exact threshold, no K alignment).
    """
    if rng is None:
        rng = np.random.default_rng(C.HP.seed)
    linkage = drac_result.linkage
    n = drac_result.n
    cv_cache: Dict[int, float] = {}

    def cv(node_id: int) -> float:
        if node_id not in cv_cache:
            cv_cache[node_id] = _cv_for_node(node_id, drac_result, vectors, rng)
        return cv_cache[node_id]

    if fixed_tau is not None:
        best_tau = float(fixed_tau)
        info = {
            "best_tau": best_tau,
            "best_k_cv": None,
            "cv_cache_size": 0,
            "K_star": K_star,
            "fixed_tau": True,
            "fallback": None,
        }
    else:
        def count_concepts(tau: float) -> int:
            if len(linkage) == 0:
                return n
            root = n + len(linkage) - 1
            stack = [root]
            c = 0
            while stack:
                cur = stack.pop()
                if cur < n:
                    c += 1
                    continue
                if cv(cur) <= tau:
                    c += 1
                    continue
                a, b = int(linkage[cur - n, 0]), int(linkage[cur - n, 1])
                stack.append(a)
                stack.append(b)
            return c

        best_tau, best_k = binary_search_tau(
            count_concepts,
            target_k=K_star,
            tau_low=tau_low,
            tau_high=tau_high,
            tolerance_pct=fallback_tol,
            max_iter=C.HP.max_iter,
            no_improve_tol=C.HP.no_improve_tol,
        )

        info = {
            "best_tau": best_tau,
            "best_k_cv": best_k,
            "cv_cache_size": len(cv_cache),
            "K_star": K_star,
            "fixed_tau": False,
        }

        # decide CV-cut vs horizontal fallback
        if abs(best_k - K_star) > fallback_tol * K_star:
            info["fallback"] = "horizontal_cut"
            labels = horizontal_cut(linkage, K_star, n)
            return labels.astype(int), info
        info["fallback"] = None

    # build final labels using DFS at best_tau (iterative, chain-safe).
    # Every visited node is a potential concept root: stop when CV <= tau.
    labels = np.full(n, -1, dtype=int)
    next_label = 0
    root = n + len(linkage) - 1
    stack = [root]
    while stack:
        node_id = stack.pop()
        if node_id < n:
            labels[node_id] = next_label
            next_label += 1
            continue
        if cv(node_id) <= best_tau:
            labels[drac_result.members[node_id]] = next_label
            next_label += 1
            continue
        a, b = int(linkage[node_id - n, 0]), int(linkage[node_id - n, 1])
        stack.append(a)
        stack.append(b)

    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(int), info




# -----------------------------------------------------------------------------
# global-eps grid search (for B1/B2)
# -----------------------------------------------------------------------------
def best_global_eps(
    vectors: np.ndarray,
    K_gt: int,
    *,
    candidates: Tuple[float, ...] = C.HP.global_eps_grid,
) -> float:
    """Greedy epsilon grid search for B1/B2 (cut at K_gt, maximise SC)."""
    best_eps, best_score = float(candidates[0]), float("-inf")
    n = len(vectors)
    for eps in candidates:
        drac = DRAC(vectors, eps_mode="global", global_eps=eps)
        res = drac.fit()
        labels = horizontal_cut(res.linkage, K_gt, n)
        sc = safe_silhouette(vectors, labels)
        score = -float("inf") if np.isnan(sc) else float(sc)
        if score > best_score:
            best_score = score
            best_eps = float(eps)
    return best_eps

# -----------------------------------------------------------------------------
# Method dispatch
# -----------------------------------------------------------------------------
def run_with_options(
    vectors: np.ndarray,
    K_gt: int,
    *,
    eps_mode: str = "global",
    global_eps: float = C.HP.global_eps_default,
    q: float = C.HP.q,
    k_neighbors=None,
    eps_min: float = C.HP.eps_min,
    use_auto_k: bool = False,
    use_dynamic_cut: bool = False,
    seed: int = C.HP.seed,
    label: str = "custom",
    lambda1: Optional[float] = None,
    lambda2: Optional[float] = None,
    lambda3: Optional[float] = None,
    w1: Optional[float] = None,
    w2: Optional[float] = None,
    fixed_tau: Optional[float] = None,
) -> MethodResult:
    """Universal method runner. Each boolean switches one stage of the base.

    Use `run_method(id)` below for the five entry points.
    Optional module-B/C hyper-parameters (lambda*, w*, fixed_tau) default to
    the values in config.HP at *call* time (NOT at definition time), so
    sensitivity scans can override them per call.
    """
    n = len(vectors)
    timings = {"A": 0.0, "B": 0.0, "C": 0.0}
    rng = np.random.default_rng(seed)

    # ---- module A: hierarchy construction --------------------------------
    t0 = time.time()
    drac = DRAC(
        vectors,
        eps_mode=eps_mode,
        global_eps=global_eps,
        q=q,
        k_neighbors=k_neighbors,
        eps_min=eps_min,
    )
    drac_res = drac.fit()
    timings["A"] = time.time() - t0

    info_b: Dict = {}
    # ---- module B: K auto selection --------------------------------------
    if use_auto_k:
        t0 = time.time()
        kw_b = {}
        if lambda1 is not None:
            kw_b["lambda1"] = lambda1
        if lambda2 is not None:
            kw_b["lambda2"] = lambda2
        if lambda3 is not None:
            kw_b["lambda3"] = lambda3
        if w1 is not None:
            kw_b["w1"] = w1
        if w2 is not None:
            kw_b["w2"] = w2
        K_star, info_b = module_b_select_k(drac_res.linkage, vectors, drac_res, **kw_b)
        timings["B"] = time.time() - t0
    else:
        K_star = K_gt

    # ---- module C: dynamic cut -------------------------------------------
    info_c: Dict = {}
    if use_dynamic_cut:
        t0 = time.time()
        labels, info_c = module_c_dynamic_cut(
            drac_res, vectors, K_star, rng=rng, fixed_tau=fixed_tau,
        )
        timings["C"] = time.time() - t0
    else:
        target_k = K_star
        labels = horizontal_cut(drac_res.linkage, target_k, n)
        info_c = {"K_star": K_star, "fallback": None}

    k_pred = int(labels.max() + 1)
    return MethodResult(
        labels=labels,
        k_pred=k_pred,
        linkage=drac_res.linkage,
        members=drac_res.members,
        drac_mode=eps_mode,
        timings=timings,
        notes={
            "label": label,
            "K_star": K_star,
            "info_b": info_b,
            "info_c": info_c,
            "global_eps_used": global_eps,
        },
    )




def run_method(
    method_id: str,
    vectors: np.ndarray,
    K_gt: int,
    *,
    global_eps: float = C.HP.global_eps_default,
    q: float = C.HP.q,
    k_neighbors=None,
    eps_min: float = C.HP.eps_min,
    seed: int = C.HP.seed,
) -> MethodResult:
    """Five-method entry point."""
    if method_id not in METHODS:
        raise ValueError(f"unknown method_id: {method_id}")
    cfg = {
        "B1": dict(eps_mode="global", use_auto_k=False, use_dynamic_cut=False, label="B1"),
        "B2": dict(eps_mode="global", use_auto_k=True,  use_dynamic_cut=False, label="B2"),
        "M1": dict(eps_mode="local",  use_auto_k=False, use_dynamic_cut=False, label="M1"),
        "M2": dict(eps_mode="local",  use_auto_k=True,  use_dynamic_cut=False, label="M2"),
        "M3": dict(eps_mode="local",  use_auto_k=True,  use_dynamic_cut=True,  label="M3"),
    }
    return run_with_options(
        vectors,
        K_gt,
        global_eps=global_eps,
        q=q,
        k_neighbors=k_neighbors,
        eps_min=eps_min,
        seed=seed,
        **cfg[method_id],
    )
