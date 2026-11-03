"""Default hyperparameters of the clustering base (modules A-C).

All experiment scripts import from here. Overriding is done through
function arguments; values below are paper defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# ---- file locations ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DATA_ROOT = Path(os.getenv("ONTOEVOLVE_DATA_ROOT", str(WORKSPACE_ROOT / "data")))

# ---- Qwen3-Embedding endpoint ----------------------------------------------
API_URL = os.getenv("ONTOEVOLVE_EMBEDDING_API_URL", "").strip()
API_KEY = os.getenv("ONTOEVOLVE_EMBEDDING_API_KEY", "").strip()
EMBED_MODEL = os.getenv("ONTOEVOLVE_EMBEDDING_MODEL", "Qwen3-Embedding-0.6B")
EMBED_DIM = int(os.getenv("ONTOEVOLVE_EMBEDDING_DIM", "1024"))


@dataclass
class HyperParams:
    """Default hyperparameters."""

    # ---- module A: local-density adaptive eps ------------------------------
    q: float = 0.20                       # quantile for epsilon
    k_min: int = 5                        # lower bound for k
    eps_min: float = 1e-8                 # floor constant for boundary cases

    # ---- module B: K selection scoring ---------------------------------------
    lambda1: float = 0.4                  # cluster-quality weight
    lambda2: float = 0.3                  # hierarchy-consistency weight
    lambda3: float = 0.3                  # ontology-granularity weight
    w1: float = 0.6                       # SC sub-weight inside Q_cluster
    w2: float = 0.4                       # (1-DBI) sub-weight inside Q_cluster
    s_star_min: float = 5.0               # floor for target mean cluster size
    k_min_K: int = 2                      # floor for candidate layer count
    k_min_ratio: float = 0.5              # K_min = max(2, floor(sqrt(n)*k_min_ratio))
    k_max_ratio: float = 3.0              # K_max = min(n, floor(sqrt(n)*k_max_ratio))

    # ---- module C: semantic granularity consistency-driven cutting ---------
    tau_init: float = 0.10                # initial threshold
    tau_low: float = 0.02                 # binary search lower bound
    tau_high: float = 0.30                # binary search upper bound
    cv_sample_size: int = 200             # pair sample size for large clusters
    cv_max_full: int = 200                # compute all pairs when |C| <= this
    max_iter: int = 20                    # binary search max iterations
    no_improve_tol: int = 3               # stop if no improvement for n iters
    tolerance_pct: float = 0.10           # binary search accept zone (±10%)

    # ---- DRAC baseline ------------------------------------------------------
    global_eps_grid: Sequence[float] = (1e-6, 1e-4, 1e-2, 0.1, 1.0)
    global_eps_default: float = 1e-4

    # ---- experiment-level ---------------------------------------------------
    seed: int = 42
    n_repeats: int = 5
    # asymptotic neighbour formula: k = ceil(log2 n); respect k_min below


HP = HyperParams()


def k_default(n: int) -> int:
    """Default: k = ceil(log_2 n); lower-bounded by HP.k_min."""
    import math
    return max(HP.k_min, math.ceil(math.log2(max(2, n))))
