"""src.metrics — quality, alignment, K-selection and timing metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


@dataclass
class MetricBag:
    """Container for a single method run's metrics."""

    sc: float = float("nan")
    dbi: float = float("nan")
    nmi: float = float("nan")
    k_pred: int = 0
    k_gt: int = 0
    k_rel_err: float = float("nan")
    time_total: float = 0.0
    time_a: float = 0.0
    time_b: float = 0.0
    time_c: float = 0.0
    notes: Dict[str, str] = field(default_factory=dict)

    def to_row(self, method: str) -> Dict[str, object]:
        return {
            "method": method,
            "SC": self.sc,
            "DBI": self.dbi,
            "NMI": self.nmi,
            "K_pred": self.k_pred,
            "K_gt": self.k_gt,
            "K_rel_err": self.k_rel_err,
            "time_total_s": self.time_total,
            "time_A_s": self.time_a,
            "time_B_s": self.time_b,
            "time_C_s": self.time_c,
        }


# -----------------------------------------------------------------------------
# clustering-quality metrics
# -----------------------------------------------------------------------------
def safe_silhouette(vectors: np.ndarray, labels: np.ndarray, dist_matrix: Optional[np.ndarray] = None) -> float:
    """Silhouette score with guards (single cluster -> NaN).

    If `dist_matrix` is given, it must be the (n, n) euclidean distance
    matrix and is used with metric='precomputed' (much faster when the
    score is evaluated for many different label vectors).
    """
    if len(set(labels.tolist())) < 2:
        return float("nan")
    if dist_matrix is not None:
        return float(silhouette_score(dist_matrix, labels, metric="precomputed"))
    return float(silhouette_score(vectors, labels, metric="euclidean"))


def safe_davies_bouldin(vectors: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(davies_bouldin_score(vectors, labels))


def safe_nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    if len(true_labels) != len(pred_labels):
        return float("nan")
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def k_relative_error(k_pred: int, k_gt: int) -> float:
    if k_gt <= 0:
        return float("nan")
    return float(abs(k_pred - k_gt) / k_gt)
