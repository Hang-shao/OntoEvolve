"""src.drac — generic dynamic-radius adaptive clustering (DRAC).

Agglomerative loop with a pluggable epsilon scheme and a strictly
scipy-compatible output linkage matrix.  Each merge produces a new internal
node with id = n + step (so leaves are 0..n-1 and the root is the last row).

eps_mode = 'global'  ->  one global epsilon everywhere (B1, B2)
eps_mode = 'local'   ->  per-cluster epsilon computed from local kNN
                          (module A)

Performance notes
-----------------
* All arrays (centres/radii/sizes/eps) are preallocated for 2n-1 nodes.
* The kNN distance table used by local-eps is computed ONCE.
* The d_hat matrix is maintained incrementally: a merge only changes one
  row/column, so the whole build is O(n^2) numpy work instead of O(n^3)
  python loops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import config as C


@dataclass
class DRACResult:
    linkage: np.ndarray        # (n-1, 4) scipy-compatible
    n: int
    eps_dict: Dict[int, float]  # local eps per internal node (only when local)
    members: List[np.ndarray]   # length 2n-1; members[id] = leaf indices under that node
    centres: np.ndarray         # (2n-1, d); centres[id] = cluster centre vector


class DRAC:
    def __init__(
        self,
        vectors: np.ndarray,
        *,
        eps_mode: str = "global",
        global_eps: float = C.HP.global_eps_default,
        q: float = C.HP.q,
        k_neighbors=None,
        eps_min: float = C.HP.eps_min,
        radius_mode: str = "mean",
    ) -> None:
        if eps_mode not in {"global", "local"}:
            raise ValueError("eps_mode must be 'global' or 'local'")
        if radius_mode not in {"mean", "max"}:
            raise ValueError("radius_mode must be 'mean' or 'max'")
        self.radius_mode = radius_mode
        self.X = np.ascontiguousarray(vectors, dtype=np.float64)
        self.n = int(self.X.shape[0])
        self.d = int(self.X.shape[1])
        self.eps_mode = eps_mode
        self.global_eps = float(global_eps) if global_eps is not None else 0.0
        self.q = float(q)
        if k_neighbors is None:
            self.k_neighbors = C.k_default(self.n)
        elif callable(k_neighbors):
            self.k_neighbors = int(k_neighbors(self.n))
        else:
            self.k_neighbors = int(k_neighbors)
        self.eps_min = float(eps_min)

        n = self.n
        # preallocated cluster state, index = scipy node id (0..2n-2)
        self._members: List[np.ndarray] = [np.array([i], dtype=int) for i in range(n)] + [None] * (n - 1)
        self._centres = np.zeros((2 * n - 1, self.d), dtype=np.float64)
        self._centres[:n] = self.X
        self._radii = np.zeros(2 * n - 1, dtype=np.float64)
        self._sizes = np.ones(2 * n - 1, dtype=np.int64)
        self._eps = np.zeros(2 * n - 1, dtype=np.float64)

        # kNN distance table for local eps (computed once)
        if self.eps_mode == "local":
            k_eff = max(1, min(self.k_neighbors, n - 1))
            nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(self.X)
            dists, _ = nn.kneighbors(self.X)          # (n, k_eff+1), col 0 == 0
            self._knn_d = dists[:, 1:]                 # (n, k_eff)
            # per-leaf eps
            for i in range(n):
                self._eps[i] = float(np.quantile(self._knn_d[i], self.q))
        else:
            self._knn_d = None

    # ------------------------------------------------------------------
    def _eps_for_members(self, members: np.ndarray) -> float:
        """quantile_q over the kNN distances of all cluster members."""
        if self._knn_d is None:
            return 0.0
        flat = self._knn_d[members].ravel()
        if flat.size == 0:
            return 0.0
        return float(np.quantile(flat, self.q))

    # ------------------------------------------------------------------
    def _d_hat_row(self, i: int, others: np.ndarray) -> np.ndarray:
        """Vectorised d_hat between cluster i and an array of other ids."""
        if others.size == 0:
            return np.zeros(0, dtype=np.float64)
        ci = self._centres[i]
        d_ij = np.linalg.norm(self._centres[others] - ci, axis=1)
        if self.eps_mode == "local":
            eps_pair = np.maximum(self._eps[i], self._eps[others])
        else:
            eps_pair = np.full(others.size, self.global_eps, dtype=np.float64)
        denom = np.maximum(
            self._radii[others] + self._radii[i],
            np.maximum(eps_pair, self.eps_min),
        )
        denom = np.where(denom > 0.0, denom, self.eps_min)
        return d_ij / denom

    # ------------------------------------------------------------------
    def fit(self) -> DRACResult:
        n = self.n
        if n == 1:
            return DRACResult(
                linkage=np.zeros((0, 4)),
                n=n,
                eps_dict={},
                members=[np.array([0])],
                centres=self._centres[:1].copy(),
            )

        linkage_rows: List[Tuple[int, int, float, int]] = []
        eps_log: Dict[int, float] = {}

        # ---- initial d_hat matrix over the n leaves --------------------
        centres = self._centres[:n]
        from scipy.spatial.distance import cdist
        D = cdist(centres, centres)
        if self.eps_mode == "local":
            eps_pair = np.maximum(self._eps[:n, None], self._eps[:n, None].T)
        else:
            eps_pair = np.full((n, n), self.global_eps, dtype=np.float64)
        denom = np.maximum(
            self._radii[:n, None] + self._radii[:n, None].T,
            np.maximum(eps_pair, self.eps_min),
        )
        denom = np.where(denom > 0.0, denom, self.eps_min)
        d_mat = D / denom
        np.fill_diagonal(d_mat, np.inf)

        # active slots: slot s <-> scipy id slots[s]; matrix is (m, m)
        slots: List[int] = list(range(n))
        m = n

        for step in range(n - 1):
            # ---- pick the globally smallest d_hat -----------------------
            flat = int(np.argmin(d_mat[:m, :m]))
            pa, pb = divmod(flat, m)
            if pa > pb:
                pa, pb = pb, pa
            i = slots[pa]
            j = slots[pb]
            best_val = float(d_mat[pa, pb])
            size_i = int(self._sizes[i])
            size_j = int(self._sizes[j])

            # ---- create the merged cluster ------------------------------
            members_new = np.concatenate([self._members[i], self._members[j]])
            centre_new = (size_i * self._centres[i] + size_j * self._centres[j]) / (size_i + size_j)
            dists_new = np.linalg.norm(self.X[members_new] - centre_new, axis=1)
            if self.radius_mode == "max":
                radius_new = float(dists_new.max())
            else:
                radius_new = float(dists_new.mean())
            if self.eps_mode == "local":
                eps_new = self._eps_for_members(members_new)
            else:
                eps_new = 0.0

            new_id = n + step
            self._members[new_id] = members_new
            self._centres[new_id] = centre_new
            self._radii[new_id] = radius_new
            self._sizes[new_id] = size_i + size_j
            self._eps[new_id] = eps_new
            if self.eps_mode == "local":
                eps_log[new_id] = eps_new

            linkage_rows.append((i, j, best_val, size_i + size_j))

            # ---- update the matrix in O(m) ------------------------------
            # move the LAST slot into pb, then reuse pa for the new cluster
            last = m - 1
            if pb != last:
                d_mat[pb, :m] = d_mat[last, :m]
                d_mat[:m, pb] = d_mat[:m, last]
                slots[pb] = slots[last]
                d_mat[pb, pb] = np.inf
            # (if pb == last, its row/col will be truncated away anyway)
            m -= 1
            # now active positions are 0..m-1 (m already decremented) with
            # position pa still holding old cluster i -> replace with new_id
            slots[pa] = new_id
            others = np.array([slots[s] for s in range(m) if s != pa], dtype=int)
            if others.size:
                row = self._d_hat_row(new_id, others)
                # write into matrix at position pa
                pos = 0
                for s in range(m):
                    if s == pa:
                        continue
                    d_mat[pa, s] = row[pos]
                    d_mat[s, pa] = row[pos]
                    pos += 1
            d_mat[pa, pa] = np.inf
            # ensure stale entries beyond m never get read: argmin is
            # restricted to [:m, :m], and slot pa/pb columns were refreshed.

        linkage = np.asarray(linkage_rows, dtype=np.float64)
        members_snapshot: List[np.ndarray] = [
            self._members[i] if i < n or self._members[i] is not None else np.array([], dtype=int)
            for i in range(2 * n - 1)
        ]
        return DRACResult(
            linkage=linkage,
            n=n,
            eps_dict=eps_log,
            members=members_snapshot,
            centres=self._centres.copy(),
        )


# -----------------------------------------------------------------------------
# horizontal cut helpers
# -----------------------------------------------------------------------------
def horizontal_cut(linkage: np.ndarray, k: int, n: int) -> np.ndarray:
    """Cut the dendrogram at exactly k clusters by performing the first
    (n - k) merges as recorded in `linkage`.  Robust to non-monotone heights
    (DRAC is normalised so d_hat is not monotone)."""
    if k >= n:
        return np.arange(n, dtype=int)
    if k <= 1:
        return np.zeros(n, dtype=int)
    parent = np.arange(2 * n - 1, dtype=int)
    n_merges = min(n - k, linkage.shape[0])
    for i in range(n_merges):
        a = int(linkage[i, 0])
        b = int(linkage[i, 1])
        new_root = n + i
        parent[a] = new_root
        parent[b] = new_root
    # find root for every leaf (iterative, no recursion)
    root = np.empty(n, dtype=int)
    for x in range(n):
        r = x
        while parent[r] != r:
            r = parent[r]
        # path compression
        c = x
        while parent[c] != c:
            nxt = parent[c]
            parent[c] = r
            c = nxt
        root[x] = r
    _, lab = np.unique(root, return_inverse=True)
    return lab.astype(int)


def linkage_to_dendro_dict(linkage: np.ndarray, n: int) -> dict:
    """Return children map (scipy_id -> (left, right)) for module C."""
    children = {}
    for idx, row in enumerate(linkage):
        children[n + idx] = (int(row[0]), int(row[1]))
    return children
