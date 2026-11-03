"""诊断：D_0 规模对模块 A/B/C 的影响，并与 scipy average-linkage 对照。

    python experiments/diag_scale.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score

from src import config as C
from src.build_ontology import run_abc
from src.controlled_stream import make_controlled_stream
from src.embedding import Embedder


def main() -> int:
    emb = Embedder(allow_fallback=False, )
    for n_init, pool in ((10, 40), (15, 40), (20, 40)):
        params = C.PARAMS.copy(n_batch=6, pool_per_leaf=pool,
                               n_init_per_leaf=n_init, n_emergence=1,
                               n_extinction=1, n_redefinition=1)
        stream = make_controlled_stream(emb, seed=0, params=params)
        X = np.vstack([stream.pool[i].vec for i in stream.d0_iids])
        y = np.array([stream.pool[i].leaf for i in stream.d0_iids])
        k_gt = len(set(y))
        labels, meta = run_abc(X, k_gt, seed=0, linkage_mode="average")
        Z = linkage(X, method="average", metric="cosine")
        lab_avg = fcluster(Z, t=k_gt, criterion="maxclust")
        print(f"n={X.shape[0]:>4} A/B/C(avg-link): K*={meta.get('K_star')} "
              f"NMI={nmi(y, labels):.3f} SC={silhouette_score(X, labels):.3f} "
              f"| avg-linkage: NMI={nmi(y, lab_avg):.3f} "
              f"SC={silhouette_score(X, lab_avg):.3f} "
              f"| ref SC@leaf={silhouette_score(X, y):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
