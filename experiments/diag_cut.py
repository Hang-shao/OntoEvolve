"""诊断：不同 D_0 规模下模块 C（动态剪切 vs 水平剪切）的表现。

    python experiments/diag_cut.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score

from src import config as C
from src.build_ontology import load_legacy, s_star_dendrogram_gap
from src.controlled_stream import make_controlled_stream
from src.embedding import Embedder


def main() -> int:
    emb = Embedder(allow_fallback=False, )
    legacy = load_legacy()
    for n_init, pool in ((10, 20), (20, 40), (30, 60)):
        params = C.PARAMS.copy(n_batch=6, pool_per_leaf=pool,
                               n_init_per_leaf=n_init, n_emergence=1,
                               n_extinction=1, n_redefinition=1)
        stream = make_controlled_stream(emb, seed=0, params=params)
        X = np.vstack([stream.pool[i].vec for i in stream.d0_iids])
        y = np.array([stream.pool[i].leaf for i in stream.d0_iids])
        k_gt = len(set(y))
        for dyn in (True, False):
            legacy.methods._target_cluster_size = s_star_dendrogram_gap
            res = legacy.methods.run_with_options(
                X, k_gt, eps_mode="local", use_auto_k=True,
                use_dynamic_cut=dyn, seed=0)
            labels = np.asarray(res.labels)
            sizes = np.bincount(labels)
            print(f"n={X.shape[0]:>4} dynamic={str(dyn):<5} K*={res.notes.get('K_star')} "
                  f"k={labels.max()+1:>3} NMI={nmi(y, labels):.3f} "
                  f"SC={silhouette_score(X, labels):.3f} "
                  f"sizes[min,med,max]=[{sizes.min()},{int(np.median(sizes))},{sizes.max()}] "
                  f"n_singleton={int((sizes == 1).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
