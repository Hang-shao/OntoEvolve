"""诊断：模块 A/B/C 在 D_0 上的粒度与聚类质量（含 PCA 预处理对比）。

    python experiments/diag_abc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import silhouette_score

from src import config as C
from src.build_ontology import load_legacy
from src.controlled_stream import make_controlled_stream
from src.embedding import Embedder, l2_normalize


def run(X, k_gt, tag, *, use_auto_k=True, seed=0):
    legacy = load_legacy()
    res = legacy.methods.run_with_options(
        X, k_gt, eps_mode="local", use_auto_k=use_auto_k,
        use_dynamic_cut=True, seed=seed)
    labels = np.asarray(res.labels)
    info_b = res.notes.get("info_b", {})
    return {
        "tag": tag, "k_pred": int(labels.max() + 1),
        "K_star": res.notes.get("K_star"),
        "K_min": info_b.get("K_min"), "s_star": info_b.get("s_star"),
    }


def main() -> int:
    params = C.PARAMS.copy(n_batch=8, pool_per_leaf=10, n_init_per_leaf=20,
                           n_emergence=1, n_extinction=1, n_redefinition=1)
    emb = Embedder(allow_fallback=False, verbose=True)
    stream = make_controlled_stream(emb, seed=0, params=params)
    X = np.vstack([stream.pool[i].vec for i in stream.d0_iids])
    y_leaf = np.array([stream.pool[i].leaf for i in stream.d0_iids])
    k_gt = len(set(y_leaf))
    print(f"n={X.shape[0]} leaves={k_gt}")
    print(f"  ref: silhouette@leaf={silhouette_score(X, y_leaf):.3f}")

    for dim in (0, 32, 64, 128):
        Z = X if dim == 0 else l2_normalize(
            PCA(n_components=min(dim, X.shape[1]), random_state=0).fit_transform(X))
        for auto in (True, False):
            r = run(Z, k_gt, f"pca={dim or 'raw'} autoK={auto}", use_auto_k=auto)
            lab = run(Z, k_gt, "", use_auto_k=auto)  # noop to keep typing simple
            legacy = load_legacy()
            res = legacy.methods.run_with_options(
                Z, k_gt, eps_mode="local", use_auto_k=auto,
                use_dynamic_cut=True, seed=0)
            labels = np.asarray(res.labels)
            sc = silhouette_score(Z, labels)
            print(f"  pca={str(dim or 'raw'):>4} autoK={str(auto):<5} "
                  f"k_pred={labels.max()+1:>3} K*={res.notes.get('K_star')} "
                  f"SC={sc:.3f} NMI(leaf)={nmi(y_leaf, labels):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
