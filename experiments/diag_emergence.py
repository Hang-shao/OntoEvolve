"""诊断：涌现检测链路（未链接实例数量、候选簇规模、与已有锚点的距离）。

    python experiments/diag_emergence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config as C
from src.build_ontology import build_initial_ontology
from src.controlled_stream import make_controlled_stream
from src.drift_detect import DriftDetector
from src.embedding import Embedder, l2_normalize


def main() -> int:
    params = C.PARAMS.copy(n_batch=14, pool_per_leaf=16, n_init_per_leaf=16,
                           base_rate=4.0)
    emb = Embedder(allow_fallback=False, verbose=True)
    stream = make_controlled_stream(emb, seed=0, params=params)
    onto, info = build_initial_ontology(
        stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
        vocab={**stream.vocab_base, **stream.vocab_emer})
    print(f"concepts={len(onto.concepts)} emergence_leaves={stream.emergence_leaves}")
    leaf_to_cid = info["leaf_to_cid"]
    det = DriftDetector(onto, params)
    cen = det.centroids

    for t, items in enumerate(stream.batches, start=1):
        assigned, unassigned = [], []
        for o in items:
            v = np.asarray(o.vec, dtype=np.float32)
            best, bs = None, -2.0
            for cid, c in cen.items():
                s = float(np.dot(v, c))
                if s > bs:
                    bs, best = s, cid
            (assigned if bs >= det.link_sim else unassigned).append((o, best, bs))
        emer = [o for o in items if o.leaf in stream.emergence_leaves]
        if emer:
            sims = []
            for o in emer:
                v = np.asarray(o.vec, dtype=np.float32)
                sims.append(max(float(np.dot(v, c)) for c in cen.values()))
            print(f"t={t:>2} emer={len(emer):>2} unassigned={len(unassigned):>2} "
                  f"max_sim_of_emer={[round(s,3) for s in sims]}")
        det.observe(t, items)
    print("candidates:")
    for c in det.candidates:
        cen_c = l2_normalize(np.mean(np.vstack([o.vec for o in c["items"]]), axis=0))
        dmin = min(1.0 - float(np.dot(cen_c, cc)) for cc in cen.values())
        print(f"   {c['id']:<6} size={len(c['items']):>3} batches={sorted(c['batches'])} "
              f"fired={c['fired']} dmin={dmin:.3f} theta_sem={params.theta_sem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
