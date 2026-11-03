"""诊断：全尺寸（T=36）配置下 O_0 质量与三类漂移的可检测性。

    python experiments/diag_full.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi

from src import config as C
from src.build_ontology import build_initial_ontology
from src.controlled_stream import make_controlled_stream
from src.drift_detect import DriftDetector
from src.embedding import Embedder
from src.eval_drift import match_events


def main() -> int:
    params = C.PARAMS  # 全尺寸默认配置
    emb = Embedder(allow_fallback=False, verbose=True)
    stream = make_controlled_stream(emb, seed=42, params=params)
    onto, info = build_initial_ontology(
        stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
        vocab={**stream.vocab_base, **stream.vocab_emer})
    y_true = [stream.pool[i].leaf for i in stream.d0_iids]
    labels = []
    cid2 = {}
    for cid, c in onto.concepts.items():
        for pos in c.meta["members"]:
            labels.append(cid)
    print(f"D_0={len(y_true)} concepts={len(onto.concepts)} "
          f"NMI={nmi(y_true, labels):.3f} K*={info.get('K_star')} "
          f"link_p05={info.get('link_sim_p05'):.3f}")
    sizes = {}
    for cid, c in onto.concepts.items():
        sizes[cid] = len(c.meta["members"])
    print("cluster sizes:", sorted(sizes.values()))

    l2c = info["leaf_to_cid"]
    print("\nGT events:")
    for e in stream.events:
        print(f"   {e.dtype:<13} t={e.t_declare:<3} leaf={e.leaf:<20} "
              f"cid={l2c.get(e.leaf, 'NEW')} diff={e.difficulty}")

    p = params.copy(link_sim=max(0.4, info["link_sim_p05"] - 0.02))
    det = DriftDetector(onto, p)
    alarms = []
    for t, items in enumerate(stream.batches, start=1):
        alarms.extend(det.observe(t, items))
    print(f"\nalarms={len(alarms)}")
    for a in alarms[:40]:
        print(f"   t={a.t:<3} {a.dtype:<13} {a.cid:<6} "
              f"ax={a.signals['ax']:.2f} sem={a.signals['sem']:.3f} "
              f"stat={a.signals['stat']:.2f} conf={a.signals['mean_conf']:.2f}")
    res = match_events(stream.events, alarms, l2c, delay_tol=p.delay_tol)
    print(f"\nmacroF1={res.macro_f1:.3f} tp={res.tp} fp={res.fp} fn={res.fn}")
    for t, d in res.per_type.items():
        print(f"   {t:<13} P={d['p']:.2f} R={d['r']:.2f} F1={d['f1']:.2f}")
    # 诊断：每个概念的每批计数
    print("\ncounts per concept (前 12 个概念):")
    for cid in list(det.counts)[:12]:
        print(f"   {cid}: {det.counts[cid]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
