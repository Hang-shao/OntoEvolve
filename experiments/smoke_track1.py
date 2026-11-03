"""轨道 1 冒烟测试：用小规模参数跑通“建池 → 构建 O_0 → 检测 → 评估”。

    python experiments/smoke_track1.py
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
from src.embedding import Embedder
from src.eval_drift import match_events


def main() -> int:
    params = C.PARAMS.copy(
        n_batch=12, pool_per_leaf=12, n_init_per_leaf=8, base_rate=3.0,
        n_emergence=2, n_extinction=2, n_redefinition=2)
    emb = Embedder(allow_fallback=False, verbose=True)
    print(f"[cfg] embed online={emb.online} dim={emb.dim}")

    stream = make_controlled_stream(emb, seed=0, params=params)
    print(f"[stream] pool={len(stream.pool)} batches={len(stream.batches)} "
          f"d0={len(stream.d0_iids)} events={len(stream.events)}")
    for e in stream.events:
        print(f"    GT  {e.dtype:<13} leaf={e.leaf:<22} t_declare={e.t_declare} "
              f"difficulty={e.difficulty}")

    onto, info = build_initial_ontology(
        stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
        run_abc_clustering=True, vocab={**stream.vocab_base, **stream.vocab_emer})
    X = np.vstack([stream.pool[i].vec for i in stream.d0_iids])
    print(f"[O_0] concepts={len(onto.concepts)} K_star={info.get('K_star')} "
          f"k_pred={info.get('k_pred')} timings={info.get('timings')}")
    print(f"      leaf_to_cid 覆盖 {len(info['leaf_to_cid'])}/{len(stream.vocab_base)} "
          f"link_sim_p05={info.get('link_sim_p05', float('nan')):.3f} "
          f"p50={info.get('link_sim_p50', float('nan')):.3f}")

    params = params.copy(
        link_sim=max(0.50, float(info.get("link_sim_p05", params.link_sim)) - 0.02))
    det = DriftDetector(onto, params)
    alarms = []
    for t, items in enumerate(stream.batches, start=1):
        alarms.extend(det.observe(t, items))
    print(f"[detect] alarms={len(alarms)}")
    for a in alarms:
        print(f"    ALARM t={a.t:<3} {a.dtype:<13} cid={a.cid:<6} "
              f"score={a.score:.3f} signals={ {k: round(v,3) for k,v in a.signals.items()} }")

    res = match_events(stream.events, alarms, info["leaf_to_cid"],
                       delay_tol=params.delay_tol)
    print(f"[eval] macroF1={res.macro_f1:.3f} tp={res.tp} fp={res.fp} "
          f"fn={res.fn} delay={res.mean_delay}")
    for t, d in res.per_type.items():
        print(f"    {t:<13} P={d['p']:.2f} R={d['r']:.2f} F1={d['f1']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
