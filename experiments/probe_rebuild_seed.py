"""诊断：seed 2 的全量重构为什么比 seed 3/4 慢 5 倍——是数据效应还是机器降速。

做法：对指定种子重建同一条受控流，分别在 t=18 与 t=36 计时一次 build_from_occurrences，
并在计时前后各做一次定长 CPU 基准。若此刻（机器确认满速）seed 2 的 t=36 单次重建仍是
150 s 量级，则是数据/算法效应，可以照实报告；若回到 30 s 量级，则 t36_allexact_b 里
seed 2 的数值是被降速污染的，必须重测。

用法：python experiments/probe_rebuild_seed.py --seeds 2,4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src import config as C                        # noqa: E402
from src.build_ontology import build_from_occurrences, items_to_occurrences  # noqa: E402
from src.controlled_stream import make_controlled_stream                     # noqa: E402
from src.embedding import Embedder                                          # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def probe(seconds=5.0):
    a = np.random.default_rng(0).random((256, 256))
    b = np.random.default_rng(1).random((256, 256))
    times, t_end = [], time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        for _ in range(40):
            (a @ b).sum()
        times.append(time.perf_counter() - t0)
    return {"n": len(times), "med_ms": round(float(np.median(times) * 1e3), 2),
            "min_ms": round(float(min(times) * 1e3), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2,4")
    ap.add_argument("--marks", default="18,36")
    args = ap.parse_args()
    marks = [int(x) for x in args.marks.split(",")]

    emb = Embedder(allow_fallback=False)
    out = {"probe_before": probe(), "seeds": {}}
    print("probe before:", out["probe_before"], flush=True)

    for s in [int(x) for x in args.seeds.split(",")]:
        stream = make_controlled_stream(emb, seed=s, params=C.PARAMS)
        vocab = {**stream.vocab_base, **stream.vocab_emer}
        k_gt = len(stream.vocab_base)
        d0_occ = items_to_occurrences(stream.pool, stream.d0_iids)

        t0 = time.perf_counter()
        onto0, info0 = build_from_occurrences(d0_occ, seed=0, k_gt=k_gt, vocab=vocab)
        setup_ms = (time.perf_counter() - t0) * 1e3
        op_vocab, dp_vocab = list(onto0.op_vocab), list(onto0.dp_vocab)
        rec = {"setup_ms": round(setup_ms, 1), "k_gt": k_gt, "marks": {}}

        occ = list(d0_occ)
        for t, batch in enumerate(stream.batches, start=1):
            occ.extend(batch)
            if t in marks:
                s0 = time.perf_counter()
                onto, info = build_from_occurrences(occ, seed=0, k_gt=k_gt, vocab=vocab,
                                                    op_vocab=op_vocab, dp_vocab=dp_vocab)
                ms = (time.perf_counter() - s0) * 1e3
                rec["marks"][t] = {"ms": round(ms, 1), "n_total": len(occ),
                                   "k_used": info.get("k_used") or info.get("k"),
                                   "n_concepts": len([c for c in onto.concepts.values()
                                                      if c.active])}
                print("seed %d  t=%d  n=%d  %.1f s  %s" %
                      (s, t, len(occ), ms / 1e3, rec["marks"][t]), flush=True)
        out["seeds"][s] = rec

    out["probe_after"] = probe()
    print("probe after:", out["probe_after"], flush=True)
    p = ROOT / "results/exp_cost/rebuild_probe.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入", p)


if __name__ == "__main__":
    main()
