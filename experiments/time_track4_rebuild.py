"""反诈轨道：周期全量重构方案的单次重建成本（用于 4.6(6) 成本—质量对照）。

只做计时，不改变任何结果：对末 P 批（默认 4 批 = 960 条实例）以与
run_track4_qa.py 相同的类型级骨架构造重建本体，重复 3 次取中位数，
并同时报出 HermiT 全本体一致性检查的单次耗时。

Run:
    python experiments/time_track4_rebuild.py --embedder api
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.consistency import consistency_report              # noqa: E402
from src.fraud_ontology import build_skeleton_ontology      # noqa: E402
from src.fraud_stream import read_materialized, to_occurrences  # noqa: E402
from src.fraud_vocab import DP_VOCAB, OP_VOCAB              # noqa: E402

RES = ROOT / "results" / "track4"


def make_embedder(kind: str):
    if kind == "api":
        from src.embedding import Embedder
        from src import config as C
        return Embedder(cache_dir=C.DIR_CACHE_API, allow_fallback=False)
    from src.local_embed import LocalEmbedder
    from src import config as C
    return LocalEmbedder(cache_dir=C.DIR_CACHE_LOCAL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=("local", "api"), default="api")
    ap.add_argument("--periodic-p", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    batches, _truth, _manifest = read_materialized()
    emb = make_embedder(args.embedder)
    streams = [to_occurrences(items, emb) for items in batches]
    P = args.periodic_p
    window = [o for b in streams[-P:] for o in b]

    build_s, check_s, hermit_ms, sizes = [], [], [], []
    for _ in range(args.repeats):
        t0 = time.time()
        onto, _info = build_skeleton_ontology(
            window, min_members=10, op_vocab=OP_VOCAB, dp_vocab=DP_VOCAB,
            granularity="type")
        build_s.append(time.time() - t0)
        t1 = time.time()
        rep = consistency_report(onto, use_cache=False)
        check_s.append(time.time() - t1)
        hermit_ms.append(float(rep.get("ms", 0.0)))
        sizes.append({"concepts": len(onto.concepts),
                      "axioms": sum(len(c.axioms) for c in onto.concepts.values()),
                      "consistent": bool(rep["consistent"])})
    out = {
        "window_batches": P, "window_instances": len(window),
        "repeats": args.repeats,
        "build_seconds_median": round(statistics.median(build_s), 2),
        "build_seconds_all": [round(x, 2) for x in build_s],
        "hermit_seconds_median": round(statistics.median(check_s), 2),
        "hermit_ms_reported_median": round(statistics.median(hermit_ms), 1),
        "sizes": sizes[0],
        "note": "单次重建（末 P 批）的外部计时；不包含嵌入（嵌入命中缓存）；一致性检查强制不走缓存（use_cache=False）",
    }
    (RES / "rebuild_timing.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
