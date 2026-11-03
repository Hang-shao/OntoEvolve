"""experiments/build_track4_stream.py — 阶段 D3：物化轨道 4 的批次与真值。

只做文本侧构造（确定性、无 API、无嵌入），产物：
    data/track4/batches/b01..b10.jsonl
    data/track4/truth.jsonl
    data/track4/stream_manifest.json
    data/track4/audit/d3_batch_report.txt

用法：
    python experiments/build_track4_stream.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import fraud_stream as FS          # noqa: E402
from src import fraud_vocab as FV           # noqa: E402
from src.fraud_vocab import steps_of        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-batch", type=int, default=FS.PER_BATCH)
    ap.add_argument("--n-batches", type=int, default=FS.N_BATCHES)
    args = ap.parse_args()

    pools = FS.load_pools()
    for k, v in pools.items():
        print(f"[pool] {k}: {len(v)}")
    batches, truth, manifest = FS.build_batches(
        pools, seed=args.seed, n_batches=args.n_batches,
        per_batch=args.per_batch)
    FS.write_materialized(batches, truth, manifest)

    rep = []
    rep.append(f"批次构造（seed={args.seed}, per_batch={args.per_batch}, "
               f"T={len(batches)}）")
    rep.append("=" * 72)
    for bi, items in enumerate(batches, start=1):
        cnt = Counter(it.ftype for it in items)
        nsteps = Counter()
        for it in items:
            for s in it.steps:
                nsteps[s] += 1
        ids = {it.iid for it in items}
        rep.append(f"b{bi:02d} n={len(items)} 类型={dict(cnt)} 唯一id={len(ids)}")
        rep.append(f"     步骤分布={dict(nsteps.most_common())}")
        # 空步骤（抽取器无命中）比例：衡量词表覆盖
        empty = sum(1 for it in items if not it.steps)
        rep.append(f"     无步骤命中 {empty}/{len(items)} = {empty/len(items):.3f}")
    rep.append("")
    rep.append("真值事件：")
    for t in truth:
        rep.append(f"  {t['event_id']} {t['dtype']:<13} b{t['batch_declare']:02d} "
                   f"目标={t['target']} synthetic={t['synthetic']} | {t['note']}")
    # 跨批重复检查
    all_ids = [it.iid for b in batches for it in b if not it.iid.startswith("cmp:")]
    dup = len(all_ids) - len(set(all_ids))
    rep.append("")
    rep.append(f"跨批重复实例数（不含复合构造）：{dup}")
    # 各类型在批内的出现序列（用于核对淘汰/涌现的可见性）
    seq = {}
    for bi, items in enumerate(batches, start=1):
        for t in {it.ftype for it in items}:
            seq.setdefault(t, []).append(bi)
    rep.append("类型出现批次：")
    for t, bs in sorted(seq.items()):
        rep.append(f"  {t}: {bs}")

    out = ROOT / "data" / "track4" / "audit" / "d3_batch_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep), encoding="utf-8")
    print(f"-> {out}")
    print(f"-> {ROOT / 'data' / 'track4' / 'stream_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
