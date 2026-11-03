# -*- coding: utf-8 -*-
"""命题 B 成本实验的多 seed 版本：串行运行 run_cost 并汇总统计。

为什么需要它：单 seed 的成本实验没有置信区间，而本文其他量化命题
（表 3/5/6/9）都给出了 10 次重复或 5 个 seed。本脚本对同一配置串行跑
多个 seed（必须串行：计时刻度对 CPU 竞争极其敏感），汇总每个指标
的均值、标准差与 bootstrap 95% 置信区间。

用法：
    python experiments/run_cost_multiseed.py --tag t36_multiseed
    python experiments/run_cost_multiseed.py --quick --seeds 0,1        # 冒烟
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                       # noqa: E402
from src.embedding import Embedder                # noqa: E402
from run_exp_cost import run_cost                 # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def boot_ci(x, n_boot: int = 5000, seed: int = 0):
    """bootstrap 95% 置信区间（与表 3 的口径一致）。"""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        v = float(x[0]) if x.size else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# 需要汇总的标量指标：(名称, 取值函数)
METRICS = [
    ("speedup_cumulative", lambda r: r["speedup_cumulative"]),
    ("speedup_per_batch", lambda r: r["speedup_per_batch"]),
    ("incremental_cum_s", lambda r: r["incremental"]["cum_ms"] / 1000.0),
    ("full_rebuild_cum_s", lambda r: r["full_rebuild"]["cum_ms"] / 1000.0),
    ("incremental_ms_mean", lambda r: r["incremental"]["ms_mean"]),
    ("full_rebuild_ms_mean", lambda r: r["full_rebuild"]["ms_mean"]),
    ("incremental_peak_mb", lambda r: r["incremental"]["peak_mem_mb"]),
    ("full_rebuild_peak_mb", lambda r: r["full_rebuild"]["peak_mem_mb"]),
    ("rollback_rate", lambda r: r["incremental"]["rollback_rate"]),
    ("n_ops", lambda r: r["incremental"]["n_ops"]),
    ("inc_concepts", lambda r: r["incremental"]["n_concepts_final"]),
    ("full_concepts", lambda r: r["full_rebuild"]["n_concepts_final"]),
    ("fit_r2", lambda r: (r.get("fit_full_rebuild") or {}).get("r2")),
    ("fit_n_measured", lambda r: (r.get("fit_full_rebuild") or {}).get("n_measured")),
    ("quality_match_rate", lambda r: (r.get("quality_match") or {}).get("match_rate")),
    ("quality_mean_sim", lambda r: (r.get("quality_match") or {}).get("mean_sim")),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="multiseed")
    ap.add_argument("--full-every", type=int, default=3)
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base = C.PARAMS
    if args.quick:
        base = base.copy(n_batch=12, n_init_per_leaf=6, pool_per_leaf=12,
                         base_rate=3.0, n_emergence=1, n_extinction=1,
                         n_redefinition=1)

    emb = Embedder(allow_fallback=False)
    outdir = C.DIR_ONTOLOGY / "exp_cost"

    per_seed = []
    for s in seeds:
        print(f"\n########## seed {s} 开始 ##########", flush=True)
        t0 = time.perf_counter()
        r = run_cost(s, base, emb, check=not args.no_check, period=args.period,
                     full_every=args.full_every, outdir=outdir)
        wall = time.perf_counter() - t0
        print(f"########## seed {s} 完成：墙钟 {wall/60:.1f} min，"
              f"累计加速 {r['speedup_cumulative']}×，单批 {r['speedup_per_batch']}×，"
              f"回退率 {r['incremental']['rollback_rate']} ##########", flush=True)
        rec = {"seed": s, "wall_s": round(wall, 1), "T": r["T"],
               "n_d0": r["n_d0"]}
        for name, fn in METRICS:
            rec[name] = fn(r)
        per_seed.append(rec)

    agg = {}
    for name, _ in METRICS:
        vals = [r[name] for r in per_seed if r[name] is not None]
        if not vals:
            continue
        lo, hi = boot_ci(vals)
        agg[name] = {"mean": round(float(np.mean(vals)), 4),
                     "sd": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                     "ci95": [round(lo, 4), round(hi, 4)],
                     "n": len(vals),
                     "values": vals}

    res = {
        "tag": args.tag,
        "config": {"seeds": seeds, "T": C.PARAMS.n_batch if not args.quick else 12,
                   "check": not args.no_check, "period": args.period,
                   "full_every": args.full_every, "quick": args.quick,
                   "n_d0_per_seed": per_seed[0]["n_d0"]},
        "per_seed": per_seed,
        "aggregate": agg,
    }

    out = C.DIR_RESULTS / "exp_cost"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"exp_cost_{args.tag}"
    (out / f"{stem}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)

    print("\n================ 汇总（%d 个 seed） ================" % len(seeds))
    print("%-24s %12s %12s %-24s" % ("指标", "均值", "标准差", "95% CI (bootstrap)"))
    for name, _ in METRICS:
        a = agg.get(name)
        if not a:
            continue
        print("%-24s %12.4f %12.4f [%.4f, %.4f]"
              % (name, a["mean"], a["sd"], a["ci95"][0], a["ci95"][1]))
    print("\n结果写入 %s" % (out / (stem + ".json")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
