"""【已废弃】合并两轮历史成本实验的结果；论文 Table 6 由 `run_cost_full5.py`
的单会话五种子结果产出，本脚本只用于核对旧归档。

用法：
    python experiments/merge_cost.py                # 合并并写文件
    python experiments/merge_cost.py --verify-only  # 只做自检（用 A 全部 5 seed 重算，比对已存 aggregate）
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "results/exp_cost"
A_TAG = "t36_allexact"
B_TAG = "t36_allexact_b"
C_TAG = "t36_allexact_c_s2"
OUT_TAG = "t36_allexact_merged"
# 干净种子的来源：seed 2 的 t36_allexact_b 值被降速污染（重测 32.8 s/批 vs 160.7 s/批），
# 改用 run_cost_guarded.py 带探针的重测（t36_allexact_c_s2，探针漂移 1.7%）。
SOURCES = {0: A_TAG, 1: A_TAG, 2: C_TAG, 3: B_TAG, 4: B_TAG}

# 与 experiments/run_cost_multiseed.py 的 METRICS 顺序一致
METRICS = [
    "speedup_cumulative", "speedup_per_batch",
    "incremental_cum_s", "full_rebuild_cum_s",
    "incremental_ms_mean", "full_rebuild_ms_mean",
    "incremental_peak_mb", "full_rebuild_peak_mb",
    "rollback_rate", "n_ops", "inc_concepts", "full_concepts",
    "fit_r2", "fit_n_measured",
    "quality_match_rate", "quality_mean_sim",
]


def boot_ci(x, n_boot: int = 5000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        v = float(x[0]) if x.size else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(per_seed):
    agg = {}
    for name in METRICS:
        vals = [r[name] for r in per_seed if r.get(name) is not None]
        if not vals:
            continue
        lo, hi = boot_ci(vals)
        agg[name] = {"mean": round(float(np.mean(vals)), 4),
                     "sd": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                     "ci95": [round(lo, 4), round(hi, 4)],
                     "n": len(vals),
                     "values": vals}
    return agg


def load(tag):
    p = DIR / f"exp_cost_{tag}.json"
    if not p.exists():
        raise SystemExit("缺少 %s" % p)
    return json.loads(p.read_text(encoding="utf-8"))


def verify():
    """用 A 的全部 5 个 seed 重算 aggregate，与文件里存的逐项比对。"""
    a = load(A_TAG)
    mine = aggregate(a["per_seed"])
    bad = 0
    for k, v in a["aggregate"].items():
        m = mine.get(k)
        if m is None:
            print("  [缺失] %s" % k)
            bad += 1
            continue
        for f in ("mean", "sd"):
            if abs(m[f] - v[f]) > 1e-6:
                print("  [不一致] %s.%s 存 %.4f 重算 %.4f" % (k, f, v[f], m[f]))
                bad += 1
        if any(abs(x - y) > 1e-6 for x, y in zip(m["ci95"], v["ci95"])):
            print("  [不一致] %s.ci95 存 %s 重算 %s" % (k, v["ci95"], m["ci95"]))
            bad += 1
    print("自检：%s（%d 处不一致）" % ("通过" if bad == 0 else "失败", bad))
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    ok = verify()
    if args.verify_only or not ok:
        return 0 if ok else 1

    a, b = load(A_TAG), load(B_TAG)
    c = load(C_TAG)
    pool = {}
    for tag, src in ((A_TAG, a), (B_TAG, b), (C_TAG, c)):
        for r in src["per_seed"]:
            if SOURCES.get(r["seed"]) == tag:
                pool[r["seed"]] = r
    seeds = sorted(pool)
    if seeds != [0, 1, 2, 3, 4]:
        raise SystemExit("种子不全：%s" % seeds)
    per_seed = [pool[s] for s in seeds]

    # 污染检查：单 seed 墙钟不应超过 45 分钟
    for r in per_seed:
        if r["wall_s"] > 2700:
            print("  [警告] seed %d 墙钟 %.1f min，可能仍有污染" % (r["seed"], r["wall_s"] / 60))

    res = {
        "tag": OUT_TAG,
        "config": {"seeds": seeds, "T": 36, "check": True, "period": 5,
                   "full_every": 1, "quick": False,
                   "n_d0_per_seed": per_seed[0]["n_d0"],
                   "provenance": {str(k): "exp_cost_%s.json" % v
                                  for k, v in SOURCES.items()},
                   "note": "seed 2 of t36_allexact_b was throttle-contaminated and "
                           "re-measured with a CPU-speed probe (t36_allexact_c_s2)"},
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }

    out = DIR / f"exp_cost_{OUT_TAG}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    with (DIR / f"exp_cost_{OUT_TAG}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)

    agr = res["aggregate"]
    print("\n合并完成：%s" % out)
    print("每 seed 墙钟(min)：", [round(r["wall_s"] / 60, 1) for r in per_seed])
    print("累计加速比：", [r["speedup_cumulative"] for r in per_seed])
    for k in ("incremental_cum_s", "full_rebuild_cum_s", "speedup_cumulative",
              "speedup_per_batch", "incremental_ms_mean", "full_rebuild_ms_mean",
              "incremental_peak_mb", "full_rebuild_peak_mb", "n_ops"):
        v = agr[k]
        print("  %-22s %.4f ± %.4f  [%.4f, %.4f]" %
              (k, v["mean"], v["sd"], v["ci95"][0], v["ci95"][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
