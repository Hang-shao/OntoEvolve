"""五种子、单会话、全批实测的成本实验（含周期重构 P = 5）——论文 Table 6 / Fig. 4。

为什么是这个口径：五个种子放在**同一次会话**里跑完，每条路径都在同一
进程、同一机器状态下计时，36 次全量重建与全部周期重构都实测，不做外推，也不跨会话
拼接；每个种子前后各跑一次定长 CPU 基准，用于判定这段时间机器是否被降速。

配置与正文一致：T = 36、默认 DriftParams、check=True、full_every=1、period=5。

平台说明：Windows 上额外压住系统休眠并关闭进程级执行速度限制（Power Throttling），
这是必要的——只压休眠挡不住 CPU 降速，会让长计时段整体变慢。其他平台只做 CPU 探针，
计时本身照常进行。

用法：
    python experiments/run_cost_full5.py --seeds 0,1,2,3,4 --tag t36_full5
    python experiments/run_cost_full5.py --smoke            # 极小配置冒烟（约 1—2 分钟）

产物：`results/exp_cost/exp_cost_<tag>.json`（per_seed / aggregate / probes）、同名 `.csv`，
以及每完成一个种子就落盘的 `cost_full5_partial_<tag>.json`。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from src import config as C                                    # noqa: E402
from src.embedding import Embedder                             # noqa: E402
from run_exp_cost import run_cost                              # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:  # pragma: no cover - 平台相关
    import ctypes
    from ctypes import wintypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    ProcessPowerThrottling = 4
    HIGH_PRIORITY_CLASS = 0x00000080

    kernel32 = ctypes.windll.kernel32
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]

    class ThrottleState(ctypes.Structure):
        _fields_ = [("Version", wintypes.DWORD), ("ControlMask", wintypes.DWORD),
                    ("StateMask", wintypes.DWORD)]


def boot_ci(x, n_boot: int = 5000, seed: int = 0):
    """bootstrap 95% 置信区间，与 `run_cost_multiseed.py` 同口径。"""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        v = float(x[0]) if x.size else float("nan")
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def keep_awake() -> None:
    """压住系统休眠（仅 Windows；其他平台计时前置条件由使用者保证）。"""
    if not IS_WINDOWS:
        print("[guard] non-Windows platform: sleep guard skipped", flush=True)
        return
    prev = kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED |
                                            ES_DISPLAY_REQUIRED)
    print("[guard] SetThreadExecutionState ->", prev, flush=True)


def no_throttle() -> None:
    """关闭进程级执行速度限制并提高优先级（仅 Windows）。"""
    if not IS_WINDOWS:
        print("[ctl] non-Windows platform: power-throttling control skipped", flush=True)
        return
    st = ThrottleState(PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                       PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
    ok = kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), ProcessPowerThrottling,
                                        ctypes.byref(st), ctypes.sizeof(st))
    prio = kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)
    print("[ctl] throttle-off:", bool(ok), " high-priority:", bool(prio), flush=True)


def probe(seconds: float = 5.0) -> dict:
    """定长浮点负载：返回迭代次数与单次耗时，用于判断机器是否被降速。"""
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


METRICS = ["speedup_cumulative", "speedup_per_batch", "incremental_cum_s",
           "full_rebuild_cum_s", "incremental_ms_mean", "full_rebuild_ms_mean",
           "incremental_peak_mb", "full_rebuild_peak_mb", "rollback_rate", "n_ops",
           "inc_concepts", "full_concepts", "quality_match_rate", "quality_mean_sim",
           "periodic_cum_s", "periodic_speedup"]


def main() -> int:
    ap = argparse.ArgumentParser(description="五种子单会话的全批实测成本实验")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="t36_full5")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="极小配置冒烟（约 1—2 分钟）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    params = C.PARAMS
    if args.smoke:
        params = C.PARAMS.copy(n_batch=3, n_init_per_leaf=6, pool_per_leaf=12,
                               base_rate=3.0, n_emergence=1, n_extinction=1,
                               n_redefinition=1)
        seeds = seeds[:1]
        args.tag += "_smoke"

    keep_awake()
    no_throttle()
    emb = Embedder(allow_fallback=False)
    outdir = C.DIR_RESULTS / "exp_cost"
    outdir.mkdir(parents=True, exist_ok=True)

    per_seed, probes = [], []
    for s in seeds:
        pre = probe()
        print("[probe] before seed %d: %s" % (s, pre), flush=True)
        t0 = time.perf_counter()
        r = run_cost(s, params, emb, check=True, period=args.period,
                     full_every=1, outdir=None)
        wall = time.perf_counter() - t0
        post = probe()
        drift = abs(post["med_ms"] - pre["med_ms"]) / pre["med_ms"]
        probes.append({"seed": s, "pre": pre, "post": post,
                       "drift": round(drift, 4)})
        print("[done] seed %d wall %.1f min  drift %.1f%%  inc %.2fs  full %.2fs  "
              "periodic %.2fs  speedup %.2fx  periodic-ratio %.2fx"
              % (s, wall / 60, drift * 100, r["incremental"]["cum_ms"] / 1e3,
                 r["full_rebuild"]["cum_ms"] / 1e3, r["periodic"]["cum_ms"] / 1e3,
                 r["speedup_cumulative"],
                 r["periodic"]["cum_ms"] / r["incremental"]["cum_ms"]), flush=True)
        per_seed.append({
            "seed": s, "wall_s": round(wall, 1), "T": r["T"], "n_d0": r["n_d0"],
            "speedup_cumulative": r["speedup_cumulative"],
            "speedup_per_batch": r["speedup_per_batch"],
            "incremental_cum_s": round(r["incremental"]["cum_ms"] / 1e3, 4),
            "full_rebuild_cum_s": round(r["full_rebuild"]["cum_ms"] / 1e3, 4),
            "incremental_ms_mean": r["incremental"]["ms_mean"],
            "full_rebuild_ms_mean": r["full_rebuild"]["ms_mean"],
            "incremental_peak_mb": r["incremental"]["peak_mem_mb"],
            "full_rebuild_peak_mb": r["full_rebuild"]["peak_mem_mb"],
            "rollback_rate": r["incremental"]["rollback_rate"],
            "n_ops": r["incremental"]["n_ops"],
            "inc_concepts": r["incremental"]["n_concepts_final"],
            "full_concepts": r["full_rebuild"]["n_concepts_final"],
            "quality_match_rate": (r.get("quality_match") or {}).get("match_rate"),
            "quality_mean_sim": (r.get("quality_match") or {}).get("mean_sim"),
            "periodic_cum_s": round(r["periodic"]["cum_ms"] / 1e3, 4),
            "periodic_speedup": round(r["periodic"]["cum_ms"] /
                                      r["incremental"]["cum_ms"], 2),
        })
        # 每完成一个种子就落盘：长任务可能被外部中断（曾因后台重启丢掉 2.8 h 进度）。
        (outdir / f"cost_full5_partial_{args.tag}.json").write_text(
            json.dumps({"per_seed": per_seed, "probes": probes},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    agg = {}
    for name in METRICS:
        vals = [r[name] for r in per_seed if r.get(name) is not None]
        if not vals:
            continue
        lo, hi = boot_ci(vals)
        agg[name] = {"mean": round(float(np.mean(vals)), 4),
                     "sd": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                     "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals),
                     "values": vals}

    res = {"tag": args.tag,
           "config": {"seeds": seeds, "T": params.n_batch, "check": True,
                      "period": args.period, "full_every": 1, "quick": bool(args.smoke),
                      "n_d0_per_seed": per_seed[0]["n_d0"],
                      "note": "single-session run; sleep guard + power-throttling off "
                              "(Windows); CPU probe before and after every seed"},
           "per_seed": per_seed, "aggregate": agg, "probes": probes}

    (outdir / f"exp_cost_{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    with (outdir / f"exp_cost_{args.tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys()))
        w.writeheader()
        w.writerows(per_seed)

    print("\n================ 汇总（%d 个 seed，同一次会话） ================" % len(seeds))
    for k in METRICS:
        v = agg.get(k)
        if v:
            print("  %-22s %.4f ± %.4f  [%.4f, %.4f]"
                  % (k, v["mean"], v["sd"], v["ci95"][0], v["ci95"][1]))
    print("  probe drifts:", [p["drift"] for p in probes])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
