# -*- coding: utf-8 -*-
"""基线校准协议因子实验：网格 × 标定流数 × 标定期告警抑制窗口。

八个格子：
  G0 原网格 / 1 流 / cooldown 3   —— V23 复现（必须对上 Table 2）
  G1 原网格 / 1 流 / cooldown 99  —— 只修标定期抑制窗口
  G2 扩展网格 / 1 流 / cooldown 3
  G3 扩展网格 / 1 流 / cooldown 99
  G4 原网格 / 2 流 / cooldown 3
  G5 原网格 / 2 流 / cooldown 99  —— 文档宣称的协议
  G6 扩展网格 / 2 流 / cooldown 3
  G7 扩展网格 / 2 流 / cooldown 99  —— 主表候选

本方法固定双流 + V23 参数，在四格中各自跑一次作为审计探针。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.baselines import BaselineMonitor  # noqa: E402
from src.build_ontology import build_initial_ontology  # noqa: E402
from src.calibrate import BASELINE_GRID, calibrate_baseline  # noqa: E402
from src.controlled_stream import make_controlled_stream  # noqa: E402
from src.embedding import Embedder  # noqa: E402
from src.eval_drift import match_events  # noqa: E402

CALIB_SEEDS = (9001, 9002)
TEST_SEED0 = 42
N_REPEATS = 10
COOLDOWN_TEST = 99

BASE = C.PARAMS

GRID_EXT = {
    "adwin": {"delta": (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)},
    "page_hinkley": {"ph_lambda": (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)},
    "embdist": {"k_std": (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)},
    "embkmeans": {"k_std": (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)},
    "inc_clustering": {"theta": (0.2, 0.25, 0.3, 0.35, 0.45),
                       "stale_window": (2, 3, 5, 8)},
}

# (名称, 网格, 标定流数, 标定期 cooldown, 标定期匹配容差)
# legacy = V23 实际执行的标定口径（cooldown 3、容差 2）；documented = 论文宣称的口径（99、3）
CELLS = [
    ("G0_v23_repro", "orig", 1, 3, 2),
    ("G1_cooldown_only", "orig", 1, 99, 2),
    ("G2_tolerance_only", "orig", 1, 3, 3),
    ("G3_documented", "orig", 1, 99, 3),
    ("G4_ext_documented", "ext", 1, 99, 3),
    ("G5_two_streams_documented", "orig", 2, 99, 3),
    ("G6_main_candidate", "ext", 2, 99, 3),
    ("G7_ext_legacy", "ext", 1, 3, 2),
    ("G8_ext_two_streams_legacy", "ext", 2, 3, 2),
    ("G9_two_streams_legacy", "orig", 2, 3, 2),
]

BASELINES = ("adwin", "page_hinkley", "embdist", "embkmeans", "inc_clustering")


def f1_baseline(onto, l2c, stream, method, kw, params=None):
    mon = BaselineMonitor(onto, method, params=params or TEST_PARAMS,
                          cooldown=COOLDOWN_TEST, **kw)
    alarms = []
    for t, items in enumerate(stream.batches, start=1):
        alarms.extend(mon.observe(t, items))
    res = match_events(stream.events, alarms, l2c,
                       delay_tol=(params or TEST_PARAMS).delay_tol, typed=False)
    return res


def f1_method(onto, l2c, stream, p05):
    """本方法用 V23 标定参数评估（与 run_exp1 的 evaluate 同口径）。"""
    from src.calibrate import evaluate
    res, _ = evaluate(stream, onto, l2c, BEST_PARAMS,
                      link_offset=BEST_OFFSET, base_p05=p05,
                      cooldown=COOLDOWN_TEST)
    return float(res.macro_f1)


def load_best():
    import json as _json
    d = _json.loads((C.DIR_RESULTS / "exp1_summary.json").read_text("utf-8"))
    cal = d["calibration"]
    fields = set(C.DriftParams.__annotations__)
    params = BASE.copy(**{k: v for k, v in cal["params"].items() if k in fields})
    return params, float(cal["link_offset"])


BEST_PARAMS, BEST_OFFSET = load_best()
TEST_PARAMS = BEST_PARAMS  # 测试阶段用标定后的测量参数（V23 行为）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.DIR_RESULTS / "baseline_protocol.json"))
    ap.add_argument("--csv", default=str(C.DIR_RESULTS / "baseline_protocol.csv"))
    ap.add_argument("--cells", default="", help="逗号分隔，默认全部")
    args = ap.parse_args()

    t0 = time.time()
    emb = Embedder(allow_fallback=False, verbose=False)
    want = {c.strip() for c in args.cells.split(",") if c.strip()}

    calib = []
    for sd in CALIB_SEEDS:
        st = make_controlled_stream(emb, seed=sd, params=BASE)
        onto, info = build_initial_ontology(
            st.pool, st.d0_iids, seed=0, k_gt=len(st.vocab_base),
            vocab={**st.vocab_base, **st.vocab_emer})
        calib.append({"stream": st, "onto": onto,
                      "leaf_to_cid": info["leaf_to_cid"],
                      "p05": float(info["link_sim_p05"])})
    print(f"[setup] calib streams={len(calib)} api_calls={emb.n_api_calls}")

    tests = []
    for r in range(N_REPEATS):
        st = make_controlled_stream(emb, seed=TEST_SEED0 + r, params=BASE)
        onto, info = build_initial_ontology(
            st.pool, st.d0_iids, seed=0, k_gt=len(st.vocab_base),
            vocab={**st.vocab_base, **st.vocab_emer})
        tests.append({"stream": st, "onto": onto,
                      "leaf_to_cid": info["leaf_to_cid"],
                      "p05": float(info["link_sim_p05"])})
    print(f"[setup] test streams={len(tests)} elapsed={time.time()-t0:.1f}s")

    results = {}
    for name, grid_key, n_streams, calib_cd, calib_tol in CELLS:
        if want and name not in want:
            continue
        grid_map = GRID_EXT if grid_key == "ext" else None
        used = calib[:n_streams]
        cell = {}
        for method in BASELINES:
            t1 = time.time()
            calib_params = BASE.copy(delay_tol=calib_tol)
            kw = calibrate_baseline(
                [c["stream"] for c in used], method, base_params=calib_params,
                verbose=False, calib_cooldown=None if calib_cd == 3 else calib_cd,
                grid=(grid_map or BASELINE_GRID)[method], mode="legacy",
                return_trials=True)
            rres = [f1_baseline(t["onto"], t["leaf_to_cid"], t["stream"],
                                method, kw["kw"], TEST_PARAMS) for t in tests]
            per = [float(r.macro_f1) for r in rres]
            mean = sum(per) / len(per)
            per_type = {k: [float(r.per_type[k]["f1"]) for r in rres]
                        for k in rres[0].per_type}
            counts = {"tp": [int(r.tp) for r in rres],
                      "fp": [int(r.fp) for r in rres],
                      "fn": [int(r.fn) for r in rres],
                      "n_alarms": [int(r.n_alarms) for r in rres]}
            cell[method] = {"kw": kw["kw"], "calib_f1": kw["macro_f1"],
                            "n_evals": kw["n_evals"], "per_repeat": per,
                            "mean": mean,
                            "seconds": round(time.time() - t1, 1)}
            print(f"[{name}] {method:<14} kw={kw['kw']} calib={kw['macro_f1']:.4f} "
                  f"test={mean:.4f}")
        # 本方法：固定 V23 参数，四格各跑一次作审计探针
        per_m = [f1_method(t["onto"], t["leaf_to_cid"], t["stream"], t["p05"])
                 for t in tests]
        cell["ontoevolve"] = {"kw": {}, "calib_f1": None,
                              "per_repeat": per_m,
                              "mean": sum(per_m) / len(per_m)}
        print(f"[{name}] ontoevolve     test={cell['ontoevolve']['mean']:.4f}")
        results[name] = {"config": {"grid": grid_key, "n_streams": n_streams,
                                    "calib_cooldown": calib_cd,
                                    "calib_tolerance": calib_tol},
                         "methods": cell}

    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    lines = ["cell,grid,streams,calib_cooldown,calib_tol,method,kw,calib_f1,test_f1"]
    for name, c in results.items():
        for m, v in c["methods"].items():
            lines.append(f"{name},{c['config']['grid']},{c['config']['n_streams']},"
                         f"{c['config']['calib_cooldown']},{c['config']['calib_tolerance']},{m},"
                         f"\"{json.dumps(v['kw'])}\",{v['calib_f1']},{v['mean']}")
    Path(args.csv).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {len(results)} cells, elapsed={time.time()-t0:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
