# -*- coding: utf-8 -*-
"""inc_clustering 的联合网格搜索独立归档（口径细节，供 G6 升格决策用）。

背景：`results/baseline_protocol.json` 的十格一律走 legacy 口径（逐键分轮搜索，
另一键固定在默认值）。ESM S8 已如实写明这一点，但"联合搜索最优(θ=0.2, stale=3,
标定 0.612)"此前只在临时输出里出现过，没有独立结果文件。本脚本把它归档。

六种配置，全部在声明口径下（标定期 cooldown=99、匹配容差=3）：
  L_orig_1 / L_orig_2   原网格 legacy，1 / 2 条标定流   —— 应与 G3 / G5 的 inc 行一致
  J_orig_1 / J_orig_2   原网格 joint，1 / 2 条标定流
  J_ext_1  / J_ext_2    扩展网格 joint，1 / 2 条标定流

只记录与核算，不改任何评估逻辑。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_baseline_protocol as R  # noqa: E402  复用流构造与测试口径

from src import config as C  # noqa: E402
from src.build_ontology import build_initial_ontology  # noqa: E402
from src.calibrate import BASELINE_GRID, calibrate_baseline  # noqa: E402
from src.controlled_stream import make_controlled_stream  # noqa: E402

METHOD = "inc_clustering"
CALIB_CD = 99      # 声明口径
CALIB_TOL = 3      # 声明口径

CONFIGS = [
    # 名称, 网格, 标定流数, 搜索模式
    ("L_orig_1", "orig", 1, "legacy"),
    ("L_orig_2", "orig", 2, "legacy"),
    ("J_orig_1", "orig", 1, "joint"),
    ("J_orig_2", "orig", 2, "joint"),
    ("J_ext_1", "ext", 1, "joint"),
    ("J_ext_2", "ext", 2, "joint"),
]


def main() -> int:
    t0 = time.time()
    emb = R.Embedder(allow_fallback=False, verbose=False)
    calib = []
    for sd in R.CALIB_SEEDS:
        st = make_controlled_stream(emb, seed=sd, params=R.BASE)
        onto, info = build_initial_ontology(
            st.pool, st.d0_iids, seed=0, k_gt=len(st.vocab_base),
            vocab={**st.vocab_base, **st.vocab_emer})
        calib.append({"stream": st, "onto": onto,
                      "leaf_to_cid": info["leaf_to_cid"]})
    tests = []
    for r in range(R.N_REPEATS):
        st = make_controlled_stream(emb, seed=R.TEST_SEED0 + r, params=R.BASE)
        onto, info = build_initial_ontology(
            st.pool, st.d0_iids, seed=0, k_gt=len(st.vocab_base),
            vocab={**st.vocab_base, **st.vocab_emer})
        tests.append({"stream": st, "onto": onto,
                      "leaf_to_cid": info["leaf_to_cid"]})
    print(f"[setup] calib={len(calib)} test={len(tests)} "
          f"api_calls={emb.n_api_calls} elapsed={time.time()-t0:.1f}s")

    out = {}
    for name, grid_key, n_streams, mode in CONFIGS:
        grid = (R.GRID_EXT if grid_key == "ext" else BASELINE_GRID)[METHOD]
        used = calib[:n_streams]
        calib_params = R.BASE.copy(delay_tol=CALIB_TOL)
        kw = calibrate_baseline(
            [c["stream"] for c in used], METHOD, base_params=calib_params,
            verbose=False, calib_cooldown=CALIB_CD, grid=grid, mode=mode,
            return_trials=True)
        rres = [R.f1_baseline(t["onto"], t["leaf_to_cid"], t["stream"],
                              METHOD, kw["kw"], R.TEST_PARAMS) for t in tests]
        per = [float(r.macro_f1) for r in rres]
        trials = sorted(kw["trials"], key=lambda t: -t["mean"])
        out[name] = {
            "config": {"grid": grid_key, "n_streams": n_streams, "mode": mode,
                       "calib_cooldown": CALIB_CD, "calib_tolerance": CALIB_TOL,
                       "n_candidates": len(kw["trials"]),
                       "tie_rule": kw["tie_rule"]},
            "kw": kw["kw"], "calib_f1": kw["macro_f1"],
            "test_f1": sum(per) / len(per), "per_repeat": per,
            "per_type": {k: sum(float(r.per_type[k]["f1"]) for r in rres) / len(rres)
                         for k in rres[0].per_type},
            "counts": {"tp": sum(int(r.tp) for r in rres) / len(rres),
                       "fp": sum(int(r.fp) for r in rres) / len(rres),
                       "fn": sum(int(r.fn) for r in rres) / len(rres),
                       "n_alarms": sum(int(r.n_alarms) for r in rres) / len(rres)},
            "top5": [{"kw": t["kw"], "calib": t["mean"]} for t in trials[:5]],
        }
        print(f"[{name}] mode={mode:<6} grid={grid_key:<4} streams={n_streams} "
              f"kw={str(kw['kw']):<26} calib={kw['macro_f1']:.4f} "
              f"cands={len(kw['trials']):2d} test={out[name]['test_f1']:.4f}")

    # ---- 一致性断言：legacy 两格必须与十格归档里的 inc 行对上 ----
    arch = json.loads((C.DIR_RESULTS / "baseline_protocol.json").read_text("utf-8"))
    fails = []
    for name, cell in (("L_orig_1", "G3_documented"), ("L_orig_2", "G5_two_streams_documented")):
        ref = arch[cell]["methods"][METHOD]
        if out[name]["kw"] != ref["kw"] or abs(out[name]["calib_f1"] - ref["calib_f1"]) > 1e-12 \
                or out[name]["per_repeat"] != ref["per_repeat"]:
            fails.append(f"{name} != {cell}（kw/calib/test 至少一项不同）")
    print("[check] legacy 两格与 G3/G5 的 inc 行一致" if not fails else
          "[check] FAIL " + "; ".join(fails))

    # ---- 联合搜索的并列情况（同分候选）----
    for name in ("J_orig_1", "J_orig_2", "J_ext_1", "J_ext_2"):
        top = out[name]["top5"][0]["calib"]
        ties = [t for t in out[name]["top5"] if abs(t["calib"] - top) < 1e-12]
        out[name]["n_tied_at_top"] = len(ties)
        if len(ties) > 1:
            print(f"[tie] {name}: {len(ties)} 个候选同分 {top:.4f} -> "
                  f"{[t['kw'] for t in ties]}，按 first_in_grid_order 取首个")

    Path(C.DIR_RESULTS / "baseline_joint.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["config,grid,streams,mode,candidates,kw,calib_f1,test_f1"]
    for name, v in out.items():
        cf = v["config"]
        lines.append(f"{name},{cf['grid']},{cf['n_streams']},{cf['mode']},"
                     f"{cf['n_candidates']},\"{json.dumps(v['kw'])}\","
                     f"{v['calib_f1']},{v['test_f1']}")
    Path(C.DIR_RESULTS / "baseline_joint.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] elapsed={time.time()-t0:.1f}s -> results/baseline_joint.json")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
