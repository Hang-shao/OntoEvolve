"""实验 1（轨道 1）：概念漂移检测 macro-F1 对比 + 阈值标定。

流程：
    1) 在独立的标定流上网格搜索模块 D 阈值与三个基线的超参数；
    2) 在 n_repeats 条独立测试流上评估（所有方法共享由 A/B/C 构建的 O_0）；
    3) 汇总 macro-F1 均值、bootstrap 95% CI、与每个基线的配对 Wilcoxon 检验。

Run:
    python experiments/run_exp1.py [--repeats 10] [--trials 240] [--quick]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config as C
from src.baselines import BaselineMonitor
from src.build_ontology import build_initial_ontology
from src.calibrate import (calibrate_baseline, calibrate_ontoevolve, evaluate)
from src.controlled_stream import make_controlled_stream
from src.embedding import Embedder
from src.eval_drift import TYPES, bootstrap_ci, match_events, wilcoxon_paired

CALIB_SEED = 9001
BASELINES = ("adwin", "page_hinkley", "embdist", "embkmeans",
             "inc_clustering")


def build_o0(stream, seed: int = 0):
    return build_initial_ontology(
        stream.pool, stream.d0_iids, seed=seed, k_gt=len(stream.vocab_base),
        vocab={**stream.vocab_base, **stream.vocab_emer})


def run_baseline(onto, leaf_to_cid, stream, method: str, kw: Dict,
                 params: C.DriftParams, cooldown: int = 99):
    kw = {**kw, "cooldown": int(cooldown)}
    mon = BaselineMonitor(onto, method, params=params, **kw)
    alarms = []
    for t, items in enumerate(stream.batches, start=1):
        alarms.extend(mon.observe(t, items))
    res = match_events(stream.events, alarms, leaf_to_cid,
                       delay_tol=params.delay_tol, typed=False)
    return res, alarms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=C.PARAMS.n_repeats)
    ap.add_argument("--trials", type=int, default=240)
    ap.add_argument("--seed", type=int, default=C.PARAMS.seed)
    ap.add_argument("--calib-streams", type=int, default=2,
                    help="用于阈值标定的独立数据流条数（取均值，降低过拟合）")
    ap.add_argument("--quick", action="store_true",
                    help="小规模快速验证（少批次、少试验）")
    ap.add_argument("--out", default=str(C.DIR_RESULTS / "exp1_main.csv"))
    ap.add_argument("--summary", default=str(C.DIR_RESULTS / "exp1_summary.json"))
    ap.add_argument("--cooldown", type=int, default=99,
                    help="告警抑制窗口（批）；同时施加于本方法与全部基线，"
                         "标定与测试共用同一取值。默认 99 即论文现口径")
    ap.add_argument("--dump-alarms", default="",
                    help="把逐条告警（repeat/method/t/cid/type）写为 JSONL，"
                         "用于离线重算匹配窗口敏感性")
    args = ap.parse_args()

    base = C.PARAMS
    if args.quick:
        base = base.copy(n_batch=18, pool_per_leaf=20, n_init_per_leaf=12,
                         n_repeats=3)
        args.trials = min(args.trials, 60)
        args.repeats = 3

    t_start = time.time()
    emb = Embedder(allow_fallback=False, verbose=True)
    print(f"[cfg] embed_online={emb.online} dim={emb.dim} "
          f"batches={base.n_batch} repeats={args.repeats}")

    # ---------- 1) 标定 ------------------------------------------------------
    calib_streams = [make_controlled_stream(emb, seed=CALIB_SEED + i, params=base)
                     for i in range(args.calib_streams)]
    print(f"[calib-stream] streams={len(calib_streams)} "
          f"batches={len(calib_streams[0].batches)} "
          f"events={[len(s.events) for s in calib_streams]}")
    calib = calibrate_ontoevolve(calib_streams, base_params=base,
                                 n_trials=args.trials, seed=args.seed,
                                 cooldown=args.cooldown)
    best_params: C.DriftParams = calib["params"]
    link_offset: float = calib["link_offset"]
    base_kw = {m: calibrate_baseline(calib_streams[0], m, base_params=base,
                                     cooldown=args.cooldown)["kw"]
               for m in BASELINES}

    # ---------- 2) 测试重复 --------------------------------------------------
    scores: Dict[str, List[float]] = {"ontoevolve": []}
    for m in BASELINES:
        scores[m] = []
    per_type: Dict[str, Dict[str, List[float]]] = {
        k: {t: [] for t in TYPES} for k in scores}
    delays: Dict[str, List[float]] = {k: [] for k in scores}
    rows: List[Dict] = []
    dumped: List[Dict] = []          # --dump-alarms 用：逐条告警

    def _dump(repeat: int, method: str, alarms) -> None:
        if not args.dump_alarms:
            return
        for a in alarms:
            dumped.append({"repeat": repeat, "method": method,
                           "t": int(a.t), "cid": str(a.cid),
                           "type": str(a.dtype)})

    for r in range(args.repeats):
        seed_r = args.seed + r
        stream = make_controlled_stream(emb, seed=seed_r, params=base)
        onto, info = build_o0(stream, seed=0)
        leaf_to_cid = info["leaf_to_cid"]
        p05 = float(info.get("link_sim_p05", best_params.link_sim))
        t0 = time.time()

        res, alarms = evaluate(stream, onto, leaf_to_cid, best_params,
                               link_offset=link_offset, base_p05=p05,
                               cooldown=args.cooldown)
        scores["ontoevolve"].append(res.macro_f1)
        delays["ontoevolve"].append(res.mean_delay)
        for t in TYPES:
            per_type["ontoevolve"][t].append(res.per_type[t]["f1"])
        rows.append({"repeat": r, "method": "ontoevolve",
                     **res.as_row(), "n_events": res.n_events})
        _dump(r, "ontoevolve", alarms)

        for m in BASELINES:
            bres, balarms = run_baseline(onto, leaf_to_cid, stream, m,
                                         base_kw[m], best_params,
                                         cooldown=args.cooldown)
            scores[m].append(bres.macro_f1)
            delays[m].append(bres.mean_delay)
            for t in TYPES:
                per_type[m][t].append(bres.per_type[t]["f1"])
            rows.append({"repeat": r, "method": m,
                         **bres.as_row(), "n_events": bres.n_events})
            _dump(r, m, balarms)
        print(f"[rep {r:>2}] ontoevolve={scores['ontoevolve'][-1]:.3f} "
              + " ".join(f"{m}={scores[m][-1]:.3f}" for m in BASELINES)
              + f"  ({time.time()-t0:.1f}s, concepts={len(onto.concepts)})")

    if args.dump_alarms and dumped:
        dp = Path(args.dump_alarms)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with dp.open("w", encoding="utf-8") as f:
            for rec in dumped:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[alarms] {len(dumped)} 条已导出 -> {dp}")

    # ---------- 3) 汇总 ------------------------------------------------------
    summary = {
        "cooldown": args.cooldown,
        "calibration": {
            "macro_f1_calib": calib["macro_f1"],
            "link_offset": link_offset,
            "params": {k: v for k, v in vars(best_params).items()},
            "baseline_kw": base_kw,
            "n_trials": calib["n_trials"],
        },
        "n_repeats": args.repeats,
        "methods": {},
    }
    def _safe_mean(xs: List[float]) -> float:
        arr = np.asarray([x for x in xs if not np.isnan(x)], dtype=float)
        return float(arr.mean()) if arr.size else float("nan")

    for m, vals in scores.items():
        ci = bootstrap_ci(vals, n_boot=C.PARAMS.n_bootstrap, seed=args.seed)
        entry = {
            "macro_f1_mean": ci["mean"],
            "macro_f1_ci95": [ci["lo"], ci["hi"]],
            "per_repeat": vals,
            "per_type_mean": {t: float(np.mean(per_type[m][t])) for t in TYPES},
            "mean_delay": _safe_mean(delays[m]),
        }
        if m != "ontoevolve":
            w = wilcoxon_paired(scores["ontoevolve"], vals)
            entry["wilcoxon_vs_ontoevolve"] = w
        summary["methods"][m] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    print("\n==== 实验 1 汇总（macro-F1，bootstrap 95% CI） ====")
    for m, e in summary["methods"].items():
        extra = ""
        if "wilcoxon_vs_ontoevolve" in e:
            w = e["wilcoxon_vs_ontoevolve"]
            extra = f"  Wilcoxon p={w['p']:.4f}"
        print(f"  {m:<14} {e['macro_f1_mean']:.3f} "
              f"[{e['macro_f1_ci95'][0]:.3f}, {e['macro_f1_ci95'][1]:.3f}]"
              f"  delay={e['mean_delay']:.2f}{extra}")
    print(f"\n-> {out}\n-> {args.summary}  ({time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
