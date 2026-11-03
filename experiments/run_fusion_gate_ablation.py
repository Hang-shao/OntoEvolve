# -*- coding: utf-8 -*-
"""补充实验：优先级分派 vs 加权融合门（轨道 1，10 次重复）。

3.2 节给出融合分数 s(c)=w_sem·s_sem+w_stat·s_stat+w_ax·s_ax 与门限 θ_score，
但实现里该门默认关闭、决策走优先级分派。本脚本把门打开，在同一测试流、
同一标定阈值（不重新调参）下比较：

    priority    门关闭（正文主实验口径）
    fused_x     门开启，θ_score = x

只影响已有概念的重定义/淘汰告警（涌现不经过该门），故差异集中在这两类。

Run:
    python experiments/run_fusion_gate_ablation.py --repeats 10
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

import numpy as np  # noqa: E402

from src import config as C                                      # noqa: E402
from src.build_ontology import build_initial_ontology             # noqa: E402
from src.calibrate import evaluate                                # noqa: E402
from src.controlled_stream import make_controlled_stream           # noqa: E402
from src.embedding import Embedder                                # noqa: E402
from src.eval_drift import TYPES, bootstrap_ci, wilcoxon_paired    # noqa: E402

OUT_DIR = C.DIR_RESULTS / "exp_fusion_gate"
VARIANTS = [("priority", False, None),
            ("fused_0.35", True, 0.35),
            ("fused_0.45", True, 0.45),
            ("fused_0.55", True, 0.55)]


def load_calibrated(path: Path):
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        cal = data["calibration"]
        return C.DriftParams(**cal["params"]), float(cal.get("link_offset", 0.0))
    print(f"[warn] 未找到 {path}，使用默认参数")
    return C.PARAMS, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary-in", default=str(C.DIR_RESULTS / "exp1_summary.json"))
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_params, link_offset = load_calibrated(Path(args.summary_in))
    emb = Embedder(allow_fallback=False, verbose=True)
    t_start = time.time()

    macro: Dict[str, List[float]] = {v: [] for v, _, _ in VARIANTS}
    per_type: Dict[str, Dict[str, List[float]]] = {
        v: {t: [] for t in TYPES} for v, _, _ in VARIANTS}
    alarms: Dict[str, List[int]] = {v: [] for v, _, _ in VARIANTS}
    hits: Dict[str, List[int]] = {v: [] for v, _, _ in VARIANTS}
    rows: List[Dict] = []

    for r in range(args.repeats):
        stream = make_controlled_stream(emb, seed=args.seed + r, params=C.PARAMS)
        onto, info = build_initial_ontology(
            stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
            vocab={**stream.vocab_base, **stream.vocab_emer})
        p05 = float(info.get("link_sim_p05", base_params.link_sim))
        for name, gate, theta in VARIANTS:
            p = base_params.copy(theta_score=float(theta)) if theta is not None \
                else base_params
            res, al = evaluate(stream, onto, info["leaf_to_cid"], p,
                               link_offset=link_offset, base_p05=p05,
                               fusion_gate=gate)
            macro[name].append(res.macro_f1)
            alerts_n = len(al)
            alarms[name].append(alerts_n)
            hits[name].append(res.tp)
            for t in TYPES:
                per_type[name][t].append(res.per_type[t]["f1"])
            rows.append({"repeat": r, "variant": name, "theta_score": theta,
                         "macro_f1": res.macro_f1, "alerts": alerts_n,
                         "hits": res.tp, "false_alarms": res.fp,
                         "misses": res.fn,
                         **{f"f1_{t}": res.per_type[t]["f1"] for t in TYPES}})
        print(f"[rep {r:>2}] " + " ".join(
            f"{n}={macro[n][-1]:.3f}(alarms {alarms[n][-1]})" for n, _, _ in VARIANTS),
            flush=True)

    summary: Dict = {"n_repeats": args.repeats, "seed_range": [args.seed,
                                                               args.seed + args.repeats - 1],
                     "link_offset": link_offset, "variants": {}}
    base_vals = macro["priority"]
    for name, gate, theta in VARIANTS:
        ci = bootstrap_ci(macro[name], n_boot=C.PARAMS.n_bootstrap, seed=args.seed)
        entry = {"fusion_gate": gate, "theta_score": theta,
                 "macro_f1": {"mean": round(float(np.mean(macro[name])), 4),
                              "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)]},
                 "per_type_f1": {t: round(float(np.mean(per_type[name][t])), 4)
                                 for t in TYPES},
                 "alerts_mean": round(float(np.mean(alarms[name])), 2),
                 "hits_mean": round(float(np.mean(hits[name])), 2)}
        if name != "priority":
            w = wilcoxon_paired(macro[name], base_vals)
            entry["delta_vs_priority"] = round(
                float(np.mean(macro[name]) - np.mean(base_vals)), 4)
            entry["wilcoxon_vs_priority"] = {"stat": round(float(w["stat"]), 3),
                                             "p": round(float(w["p"]), 6)}
        summary["variants"][name] = entry

    stem = f"fusion_gate_{args.tag}"
    (OUT_DIR / f"{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT_DIR / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n==== 融合门 vs 优先级分派（%d 次重复，用时 %.1f min）===="
          % (args.repeats, (time.time() - t_start) / 60))
    for name, _, _ in VARIANTS:
        e = summary["variants"][name]
        line = ("  %-11s F1=%.3f [%.3f, %.3f]  em=%.3f ex=%.3f re=%.3f  "
                "告警=%.1f 命中=%.1f"
                % (name, e["macro_f1"]["mean"], e["macro_f1"]["ci95"][0],
                   e["macro_f1"]["ci95"][1], e["per_type_f1"]["emergence"],
                   e["per_type_f1"]["extinction"], e["per_type_f1"]["redefinition"],
                   e["alerts_mean"], e["hits_mean"]))
        if "wilcoxon_vs_priority" in e:
            line += "  Δ=%+.3f p=%.3g" % (e["delta_vs_priority"],
                                          e["wilcoxon_vs_priority"]["p"])
        print(line)
    print(f"-> {OUT_DIR / (stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
