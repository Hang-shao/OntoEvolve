# -*- coding: utf-8 -*-
"""逐类型 precision / recall / F1 与事件数（Table 3 的支撑表 = ESM Table S20）。

数据来源：
  * `results/baseline_protocol_detail_G6pr.json` —— 由
    `experiments/run_baseline_protocol.py --cells G6_main_candidate --detail ...` 产出，
    含每个方法、每个 repeat 的 per-type p / r / f1 / tp / fp / fn（含 fp_other 三分摊）与 n_events；
  * `results/baseline_protocol_detail_all.json` 的 G6 单元 —— 用作**对账**：逐 repeat 的
    per-type F1 必须逐位相同，否则口径已漂。

口径说明（写进 ESM Table S20 的表注）：
  * 一个流注入 9 个事件（涌现、消亡、重定义各 3），十个 repeat 即每类型 30 个事件；
  * fp 按 §4.2.4 归因：落在无事件概念上的告警三等分计入三个类型（因此 fp 是小数）；
  * p、r、f1 是十次 repeat 的均值，不是由均值 tp/fp/fn 反算出来的。

用法：
    python experiments/analyze_per_type_pr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TYPES = ("emergence", "extinction", "redefinition")
METHODS = ("ontoevolve", "adwin", "inc_clustering", "page_hinkley", "embdist", "embkmeans")


def main() -> int:
    pr = json.loads((RES / "baseline_protocol_detail_G6pr.json").read_text(encoding="utf-8"))
    ref = json.loads((RES / "baseline_protocol_detail_all.json").read_text(encoding="utf-8"))
    cell_pr = pr["G6_main_candidate"]
    cell_ref = ref["G6_main_candidate"]

    # ---- 对账：逐 repeat 的 per-type F1 与 per-repeat macro-F1 必须一致
    recon = {"f1_matches_archive": True, "macro_matches_archive": True, "max_abs_diff": 0.0}
    for m in METHODS:
        a, b = cell_pr["methods"][m], cell_ref["methods"][m]
        for t in TYPES:
            for x, y in zip(a["per_type"][t], b["per_type"][t]):
                d = abs(x - y)
                recon["max_abs_diff"] = max(recon["max_abs_diff"], d)
                if d > 1e-12:
                    recon["f1_matches_archive"] = False
        for x, y in zip(a["per_repeat"], b["per_repeat"]):
            if abs(x - y) > 1e-12:
                recon["macro_matches_archive"] = False
    recon["max_abs_diff"] = round(recon["max_abs_diff"], 15)

    # ---- 汇总
    out = {"protocol": {"source": "results/baseline_protocol_detail_G6pr.json",
                        "cell": "G6_main_candidate",
                        "grid": cell_pr["config"]["grid"],
                        "n_streams_calib": cell_pr["config"]["n_streams"],
                        "calib_cooldown": cell_pr["config"]["calib_cooldown"],
                        "calib_tolerance": cell_pr["config"]["calib_tolerance"],
                        "n_repeats": len(cell_pr["methods"]["ontoevolve"]["per_repeat"]),
                        "events_per_stream": sorted({e for m in METHODS
                                                     for e in cell_pr["methods"][m]["n_events"]}),
                        "events_per_type_total": 3 * len(cell_pr["methods"]["ontoevolve"]["n_events"]),
                        "fp_attribution": "alarms on concepts with no event are split equally over "
                                          "the three types (Section 4.2.4), so fp is fractional",
                        "means": "p, r and f1 are means over the ten repeats, not recomputed "
                                 "from the mean tp/fp/fn"},
           "reconciliation": recon, "methods": {}}

    for m in METHODS:
        entry = cell_pr["methods"][m]
        types = {}
        for t in TYPES:
            d = entry["per_type_pr"][t]
            types[t] = {
                "precision_mean": round(float(sum(d["p"]) / len(d["p"])), 4),
                "recall_mean": round(float(sum(d["r"]) / len(d["r"])), 4),
                "f1_mean": round(float(sum(entry["per_type"][t]) / len(entry["per_type"][t])), 4),
                "tp_mean_per_stream": round(float(sum(d["tp"]) / len(d["tp"])), 2),
                "fp_mean_per_stream": round(float(sum(d["fp"]) / len(d["fp"])), 2),
                "fn_mean_per_stream": round(float(sum(d["fn"]) / len(d["fn"])), 2),
            }
        out["methods"][m] = {"kw": entry["kw"], "n_evals": entry.get("n_evals"),
                             "macro_f1_mean": round(entry["mean"], 4),
                             "n_events_per_stream": entry["n_events"], "types": types}

    (RES / "per_type_pr.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    print("对账", recon)
    for m in METHODS:
        e = out["methods"][m]
        row = " ".join(f"{t[:3]} P{e['types'][t]['precision_mean']:.3f}/"
                       f"R{e['types'][t]['recall_mean']:.3f}/"
                       f"F{e['types'][t]['f1_mean']:.3f}" for t in TYPES)
        print(f"{m:<15} {row}  macro={e['macro_f1_mean']:.4f}")
    print("写入", RES / "per_type_pr.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
