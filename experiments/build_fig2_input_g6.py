# -*- coding: utf-8 -*-
"""由 G6（声明口径）归档生成 Fig. 2 的输入，schema 与 results/exp1_summary.json 一致。

`results/exp1_summary.json` 是执行口径（V23）的结果；Table 3 升到声明口径后，
Fig. 2 必须用同一份数据，否则图与表不一致（审稿意见 Minor 7）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_drift import bootstrap_ci  # noqa: E402

CELL = "G6_main_candidate"
SEED = 42
SRC = ROOT / "results/baseline_protocol_detail_all.json"
DST = ROOT / "results/exp1_summary_g6.json"

d = json.loads(SRC.read_text("utf-8"))[CELL]["methods"]
out = {"source": {"cell": CELL, "file": SRC.name,
                  "note": "declared protocol; Table 3 of V25 and Fig. 2 share this file"},
       "methods": {}}
for m, v in d.items():
    per = [float(x) for x in v["per_repeat"]]
    ci = bootstrap_ci(per, n_boot=1000, seed=SEED)
    out["methods"][m] = {
        "macro_f1_mean": sum(per) / len(per),
        "macro_f1_ci95": [ci["lo"], ci["hi"]],
        "per_type_mean": {k: sum(x) / len(x) for k, x in v["per_type"].items()},
        "per_repeat": per,
    }
DST.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", DST, "methods:", len(out["methods"]))
for m, v in out["methods"].items():
    print(f"  {m:<15} {v['macro_f1_mean']:.4f} "
          f"[{v['macro_f1_ci95'][0]:.3f}, {v['macro_f1_ci95'][1]:.3f}]")
