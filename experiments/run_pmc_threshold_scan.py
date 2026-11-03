# -*- coding: utf-8 -*-
"""轨道 3（PMC）重定义阈值的敏感性扫描。

背景：论文里轨道 3 的重定义阈值只能由无监督的 μ+zσ 估计（z=2 时 θ≈64.78），
而论文自认该阈值“被推高、跨轨道不可外推”。本脚本在**与表 7 轨道 3 相同的
数据口径（per-slice=1200）**下扫描 z，检验：

  · 检出结果（涌现/灭绝/重定义告警数）对阈值是否敏感；
  · 阈值提高是否改变最终本体规模（概念数/公理数）。

注意：曾用默认 per-slice=600 跑过一遍，与论文基线口径不一致（基线 exp_pmc.json
为 per_slice=1200、14400 句），故本脚本显式固定 per_slice=1200 以保证可比。

用法：
    python experiments/run_pmc_threshold_scan.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ZS = ("1.0", "1.5", "2.0", "3.0")
PER_SLICE = "1200"

sys.stdout.reconfigure(encoding="utf-8")


def run_one(z: str) -> dict:
    out_dir = ROOT / "results" / ("exp_pmc_ps1200_z%s" % z)
    t0 = time.perf_counter()
    r = subprocess.run([PY, "experiments/run_exp_pmc.py", "--z", z,
                        "--per-slice", PER_SLICE, "--out-dir", str(out_dir)],
                       cwd=str(ROOT))
    el = time.perf_counter() - t0
    rep = json.loads((out_dir / "exp_pmc.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((out_dir / "exp_pmc.csv").open(encoding="utf-8")))
    s: Counter = Counter()
    for row in rows:
        for k in ("alarms", "n_emergence", "n_extinction", "n_redefinition"):
            s[k] += int(row[k])
    return {
        "z": float(z),
        "theta_red": round(float(rep["theta_red_unsupervised"]), 2),
        "alarms": s["alarms"],
        "emergence": s["n_emergence"],
        "extinction": s["n_extinction"],
        "redefinition": s["n_redefinition"],
        "final_concepts": rep["final"]["concepts"],
        "final_axioms": rep["final"]["axioms"],
        "consistency_rate": rep["consistency_rate"],
        "rollback_rate": rep["rollback_rate"],
        "n_slices": len(rows),
        "exit_code": r.returncode,
        "seconds": round(el, 1),
    }


def main() -> int:
    recs = []
    for z in ZS:
        print("\n===== 轨道3 阈值扫描 z=%s（per-slice=%s）=====" % (z, PER_SLICE),
              flush=True)
        rec = run_one(z)
        recs.append(rec)
        print("  θ_red=%.2f alarms=%d(emer %d/ext %d/redef %d) "
              "final concepts=%d axioms=%d cons=%.2f rb=%.2f（%.0fs，退出码 %s）"
              % (rec["theta_red"], rec["alarms"], rec["emergence"],
                 rec["extinction"], rec["redefinition"], rec["final_concepts"],
                 rec["final_axioms"], rec["consistency_rate"],
                 rec["rollback_rate"], rec["seconds"], rec["exit_code"]),
              flush=True)

    out = ROOT / "results" / "exp_pmc_threshold_scan.json"
    out.write_text(json.dumps({"per_slice": int(PER_SLICE), "records": recs},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = ROOT / "results" / "exp_pmc_threshold_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    print("\n======== 轨道3 阈值敏感性（per-slice=%s）========" % PER_SLICE)
    print("%-6s %-9s %-7s %-7s %-7s %-8s %-9s %-8s"
          % ("z", "θ_red", "alarms", "涌现", "灭绝", "重定义", "概念数", "公理数"))
    for r in recs:
        print("%-6.1f %-9.2f %-7d %-7d %-7d %-8d %-9d %-8d"
              % (r["z"], r["theta_red"], r["alarms"], r["emergence"],
                 r["extinction"], r["redefinition"], r["final_concepts"],
                 r["final_axioms"]))
    print("\n结果写入 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
