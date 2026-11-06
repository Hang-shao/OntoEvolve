"""QA 的 seed 级推断（ESM S14 / Table S19）。

归档 `results/exp_qa_periodic/answers.jsonl` 同时含 dynamic / frozen / periodic 三臂
（5 seeds × 2 版本 × 三类知识 × 每类 100 题）。本脚本把**推断单位**从"问题对"换成"seed"：

  - 每个 seed、每个 (知识类型, 版本) 单元格的准确率；
  - seed 级配对差 dynamic − frozen、dynamic − periodic；
  - seed 级精确符号检验（n = 5，双侧；最小可达 p = 2/32 = 0.0625）；
  - 以 seed 为重抽样单位的 cluster bootstrap（B = 10000，标记为探索性）。

它不改任何数字：问题级配对区间仍由 `analyze_qa_pairs.py` 产出，本脚本只回答
"五个 seed 的方向是否一致，以及这个样本量能支持多强的推断"。

用法：
    python experiments/analyze_qa_seed_level.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

ANSWERS = ROOT / "results" / "exp_qa_periodic" / "answers.jsonl"
OUT = ROOT / "results" / "qa_seed_level.json"
KCLASS = ("stable", "new", "redefined")
VERSIONS = ("a", "b")
SYSTEMS = ("dynamic", "frozen", "periodic")


def sign_test_two_sided(pos: int, neg: int) -> float:
    """精确符号检验（双侧，忽略并列）。"""
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    rows = [json.loads(l) for l in ANSWERS.read_text(encoding="utf-8").splitlines() if l.strip()]
    seeds = sorted({r["seed"] for r in rows})

    # (system, kclass, version, seed) -> 正确率
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["system"], r["kclass"], r["version"], r["seed"])
        acc[key][0] += int(r["correct"])
        acc[key][1] += 1

    cells, summary = {}, []
    rng = np.random.default_rng(0)
    for kc in KCLASS:
        for ver in VERSIONS:
            cell = {"kclass": kc, "version": ver, "per_seed": {}}
            for sysname in SYSTEMS:
                cell["per_seed"][sysname] = [
                    round(acc[(sysname, kc, ver, s)][0] / max(1, acc[(sysname, kc, ver, s)][1]), 4)
                    for s in seeds]
            for ref in ("frozen", "periodic"):
                d = [a - b for a, b in zip(cell["per_seed"]["dynamic"], cell["per_seed"][ref])]
                pos, neg = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
                boot = rng.choice(np.asarray(d, dtype=float),
                                  size=(10000, len(d)), replace=True).mean(axis=1)
                cell[f"diff_vs_{ref}"] = {
                    "per_seed": [round(x, 4) for x in d],
                    "mean": round(float(np.mean(d)), 4),
                    "min": round(min(d), 4),
                    "max": round(max(d), 4),
                    "seeds_better": pos,
                    "seeds_worse": neg,
                    "seeds_tied": len(d) - pos - neg,
                    "exact_sign_p": round(sign_test_two_sided(pos, neg), 4),
                    "cluster_bootstrap_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                                               round(float(np.percentile(boot, 97.5)), 4)],
                    "exploratory": True,
                }
            cells[f"{kc}/{ver}"] = cell

    # 汇总：方向一致性（五个 seed 全正 / 全负 / 不一致）
    for kc in KCLASS:
        for ver in VERSIONS:
            for ref in ("frozen", "periodic"):
                blk = cells[f"{kc}/{ver}"][f"diff_vs_{ref}"]
                summary.append({"kclass": kc, "version": ver, "reference": ref,
                                "mean": blk["mean"], "seeds_better": blk["seeds_better"],
                                "seeds_worse": blk["seeds_worse"],
                                "exact_sign_p": blk["exact_sign_p"]})

    out = {
        "protocol": {"source": "results/exp_qa_periodic/answers.jsonl",
                     "seeds": seeds, "n_seeds": len(seeds),
                     "inference_unit": "seed",
                     "seed_level_floor_p": round(2 / 2 ** len(seeds), 4),
                     "item_level_intervals": "reported separately by analyze_qa_pairs.py "
                                             "(conditional on the realized question set)",
                     "bootstrap": {"B": 10000, "unit": "seed", "status": "exploratory"}},
        "cells": cells,
        "summary": summary,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in summary:
        print(f"{s['kclass']:10s} v{s['version']} vs {s['reference']:9s} "
              f"mean={s['mean']:+.3f} better={s['seeds_better']} worse={s['seeds_worse']} "
              f"p={s['exact_sign_p']:.4f}")
    print("写入", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
