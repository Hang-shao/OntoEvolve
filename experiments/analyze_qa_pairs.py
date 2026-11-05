# -*- coding: utf-8 -*-
"""配对表：每个比较的 N、n+、n- 与精确 McNemar p（供 ESM 1 直接重算）。

比较对象：dynamic vs frozen（Table 7）、dynamic vs periodic（Section 4.5.2），
按 (seed, version, kclass, qid) 配对，逐版本与池化两种口径。

Run:
    python experiments/analyze_qa_pairs.py --out results/qa_pair_counts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from scipy.stats import binomtest
except Exception:                                   # pragma: no cover
    binomtest = None


def exact_mcnemar(b: int, c: int) -> float:
    """两侧精确 McNemar：2 × min{P(X≤min(b,c))} 截断到 1（X ~ Bin(b+c, 0.5)）。"""
    n = b + c
    if n == 0:
        return 1.0
    if binomtest is not None:
        return float(binomtest(min(b, c), n, 0.5).pvalue)
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(ROOT / "results" / "exp_qa_periodic" / "answers.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "qa_pair_counts.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.answers).read_text(encoding="utf-8").splitlines() if l.strip()]
    keyed = defaultdict(dict)
    for r in rows:
        keyed[(r["seed"], r["version"], r["kclass"], r["qid"])][r["system"]] = int(r["correct"])

    out = {"n_answers": len(rows),
           "systems": sorted({r["system"] for r in rows}),
           "comparisons": {}}
    for other in ("frozen", "periodic"):
        cells = defaultdict(lambda: {"n": 0, "plus": 0, "minus": 0, "both": 0, "neither": 0})
        for (seed, ver, kc, qid), d in keyed.items():
            if "dynamic" not in d or other not in d:
                continue
            cell = cells[(ver, kc)]
            a, b = d["dynamic"], d[other]
            cell["n"] += 1
            if a and not b:
                cell["plus"] += 1
            elif b and not a:
                cell["minus"] += 1
            elif a and b:
                cell["both"] += 1
            else:
                cell["neither"] += 1
        for ver in ("a", "b"):
            for kc in ("stable", "new", "redefined"):
                if (ver, kc) not in cells:
                    continue
                c = cells[(ver, kc)]
                p = exact_mcnemar(c["plus"], c["minus"])
                cells[(ver, kc)].update({
                    "p_exact": float(f"{p:.3g}"),
                    "paired_accuracy_difference":
                        round((c["plus"] - c["minus"]) / c["n"], 4) if c["n"] else None})
        out["comparisons"][f"dynamic_vs_{other}"] = {
            f"{ver}/{kc}": cells[(ver, kc)]
            for ver in ("a", "b") for kc in ("stable", "new", "redefined") if (ver, kc) in cells}
        # 池化（两个版本合并，按 kclass）
        pooled = {}
        for kc in ("stable", "new", "redefined"):
            n = sum(cells[(v, kc)]["n"] for v in ("a", "b") if (v, kc) in cells)
            plus = sum(cells[(v, kc)]["plus"] for v in ("a", "b") if (v, kc) in cells)
            minus = sum(cells[(v, kc)]["minus"] for v in ("a", "b") if (v, kc) in cells)
            if not n:
                continue
            pooled[kc] = {"n": n, "plus": plus, "minus": minus,
                          "p_exact": float(f"{exact_mcnemar(plus, minus):.3g}"),
                          "paired_accuracy_difference": round((plus - minus) / n, 4)}
        out["comparisons"][f"dynamic_vs_{other}"]["pooled"] = pooled

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, comp in out["comparisons"].items():
        print(f"\n== {name} ==")
        for k, v in comp.items():
            if k == "pooled":
                for kc, pv in v.items():
                    print(f"  pooled/{kc:<10} N={pv['n']:<5} n+={pv['plus']:<4} "
                          f"n-={pv['minus']:<4} p={pv['p_exact']:<10} "
                          f"diff={pv['paired_accuracy_difference']}")
                continue
            print(f"  {k:<12} N={v['n']:<5} n+={v['plus']:<4} n-={v['minus']:<4} "
                  f"p={v['p_exact']:<10} diff={v['paired_accuracy_difference']}")
    print("->", args.out)


if __name__ == "__main__":
    main()
