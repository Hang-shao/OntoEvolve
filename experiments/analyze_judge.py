# -*- coding: utf-8 -*-
"""LLM-Judge 交叉验证分析：确定性判据 vs LLM-Judge（Qwen3-32B）。

用法：python _analyze_judge.py <scores.csv> <输出 json> <模型名>
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    src, out, model = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    n = len(rows)

    det = [int(r["correct"]) for r in rows]
    jr = [int(r["judge"]) for r in rows]
    llm = [1 if v >= 2 else 0 for v in jr]

    pairs = list(zip(det, llm))
    agree = sum(1 for a, b in pairs if a == b) / n
    p1 = sum(det) / n
    p2 = sum(llm) / n
    exp = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = None if exp >= 1.0 else (agree - exp) / (1 - exp)

    conf = Counter(pairs)
    dist = Counter(jr)
    n_fail = sum(1 for v in jr if v < 0)

    # 分组一致性
    groups: dict = {}
    for r in rows:
        g = f'{r["kclass"]}|{r["version"]}|{r["system"]}'
        e = groups.setdefault(g, {"n": 0, "det": 0, "llm": 0, "agree": 0})
        e["n"] += 1
        e["det"] += int(r["correct"])
        e["llm"] += 1 if int(r["judge"]) >= 2 else 0
        e["agree"] += int(int(r["correct"]) == (1 if int(r["judge"]) >= 2 else 0))

    per = {}
    for g, e in sorted(groups.items()):
        per[g] = {
            "n": e["n"],
            "deterministic_acc": round(e["det"] / e["n"], 4),
            "llm_judge_acc": round(e["llm"] / e["n"], 4),
            "agreement": round(e["agree"] / e["n"], 4),
        }

    # 分歧样本（供人工核查）
    dis = [{"qid": r["qid"], "kclass": r["kclass"], "system": r["system"],
            "version": r["version"], "gold": r["gold"], "pred": r["pred_value"],
            "det": int(r["correct"]), "judge": int(r["judge"])}
           for r in rows
           if int(r["correct"]) != (1 if int(r["judge"]) >= 2 else 0)]

    res = {
        "n_scored": n,
        "judge_model": model,
        "judge_scale": "0-3, threshold >=2 counted as correct",
        "parse_failures": n_fail,
        "agreement": round(agree, 6),
        "expected_agreement": round(exp, 6),
        "cohens_kappa": None if kappa is None else round(kappa, 6),
        "confusion": {str(k): v for k, v in sorted(conf.items())},
        "judge_score_dist": {str(k): v for k, v in sorted(dist.items())},
        "per_class_version_system": per,
        "disagreements": dis[:50],
        "n_disagreements": len(dis),
    }
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_class_version_system", "disagreements")},
                     ensure_ascii=False, indent=1))
    print("\n--- 分组 ---")
    for g, e in per.items():
        print(f"{g:>26}: n={e['n']:>4} det={e['deterministic_acc']:.4f} "
              f"llm={e['llm_judge_acc']:.4f} agree={e['agreement']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
