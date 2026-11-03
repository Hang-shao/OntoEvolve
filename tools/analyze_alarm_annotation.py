"""analyze_alarm_annotation - 合并人工复核与 LLM 首轮标注，输出论文所需指标。

输入：
    --pack  人工复核后的标注表（含 is_true_drift / dtype_true / confidence / note）
    --llm   llm_annotate_alarms.py 的输出
输出：
    --wide  合并宽表（逐条对照，供申诉与附表）
    控制台：人工精确率 + Wilson 95% CI、LLM 与人工一致率与 Cohen's κ、
            真漂移子集上的类型一致率、分歧清单条目数

Run:
    python tools/analyze_alarm_annotation.py \
        --pack annotations/alarm_pack_human.csv \
        --llm annotations/alarm_pack_llm.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
POSITIVE, NEGATIVE, UNSURE = "是", "否", "无法判定"


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohen_kappa(a: List[str], b: List[str]) -> Optional[float]:
    """仅对双方都给出判定的条目计算 κ。"""
    pairs = [(x, y) for x, y in zip(a, b)
             if x in (POSITIVE, NEGATIVE) and y in (POSITIVE, NEGATIVE)]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def precision(rows: List[Dict], key: str) -> Dict:
    yes = sum(1 for r in rows if r[key] == POSITIVE)
    no = sum(1 for r in rows if r[key] == NEGATIVE)
    un = sum(1 for r in rows if r[key] == UNSURE)
    denom = yes + no
    lo, hi = wilson(yes, denom)
    return {"yes": yes, "no": no, "unsure": un, "denom": denom,
            "p": (yes / denom if denom else float("nan")), "ci": (lo, hi)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(ROOT / "annotations" / "alarm_pack_human.csv"))
    ap.add_argument("--llm", default=str(ROOT / "annotations" / "alarm_pack_llm.csv"))
    ap.add_argument("--wide", default=str(ROOT / "annotations" / "alarm_pack_wide.csv"))
    args = ap.parse_args()

    human = {r["alarm_id"]: r for r in csv.DictReader(open(args.pack, encoding="utf-8-sig"))}
    rows = list(human.values())
    missing = sum(1 for r in rows if not r.get("is_true_drift"))
    print(f"[check] 人工标注 {len(rows)} 条，未填写 {missing} 条")
    if missing:
        print("[warn] 存在未填写的条目，精确率分母将只统计已填写的'是/否'")

    for tag in ("ICEWS18", "PMC"):
        sub = [r for r in rows if r["track"] == tag]
        st = precision(sub, "is_true_drift")
        print(f"[human] {tag}: 真 {st['yes']} / 误报 {st['no']} / 无法判定 {st['unsure']}"
              f"  → 精确率 {st['p']:.3f}  95%CI [{st['ci'][0]:.3f}, {st['ci'][1]:.3f}]")
    allst = precision(rows, "is_true_drift")
    print(f"[human] 合计: 精确率 {allst['p']:.3f} "
          f"95%CI [{allst['ci'][0]:.3f}, {allst['ci'][1]:.3f}]")

    llm_path = Path(args.llm)
    if llm_path.exists():
        llm = {r["alarm_id"]: r for r in csv.DictReader(open(llm_path, encoding="utf-8-sig"))}
        a = [human[i].get("is_true_drift", "") for i in human if i in llm]
        b = [llm[i].get("llm_is_true_drift", "") for i in human if i in llm]
        k = cohen_kappa(a, b)
        pair = [(x, y) for x, y in zip(a, b)
                if x in (POSITIVE, NEGATIVE) and y in (POSITIVE, NEGATIVE)]
        agree = (sum(1 for x, y in pair if x == y) / len(pair)) if pair else float("nan")
        print(f"[agree] LLM vs 人工：可比 {len(pair)} 条，一致率 {agree:.3f}，"
              f"Cohen's κ = {('n/a' if k is None else f'{k:.3f}')}")
        ll = precision([{"v": llm[i].get("llm_is_true_drift", "")} for i in human
                        if i in llm], "v")
        print(f"[llm] 首轮判断：真 {ll['yes']} / 否 {ll['no']} / 无法判定 {ll['unsure']}"
              f"  → 精确率 {ll['p']:.3f}")

        wide = []
        for i, h in human.items():
            l = llm.get(i, {})
            wide.append({**h,
                         "llm_is_true_drift": l.get("llm_is_true_drift", ""),
                         "llm_dtype": l.get("llm_dtype", ""),
                         "llm_confidence": l.get("llm_confidence", ""),
                         "llm_rationale": l.get("llm_rationale", ""),
                         "agree": ("一致" if h.get("is_true_drift")
                                   and h.get("is_true_drift") == l.get("llm_is_true_drift")
                                   else "分歧")})
        dis = sum(1 for r in wide if r["agree"] == "分歧")
        with open(args.wide, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(wide[0].keys()))
            w.writeheader()
            w.writerows(wide)
        print(f"[wide] 分歧 {dis} 条 -> {args.wide}（可直接作为仲裁工作表）")

    # 真漂移子集上的类型一致率
    both = [(r.get("dtype_true", ""), (llm.get(r["alarm_id"], {}).get("llm_dtype", "")
            if llm_path.exists() else "")) for r in rows
            if r.get("is_true_drift") == POSITIVE]
    if llm_path.exists() and both:
        ok = sum(1 for x, y in both if x and y and x == y)
        print(f"[type] 真漂移子集 {len(both)} 条，类型一致 {ok} 条"
              f"（{ok / len(both):.3f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
