# -*- coding: utf-8 -*-
"""人工标注完成后，计算人工评分与各自动判据的一致性。

输入：annotations/human_eval_for_scoring.csv（human_score 列已填 0–3 整数；
      若只做二值标注，可填 0/1，脚本同时给出二值口径）。

输出：
  1) 人工 vs 确定性判据（"human_score >= 2" 对 "det_correct"） 的 Cohen's kappa
  2) 人工 vs DeepSeek-Judge、人工 vs Qwen3-Judge 的 kappa（0–3 分整数口径）
  3) 混淆矩阵与百分比一致率，并给出 kappa 的 95% CI（bootstrap）

用法：python compute_human_kappa.py [--file annotations/human_eval_for_scoring.csv]
"""
import argparse, csv, math, random
from collections import Counter, defaultdict

random.seed(0)


def kappa(pairs):
    """pairs: list of (a, b) -> Cohen's kappa"""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1:
        return 1.0, 1.0
    return (po - pe) / (1 - pe), po


def boot_ci(pairs, rounds=2000):
    vals = []
    n = len(pairs)
    for _ in range(rounds):
        s = [pairs[random.randrange(n)] for _ in range(n)]
        k, _ = kappa(s)
        if not math.isnan(k):
            vals.append(k)
    vals.sort()
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def load(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("human_score", "")).strip() == "":
                continue
            rows.append(r)
    return rows


def to_int(x):
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="annotations/human_eval_for_scoring.csv")
    ap.add_argument("--threshold", type=int, default=2,
                    help="人工 0–3 分转为二值时的阈值（>=threshold 记为正确）")
    a = ap.parse_args()

    rows = load(a.file)
    print("已标注 %d / 目标题数" % len(rows))
    if not rows:
        print("human_score 列尚未填写，退出。")
        return

    print("\n== 一、多档口径（0–3 分整数）==")
    for name, key in (("DeepSeek-Judge", "judge_deepseek"), ("Qwen3-Judge", "judge_qwen3")):
        pairs = [(to_int(r["human_score"]), to_int(r.get(key))) for r in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        k, po = kappa(pairs)
        lo, hi = boot_ci(pairs)
        print(f"  人工 vs {name:14s} n={len(pairs):3d} kappa={k:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"一致率={po:.3f}")

    print("\n== 二、二值口径（人工 >=%d 记为正确）==" % a.threshold)
    for name, key in (("确定性判据", "det_correct"), ("DeepSeek-Judge", "judge_deepseek"),
                      ("Qwen3-Judge", "judge_qwen3")):
        pairs = []
        for r in rows:
            h = to_int(r["human_score"])
            hb = None if h is None else int(h >= a.threshold)
            if key == "det_correct":
                ab = to_int(r.get("det_correct"))
            else:
                v = to_int(r.get(key))
                ab = None if v is None else int(v >= a.threshold)
            pairs.append((hb, ab))
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        k, po = kappa(pairs)
        lo, hi = boot_ci(pairs)
        print(f"  人工 vs {name:14s} n={len(pairs):3d} kappa={k:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"一致率={po:.3f}")

    print("\n== 三、人工评分分布 ==")
    print("  ", dict(sorted(Counter(to_int(r["human_score"]) for r in rows).items(),
                           key=lambda kv: (kv[0] is None, kv[0]))))
    print("\n== 四、分歧最多的分层（按知识类型）==")
    bad = defaultdict(lambda: [0, 0])
    for r in rows:
        h, q = to_int(r["human_score"]), to_int(r.get("judge_qwen3"))
        if h is None or q is None:
            continue
        bad[r["kclass"]][0] += 1
        if int(h >= a.threshold) != int(q >= a.threshold):
            bad[r["kclass"]][1] += 1
    for k, (tot, dis) in sorted(bad.items()):
        print(f"  {k:10s} n={tot:3d} 二值分歧 {dis:3d} ({dis / tot:.1%})")


if __name__ == "__main__":
    main()
