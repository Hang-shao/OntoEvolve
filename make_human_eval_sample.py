# -*- coding: utf-8 -*-
"""生成人工双评抽样表（供 §5/§6.3 所述人工 Cohen's kappa 使用）。

分层：知识类型(3) × 确定性判定(对/错)，并**强制纳入**所有 LLM-Judge 给中间分
（1 分、2 分）与陈旧回答的题；输出 annotations/human_eval_sample.csv。

用法：
    python make_human_eval_sample.py --per-class 50 --out annotations/human_eval_sample.csv
"""
import argparse, csv, json, os, random

random.seed(20261008)

QDIR = "results/exp_qa_judge_qwen"


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--baseline", default="results/exp_qa")
    ap.add_argument("--out", default="annotations/human_eval_sample.csv")
    ap.add_argument("--csv-out", default="annotations/human_eval_for_scoring.csv")
    a = ap.parse_args()

    ans = load_jsonl(os.path.join(QDIR, "answers.jsonl"))
    qs = {q["qid"]: q for q in load_jsonl(os.path.join(QDIR, "questions.jsonl"))}
    # 确定性判据（无 judge 列）的同一批答案
    det = {}
    with open(os.path.join(a.baseline, "scores.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            det[(r["seed"], r["qid"], r["system"], r["version"])] = int(r["correct"])
    # DeepSeek 族 judge（若存在）
    ds = {}
    dsp = "results/exp_qa_judge/scores.csv"
    if os.path.exists(dsp):
        with open(dsp, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "judge" in r and r["judge"]:
                    ds[(r["seed"], r["qid"], r["system"], r["version"])] = r["judge"]

    # 去重：每个 (qid, system, version) 取 seed 最小的一条
    uniq = {}
    for r in ans:
        k = (r["qid"], r["system"], r["version"])
        if k not in uniq or int(r["seed"]) < int(uniq[k]["seed"]):
            uniq[k] = r

    must, pool = [], {c: [] for c in ("stable", "new", "redefined")}
    for r in uniq.values():
        key = (str(r["seed"]), r["qid"], r["system"], r["version"])
        j = r.get("judge")
        rec = {
            "qid": r["qid"], "kclass": r["kclass"], "kind": r["kind"],
            "system": r["system"], "version": r["version"], "seed": r["seed"],
            "question": qs[r["qid"]].get("q_%s" % r["version"], ""),
            "gold": r["gold"], "model_answer": r["pred_value"],
            "det_correct": det.get(key, r["correct"]),
            "judge_deepseek": ds.get(key, ""),
            "judge_qwen3": j if j not in ("", None) else "",
            "human_score": "", "human_note": "",
        }
        try:
            jj = int(j)
        except (TypeError, ValueError):
            jj = None
        if r.get("stale_hit") or jj in (1, 2):
            must.append(rec)
        else:
            pool[r["kclass"]].append(rec)

    # 强制纳入（按 (qid,system,version) 去重）
    chosen, seen = [], set()
    for rec in must:
        k = (rec["qid"], rec["system"], rec["version"])
        if k not in seen:
            seen.add(k)
            chosen.append(rec)
    # 分层补齐：每类对/错各半
    for cls, items in pool.items():
        n = max(0, a.per_class - sum(1 for c in chosen if c["kclass"] == cls))
        if n <= 0:
            continue
        ok = [r for r in items if r["det_correct"] == 1 and (r["qid"], r["system"], r["version"]) not in seen]
        bad = [r for r in items if r["det_correct"] == 0 and (r["qid"], r["system"], r["version"]) not in seen]
        random.shuffle(ok)
        random.shuffle(bad)
        half = n // 2
        pick = ok[:half] + bad[:n - half]
        for r in pick:
            seen.add((r["qid"], r["system"], r["version"]))
            chosen.append(r)

    chosen.sort(key=lambda r: (r["kclass"], r["qid"], r["system"], r["version"]))
    os.makedirs(os.path.dirname(a.csv_out), exist_ok=True)
    with open(a.csv_out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chosen[0].keys()))
        w.writeheader()
        w.writerows(chosen)

    # 统计
    from collections import Counter
    print("抽样 %d 题 ->" % len(chosen), a.csv_out)
    print("  按知识类型:", dict(Counter(r["kclass"] for r in chosen)))
    print("  按判定:", dict(Counter(r["det_correct"] for r in chosen)))
    print("  按臂:", dict(Counter(r["system"] for r in chosen)))
    print("  judge 给中间分的题:", sum(1 for r in chosen if r["judge_qwen3"] in ("1", "2", 1, 2)))
    print("  两族 judge 分值不同者:",
          sum(1 for r in chosen if r["judge_deepseek"] not in ("", None)
              and str(r["judge_deepseek"]) != str(r["judge_qwen3"])))


if __name__ == "__main__":
    main()
