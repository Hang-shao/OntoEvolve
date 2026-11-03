# -*- coding: utf-8 -*-
"""统一告警选择规则分析：从无抑制原始告警流出发，对所有检测器施加同一套规则。

动机：主结果里的告警抑制（cooldown = window）对两侧的削减幅度不同，因此
"同等抑制窗口"并不等于同等处理。要回答"优势来自告警更少，还是来自选得更好"，
需要在原始（未抑制）告警流上施加与真值无关的统一规则。

规则（全部不看真值）：
    raw              不选择（原始检测输出）
    capN             按告警产生顺序取前 N 条
    dedup_cid        每个概念每流最多保留 1 条
    dedup_cid+capN   先去重再截断

输出 results/uniform_selection.json；对应论文 ESM 1 的 Table S5。

Run:
    python experiments/analyze_selection.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config as C
from src.build_ontology import build_initial_ontology
from src.controlled_stream import make_controlled_stream
from src.drift_detect import Alarm
from src.embedding import Embedder
from src.eval_drift import bootstrap_ci, match_events, wilcoxon_paired

METHODS = ["ontoevolve", "adwin", "inc_clustering", "page_hinkley",
           "embdist", "embkmeans"]
CAPS = [5, 10, 20, 35]
DELAY_TOL = 3          # 标定值（exp1_summary.json: calibration.params.delay_tol）

RULES = [("raw", None)] + \
        [(f"cap{c}", ("cap", c)) for c in CAPS] + \
        [("dedup_cid", ("dedup", None))] + \
        [(f"dedup_cid+cap{c}", ("dedup+cap", c)) for c in CAPS]


def load_alarms(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        out.setdefault((d["repeat"], d["method"]), []).append(
            Alarm(t=int(d["t"]), cid=str(d["cid"]), dtype=str(d["type"]), score=0.0))
    return out


def dedup_cid(alarms):
    seen, out = set(), []
    for a in alarms:
        if a.cid in seen:
            continue
        seen.add(a.cid)
        out.append(a)
    return out


def apply_rule(alarms, spec):
    if spec is None:
        return alarms
    kind, cap = spec
    if kind == "cap":
        return alarms[:cap]
    if kind == "dedup":
        return dedup_cid(alarms)
    if kind == "dedup+cap":
        return dedup_cid(alarms)[:cap]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    res = Path(args.results)

    raw0 = load_alarms(res / "alarms_cd0.jsonl")
    raw99 = load_alarms(res / "alarms_cd99.jsonl")
    params = C.PARAMS
    emb = Embedder(allow_fallback=False, verbose=False)

    out = {}
    for r in range(args.repeats):
        stream = make_controlled_stream(emb, seed=params.seed + r, params=params)
        _, info = build_initial_ontology(
            stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
            vocab={**stream.vocab_base, **stream.vocab_emer})
        l2c = info["leaf_to_cid"]
        for m in METHODS:
            typed = (m == "ontoevolve")
            for label, spec in RULES:
                al = apply_rule(raw0.get((r, m), []), spec)
                rr = match_events(stream.events, al, l2c, delay_tol=DELAY_TOL, typed=typed)
                rec = out.setdefault(f"cd0|{m}|{label}", {"f1": [], "na": []})
                rec["f1"].append(rr.macro_f1)
                rec["na"].append(len(al))
            for label, spec in ([("suppressed_raw", None)] +
                                [(f"suppressed_cap{c}", ("cap", c)) for c in CAPS]):
                al = apply_rule(raw99.get((r, m), []), spec)
                rr = match_events(stream.events, al, l2c, delay_tol=DELAY_TOL, typed=typed)
                rec = out.setdefault(f"cd99|{m}|{label}", {"f1": [], "na": []})
                rec["f1"].append(rr.macro_f1)
                rec["na"].append(len(al))
        print(f"[repeat {r}] done", flush=True)

    summary = {}
    for key, rec in out.items():
        cd, m, label = key.split("|")
        ci = bootstrap_ci(rec["f1"], n_boot=1000, seed=params.seed)
        summary.setdefault(cd, {}).setdefault(m, {})[label] = {
            "macro_f1_mean": round(ci["mean"], 4),
            "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
            "mean_alarms": round(float(np.mean(rec["na"])), 2),
            "per_repeat": [round(x, 4) for x in rec["f1"]],
        }
    cd0_labels = [lab for lab, _ in RULES]
    cd99_labels = ["suppressed_raw"] + [f"suppressed_cap{c}" for c in CAPS]
    for cd, labels in (("cd0", cd0_labels), ("cd99", cd99_labels)):
        for label in labels:
            summary[cd].setdefault("_wilcoxon_ours_vs_adwin", {})[label] = wilcoxon_paired(
                summary[cd]["ontoevolve"][label]["per_repeat"],
                summary[cd]["adwin"][label]["per_repeat"])

    dest = res / "uniform_selection.json"
    dest.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'rule':<20}" + "".join(f"{m[:10]:>16}" for m in METHODS))
    for label in cd0_labels:
        row = f"{label:<20}"
        for m in METHODS:
            v = summary["cd0"][m][label]
            row += f"{v['macro_f1_mean']:>9.3f}/{v['mean_alarms']:>6.1f}"
        print(row)
    print("->", dest)


if __name__ == "__main__":
    main()
