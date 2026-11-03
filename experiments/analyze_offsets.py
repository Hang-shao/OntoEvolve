# -*- coding: utf-8 -*-
"""匹配偏移（matched-alarm offset）的分解：巧合命中 / 有效提前 / 同步或滞后。

对称匹配窗口 |t_alert - t_declare| <= delay_tol 会把"事件声明之前"的告警也算作命中。
但 t_declare 不是人为标签：它由 controlled_stream._declare_batch 从数据里算出，
是"最早可判定批次"，平均比漂移真正开始的 t_start 晚约 2 批。因此每条命中要分三类：

    (a) t_alert < t_start            事件在数据里还没开始 —— 窗口内的巧合命中
    (b) t_start <= t_alert, offset>0 漂移已开始、尚未满足判定标准 —— 真正的提前检测
    (c) 同步或滞后

同时给出因果口径（只接受 0 <= t_alert - t_ref <= delay_tol）下的 macro-F1，
分别以 t_declare（可判定批次，方法相关）和 t_start（物理起始批次）为参照：
前者是方法相关参照，后者与检测器的准入条件无关，两者一起给出才能看出
因果窗口结论对参照点的依赖程度。

输出 results/offset_detail.json；对应论文 ESM 1 的 Table S6。

Run:
    python experiments/analyze_offsets.py
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
from src.eval_drift import TYPES, _cid_match, bootstrap_ci, match_events, wilcoxon_paired

METHODS = ["ontoevolve", "adwin", "inc_clustering", "page_hinkley",
           "embdist", "embkmeans"]
TOL = 3


def match_detail(events, alarms, leaf_to_cid, *, tol, typed, causal=False, ref="declare"):
    """与 src.eval_drift.match_events 同构，额外返回逐条命中记录与 macro-F1。

    ref="declare" 以 t_declare 为参照（与方法准入条件相关）；
    ref="start"   以 t_start 为参照（物理起始批次，与方法无关）。
    """
    used = [False] * len(alarms)
    tp_by = {t: 0 for t in TYPES}
    fn_by = {t: 0 for t in TYPES}
    rows = []
    for ev in events:
        hit = None
        base = ev.t_declare if ref == "declare" else ev.t_start
        for j, a in enumerate(alarms):
            if used[j]:
                continue
            d = a.t - base
            if causal:
                if d < 0 or d > tol:
                    continue
            elif abs(d) > tol:
                continue
            if typed and a.dtype != ev.dtype:
                continue
            if ev.dtype == "emergence":
                ok = a.cid.startswith("NEW") if typed else True
            else:
                ok = _cid_match(a, ev.leaf, leaf_to_cid)
            if ok:
                hit = j
                break
        if hit is None:
            fn_by[ev.dtype] += 1
            continue
        used[hit] = True
        tp_by[ev.dtype] += 1
        a = alarms[hit]
        rows.append({"leaf": ev.leaf, "dtype": ev.dtype,
                     "d_declare": int(a.t - ev.t_declare),
                     "d_start": int(a.t - ev.t_start)})
    concept_type = {}
    for ev in events:
        if ev.leaf in leaf_to_cid:
            concept_type[leaf_to_cid[ev.leaf]] = ev.dtype
    fp_by = {t: 0 for t in TYPES}
    fp_other = 0
    for j, a in enumerate(alarms):
        if used[j]:
            continue
        if a.cid.startswith("NEW"):
            fp_by["emergence"] += 1
            continue
        t = concept_type.get(a.cid)
        if t is None:
            fp_other += 1
        else:
            fp_by[t] += 1
    share = fp_other / 3.0
    f1s = []
    for t in TYPES:
        tp, fn = tp_by[t], fn_by[t]
        fp = fp_by[t] + share
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    res = Path(args.results)

    raw = {}
    for line in (res / "alarms_cd99.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        raw.setdefault((d["repeat"], d["method"]), []).append(
            Alarm(t=int(d["t"]), cid=str(d["cid"]), dtype=str(d["type"]), score=0.0))

    params = C.PARAMS
    emb = Embedder(allow_fallback=False, verbose=False)

    rows = {m: [] for m in METHODS}
    sym_f1 = {m: [] for m in METHODS}
    cau_f1 = {m: [] for m in METHODS}
    cau_start_f1 = {m: [] for m in METHODS}
    gaps = []
    maxdiff = 0.0
    for r in range(args.repeats):
        stream = make_controlled_stream(emb, seed=params.seed + r, params=params)
        _, info = build_initial_ontology(
            stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
            vocab={**stream.vocab_base, **stream.vocab_emer})
        l2c = info["leaf_to_cid"]
        gaps.extend(int(ev.t_declare - ev.t_start) for ev in stream.events)
        for m in METHODS:
            typed = (m == "ontoevolve")
            al = raw.get((r, m), [])
            ref = match_events(stream.events, al, l2c, delay_tol=TOL, typed=typed)
            f_sym, rr = match_detail(stream.events, al, l2c, tol=TOL, typed=typed, causal=False)
            f_cau, _ = match_detail(stream.events, al, l2c, tol=TOL, typed=typed, causal=True)
            f_caus, _ = match_detail(stream.events, al, l2c, tol=TOL, typed=typed,
                                     causal=True, ref="start")
            maxdiff = max(maxdiff, abs(f_sym - ref.macro_f1))
            rows[m].extend(rr)
            sym_f1[m].append(f_sym)
            cau_f1[m].append(f_cau)
            cau_start_f1[m].append(f_caus)
        print(f"[repeat {r}] done", flush=True)

    out = {"selfcheck_max_abs_diff_vs_eval_drift": maxdiff,
           "declare_minus_start": {"hist": {str(k): gaps.count(k) for k in sorted(set(gaps))},
                                   "mean": round(float(np.mean(gaps)), 3), "n": len(gaps)}}
    for m in METHODS:
        rr = rows[m]
        n = len(rr)
        pre = sum(1 for x in rr if x["d_start"] < 0)
        early = sum(1 for x in rr if x["d_start"] >= 0 and x["d_declare"] > 0)
        dd = [x["d_declare"] for x in rr]
        ds = [x["d_start"] for x in rr]
        cs = bootstrap_ci(sym_f1[m], n_boot=1000, seed=params.seed)
        cc = bootstrap_ci(cau_f1[m], n_boot=1000, seed=params.seed)
        ccs = bootstrap_ci(cau_start_f1[m], n_boot=1000, seed=params.seed)
        out[m] = {
            "hits": n,
            "pre_start": pre, "pre_start_share": round(pre / n, 4) if n else None,
            "early_but_valid": early, "early_but_valid_share": round(early / n, 4) if n else None,
            "on_or_after": n - pre - early,
            "mean_offset": round(float(np.mean(dd)), 3) if n else None,
            "median_offset": int(np.median(dd)) if n else None,
            "mean_offset_from_start": round(float(np.mean(ds)), 3) if n else None,
            "offset_hist": {str(k): dd.count(k) for k in range(-TOL, TOL + 1)},
            "sym_f1_mean": round(cs["mean"], 4), "sym_ci95": [round(cs["lo"], 4), round(cs["hi"], 4)],
            "causal_f1_mean": round(cc["mean"], 4), "causal_ci95": [round(cc["lo"], 4), round(cc["hi"], 4)],
            "causal_start_f1_mean": round(ccs["mean"], 4),
            "causal_start_ci95": [round(ccs["lo"], 4), round(ccs["hi"], 4)],
            "per_repeat_sym": [round(x, 4) for x in sym_f1[m]],
            "per_repeat_causal": [round(x, 4) for x in cau_f1[m]],
            "per_repeat_causal_start": [round(x, 4) for x in cau_start_f1[m]],
        }
    out["_wilcoxon_sym"] = wilcoxon_paired(sym_f1["ontoevolve"], sym_f1["adwin"])
    out["_wilcoxon_causal"] = wilcoxon_paired(cau_f1["ontoevolve"], cau_f1["adwin"])
    out["_wilcoxon_causal_start"] = wilcoxon_paired(cau_start_f1["ontoevolve"],
                                                    cau_start_f1["adwin"])

    dest = res / "offset_detail.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n[self-check] max |local - eval_drift| = {maxdiff:.2e}")
    print(f"t_declare - t_start: mean={np.mean(gaps):.2f} hist={out['declare_minus_start']['hist']}")
    print(f"\n{'method':<14}{'hits':>5}{'pre-start':>11}{'early-valid':>13}{'on/after':>9}"
          f"{'offset':>8}{'symF1':>7}{'cauF1':>7}{'cauStart':>9}")
    for m in METHODS:
        v = out[m]
        print(f"{m:<14}{v['hits']:>5}{v['pre_start_share']:>11.3f}{v['early_but_valid_share']:>13.3f}"
              f"{v['on_or_after']:>9}{v['mean_offset']:>8.2f}{v['sym_f1_mean']:>7.3f}{v['causal_f1_mean']:>7.3f}"
              f"{v['causal_start_f1_mean']:>9.3f}")
    print("[wilcoxon] causal(start-ref) vs adwin:", out["_wilcoxon_causal_start"])
    print("->", dest)


if __name__ == "__main__":
    main()
