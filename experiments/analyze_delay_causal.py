# -*- coding: utf-8 -*-
"""匹配偏移的因果口径分析：对称窗与因果窗下的 macro-F1，逐重复记录。

需要嵌入服务（重建受控流与 O_0）；离线复核可用已归档的 results/delay_causal.json。

与 src.eval_drift.match_events 同构的本地实现，额外返回逐条 signed offset，
以便区分：对称口径（正文现状，|offset|<=tol）与因果口径（0<=offset<=tol）。

自检：本地 sym 实现的 macro-F1 必须与 match_events 一致（误差 <1e-9）。
"""
from __future__ import annotations

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


def match_local(events, alarms, leaf_to_cid, *, tol, typed, causal=False):
    used = [False] * len(alarms)
    tp_by = {t: 0 for t in TYPES}
    fn_by = {t: 0 for t in TYPES}
    delays, ev_hit = [], []
    for ev in events:
        hit = None
        for j, a in enumerate(alarms):
            if used[j]:
                continue
            d = a.t - ev.t_declare
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
            ev_hit.append(0)
        else:
            used[hit] = True
            tp_by[ev.dtype] += 1
            delays.append(int(alarms[hit].t - ev.t_declare))
            ev_hit.append(1)
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
    return float(np.mean(f1s)), delays


def main():
    raw = {}
    for line in (ROOT / "results" / "alarms_cd99.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        raw.setdefault((d["repeat"], d["method"]), []).append(
            Alarm(t=int(d["t"]), cid=str(d["cid"]), dtype=str(d["type"]), score=0.0))

    params = C.PARAMS
    emb = Embedder(allow_fallback=False, verbose=False)

    acc = {m: {"sym_f1": [], "cau_f1": [], "offsets": [], "cau_offsets": [],
               "n_hit": [], "n_events": []} for m in METHODS}
    maxdiff = 0.0

    for r in range(10):
        seed_r = params.seed + r
        stream = make_controlled_stream(emb, seed=seed_r, params=params)
        onto, info = build_initial_ontology(
            stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
            vocab={**stream.vocab_base, **stream.vocab_emer})
        l2c = info["leaf_to_cid"]
        for m in METHODS:
            typed = (m == "ontoevolve")
            al = raw.get((r, m), [])
            ref = match_events(stream.events, al, l2c, delay_tol=TOL, typed=typed)
            sym_f1, sym_delays = match_local(stream.events, al, l2c, tol=TOL, typed=typed, causal=False)
            cau_f1, cau_delays = match_local(stream.events, al, l2c, tol=TOL, typed=typed, causal=True)
            maxdiff = max(maxdiff, abs(sym_f1 - ref.macro_f1))
            acc[m]["sym_f1"].append(sym_f1)
            acc[m]["cau_f1"].append(cau_f1)
            acc[m]["offsets"].extend(sym_delays)
            acc[m]["cau_offsets"].extend(cau_delays)
            acc[m]["n_hit"].append(len(sym_delays))
            acc[m]["n_events"].append(len(stream.events))
        print(f"[seed {seed_r}] done", flush=True)

    print(f"\n[self-check] max |local sym F1 - match_events F1| = {maxdiff:.3e}")

    out = {"_selfcheck_max_abs_diff": maxdiff}
    print(f"\n{'method':<14}{'symF1':>8}{'cauF1':>8}{'hit':>6}{'early':>7}{'sync':>6}{'late':>6}"
          f"{'symmean':>9}{'caumean':>9}")
    for m in METHODS:
        o = np.array(acc[m]["offsets"])
        n_hit, n_ev = sum(acc[m]["n_hit"]), sum(acc[m]["n_events"])
        early, sync, late = int((o < 0).sum()), int((o == 0).sum()), int((o > 0).sum())
        cs = bootstrap_ci(acc[m]["sym_f1"], n_boot=1000, seed=params.seed)
        cc = bootstrap_ci(acc[m]["cau_f1"], n_boot=1000, seed=params.seed)
        out[m] = {
            "sym_f1_mean": round(cs["mean"], 4), "sym_ci95": [round(cs["lo"], 4), round(cs["hi"], 4)],
            "causal_f1_mean": round(cc["mean"], 4), "causal_ci95": [round(cc["lo"], 4), round(cc["hi"], 4)],
            "hits_sym": n_hit, "events": n_ev,
            "early": early, "sync": sync, "late": late,
            "early_share": round(early / n_hit, 4) if n_hit else None,
            "mean_offset_sym": round(float(o.mean()), 3) if n_hit else None,
            "mean_offset_nonneg": round(float(o[o >= 0].mean()), 3) if (o >= 0).any() else None,
            "offset_hist": {str(k): int((o == k).sum()) for k in range(-TOL, TOL + 1)},
            "per_repeat_sym": [round(x, 4) for x in acc[m]["sym_f1"]],
            "per_repeat_causal": [round(x, 4) for x in acc[m]["cau_f1"]],
        }
        print(f"{m:<14}{cs['mean']:>8.3f}{cc['mean']:>8.3f}{n_hit:>6}{early:>7}{sync:>6}{late:>6}"
              f"{o.mean():>9.2f}{(o[o>=0].mean() if (o>=0).any() else float('nan')):>9.2f}")

    out["_wilcoxon_sym"] = wilcoxon_paired(acc["ontoevolve"]["sym_f1"], acc["adwin"]["sym_f1"])
    out["_wilcoxon_causal"] = wilcoxon_paired(acc["ontoevolve"]["cau_f1"], acc["adwin"]["cau_f1"])
    out["_n_events_per_repeat"] = acc["ontoevolve"]["n_events"]
    (ROOT / "results" / "delay_causal.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", ROOT / "results" / "delay_causal.json")


if __name__ == "__main__":
    main()
