# -*- coding: utf-8 -*-
"""检测评分对匹配规则的敏感性：贪心（原协议）对最大基数匹配。

为什么需要这项分析
------------------
`src/eval_drift.py::match_events` 按事件声明顺序逐事件取“第一条仍空闲的可行
告警”，不是最大基数匹配。因此 TP 既是检测器的性质，也是这条规则的性质。
本脚本用**存档告警**（`results/alarms_cd99.jsonl`）重评分，不重跑任何检测器：

* greedy     —— 与 `match_events` 同构，先自检复现存档值（`results/offset_detail.json`）；
* maxcard    —— Kuhn 增广路的精确最大基数匹配。最大基数匹配的解不唯一，
               宏 F1 依赖 tie-break，故对随机事件顺序重复 `--tiebreak` 次，
               报告均值与[min, max]。

事件表不需要嵌入服务即可确定性重建
------------------------------------
`build_pool` 的结构（leaf/variant/iid）只由词表与 `pool_per_leaf` 决定，
与嵌入向量取值无关；`generate_stream` 的随机序列同理。故用确定性假嵌入即可
复现全部事件（leaf/dtype/t_start/t_declare）。脚本用 offset_detail.json 的
`declare_minus_start` 直方图作自检，不一致即失败退出。

前提：`leaf_to_cid`。O_0 由固定 `d0_iids` 构建，跨重复相同，因此取
ESM 2 的 `labels.csv`（cid → leaf 标签）即可，无需重新聚类。

Run:
    python experiments/analyze_match_sensitivity.py \
        --labels "../Discover Computing/ESM_2/labels.csv" \
        --out results/match_sensitivity.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C                      # noqa: E402
from src.controlled_stream import make_controlled_stream   # noqa: E402
from src.drift_detect import Alarm               # noqa: E402

TYPES = ["emergence", "extinction", "redefinition"]
METHODS = ["ontoevolve", "adwin", "inc_clustering", "page_hinkley",
           "embdist", "embkmeans"]
LABEL_CANDIDATES = [
    ROOT.parent / "Discover Computing" / "ESM_2" / "labels.csv",
    ROOT.parent / "ESM_2" / "labels.csv",
    Path(r"D:/code/AIP/动态本体构建/Discover Computing/ESM_2/labels.csv"),
]


class _DeterministicEmbedder:
    """确定性假嵌入：只用于重建事件表，向量值不参与任何计算。"""

    def __init__(self, dim: int = None):
        self.dim = int(dim or C.EMBED_DIM)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for s in texts:
            rng = np.random.default_rng(abs(hash(s)) % (2 ** 31))
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-12)
            out.append(v)
        return np.vstack(out)


def load_labels(path: Path) -> Dict[str, str]:
    return {row["leaf_term_label"]: row["cid"]
            for row in csv.DictReader(open(path, encoding="utf-8"))}


def load_alarms(path: Path) -> Dict[tuple, List[Alarm]]:
    raw: Dict[tuple, List[Alarm]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        raw.setdefault((d["repeat"], d["method"]), []).append(
            Alarm(t=int(d["t"]), cid=str(d["cid"]), dtype=str(d["type"]), score=0.0))
    return raw


def edge_ok(ev, a, l2c, typed, causal, tol) -> bool:
    d = a.t - ev.t_declare
    if causal:
        if d < 0 or d > tol:
            return False
    elif abs(d) > tol:
        return False
    if typed and a.dtype != ev.dtype:
        return False
    if ev.dtype == "emergence":
        return a.cid.startswith("NEW") if typed else True
    if ev.leaf not in l2c:
        return a.cid.startswith("NEW")
    return a.cid == l2c.get(ev.leaf)


def _assign(events, alarms, l2c, typed, causal, tol):
    """按有/无概念约束构造可行边集合。"""
    return [[j for j, a in enumerate(alarms) if edge_ok(ev, a, l2c, typed, causal, tol)]
            for ev in events]


def _score(events, alarms, used, l2c):
    """与 match_events 同构的评分：TP/FN 按事件，FP 按未用告警的类型归因。"""
    concept_type = {l2c[ev.leaf]: ev.dtype for ev in events if ev.leaf in l2c}
    tp_by = {t: 0 for t in TYPES}
    fn_by = {t: 0 for t in TYPES}
    for ev, j in zip(events, used):
        if j is None:
            fn_by[ev.dtype] += 1
        else:
            tp_by[ev.dtype] += 1
    taken = {j for j in used if j is not None}
    fp_by = {t: 0 for t in TYPES}
    fp_other = 0
    for j, a in enumerate(alarms):
        if j in taken:
            continue
        if a.cid.startswith("NEW"):
            fp_by["emergence"] += 1
        else:
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
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
    return float(np.mean(f1s)), int(sum(tp_by.values())), len(alarms) - len(taken)


def greedy_match(events, alarms, l2c, typed, causal, tol):
    used_a = [False] * len(alarms)
    out: List[Optional[int]] = []
    for ev in events:
        hit = None
        for j, a in enumerate(alarms):
            if not used_a[j] and edge_ok(ev, a, l2c, typed, causal, tol):
                hit = j
                break
        if hit is not None:
            used_a[hit] = True
        out.append(hit)
    return out


def maxcard_match(events, alarms, l2c, typed, causal, tol, order=None):
    adj = _assign(events, alarms, l2c, typed, causal, tol)
    match_a = [-1] * len(alarms)
    match_e = [-1] * len(events)
    order = list(range(len(events))) if order is None else list(order)

    def try_k(e, seen):
        for j in adj[e]:
            if seen[j]:
                continue
            seen[j] = True
            if match_a[j] == -1 or try_k(match_a[j], seen):
                match_a[j] = e
                match_e[e] = j
                return True
        return False

    for e in order:
        try_k(e, [False] * len(alarms))
    return match_e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alarms", default=str(ROOT / "results" / "alarms_cd99.jsonl"))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--offset-detail", default=str(ROOT / "results" / "offset_detail.json"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--tiebreak", type=int, default=200)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "match_sensitivity.json"))
    args = ap.parse_args()

    lab = Path(args.labels) if args.labels else next(
        (p for p in LABEL_CANDIDATES if p.exists()), None)
    if lab is None or not lab.exists():
        raise SystemExit("找不到 labels.csv；请用 --labels 指定 ESM 2 的 labels.csv")
    l2c = load_labels(lab)
    raw = load_alarms(Path(args.alarms))

    emb = _DeterministicEmbedder()
    streams = {r: make_controlled_stream(emb, seed=C.PARAMS.seed + r, params=C.PARAMS)
               for r in range(args.repeats)}

    # ---- 自检 1：事件表与存档的 t_declare - t_start 分布一致 -----------------
    ref = json.loads(Path(args.offset_detail).read_text(encoding="utf-8"))
    gaps = [int(e.t_declare - e.t_start) for st in streams.values() for e in st.events]
    mine = {str(k): gaps.count(k) for k in sorted(set(gaps))}
    ref_hist = {k: v for k, v in ref["declare_minus_start"]["hist"].items() if v}
    if mine != ref_hist:
        raise SystemExit(f"事件表重建与存档不符：{mine} != {ref_hist}")

    rng = np.random.default_rng(args.seed)
    out = {
        "config": {"alarms": str(Path(args.alarms).name),
                   "delay_tol": args.tol,
                   "repeats": args.repeats,
                   "tiebreak_draws": args.tiebreak,
                   "labels": lab.name},
        "selfcheck": {"declare_minus_start": mine,
                      "n_events": len(gaps),
                      "mean_declare_minus_start": round(float(np.mean(gaps)), 3)},
    }

    greedy_sym_all: Dict[str, float] = {}
    for m in METHODS:
        typed = (m == "ontoevolve")
        entry = {"typed": typed, "greedy": {}, "maxcard": {}}
        for causal, tag in ((False, "sym"), (True, "cau")):
            f_all, tp_all, fp_all = [], 0, 0
            for r in range(args.repeats):
                ev = streams[r].events
                al = raw.get((r, m), [])
                used = greedy_match(ev, al, l2c, typed, causal, args.tol)
                f, tp, fp = _score(ev, al, used, l2c)
                f_all.append(f)
                tp_all += tp
                fp_all += fp
            entry["greedy"][tag] = {
                "macro_f1_mean": round(float(np.mean(f_all)), 4),
                "macro_f1_ci95": [round(float(x), 4) for x in
                                  np.percentile(f_all, [2.5, 97.5])] if len(f_all) > 2 else None,
                "tp_total": tp_all, "fp_total": fp_all,
                "per_repeat": [round(x, 4) for x in f_all]}
            if tag == "sym":
                greedy_sym_all[m] = round(float(np.mean(f_all)), 4)

        for causal, tag in ((False, "sym"), (True, "cau")):
            draws_f, draws_tp = [], []
            for _ in range(args.tiebreak):
                tot_f, tot_tp = [], 0
                for r in range(args.repeats):
                    ev = streams[r].events
                    al = raw.get((r, m), [])
                    perm = rng.permutation(len(ev))
                    ev_p = [ev[i] for i in perm]
                    used = maxcard_match(ev_p, al, l2c, typed, causal, args.tol)
                    f, tp, _ = _score(ev_p, al, used, l2c)
                    tot_f.append(f)
                    tot_tp += tp
                draws_f.append(float(np.mean(tot_f)))
                draws_tp.append(tot_tp)
            entry["maxcard"][tag] = {
                "macro_f1_mean": round(float(np.mean(draws_f)), 4),
                "macro_f1_min": round(float(np.min(draws_f)), 4),
                "macro_f1_max": round(float(np.max(draws_f)), 4),
                "tp_total_mean": round(float(np.mean(draws_tp)), 1),
                "tp_total_min": int(np.min(draws_tp)),
                "tp_total_max": int(np.max(draws_tp)),
            }
        out[m] = entry

    # ---- 自检 2：贪心口径复现存档值 -----------------------------------------
    ref_sym = {m: ref[m]["sym_f1_mean"] for m in METHODS if m in ref}
    diff = max(abs(greedy_sym_all[m] - ref_sym[m]) for m in ref_sym)
    out["selfcheck"]["greedy_sym_vs_offset_detail_max_abs_diff"] = round(float(diff), 4)
    if diff > 5e-4:
        raise SystemExit(f"贪心口径未能复现存档值：{greedy_sym_all} != {ref_sym}")

    # ---- 涌现事件共享 NEW 告警池：竞争实例 ---------------------------------
    comp = {"sym_hits": 0, "cau_hits": 0, "n_emergence": 0, "examples": []}
    for r in range(args.repeats):
        ev = [e for e in streams[r].events if e.dtype == "emergence"]
        al = [a for a in raw.get((r, "ontoevolve"), []) if a.dtype == "emergence"]
        s = greedy_match(ev, al, l2c, True, False, args.tol)
        c = greedy_match(ev, al, l2c, True, True, args.tol)
        comp["sym_hits"] += sum(1 for j in s if j is not None)
        comp["cau_hits"] += sum(1 for j in c if j is not None)
        comp["n_emergence"] += len(ev)
        rows = []
        for ev_i, (js, jc) in enumerate(zip(s, c)):
            ev_x = ev[ev_i]
            rows.append({
                "leaf": ev_x.leaf, "t_declare": int(ev_x.t_declare),
                "sym": None if js is None else
                {"t": int(al[js].t), "offset": int(al[js].t - ev_x.t_declare)},
                "causal": None if jc is None else
                {"t": int(al[jc].t), "offset": int(al[jc].t - ev_x.t_declare)}})
        if sum(1 for j in s if j is None) and sum(1 for j in c if j is not None):
            comp["examples"].append({
                "repeat": r,
                "new_alarm_batches": [int(a.t) for a in al],
                "events": rows})

    out["emergence_competition"] = comp
    dest = Path(args.out)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[selfcheck] declare_minus_start = {mine} (n={len(gaps)})")
    print(f"[selfcheck] greedy sym vs offset_detail: max|diff| = {diff:.2e}")
    print(f"\n{'method':<16}{'greedy_sym':>11}{'max_mean':>10}{'max_min':>9}{'max_max':>9}"
          f"{'greedy_cau':>11}{'maxC_mean':>11}")
    for m in METHODS:
        e = out[m]
        print(f"{m:<16}{e['greedy']['sym']['macro_f1_mean']:>11.4f}"
              f"{e['maxcard']['sym']['macro_f1_mean']:>10.4f}"
              f"{e['maxcard']['sym']['macro_f1_min']:>9.4f}"
              f"{e['maxcard']['sym']['macro_f1_max']:>9.4f}"
              f"{e['greedy']['cau']['macro_f1_mean']:>11.4f}"
              f"{e['maxcard']['cau']['macro_f1_mean']:>11.4f}")
    print(f"\nemergence competition: sym {comp['sym_hits']}/{comp['n_emergence']}"
          f" -> causal {comp['cau_hits']}/{comp['n_emergence']}"
          f"  (examples recorded: {len(comp['examples'])})")
    print("->", dest)


if __name__ == "__main__":
    main()
