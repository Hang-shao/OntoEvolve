# -*- coding: utf-8 -*-
"""B2：post-hoc 告警预算敏感性（不重跑检测器，不改在线行为）。

协议（与 `V24_B_告警预算协议草案.md` 一致，先定后跑）
------------------------------------------------------
* 口径：**每条流**上限 k，跨批累计；k = 1, 2, 3, 5, 8, 10, 15, 20 与无上限。
* 起点：Section 4.3 评测阶段（抑制窗口 99）的归档告警，六个方法同一批流。
* 排序：**到达顺序**（批次 t 升序），并列按 (t, cid, type) 字典序。
  **不使用任何 `Alarm.score`**，因为各方法的 score 量纲不同（融合分 / 计数比 / 距离 /
  微簇大小），跨方法排序会把"score 恰好小"当成"更不重要"。
* 信息约束：截断只看告警自身的批次与标识，**不看真值标签**，因此不含未来信息。
* 第二条规则：先按概念去重（每概念每流一条，取最早），再截断。
* 指标：macro-F1（与主表同口径）、precision = tp/(tp+fp)、recall = tp/(tp+fn)、告警数。

命名：post-hoc alert-budget sensitivity，**不是** online shared alert budget。

Run:
    python experiments/run_alert_budget.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import config as C  # noqa: E402
from src.build_ontology import build_initial_ontology  # noqa: E402
from src.controlled_stream import make_controlled_stream  # noqa: E402
from src.embedding import Embedder  # noqa: E402
from src.eval_drift import bootstrap_ci, match_events, wilcoxon_paired  # noqa: E402

METHODS = ["ontoevolve", "adwin", "inc_clustering", "page_hinkley", "embdist", "embkmeans"]
BUDGETS = [1, 2, 3, 5, 8, 10, 15, 20, 35, 10 ** 9]
DELAY_TOL = 3
SEED = 42
RES = ROOT / "results"


def load_alarms(path: Path) -> dict:
    from src.drift_detect import Alarm
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.setdefault((int(d["repeat"]), str(d["method"])), []).append(
            Alarm(t=int(d["t"]), cid=str(d["cid"]), dtype=str(d["type"]), score=0.0))
    # 到达顺序 = 归档中记录的顺序（同一批内即检测器实际抛出的顺序）。
    # 不按 cid 重排：那会改变同批内的先后，使"前 k 条"不再是真正的先到者。
    # 该顺序只来自告警自身，不含任何真值信息。
    return out


def dedup_concept(alarms: list) -> list:
    seen, out = set(), []
    for a in alarms:
        if a.cid in seen:
            continue
        seen.add(a.cid)
        out.append(a)
    return out


def apply_rule(alarms: list, budget: int, dedup: bool) -> list:
    seq = dedup_concept(alarms) if dedup else list(alarms)
    return seq if budget >= 10 ** 9 else seq[:budget]


def main() -> int:
    emb = Embedder(allow_fallback=False, verbose=False)
    streams = []
    for r in range(10):
        st = make_controlled_stream(emb, seed=SEED + r, params=C.PARAMS)
        _, info = build_initial_ontology(
            st.pool, st.d0_iids, seed=0, k_gt=len(st.vocab_base),
            vocab={**st.vocab_base, **st.vocab_emer})
        streams.append((st, info["leaf_to_cid"]))
    print(f"[setup] 10 streams, api_calls={emb.n_api_calls}", flush=True)

    alarms = load_alarms(RES / "alarms_cd99.jsonl")
    acc: dict = {}
    for r, (st, l2c) in enumerate(streams):
        for m in METHODS:
            typed = (m == "ontoevolve")
            for dedup in (False, True):
                for b in BUDGETS:
                    al = apply_rule(alarms.get((r, m), []), b, dedup)
                    rr = match_events(st.events, al, l2c, delay_tol=DELAY_TOL, typed=typed)
                    tp, fp, fn = int(rr.tp), int(rr.fp), int(rr.fn)
                    rec = acc.setdefault((m, dedup, b), {"f1": [], "prec": [], "rec": [],
                                                         "na": [], "tp": [], "fp": [], "fn": []})
                    rec["f1"].append(float(rr.macro_f1))
                    rec["prec"].append(tp / (tp + fp) if tp + fp else float("nan"))
                    rec["rec"].append(tp / (tp + fn) if tp + fn else float("nan"))
                    rec["na"].append(len(al))
                    rec["tp"].append(tp)
                    rec["fp"].append(fp)
                    rec["fn"].append(fn)
        if r % 3 == 0:
            print(f"[repeat {r}] done", flush=True)

    def agg(v):
        return float(np.mean([x for x in v if not np.isnan(x)])) if any(
            not np.isnan(x) for x in v) else float("nan")

    out: dict = {"protocol": {
        "level": "post-hoc alert-budget sensitivity (not an online shared budget)",
        "unit": "per stream, cumulative across batches",
        "order": ("arrival order as recorded in the archive: batch order, and "
                  "within a batch the order in which the detector raised them"),
        "uses_alarm_score": False,
        "uses_ground_truth": False,
        "start": "evaluated alarm streams (suppression window 99), Section 4.3",
        "matching": f"delay_tol={DELAY_TOL}, typed for OntoEvolve, untyped for baselines",
    }, "methods": {}}

    for m in METHODS:
        out["methods"][m] = {"arrival_cap": {}, "dedup_then_cap": {}}
        for dedup, key in ((False, "arrival_cap"), (True, "dedup_then_cap")):
            for b in BUDGETS:
                rec = acc[(m, dedup, b)]
                ci = bootstrap_ci(rec["f1"], n_boot=1000, seed=SEED)
                out["methods"][m][key][str(b)] = {
                    "macro_f1_mean": agg(rec["f1"]),
                    "macro_f1_ci95": [ci["lo"], ci["hi"]],
                    "precision_mean": agg(rec["prec"]),
                    "recall_mean": agg(rec["rec"]),
                    "alarms_mean": agg(rec["na"]),
                    "tp_mean": agg(rec["tp"]),
                    "fp_mean": agg(rec["fp"]),
                    "fn_mean": agg(rec["fn"]),
                    "per_repeat": rec["f1"],
                }

    # 每个预算下：本方法对最强基线的领先（用 F1 比值）
    lead = {}
    for b in BUDGETS:
        ours = out["methods"]["ontoevolve"]["arrival_cap"][str(b)]["macro_f1_mean"]
        best = max(METHODS[1:], key=lambda m: out["methods"][m]["arrival_cap"][str(b)]["macro_f1_mean"])
        lead[str(b)] = {"best_baseline": best,
                        "best_f1": out["methods"][best]["arrival_cap"][str(b)]["macro_f1_mean"],
                        "ratio": ours / out["methods"][best]["arrival_cap"][str(b)]["macro_f1_mean"]}
    out["lead_at_budget"] = lead

    # 每个预算下的配对检验：本方法 vs ADWIN，以及 vs 该预算下最强的基线。
    # 只用每个重复的 macro-F1（已归档于 per_repeat），不引入任何新信息。
    for b in BUDGETS:
        ours = out["methods"]["ontoevolve"]["arrival_cap"][str(b)]["per_repeat"]
        best = lead[str(b)]["best_baseline"]
        for tag, other in (("adwin", "adwin"), ("best", best)):
            ref = out["methods"][other]["arrival_cap"][str(b)]["per_repeat"]
            w = wilcoxon_paired(ours, ref)
            out["methods"]["ontoevolve"]["arrival_cap"][str(b)][
                f"paired_vs_{tag}"] = {"against": other, "p": w["p"],
                                       "wins": int(sum(x > y for x, y in zip(ours, ref))),
                                       "losses": int(sum(x < y for x, y in zip(ours, ref))),
                                       "n": len(ours)}

    # --- 与已有 budget_analysis.json 对账（cap 5/10/20/35，同一归档同一口径）
    ref_p = RES / "budget_analysis.json"
    if ref_p.exists():
        ref = json.loads(ref_p.read_text("utf-8"))
        worst = 0.0
        for b in (5, 10, 20, 35):
            for m in METHODS:
                if str(b) not in ref.get(m, {}):
                    continue
                want = float(ref[m][str(b)]["macro_f1_mean"])
                got = out["methods"][m]["arrival_cap"][str(b)]["macro_f1_mean"]
                worst = max(worst, abs(want - got))
        print(f"[check] 与 budget_analysis.json 最大偏差 {worst:.2e}")
        out["reconcile_budget_analysis_max_abs_diff"] = worst

    (RES / "alert_budget.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                           encoding="utf-8")

    print()
    print(f"{'budget':>7} " + "".join(f"{m[:9]:>10}" for m in METHODS) + "   best/ratio")
    for b in BUDGETS:
        row = f"{b if b < 10**9 else 'inf':>7} "
        for m in METHODS:
            row += f"{out['methods'][m]['arrival_cap'][str(b)]['macro_f1_mean']:>10.3f}"
        ld = lead[str(b)]
        row += f"   {ld['best_baseline'][:6]} {ld['ratio']:.2f}x"
        print(row)
    print("\nPaired tests against ADWIN (arrival cap):")
    for b in BUDGETS:
        v = out["methods"]["ontoevolve"]["arrival_cap"][str(b)]["paired_vs_adwin"]
        tag = "inf" if b >= 10 ** 9 else str(b)
        print(f"  k={tag:<5} p={v['p']:.3g} wins={v['wins']}/{v['n']} "
              f"losses={v['losses']}")

    print("\nPrecision / recall at budget 10 (arrival cap):")
    for m in METHODS:
        v = out["methods"][m]["arrival_cap"]["10"]
        print(f"  {m:<15} P={v['precision_mean']:.3f} R={v['recall_mean']:.3f} "
              f"F1={v['macro_f1_mean']:.3f} alarms={v['alarms_mean']:.1f}")
    print(f"\n-> {RES / 'alert_budget.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
