"""重定义覆盖的四格账本（ESM S13 / Table S17）——受控轨道上的事件级真值统计。

四格：
  ① 签名变 × 真重定义   —— 覆盖（公理通道可检出）
  ② 签名不变 × 真重定义 —— 签名碰撞（该通道漏报）
  ③ 签名变 × 非重定义   —— 误报侧信号（阈值被跨过的概念批次数）
  ④ 均无变化 × 非重定义 —— 无信号

真值由生成器构造（`DriftEvent.dtype == "redefinition"`），不由公理变化反推；d_ax 走检测器
自身通路（`DriftDetector` 每批每概念记录 `ax` 与 `mean_conf`），阈值取 θ_red^eff。

同时输出三项对账：
  1. 同一跑法下的 per-type 重定义 F1（type-agnostic 匹配，与正文同口径）；
  2. 重定义**类型**告警里有多少条真的落在重定义事件上（typed 精度）；
  3. 与归档 `baseline_protocol_detail_all.json` 的每种子重定义 F1 逐位比较。

用法：
    python experiments/run_redefinition_coverage.py
    python experiments/run_redefinition_coverage.py --seeds 42,43
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C                              # noqa: E402
from src.build_ontology import build_initial_ontology    # noqa: E402
from src.controlled_stream import make_controlled_stream  # noqa: E402
from src.drift_detect import DriftDetector               # noqa: E402
from src.embedding import Embedder                       # noqa: E402
from src.eval_drift import match_events                  # noqa: E402
from src.extract import extract_candidates               # noqa: E402


def load_best() -> tuple[C.DriftParams, float]:
    d = json.loads((ROOT / "results/exp1_summary.json").read_text("utf-8"))
    cal = d["calibration"]
    fields = set(C.DriftParams.__annotations__)
    params = C.PARAMS.copy(**{k: v for k, v in cal["params"].items() if k in fields})
    return params, float(cal["link_offset"])


def axiom_kinds(onto, cid: str) -> dict:
    ops, rng, dps, card = set(), set(), set(), set()
    for a in onto.axioms_of(cid):
        if a.kind == "OP":
            ops.add(a.pred)
            rng.add((a.pred, a.obj))
        elif a.kind == "DP":
            dps.add((a.pred, a.obj))
        elif a.kind == "CARD":
            card.add((a.card_min, a.card_max))
    return {"op": ops, "range": rng, "dp": dps, "card": card}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--out", default=str(ROOT / "results" / "redefinition_coverage.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    params, link_offset = load_best()
    arch = json.loads((ROOT / "results" / "baseline_protocol_detail_all.json").read_text("utf-8"))
    arch_f1 = arch["G6_main_candidate"]["methods"]["ontoevolve"]["per_type"]["redefinition"]
    emb = Embedder(allow_fallback=False, verbose=False)

    cells = {"detected": 0, "collision": 0, "false_signal": 0, "no_signal": 0}
    fam = {"op_changed": 0, "range_changed": 0, "dp_changed": 0, "card_changed": 0}
    per_seed, theta_eff = [], set()
    typed_alarms_total = typed_on_events = 0
    f1_checks = []
    attribution = {"hit": 0, "same_concept_late": 0, "other_concept": 0}

    for i, seed in enumerate(seeds):
        st = make_controlled_stream(emb, seed=seed, params=C.PARAMS)
        onto, info = build_initial_ontology(
            st.pool, st.d0_iids, seed=0, k_gt=len(st.vocab_base),
            vocab={**st.vocab_base, **st.vocab_emer})
        l2c = info["leaf_to_cid"]
        p = params.copy(link_sim=max(0.40, float(info["link_sim_p05"]) - link_offset))
        det = DriftDetector(onto, p, support_decay=0.5, cooldown=99)
        alarms = []
        for t, batch in enumerate(st.batches, start=1):
            alarms.extend(det.observe(t, batch))
        theta_eff.add(round(float(det.theta_red_eff), 4))

        # 对账 1：与正文同口径的 per-type 重定义 F1
        res = match_events(st.events, alarms, l2c, delay_tol=p.delay_tol)
        red = res.per_type["redefinition"]
        if i < len(arch_f1):
            f1_checks.append(abs(float(red["f1"]) - float(arch_f1[i])) < 1e-9)

        # 真值：生成器构造的重定义事件
        red_events = [e for e in st.events if e.dtype == "redefinition"]
        red_cids = {l2c[e.leaf]: e for e in red_events if e.leaf in l2c}

        # ①②：签名是否在事件起点之后跨过阈值
        events_out = []
        for cid, e in red_cids.items():
            rows = [r for r in det.log if r["cid"] == cid and r["t"] >= e.t_start]
            hit = [r for r in rows if r["signals"]["ax"] > det.theta_red_eff
                   and r["signals"]["mean_conf"] >= p.tau_conf]
            events_out.append({"cid": cid, "leaf": e.leaf, "t_start": e.t_start,
                               "t_declare": e.t_declare,
                               "max_ax": round(max([r["signals"]["ax"] for r in rows] or [0.0]), 3),
                               "signature_crossed": bool(hit),
                               "first_t": (hit[0]["t"] if hit else None)})
            cells["detected" if hit else "collision"] += 1

        # 类型为 redefinition 的告警：命中事件 / 同一概念但在窗外 / 别的概念
        typed = [a for a in alarms if a.dtype == "redefinition"]
        typed_alarms_total += len(typed)
        seed_attr = {"hit": 0, "same_concept_late": 0, "other_concept": 0}
        for a in typed:
            ev_cids = {l2c.get(e.leaf) for e in st.events if e.dtype == "redefinition"}
            if any(e.dtype == "redefinition" and abs(a.t - e.t_declare) <= p.delay_tol
                   and l2c.get(e.leaf) == a.cid for e in st.events):
                typed_on_events += 1
                seed_attr["hit"] += 1
            elif a.cid in ev_cids:
                seed_attr["same_concept_late"] += 1
            else:
                seed_attr["other_concept"] += 1
        for _k, _v in seed_attr.items():
            attribution[_k] += _v

        # 族的构成：alt 实例首次出现时，投影公理相对现存公理变了哪些维度
        for cid, e in red_cids.items():
            for t, batch in enumerate(st.batches, start=1):
                if t < e.t_start:
                    continue
                lst = [o for o in batch if o.leaf == e.leaf and o.variant == "alt"]
                if not lst:
                    continue
                conc = onto.get(cid)
                delta = extract_candidates(
                    cid, [o.as_dict() for o in lst], parent=conc.parent,
                    transitive_ops=[a.pred for a in conc.axioms if a.kind == "TRANSITIVE"],
                    disjoint=[a.obj for a in conc.axioms if a.kind == "DISJOINT"])
                proj = onto.project(delta, cid, tau_conf=p.tau_conf, support_decay=0.5)
                before, after = axiom_kinds(onto, cid), axiom_kinds(proj, cid)
                for key, name in (("op", "op_changed"), ("range", "range_changed"),
                                  ("dp", "dp_changed"), ("card", "card_changed")):
                    if before[key] != after[key]:
                        fam[name] += 1
                break

        # ③④：非重定义概念上的签名信号
        for r in det.log:
            if r["cid"] in red_cids:
                continue
            if r["signals"]["ax"] > det.theta_red_eff and r["signals"]["mean_conf"] >= p.tau_conf:
                cells["false_signal"] += 1
            else:
                cells["no_signal"] += 1

        per_seed.append({"seed": seed, "n_redefinition": len(red_events),
                         "signature_crossed": sum(1 for x in events_out if x["signature_crossed"]),
                         "red_typed_alarms": len(typed),
                         "red_typed_on_events": sum(
                             1 for a in typed
                             if any(e.dtype == "redefinition" and abs(a.t - e.t_declare) <= p.delay_tol
                                    and l2c.get(e.leaf) == a.cid for e in st.events)),
                         "reported_red_f1": round(float(red["f1"]), 4),
                         "archived_red_f1": (round(float(arch_f1[i]), 4)
                                             if i < len(arch_f1) else None),
                         "events": events_out})
        print(f"[seed {seed}] crossed={per_seed[-1]['signature_crossed']}/{len(red_events)} "
              f"red_alarms={len(typed)} on_events={per_seed[-1]['red_typed_on_events']} "
              f"f1={per_seed[-1]['reported_red_f1']} (archived {per_seed[-1]['archived_red_f1']})",
              flush=True)

    total = cells["detected"] + cells["collision"]
    out = {
        "protocol": {"seeds": seeds, "theta_red_eff": sorted(theta_eff),
                     "tau_conf": float(params.tau_conf),
                     "signature_components": int(onto.M),
                     "ground_truth": "constructor-injected DriftEvent.dtype",
                     "distance": "L1 between axiom_signature(Project(O_t, delta_A), c) and "
                                 "axiom_signature(O_t, c), the detector's own path"},
        "cells": cells,
        "totals": {"n_redefinition_events": total,
                   "coverage": cells["detected"] / total if total else None,
                   "collision_rate": cells["collision"] / total if total else None,
                   "no_signal_pairs": cells["no_signal"],
                   "non_redefinition_pairs": cells["false_signal"] + cells["no_signal"]},
        "typed_ledger": {"red_typed_alarms": typed_alarms_total,
                         "attribution": attribution,
                         "on_redefinition_events": typed_on_events,
                         "typed_precision": typed_on_events / max(1, typed_alarms_total)},
        "family_profile": fam,
        "reconciliation": {"f1_matches_archive": all(f1_checks), "n_seeds": len(f1_checks)},
        "missed_events_max_ax": sorted(round(e["max_ax"], 3) for ps in per_seed
                                       for e in ps["events"] if not e["signature_crossed"]),
        "per_seed": per_seed,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("cells", "totals", "typed_ledger",
                                          "family_profile", "reconciliation")},
                     ensure_ascii=False, indent=1))
    print("写入", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
