# -*- coding: utf-8 -*-
"""重定义子类型的受控扩展（ESM S16 / Table S21）。

回答 S12 只能"手工构造"的问题：把 range-only / DP-value-only / cardinality-only 三个族
真的注入到 Track 1 的数据流里，公理通道会转出多少条重定义告警？

口径（与 S13 一致，只是子类型不同）：
  * 真值 = 生成器放在 DriftEvent 上的 dtype + subtype；
  * d_ax 走检测器自身门控通路（`DriftDetector` 的每批每概念 `signals["ax"]`）；
  * 阈值取 θ_red^eff，τ_conf 取标定值；
  * 检测参数固定为 §4.2.3 的标定值，**不在变体流上重新调参**；
  * 每个族 10 条流（seeds 42–51），每条流 3 个重定义事件 → 每族 30 个事件。

`default` 族作为对照跑同一条代码路径，必须复现 S13 的 23/30 与归档逐位相同的重定义 F1。

用法：
    python experiments/run_redefinition_subtypes.py
    python experiments/run_redefinition_subtypes.py --seeds 42,43 --families default,dp_only
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
from src.controlled_stream import (make_controlled_stream,  # noqa: E402
                                   subtype_alt_schema)
from src.drift_detect import DriftDetector               # noqa: E402
from src.embedding import Embedder                       # noqa: E402
from src.local_embed import LocalEmbedder                # noqa: E402
from src.eval_drift import match_events                  # noqa: E402
from src.extract import extract_candidates               # noqa: E402
from src.ontology_model import Concept, Ontology          # noqa: E402

FAMILIES = ("default", "range_only", "dp_only", "card_only")


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


def signature_of(onto, cid: str, axioms) -> "np.ndarray":
    """把一组公理装进临时本体，算它们的签名（只用于隔离"变异本身"的距离）。"""
    tmp = Ontology(op_vocab=onto.op_vocab, dp_vocab=onto.dp_vocab)
    tmp.add(Concept(cid=cid, name=cid, axioms=list(axioms)))
    return tmp.axiom_signature(cid)


def mutation_delta(onto, schema, cid: str, subtype: str, parent: str, *, k: int,
                   transitive_ops, disjoint) -> float:
    """只把槽位模式换成该子类型的 alt 版本，量签名移动了多少。

    与流内 d_ax 不同：这里不做投影、不含 support decay 与证据稀疏的副作用，
    所以它回答的是"这个族本身能不能进签名"，而不是"这一批检测器看到了什么"。
    """
    import numpy as np
    alt = subtype_alt_schema(schema, subtype or "default")
    base_items = [{"text": "", "op_slots": [tuple(x) for x in schema.ops],
                   "dp_slots": [tuple(x) for x in schema.dps]} for _ in range(k)]
    alt_items = [{"text": "", "op_slots": [tuple(x) for x in alt.alt_ops],
                  "dp_slots": [tuple(x) for x in alt.dps]} for _ in range(k)]
    kw = dict(parent=parent, transitive_ops=transitive_ops, disjoint=disjoint)
    a = extract_candidates(cid, base_items, **kw)
    b = extract_candidates(cid, alt_items, **kw)
    return float(np.abs(signature_of(onto, cid, b) - signature_of(onto, cid, a)).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--out", default=str(ROOT / "results" / "redefinition_subtypes.json"))
    ap.add_argument("--embedder", default="local", choices=("local", "api"),
                    help="子类型变体的文本不在缓存里，需要真正嵌入：默认用本地同模型推理"
                         "（与 API 缓存同键；API 端点自 2026-11-03 起 401）")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for f in families:
        assert f in FAMILIES, f

    params, link_offset = load_best()
    arch = json.loads((ROOT / "results" / "baseline_protocol_detail_all.json").read_text("utf-8"))
    arch_f1 = arch["G6_main_candidate"]["methods"]["ontoevolve"]["per_type"]["redefinition"]
    emb = (LocalEmbedder(cache_dir=C.DIR_CACHE, verbose=False) if args.embedder == "local"
           else Embedder(allow_fallback=False, verbose=False))

    out = {"protocol": {"seeds": seeds, "families": families,
                        "theta_red_eff": None, "tau_conf": float(params.tau_conf),
                        "ground_truth": "constructor-injected DriftEvent.dtype and .subtype",
                        "distance": "L1 between axiom_signature(Project(O_t, delta_A), c) and "
                                    "axiom_signature(O_t, c), the detector's own path",
                        "tuning": "the calibrated parameters of Section 4.2.3, held fixed; "
                                  "nothing is re-tuned on the variant streams",
                        "embedder": ("local inference of the same model for the new alt texts, "
                                     "sharing the disk cache with the API vectors"
                                     if args.embedder == "local" else "API"),
                        "redefinitions_per_stream": 3},
           "families": {}, "per_event": [], "reconciliation": {}}

    for fam in families:
        n_events = crossed = 0
        typed_on_event = typed_total = 0
        max_ax_all, axiom_change = [], {"op": 0, "range": 0, "dp": 0, "card": 0}
        mutation_deltas: list = []
        red_f1s, macros, theta = [], [], set()
        f1_check = []
        for i, seed in enumerate(seeds):
            n0, c0 = n_events, crossed
            st = make_controlled_stream(
                emb, seed=seed, params=C.PARAMS,
                redef_subtypes=None if fam == "default" else [fam])
            onto, info = build_initial_ontology(
                st.pool, st.d0_iids, seed=0, k_gt=len(st.vocab_base),
                vocab={**st.vocab_base, **st.vocab_emer})
            l2c = info["leaf_to_cid"]
            p = params.copy(link_sim=max(0.40, float(info["link_sim_p05"]) - link_offset))
            det = DriftDetector(onto, p, support_decay=0.5, cooldown=99)
            alarms = []
            for t, batch in enumerate(st.batches, start=1):
                alarms.extend(det.observe(t, batch))
            theta.add(round(float(det.theta_red_eff), 4))

            res = match_events(st.events, alarms, l2c, delay_tol=p.delay_tol)
            red_f1 = float(res.per_type["redefinition"]["f1"])
            red_f1s.append(red_f1)
            macros.append(float(res.macro_f1))
            if fam == "default" and i < len(arch_f1):
                f1_check.append(abs(red_f1 - float(arch_f1[i])) < 1e-9)

            typed = [a for a in alarms if a.dtype == "redefinition"]
            typed_total += len(typed)

            for e in st.events:
                if e.dtype != "redefinition":
                    continue
                n_events += 1
                cid = l2c.get(e.leaf)
                rows = [r for r in det.log if r["cid"] == cid and r["t"] >= e.t_start]
                hit = [r for r in rows if r["signals"]["ax"] > det.theta_red_eff
                       and r["signals"]["mean_conf"] >= p.tau_conf]
                max_ax = max([r["signals"]["ax"] for r in rows] or [0.0])
                max_ax_all.append(round(max_ax, 3))
                on_event = [a for a in typed if a.cid == cid
                            and abs(a.t - e.t_declare) <= p.delay_tol]
                sch = st.vocab_base[str(e.leaf)]
                conc0 = onto.get(cid)
                k_alt = 0
                for t, batch in enumerate(st.batches, start=1):
                    if t >= e.t_start:
                        k_alt = len([o for o in batch if o.leaf == e.leaf
                                     and o.variant == "alt"])
                        if k_alt:
                            break
                mut = mutation_delta(
                    onto, sch, cid, e.subtype, conc0.parent, k=max(1, k_alt),
                    transitive_ops=[a.pred for a in conc0.axioms if a.kind == "TRANSITIVE"],
                    disjoint=[a.obj for a in conc0.axioms if a.kind == "DISJOINT"])
                mutation_deltas.append(round(mut, 3))
                if hit:
                    crossed += 1
                if on_event:
                    typed_on_event += 1

                # 该族的变更是否真的出现在投影公理里（不靠签名距离判断）
                changed = set()
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
                    changed = {k for k in ("op", "range", "dp", "card")
                               if before[k] != after[k]}
                    break
                for k in changed:
                    axiom_change[k] += 1

                out["per_event"].append({
                    "family": fam, "seed": seed, "leaf": e.leaf, "cid": cid,
                    "subtype": e.subtype, "t_start": e.t_start, "t_declare": e.t_declare,
                    "max_ax": round(max_ax, 3), "mutation_delta": round(mut, 3),
                    "signature_crossed": bool(hit),
                    "axiom_changed": sorted(changed),
                    "red_alarm_on_event": bool(on_event)})
            print(f"[{fam} seed {seed}] crossed={crossed - c0}/{n_events - n0} "
                  f"red_f1={red_f1:.3f} macro={macros[-1]:.3f}", flush=True)

        out["families"][fam] = {
            "n_events": n_events,
            "signature_crossed": crossed,
            "converted_rate": crossed / n_events if n_events else None,
            "typed_redefinition_alarms": typed_total,
            "typed_alarm_on_event": typed_on_event,
            "axiom_change": axiom_change,
            "mutation_delta_values": sorted(set(mutation_deltas)),
            "mutation_delta_median": (sorted(mutation_deltas)[len(mutation_deltas) // 2]
                                      if mutation_deltas else None),
            "max_ax_values": sorted(set(max_ax_all)),
            "max_ax_histogram": {str(v): max_ax_all.count(v) for v in sorted(set(max_ax_all))},
            "redefinition_f1_mean": sum(red_f1s) / len(red_f1s),
            "macro_f1_mean": sum(macros) / len(macros),
            "redefinition_f1_per_seed": [round(x, 4) for x in red_f1s],
        }
        if fam == "default":
            out["reconciliation"] = {
                "default_crossed": crossed,
                "default_matches_s13_crossed_23": crossed == 23 and n_events == 30,
                "default_red_f1_matches_archive": all(f1_check) and len(f1_check) > 0,
                "n_seeds_checked": len(f1_check)}
        out["protocol"]["theta_red_eff"] = sorted(theta)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({f: {k: out["families"][f][k] for k in
                         ("n_events", "signature_crossed", "axiom_change",
                          "max_ax_values", "redefinition_f1_mean", "macro_f1_mean")}
                      for f in out["families"]}, ensure_ascii=False, indent=1))
    print("reconciliation", out["reconciliation"])
    print("写入", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
