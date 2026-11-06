# -*- coding: utf-8 -*-
"""Major 4：公理签名的有损性实证 —— 构造 d_ax = 0 的碰撞族。

问题
----
Definition 2 承认签名 σ_O(c) 有损、非单射：它由计数与截断聚合构成，凡是不改变
计数与截断值的改动都不改变签名，于是“重定义”通道保持沉默。审稿人（Major 4）要求
把这个说法变成实证，而不是只写在定义里。

做法（不重跑检测器、不训练、不调用嵌入接口）
--------------------------------------------
* 用真实的 `Ontology.axiom_signature` 与 `L1_distance` 计算 d_ax；
* 阈值取检测器实际用的 θ_red^eff = θ_red · max(1, M/18)，M = 18 时即 1.00；
* 构造五族碰撞 + 两个对照（一个越阈、一个非零但低于阈值）；
* 再把碰撞族放到 O₀ 的 30 个概念上逐一验证“存在性”，并统计各族在 O₀ 里是否可达。

输出 `results/lossy_signature.json`。
对应论文 ESM 1 的 Section S12 与 Table S16。

Run:
    python experiments/run_lossy_signature.py
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.build_ontology import build_initial_ontology  # noqa: E402
from src.controlled_stream import make_controlled_stream  # noqa: E402
from src.embedding import Embedder  # noqa: E402
from src.ontology_model import Axiom, Concept, L1_distance, Ontology  # noqa: E402
from src.vocab import DP_VOCAB, OP_VOCAB  # noqa: E402

RES = ROOT / "results"
CID = "probe"


def sig(axioms, parent="Animal"):
    """用真实实现算签名（同一批词表，M = 18）。"""
    c = Concept(cid=CID, name="probe", parent=parent, axioms=list(axioms))
    o = Ontology({CID: c}, op_vocab=OP_VOCAB, dp_vocab=DP_VOCAB)
    return o.axiom_signature(CID)


def dax(a, b):
    return L1_distance(sig(a), sig(b))


def base_axioms():
    """一条接近受控轨道真实结构的公理集（两 OP、三 DP、一 DOMAIN、一 RANGE×2、
    一 CARD、一 TRANSITIVE、一 DISJOINT）。"""
    return [
        Axiom(kind="OP", subj=CID, pred="livesIn", obj="Forest"),
        Axiom(kind="OP", subj=CID, pred="eats", obj="Prey"),
        Axiom(kind="DP", subj=CID, pred="size", obj="small"),
        Axiom(kind="DP", subj=CID, pred="weight", obj="light"),
        Axiom(kind="DP", subj=CID, pred="color", obj="pale"),
        Axiom(kind="DOMAIN", subj=CID, obj="Animal"),
        Axiom(kind="RANGE", subj=CID, obj="Forest"),
        Axiom(kind="RANGE", subj=CID, obj="Prey"),
        Axiom(kind="CARD", subj=CID, card_min=1, card_max=3),
        Axiom(kind="TRANSITIVE", subj=CID, pred="eats"),
        Axiom(kind="DISJOINT", subj=CID, obj="Plant"),
    ]


def retarget(axioms, old, new):
    """把公理集里所有出现 old 的 obj 改成 new（保持条数与集合势不变）。"""
    return [replace(a, obj=new) if a.obj == old else a for a in axioms]


def main() -> int:
    emb = Embedder(allow_fallback=False, verbose=False)
    st = make_controlled_stream(emb, seed=42, params=C.PARAMS)
    onto, _ = build_initial_ontology(st.pool, st.d0_iids, seed=0,
                                     k_gt=len(st.vocab_base),
                                     vocab={**st.vocab_base, **st.vocab_emer})
    theta = C.PARAMS.theta_red * max(1.0, onto.M / 18.0)
    print(f"[setup] O0 concepts={len(onto.concepts)} M={onto.M} "
          f"theta_red_eff={theta:.2f} api_calls={emb.n_api_calls}", flush=True)

    b = base_axioms()
    cases = []

    def add(family, name, what, a, bset, reachable, note=""):
        d = dax(a, bset)
        cases.append({"family": family, "pair": name, "differs_in": what,
                      "d_ax": d, "fires": bool(d > theta),
                      "reachable_in_O0": reachable, "note": note})

    # ---- 五族碰撞 -------------------------------------------------------
    add("A range re-target", "R1",
        "livesIn ranges over Market instead of Forest (both classes exist)",
        b, retarget(b, "Forest", "Market"),
        True, "Set size of the range component is unchanged")
    add("B domain re-target", "D1",
        "the concept is declared in Plant instead of Animal",
        b, [replace(a, obj="Plant") if a.kind == "DOMAIN" else a for a in b],
        True, "The count of declared domains is unchanged")
    add("C value change", "V1",
        "the data-property value of size becomes large",
        b, [replace(a, obj="large") if a.kind == "DP" and a.pred == "size" else a
            for a in b],
        True, "DP terms are presence flags; the value is not in the signature")
    two = b + [Axiom(kind="CARD", subj=CID, card_min=2, card_max=2)]
    two_swapped = [replace(a, card_min=2, card_max=2) if a.card_min == 1 and a.card_max == 3
                   else replace(a, card_min=1, card_max=3) if a.card_min == 2 and a.card_max == 2
                   else a for a in two]
    add("D cardinality exchange", "C1",
        "two cardinality axioms exchange their bounds, (1,3)+(2,2) -> (2,2)+(1,3)",
        two, two_swapped, False,
        "Signature keeps max(min) and max(max); O0 has one CARD axiom per concept")
    add("E clipping", "K1",
        "minimum cardinality 8 -> 9 (both clip at the cap of 8)",
        [replace(x, card_min=8, card_max=3) for x in b if x.kind == "CARD"] +
        [x for x in b if x.kind != "CARD"],
        [replace(x, card_min=9, card_max=3) for x in b if x.kind == "CARD"] +
        [x for x in b if x.kind != "CARD"],
        False, "Both values saturate the component at 1.0")
    add("E clipping", "K2",
        "maximum cardinality unbounded (stored 0) -> 8",
        [replace(x, card_min=1, card_max=0) for x in b if x.kind == "CARD"] +
        [x for x in b if x.kind != "CARD"],
        [replace(x, card_min=1, card_max=8) for x in b if x.kind == "CARD"] +
        [x for x in b if x.kind != "CARD"],
        False, "0 is stored as the cap, so both components read 1.0")

    # ---- 两个对照 -------------------------------------------------------
    extra = b + [Axiom(kind="OP", subj=CID, pred="locatedNear", obj="Market")]
    add("control F count change", "F1",
        "a new object-property axiom with a range class not yet declared",
        b, extra, True, "Above the threshold: 1.00 + 0.25")
    add("control F sub-threshold", "F2",
        "minimum cardinality 3 -> 4",
        [replace(x, card_min=3, card_max=3) for x in b if x.kind == "CARD"] +
        [x for x in b if x.kind != "CARD"],
        [replace(x, card_min=4, card_max=3) for x in b if x.kind == "CARD"] +
        [x for x in b if x.kind != "CARD"],
        True, "Non-zero but below the threshold: 0.125")

    for c in cases:
        print(f"  {c['family']:<26} {c['pair']:<4} d_ax={c['d_ax']:.4f} "
              f"fires={str(c['fires']):<5} O0={c['reachable_in_O0']}")

    # ---- 断言：碰撞族必须逐位为 0 -------------------------------------
    coll = [c for c in cases if c["family"][0] in "ABCDE"]
    assert all(c["d_ax"] == 0.0 for c in coll), [c for c in coll if c["d_ax"] != 0.0]
    control = [c for c in cases if c["family"].startswith("control")]
    assert control[0]["fires"] is True and control[1]["d_ax"] > 0.0, control

    # ---- O₀ 覆盖：碰撞族在真实本体上是否可达 ---------------------------
    concept_ids = sorted(onto.concepts)
    range_names = sorted({a.obj for c in onto.concepts.values() for a in c.axioms
                          if a.kind in ("OP", "RANGE")})
    parents = sorted({c.parent for c in onto.concepts.values() if c.parent})
    cards = [x for c in onto.concepts.values() for x in c.axioms if x.kind == "CARD"]
    cover = {
        "concepts": len(concept_ids),
        "range_swap_available": 0, "range_swap_silent": 0,
        "domain_swap_available": 0, "domain_swap_silent": 0,
        "value_swap_available": 0, "value_swap_silent": 0,
        "card_axioms_per_concept": {},
        "card_min_values": sorted({x.card_min for x in cards}),
        "card_max_values": sorted({x.card_max for x in cards}),
        "clipping_reachable": False,
    }
    for cid in concept_ids:
        ax = onto.concepts[cid].axioms
        own_ranges = {a.obj for a in ax if a.kind in ("OP", "RANGE")}
        # A：把一个不在本概念 range 集合里的类换进来
        cand = [r for r in range_names if r not in own_ranges]
        if cand and own_ranges:
            cover["range_swap_available"] += 1
            old = sorted(own_ranges)[0]
            new_ax = [replace(a, obj=cand[0]) if a.obj == old else a for a in ax]
            if dax(ax, new_ax) == 0.0:
                cover["range_swap_silent"] += 1
        # B：父类换成一个别的父类
        cand_p = [p for p in parents if p != onto.concepts[cid].parent]
        if cand_p:
            cover["domain_swap_available"] += 1
            new_ax = [replace(a, obj=cand_p[0]) if a.kind == "DOMAIN" else a for a in ax]
            if dax(ax, new_ax) == 0.0:
                cover["domain_swap_silent"] += 1
        # C：DP 值改动
        dp = [a for a in ax if a.kind == "DP"]
        if dp:
            cover["value_swap_available"] += 1
            new_ax = [replace(a, obj="UNSEEN-VALUE") if a.pred == dp[0].pred
                      and a.kind == "DP" else a for a in ax]
            if dax(ax, new_ax) == 0.0:
                cover["value_swap_silent"] += 1
        n_card = sum(1 for a in ax if a.kind == "CARD")
        cover["card_axioms_per_concept"][str(n_card)] = \
            cover["card_axioms_per_concept"].get(str(n_card), 0) + 1

    cover["clipping_reachable"] = any(
        x.card_min >= int(8) or x.card_max in (0, 8) for x in cards)
    print("\n[O0 coverage]", json.dumps(cover, ensure_ascii=False))

    out = {"protocol": {
        "level": "signature-level demonstration (no detector, no embedder call)",
        "signature": "Ontology.axiom_signature, M = |P_op| + |P_dp| + 6",
        "distance": "L1_distance (normalised [0,1] components)",
        "threshold": theta,
        "threshold_note": "theta_red * max(1, M/18); 1.00 at M = 18",
        "fired_rule": "d_ax > theta_red_eff",
        "stream": "track 1, seed 42, O0 built with build_initial_ontology",
    }, "cases": cases, "coverage_O0": cover}
    (RES / "lossy_signature.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {RES / 'lossy_signature.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
