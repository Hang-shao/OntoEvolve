"""实验 3：OWL-DL 推理机（HermiT）一致性保障与回退率。

两部分：
  A. 演化过程：受控数据流 → 检测 → 演化算子 → 每步一致性检查（HermiT）
     指标：一致率、回退率、推理耗时（均值/p95）、不可满足类数
  B. 注入矛盾对照：构造 5 类本体（1 一致 + 4 矛盾），比较 HermiT 与 z3 的
     判定，量化“受限近似 vs 完整 OWL-DL 推理”的差距

用法：
    python experiments/run_exp3.py                # 全量
    python experiments/run_exp3.py --quick        # 小规模快速验证
    python experiments/run_exp3.py --engine z3    # 强制 z3 降级对照
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                    # noqa: E402
from src.build_ontology import build_initial_ontology          # noqa: E402
from src.consistency import consistency_report                 # noqa: E402
from src.controlled_stream import make_controlled_stream       # noqa: E402
from src.drift_detect import DriftDetector                     # noqa: E402
from src.embedding import Embedder                             # noqa: E402
from src.evolve import OntologyEvolver                         # noqa: E402
from src.extract import extract_candidates                     # noqa: E402
from src.ontology_model import Axiom, Concept, Ontology        # noqa: E402
from src.owl_export import export_owl                          # noqa: E402
from src.reasoner import find_java, get_reasoner, java_version  # noqa: E402

TYPES = ("emergence", "extinction", "redefinition")


# ---------------------------------------------------------------------------
# A. 演化过程
# ---------------------------------------------------------------------------
def run_evolution(seed: int, engine: str, params: C.DriftParams,
                  emb: Embedder, verbose: bool = True) -> Dict:
    stream = make_controlled_stream(emb, seed=seed, params=params)
    onto, info = build_initial_ontology(
        stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
        vocab={**stream.vocab_base, **stream.vocab_emer})
    p = params.copy(link_sim=max(0.50, float(info.get("link_sim_p05",
                                                       params.link_sim)) - 0.02))
    det = DriftDetector(onto, p)
    evolver = OntologyEvolver(onto)

    owls = C.DIR_ONTOLOGY / f"exp3_seed{seed}"
    times: List[float] = []
    engines: Dict[str, int] = {}
    unsat_total = 0
    new_cid = len(onto.concepts)

    for t, batch in enumerate(stream.batches, start=1):
        alarms = det.observe(t, batch)
        for a in alarms:
            if a.dtype == "emergence":
                new_cid += 1
                cid = f"C{new_cid}"
                items = list(a.detail.get("items", []))
                if not items:
                    continue
                axs = extract_candidates(
                    cid, [o.as_dict() for o in items], parent=None)
                cen = a.detail.get("centroid")
                ok = evolver.add_concept(
                    cid, a.cid, centroid=np.asarray(cen, dtype=np.float32)
                    if cen is not None else np.mean(
                        [o.vec for o in items], axis=0),
                    axioms=axs)
                # 新增概念可能因不一致被回滚，此时不应登记到检测器
                if ok and cid in onto.concepts:
                    det.centroids[cid] = np.asarray(
                        onto.concepts[cid].centroid, dtype=np.float32)
                    det.counts[cid] = []
                    det.weights[cid] = []
            elif a.dtype == "redefinition":
                axs = det.last_projected.get(a.cid) or []
                if axs:
                    evolver.redefine(a.cid, axs)
            elif a.dtype == "extinction":
                evolver.retire(a.cid)

    # 演化统计：evolve 内部已逐步做一致性检查，这里重放一次以取引擎与耗时
    rep = consistency_report(onto, engine=engine,
                             owl_out=owls / "final.owl")
    engines[rep.get("engine", "?")] = engines.get(rep.get("engine", "?"), 0) + 1
    times.append(float(rep.get("ms", 0.0)))
    unsat_total += len(rep.get("unsatisfiable", []))
    export_owl(onto, owls / "final.owl")

    st = evolver.stats
    return {
        "seed": seed,
        "n_batches": len(stream.batches),
        "n_events": len(stream.events),
        "n_concepts_final": len([c for c in onto.concepts.values() if c.active]),
        "versions": st.versions,
        "applied": dict(st.applied),
        "rollbacks": dict(st.rollbacks),
        "rollback_rate": st.rollback_rate,
        "inconsistent_versions": st.inconsistent_versions,
        "consistent_rate": 1.0 - (st.inconsistent_versions / max(st.versions, 1)),
        "final_consistent": bool(rep["consistent"]),
        "engines": engines,
        "reasoner_ms_mean": float(np.mean(times)) if times else 0.0,
        "reasoner_ms_p95": float(np.percentile(times, 95)) if times else 0.0,
        "unsat_classes": unsat_total,
        "owl": str(owls / "final.owl"),
    }


# ---------------------------------------------------------------------------
# B. 注入矛盾对照
# ---------------------------------------------------------------------------
def _onto(axs_by_cid: Dict[str, List[Axiom]], parents=None) -> Ontology:
    concepts = {}
    for cid, axs in axs_by_cid.items():
        concepts[cid] = Concept(cid=cid, name=cid, parent=(parents or {}).get(cid),
                                centroid=np.zeros(2, np.float32),
                                axioms=list(axs), active=True)
    return Ontology(concepts, version=1)


def injected_cases() -> Dict[str, Dict]:
    """返回 {用例名: {"onto": ..., "expect_consistent": bool}}。"""
    ok = _onto({
        "C0": [Axiom("OP", "C0", pred="livesIn", obj="Forest"),
               Axiom("DP", "C0", pred="size"),
               Axiom("CARD", "C0", pred="livesIn", card_min=1, card_max=3)],
        "C1": [Axiom("OP", "C1", pred="eats", obj="Farm")],
    })
    card_empty = _onto({
        "C0": [Axiom("OP", "C0", pred="livesIn", obj="Forest"),
               Axiom("CARD", "C0", pred="livesIn", card_min=3, card_max=1)],
    })
    domain_disjoint = _onto({
        "C0": [Axiom("DOMAIN", "C0", obj="C1"), Axiom("DISJOINT", "C0", obj="C1")],
        "C1": [Axiom("OP", "C1", pred="eats", obj="Farm")],
    })
    # 需要推理：C0 ⊑ P0 且 P0 ⊓ C0 ⊑ ⊥（结构性检查只看直接公理会漏掉）
    inherited = _onto({
        "C0": [Axiom("OP", "C0", pred="livesIn", obj="Forest")],
        "P0": [Axiom("DISJOINT", "P0", obj="C0")],
    }, parents={"C0": "P0"})
    # 需要推理：传递属性 + 不相交（A ⊑ ∃p.B, B ⊑ ∃p.C, A ⊓ C ⊑ ⊥）
    transitive = _onto({
        "A": [Axiom("OP", "A", pred="partOf", obj="B"),
              Axiom("TRANSITIVE", "A", pred="partOf"),
              Axiom("DISJOINT", "A", obj="C")],
        "B": [Axiom("OP", "B", pred="partOf", obj="C")],
        "C": [Axiom("OP", "C", pred="madeOf", obj="Forest")],
    })
    return {
        "consistent_baseline": {"onto": ok, "expect_consistent": True},
        "card_empty_interval": {"onto": card_empty, "expect_consistent": False},
        "domain_vs_disjoint": {"onto": domain_disjoint, "expect_consistent": False},
        "inherited_disjoint": {"onto": inherited, "expect_consistent": False},
        "transitive_disjoint": {"onto": transitive, "expect_consistent": True},
    }


def run_injected(engine: str, outdir: Path) -> Dict:
    rows = {}
    for name, case in injected_cases().items():
        onto = case["onto"]
        path = export_owl(onto, outdir / f"{name}.owl")
        rep_owl = consistency_report(onto, engine=engine, owl_out=path)
        rep_z3 = consistency_report(onto, engine="z3")
        rows[name] = {
            "expect_consistent": case["expect_consistent"],
            "owl_consistent": bool(rep_owl["consistent"]),
            "owl_engine": rep_owl.get("engine"),
            "owl_ms": round(float(rep_owl.get("ms", 0.0)), 1),
            "owl_unsat": list(rep_owl.get("unsatisfiable", []))[:5],
            "z3_consistent": bool(rep_z3["consistent"]),
            "structural": list(rep_z3.get("structural", [])),
            "owl": str(path),
        }
    n = len(rows)
    hit_owl = sum(int(r["owl_consistent"] == r["expect_consistent"]) for r in rows.values())
    hit_z3 = sum(int(r["z3_consistent"] == r["expect_consistent"]) for r in rows.values())
    return {"rows": rows, "n": n, "owl_correct": hit_owl,
            "z3_correct": hit_z3, "owl_acc": hit_owl / n, "z3_acc": hit_z3 / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--engine", default="auto")
    args = ap.parse_args()

    base = C.PARAMS
    if args.quick:
        base = base.copy(n_batch=12, n_init_per_leaf=6, pool_per_leaf=12,
                         base_rate=3.0, n_emergence=1, n_extinction=1,
                         n_redefinition=1)

    print("== 推理机环境 ==")
    print("java:", find_java())
    print("version:", java_version())
    r = get_reasoner(args.engine if args.engine != "auto" else None)
    print("reasoner:", r.name if r else None)

    emb = Embedder(allow_fallback=False, )
    outdir = C.DIR_RESULTS / "exp3"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    evo = [run_evolution(args.seed + i, args.engine, base, emb)
           for i in range(max(1, args.repeats))]
    inj = run_injected(args.engine, C.DIR_ONTOLOGY / "exp3_injected")

    print("\n== A. 演化一致性 ==")
    for e in evo:
        print(f"seed={e['seed']} batches={e['n_batches']} events={e['n_events']} "
              f"concepts={e['n_concepts_final']}")
        print(f"  versions={e['versions']} applied={e['applied']} "
              f"rollbacks={e['rollbacks']}")
        print(f"  consistent_rate={e['consistent_rate']:.3f} "
              f"rollback_rate={e['rollback_rate']:.3f} "
              f"final_consistent={e['final_consistent']}")
        print(f"  engines={e['engines']} ms_mean={e['reasoner_ms_mean']:.0f} "
              f"unsat={e['unsat_classes']}")

    print("\n== B. 注入矛盾对照 ==")
    for name, row in inj["rows"].items():
        print(f"{name:>20}: expect_consistent={str(row['expect_consistent']):>5} "
              f"OWL({row['owl_engine']})={str(row['owl_consistent']):>5} "
              f"z3={str(row['z3_consistent']):>5} "
              f"unsat={row['owl_unsat']}")
    print(f"  OWL 正确 {inj['owl_correct']}/{inj['n']} "
          f"({inj['owl_acc']:.2f}) | z3 正确 {inj['z3_correct']}/{inj['n']} "
          f"({inj['z3_acc']:.2f})")

    out = {"engine": args.engine, "quick": args.quick,
           "evolution": evo, "injected": inj,
           "elapsed_s": round(time.perf_counter() - t0, 1)}
    path = C.DIR_RESULTS / "exp3.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
