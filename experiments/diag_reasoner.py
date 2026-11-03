"""诊断脚本：验证 Java/HermiT 是否可用，以及 OWL 导出 + 一致性判定的行为。

用法：python experiments/diag_reasoner.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                  # noqa: E402
from src.ontology_model import Axiom, Concept, Ontology      # noqa: E402
from src.owl_export import export_owl, owl_text              # noqa: E402
from src.reasoner import (find_java, get_reasoner, hermit_available,  # noqa: E402
                          java_version)


def _onto(axioms_by_cid, parents=None, active=True):
    concepts = {}
    for cid, axs in axioms_by_cid.items():
        concepts[cid] = Concept(cid=cid, name=cid,
                                parent=(parents or {}).get(cid),
                                centroid=np.zeros(2, np.float32),
                                axioms=list(axs), active=active)
    return Ontology(concepts, version=1)


def main() -> None:
    print("== 环境 ==")
    print("java:", find_java())
    print("version:", java_version())
    print("hermit jars:", C.HERMIT_JAR.name, hermit_available())
    prefer = sys.argv[1] if len(sys.argv) > 1 else None
    r = get_reasoner(prefer)
    print("reasoner:", r.name if r else None)

    out = C.DIR_TEMP / "diag_reasoner"
    out.mkdir(parents=True, exist_ok=True)

    cases = {
        # 一致：C0 有属性与值域，C1 独立
        "consistent": _onto({
            "C0": [Axiom("OP", "C0", pred="livesIn", obj="Forest"),
                   Axiom("DP", "C0", pred="size"),
                   Axiom("CARD", "C0", pred="livesIn", card_min=1, card_max=3)],
            "C1": [Axiom("OP", "C1", pred="eats", obj="Farm")],
        }),
        # 不一致：DOMAIN(C1) 与 DISJOINT(C1) 同时成立 → C0 不可满足
        "domain_disjoint": _onto({
            "C0": [Axiom("DOMAIN", "C0", obj="C1"),
                   Axiom("DISJOINT", "C0", obj="C1")],
            "C1": [Axiom("OP", "C1", pred="eats", obj="Farm")],
        }),
        # 不一致：基数区间为空 min 3 > max 1
        "card_empty": _onto({
            "C0": [Axiom("OP", "C0", pred="livesIn", obj="Forest"),
                   Axiom("CARD", "C0", pred="livesIn", card_min=3, card_max=1)],
        }),
        # 需要推理：父子类链上的不相交（结构检查只看直接公理会漏掉）
        "inherited_disjoint": _onto({
            "C0": [Axiom("OP", "C0", pred="livesIn", obj="Forest")],
            "P0": [Axiom("DISJOINT", "P0", obj="C0")],
        }, parents={"C0": "P0"}),
    }

    for name, onto in cases.items():
        path = export_owl(onto, out / f"{name}.owl")
        t0 = time.perf_counter()
        res = r.check(path) if r else None
        print(f"\n== {name} == ({len(owl_text(onto))} B, {path.name})")
        if res is None:
            print("  no reasoner available")
            continue
        print(f"  engine={res.engine} ok={res.ok} consistent={res.consistent} "
              f"ms={res.ms:.0f}")
        print(f"  unsat={res.unsatisfiable[:5]}")
        print("  raw:", (res.raw or "").replace("\n", " | ")[:300])


if __name__ == "__main__":
    main()
