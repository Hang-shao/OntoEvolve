"""一致性与演化算子（含回滚）的单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _env_checks import requires_reasoner

from src.consistency import consistency_report, structural_conflicts
from src.evolve import OntologyEvolver
from src.ontology_model import Axiom, Concept, Ontology


def _onto() -> Ontology:
    c = Concept(cid="C0", name="Dog", parent="Animal",
                centroid=np.array([1.0, 0.0], np.float32),
                axioms=[Axiom(kind="OP", subj="C0", pred="livesIn", obj="Forest",
                              confidence=0.9),
                        Axiom(kind="CARD", subj="C0", card_min=1, card_max=3,
                              confidence=0.9)])
    return Ontology({"C0": c})


def test_healthy_ontology_is_consistent():
    rep = consistency_report(_onto())
    assert rep["consistent"] is True
    assert rep["structural"] == []


@requires_reasoner
def test_empty_cardinality_is_conflict():
    o = _onto()
    o.get("C0").axioms.append(Axiom(kind="CARD", subj="C0", card_min=5,
                                    card_max=2, confidence=0.9))
    assert structural_conflicts(o)
    assert consistency_report(o)["consistent"] is False


def test_domain_disjoint_conflict():
    o = _onto()
    c = o.get("C0")
    c.axioms.append(Axiom(kind="DOMAIN", subj="C0", obj="C1", confidence=0.9))
    c.axioms.append(Axiom(kind="DISJOINT", subj="C0", obj="C1", confidence=0.9))
    assert structural_conflicts(o)


def test_add_concept_commits_and_counts():
    ev = OntologyEvolver(_onto())
    ok = ev.add_concept("C1", "Rocket", centroid=np.array([0.0, 1.0], np.float32))
    assert ok and len(ev.onto.concepts) == 2
    assert ev.stats.applied["emergence"] == 1
    assert ev.onto.version == 1


def test_redefine_replaces_axioms():
    ev = OntologyEvolver(_onto())
    new = [Axiom(kind="OP", subj="C0", pred="eats", obj="Meat", confidence=0.9),
           Axiom(kind="CARD", subj="C0", card_min=1, card_max=2, confidence=0.9)]
    assert ev.redefine("C0", new)
    preds = {a.pred for a in ev.onto.get("C0").axioms if a.kind == "OP"}
    assert preds == {"eats"}


def test_retire_marks_inactive():
    ev = OntologyEvolver(_onto())
    assert ev.retire("C0")
    assert ev.onto.get("C0").active is False


@requires_reasoner
def test_inconsistent_change_is_rolled_back():
    ev = OntologyEvolver(_onto())
    before = len(ev.onto.concepts)
    bad = [Axiom(kind="CARD", subj="C0", card_min=9, card_max=1,
                 confidence=0.9)]     # 空基数区间 → 不一致
    ok = ev.redefine("C0", bad)
    assert ok is False
    assert ev.stats.rollbacks["redefinition"] == 1
    assert ev.stats.rollback_rate == 1.0
    # 回滚后公理恢复原状
    assert len(ev.onto.concepts) == before
    assert any(a.pred == "livesIn" for a in ev.onto.get("C0").axioms)
