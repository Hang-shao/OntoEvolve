# -*- coding: utf-8 -*-
"""fail-closed：没有 OWL-DL 裁决时不得提交。

对应论文 §3.5 的保证："a version is committed only if a full OWL-DL
reasoner returned a verdict and that verdict was consistent"。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.consistency import has_owl_dl_verdict
from src.evolve import OntologyEvolver
from src.ontology_model import Axiom, Concept, Ontology


@pytest.mark.parametrize("engine,expected", [
    ("hermit", True),
    ("owlready2", True),
    ("z3", False),
    ("z3(fallback:timeout)", False),
    ("structural", False),
    ("structural-only", False),
])
def test_verdict_classification(engine, expected):
    assert has_owl_dl_verdict({"engine": engine}) is expected


@pytest.fixture()
def onto():
    o = Ontology(version=0, op_vocab=["uses"], dp_vocab=["at"])
    o.add(Concept(cid="C0", name="c0", parent=None,
                  centroid=np.ones(4, np.float32), active=True))
    return o


def test_restricted_verdict_does_not_commit(onto, monkeypatch):
    """受限层说"一致"也不能提交：版本与概念数都不变。"""
    import src.evolve as ev

    monkeypatch.setattr(ev, "consistency_report",
                        lambda *a, **k: {"consistent": True, "engine": "z3",
                                         "reason": None})
    evo = OntologyEvolver(onto)
    ok = evo.add_concept("C1", "c1", centroid=np.ones(4, np.float32))
    assert ok is False
    assert "C1" not in onto.concepts
    assert onto.version == 0
    assert evo.stats.applied == {}
    assert evo.stats.no_verdict == 1
    assert evo.stats.rollback_rate == 1.0


def test_owl_dl_verdict_commits(onto, monkeypatch):
    import src.evolve as ev

    monkeypatch.setattr(ev, "consistency_report",
                        lambda *a, **k: {"consistent": True, "engine": "hermit",
                                         "reason": None})
    evo = OntologyEvolver(onto)
    assert evo.add_concept("C1", "c1", centroid=np.ones(4, np.float32)) is True
    assert "C1" in onto.concepts
    assert evo.stats.applied == {"emergence": 1}
    assert evo.stats.no_verdict == 0


def test_inconsistent_owl_dl_verdict_still_rolls_back(onto, monkeypatch):
    import src.evolve as ev

    monkeypatch.setattr(ev, "consistency_report",
                        lambda *a, **k: {"consistent": False, "engine": "hermit",
                                         "reason": "unsat-class"})
    evo = OntologyEvolver(onto)
    assert evo.add_concept("C1", "c1", centroid=np.ones(4, np.float32)) is False
    assert evo.stats.inconsistent_versions == 1
    assert evo.stats.no_verdict == 0
