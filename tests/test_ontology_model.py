"""本体模型 / 公理签名 / Project / L1 距离的单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ontology_model import (Axiom, Concept, L1_distance, Ontology,
                                concepts_of_axiom, resolve_conflicts,
                                source_evidence_count_factor)
from src.vocab import DP_VOCAB, OP_VOCAB


def _onto() -> Ontology:
    c = Concept(cid="C0", name="Dog", parent="Animal",
                centroid=np.array([1.0, 0.0], np.float32),
                axioms=[
                    Axiom(kind="OP", subj="C0", pred="livesIn", obj="Forest",
                          confidence=0.9),
                    Axiom(kind="DP", subj="C0", pred="size", obj="large",
                          confidence=0.9),
                    Axiom(kind="DOMAIN", subj="C0", obj="Animal", confidence=0.9),
                    Axiom(kind="CARD", subj="C0", card_min=1, card_max=3,
                          confidence=0.9),
                    Axiom(kind="DISJOINT", subj="C0", obj="Cat", confidence=0.9),
                ])
    return Ontology({"C0": c})


def test_signature_is_fixed_length_and_normalised():
    o = _onto()
    sig = o.axiom_signature("C0")
    assert sig.shape == (len(OP_VOCAB) + len(DP_VOCAB) + 6,)
    assert sig.min() >= 0.0 and sig.max() <= 1.0
    assert sig[OP_VOCAB.index("livesIn")] == 1.0
    assert sig[len(OP_VOCAB) + DP_VOCAB.index("size")] == 1.0


def test_signature_of_unknown_concept_is_zero():
    o = _onto()
    assert np.allclose(o.axiom_signature("nope"), 0.0)


def test_project_keeps_highest_confidence():
    o = _onto()
    delta = [Axiom(kind="OP", subj="C0", pred="livesIn", obj="Ocean",
                   confidence=0.95)]
    proj = o.project(delta, "C0", tau_conf=0.0)
    ops = [a for a in proj.axioms_of("C0") if a.kind == "OP"]
    assert len(ops) == 1 and ops[0].obj == "Ocean"


def test_project_support_decay_drops_unsupported_axiom():
    o = _onto()
    delta = [Axiom(kind="OP", subj="C0", pred="eats", obj="Forest",
                   confidence=0.9)]
    proj = o.project(delta, "C0", tau_conf=0.5, support_decay=0.4)
    preds = {a.pred for a in proj.axioms_of("C0") if a.kind == "OP"}
    assert "eats" in preds and "livesIn" not in preds


def test_L1_distance_and_shape_check():
    assert L1_distance([0.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        L1_distance([0.0, 1.0], [1.0])


def test_conflict_resolution_and_concepts_of_axiom():
    axs = [Axiom(kind="OP", subj="C0", pred="p", obj="A", confidence=0.4),
           Axiom(kind="OP", subj="C0", pred="p", obj="B", confidence=0.8),
           Axiom(kind="OP", subj="C0", pred="q", obj="C", confidence=0.1)]
    out = resolve_conflicts(axs)
    assert len(out) == 2
    assert [a.obj for a in out if a.pred == "p"] == ["B"]
    assert concepts_of_axiom(Axiom(kind="DISJOINT", subj="C0", obj="C1")) == {"C0", "C1"}


def test_evidence_factor_monotone_and_bounded():
    weak = Axiom(kind="OP", subj="C0", pred="p", obj="A",
                 evidence_count=1, source_doc_count=1, batch_cover_count=1)
    strong = Axiom(kind="OP", subj="C0", pred="p", obj="A",
                   evidence_count=10, source_doc_count=10, batch_cover_count=5)
    fw = source_evidence_count_factor(weak)
    fs = source_evidence_count_factor(strong)
    assert 0.0 <= fw < fs <= 1.0


def test_evidence_factor_uses_relative_strength():
    """参考基准=批内最强公理的证据量时，相对强度相同的公理置信度相同。"""
    small = Axiom(kind="OP", subj="C0", pred="p", obj="A",
                  evidence_count=1, source_doc_count=1, batch_cover_count=1)
    large = Axiom(kind="OP", subj="C0", pred="p", obj="A",
                  evidence_count=25, source_doc_count=25, batch_cover_count=1)
    # 弱证据都只占该批最强证据的 1/4
    assert source_evidence_count_factor(small, ref_count=4, ref_docs=4) == \
        pytest.approx(source_evidence_count_factor(large, ref_count=100,
                                                   ref_docs=100))
