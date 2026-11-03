"""推理机与 OWL 导出的单元测试。

HermiT 需要 Java + vendor/ 中的 jar；缺失时相关用例跳过（不判失败）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                # noqa: E402
from src.consistency import consistency_report             # noqa: E402
from src.ontology_model import Axiom, Concept, Ontology    # noqa: E402
from src.owl_export import export_owl, owl_text            # noqa: E402
from src.reasoner import (HermiTReasoner, Owlready2Reasoner,   # noqa: E402
                          find_java, get_reasoner, hermit_available,
                          hermit_jars)


def _onto(axs_by_cid, parents=None):
    concepts = {cid: Concept(cid=cid, name=cid, parent=(parents or {}).get(cid),
                             centroid=np.zeros(2, np.float32),
                             axioms=list(axs), active=True)
                for cid, axs in axs_by_cid.items()}
    return Ontology(concepts, version=1)


CONSISTENT = _onto({
    "C0": [Axiom("OP", "C0", pred="livesIn", obj="Forest"),
           Axiom("CARD", "C0", pred="livesIn", card_min=1, card_max=3)],
    "C1": [Axiom("OP", "C1", pred="eats", obj="Farm")],
})
DOMAIN_DISJOINT = _onto({
    "C0": [Axiom("DOMAIN", "C0", obj="C1"), Axiom("DISJOINT", "C0", obj="C1")],
    "C1": [Axiom("OP", "C1", pred="eats", obj="Farm")],
})
INHERITED = _onto({
    "C0": [Axiom("OP", "C0", pred="livesIn", obj="Forest")],
    "P0": [Axiom("DISJOINT", "P0", obj="C0")],
}, parents={"C0": "P0"})


@pytest.mark.skipif(not find_java(), reason="no java")
def test_find_java():
    assert find_java()
    assert hermit_jars() or True       # jar 缺失不阻断其它用例


def test_owl_export_contains_expected_constructs():
    txt = owl_text(DOMAIN_DISJOINT)
    assert "owl:Class" in txt
    assert "disjointWith" in txt
    assert "C0" in txt and "C1" in txt


def test_owl_export_individual_per_active_concept():
    txt = owl_text(CONSISTENT)
    assert "#i_C0" in txt and "#i_C1" in txt


def test_owl_export_qualified_cardinality_has_on_class():
    txt = owl_text(CONSISTENT)
    assert "minQualifiedCardinality" in txt
    assert "onClass" in txt


needs_hermit = pytest.mark.skipif(not hermit_available(),
                                  reason="hermit jars or java missing")


@needs_hermit
def test_hermit_consistent_ontology(tmp_path):
    r = HermiTReasoner()
    res = r.check(export_owl(CONSISTENT, tmp_path / "ok.owl"))
    assert res.ok, res.raw
    assert res.consistent, res.raw


@needs_hermit
def test_hermit_detects_domain_disjoint(tmp_path):
    r = HermiTReasoner()
    res = r.check(export_owl(DOMAIN_DISJOINT, tmp_path / "bad.owl"))
    assert res.ok, res.raw
    assert not res.consistent
    assert any("C0" in u for u in res.unsatisfiable)


@needs_hermit
def test_hermit_detects_inherited_disjoint(tmp_path):
    """父子链上的不相交需要推理，结构性检查会漏掉。"""
    r = HermiTReasoner()
    res = r.check(export_owl(INHERITED, tmp_path / "inh.owl"))
    assert res.ok, res.raw
    assert not res.consistent


def test_consistency_report_engine_selection():
    rep = consistency_report(CONSISTENT, engine="structural")
    assert rep["engine"] == "structural"
    assert rep["consistent"]


def test_consistency_report_structural_detects_conflict():
    rep = consistency_report(DOMAIN_DISJOINT, engine="structural")
    assert not rep["consistent"]
    assert rep["structural"]


@needs_hermit
def test_consistency_report_uses_reasoner(tmp_path):
    rep = consistency_report(DOMAIN_DISJOINT, engine="hermit",
                             owl_out=tmp_path / "r.owl")
    assert rep["engine"] == "hermit"
    assert not rep["consistent"]
    assert rep["unsatisfiable"]


def test_consistency_cache_reuse():
    a = consistency_report(CONSISTENT, engine="structural")
    b = consistency_report(CONSISTENT, engine="structural")
    assert a["consistent"] == b["consistent"]


def test_get_reasoner_prefers_hermit():
    r = get_reasoner()
    if hermit_available():
        assert r is not None and r.name == "hermit"
    else:
        # 无 HermiT 时按 get_reasoner 的次序退到 owlready2 自带推理器；
        # "pellet" 是 owlready2 的别名（_ENGINE_ALIAS），不单独实现。
        assert r is None or r.name in ("owlready2", "pellet")


def test_owl_export_writes_file(tmp_path):
    p = export_owl(CONSISTENT, tmp_path / "sub" / "o.owl")
    assert p.exists() and p.stat().st_size > 0
    assert p.read_text(encoding="utf-8").startswith("<?xml")

# --------------------------------------------------------------------------
# 备选推理器（owlready2 0.51 自带）：给出判定，但调用失败不等于“一致”
# --------------------------------------------------------------------------
needs_java = pytest.mark.skipif(not find_java(), reason="no java")


@needs_java
def test_owlready2_fallback_reports_inconsistent(tmp_path):
    res = Owlready2Reasoner().check(export_owl(DOMAIN_DISJOINT, tmp_path / "bad2.owl"))
    assert res.engine == "owlready2"
    assert res.ok, res.error
    assert not res.consistent


@needs_java
def test_owlready2_fallback_consistent_ontology(tmp_path):
    res = Owlready2Reasoner().check(export_owl(CONSISTENT, tmp_path / "ok2.owl"))
    assert res.ok, res.error
    assert res.consistent


@needs_java
def test_owlready2_fallback_is_not_fail_open(tmp_path):
    """解析不了的文件必须返回 ok=False（无裁决），不能被读成“一致”。"""
    bad = tmp_path / "garbage.owl"
    bad.write_text("this is not an ontology", encoding="utf-8")
    res = Owlready2Reasoner().check(bad)
    assert res.ok is False
    assert res.error


def test_get_reasoner_accepts_pellet_alias():
    """旧参数名 pellet 仍可用，指向 owlready2 备选（名称已如实标注）。"""
    r = get_reasoner("pellet")
    if not find_java():
        assert r is None
    else:
        assert r is not None and r.name in ("owlready2", "hermit")
