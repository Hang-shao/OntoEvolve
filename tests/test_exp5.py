"""实验 5 相关能力的单元测试：演化引擎可配置、耗时统计、注入矛盾与回滚、
OWL 2 DL 的 simple property 约束、推理机异常不被当作“一致”。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                # noqa: E402
from src.consistency import consistency_report             # noqa: E402
from src.evolve import OntologyEvolver                     # noqa: E402
from src.ontology_model import Axiom, Concept, Ontology    # noqa: E402
from src.owl_export import ontology_to_graph, owl_text     # noqa: E402
from src.reasoner import ReasonResult, hermit_available    # noqa: E402


def _onto(axs_by_cid, parents=None):
    concepts = {cid: Concept(cid=cid, name=cid, parent=(parents or {}).get(cid),
                             centroid=np.zeros(2, np.float32),
                             axioms=list(axs), active=True)
                for cid, axs in axs_by_cid.items()}
    return Ontology(concepts, version=1)


BAD_CARD = {"C0": [Axiom("CARD", "C0", pred="rel", card_min=9, card_max=1)]}


def test_evolver_engine_is_configurable():
    """引擎可配置；且 fail-closed：受限层裁决不得提交。"""
    onto = _onto({"C0": [Axiom("OP", "C0", pred="eats", obj="Farm")]})
    evo = OntologyEvolver(onto, engine="structural", use_cache=False)
    assert not evo.redefine("C0", [Axiom("OP", "C0", pred="eats", obj="Forest")])
    assert "structural" in evo.stats.engines
    assert evo.stats.no_verdict == 1
    assert onto.version == 1          # 版号只在成功提交时递增


def test_control_arm_may_relax_fail_closed():
    """exp5 的对照分支（z3/structural）可关闭 fail-closed：受限层按自身裁决提交。

    部署规则不变（默认 require_dl_verdict=True，见上一个测试）；关闭只用于对照，
    否则这些分支一次都提交不了，无法比较各检查策略的检出能力。
    """
    onto = _onto({"C0": [Axiom("OP", "C0", pred="eats", obj="Farm")]})
    evo = OntologyEvolver(onto, engine="structural", use_cache=False,
                          require_dl_verdict=False)
    assert evo.redefine("C0", [Axiom("OP", "C0", pred="eats", obj="Forest")])
    assert evo.stats.no_verdict == 0
    assert onto.version == 2


def test_evolver_rolls_back_inconsistent_change():
    onto = _onto({"C0": [Axiom("OP", "C0", pred="eats", obj="Farm")]})
    evo = OntologyEvolver(onto, engine="structural", use_cache=False)
    assert not evo.redefine("C0", BAD_CARD["C0"])
    assert evo.stats.rollbacks.get("redefinition") == 1
    assert evo.stats.rollback_rate == 1.0
    # 回滚后仍是原公理
    assert onto.concepts["C0"].axioms[0].obj == "Farm"


def test_evolver_no_check_keeps_inconsistent():
    onto = _onto({"C0": [Axiom("OP", "C0", pred="eats", obj="Farm")]})
    evo = OntologyEvolver(onto, check=False)
    assert evo.redefine("C0", BAD_CARD["C0"])
    assert evo.stats.rollback_rate == 0.0
    assert not consistency_report(onto, engine="structural")["consistent"]


def test_evolver_records_check_time():
    onto = _onto({"C0": [Axiom("OP", "C0", pred="eats", obj="Farm")],
                  "C1": [Axiom("OP", "C1", pred="eats", obj="Farm")]})
    evo = OntologyEvolver(onto, engine="z3", use_cache=False)
    evo.redefine("C0", [Axiom("OP", "C0", pred="eats", obj="Forest")])
    evo.redefine("C1", [Axiom("OP", "C1", pred="eats", obj="Farm")])
    assert evo.stats.ms_check >= 0.0
    assert evo.stats.versions == 2


def test_owl_export_skips_transitive_on_cardinality_property():
    """OWL 2 DL：基数限制只能作用于 simple property。"""
    o = _onto({"C0": [Axiom("OP", "C0", pred="partOf", obj="B"),
                      Axiom("TRANSITIVE", "C0", pred="partOf"),
                      Axiom("CARD", "C0", pred="partOf", card_min=1,
                            card_max=3)]})
    notes: list = []
    g = ontology_to_graph(o, notes=notes)
    txt = g.serialize(format="turtle")
    assert "TransitiveProperty" not in txt
    assert any("simple property" in n for n in notes)


def test_owl_export_keeps_transitive_without_cardinality():
    o = _onto({"C0": [Axiom("OP", "C0", pred="partOf", obj="B"),
                      Axiom("TRANSITIVE", "C0", pred="partOf")]})
    assert "TransitiveProperty" in owl_text(o)


def test_reasoner_no_verdict_is_not_consistent():
    """推理机没有给出判定（抛异常）时不能当作一致。"""
    from src.reasoner import HermiTReasoner
    res = ReasonResult(engine="hermit", ok=True, consistent=True, ms=1.0,
                       raw="Exception in thread main java.lang.IllegalArgumentException")
    HermiTReasoner._parse(res)
    assert res.ok is False
    assert res.error
    # 一致性检查因此降级，而不是把异常当通过
    assert issubclass(type(res), ReasonResult)


@pytest.mark.skipif(not hermit_available(), reason="hermit jars or java missing")
def test_inherited_disjoint_is_detected_only_by_reasoner(tmp_path):
    """C ⊑ D 且 D ⊓ C ⊑ ⊥：结构性检查看不见，推理机能发现。"""
    concepts = {
        "C": Concept("C", "C", None, np.zeros(2, np.float32),
                     [Axiom("DOMAIN", "C", obj="D")], active=True),
        "D": Concept("D", "D", None, np.zeros(2, np.float32),
                     [Axiom("DISJOINT", "D", obj="C")], active=True),
    }
    onto = Ontology(concepts, version=1)
    assert consistency_report(onto, engine="structural")["consistent"]
    rep = consistency_report(onto, engine="hermit",
                             owl_out=tmp_path / "inh.owl")
    assert rep["engine"] == "hermit"
    assert not rep["consistent"]


def test_exp5_inject_conflict_shapes():
    """注入的三种矛盾形态都能被构造出来。"""
    from experiments.run_exp5 import _inject_conflict, _parent_of
    onto = _onto({"C0": [Axiom("OP", "C0", pred="eats", obj="Farm")],
                  "C1": [Axiom("OP", "C1", pred="livesIn", obj="Forest")]})
    rng = np.random.default_rng(0)
    a0 = _inject_conflict(onto, "C0", 0, rng)
    assert a0 and a0[0].kind == "CARD" and a0[0].card_min > a0[0].card_max
    a1 = _inject_conflict(onto, "C0", 1, rng)
    assert any(a.kind == "DOMAIN" for a in a1)
    assert any(a.kind == "DISJOINT" for a in a1)
    assert _parent_of(onto, "C0") is None       # 无父时由调用方另行构造


def test_summarize_aggregates_runs():
    from experiments.run_exp5 import summarize

    rows = [
        {"engine": "hermit", "alpha": 0.5, "rollback_rate": 0.20, "versions": 10,
         "n_injected_conflicts": 3, "n_injected_accepted": 0,
         "final_consistent": True, "check_ms_per_version": 700.0,
         "ms_end2end": 20000.0, "peak_mem_mb": 260.0},
        {"engine": "hermit", "alpha": 0.5, "rollback_rate": 0.10, "versions": 12,
         "n_injected_conflicts": 3, "n_injected_accepted": 0,
         "final_consistent": True, "check_ms_per_version": 720.0,
         "ms_end2end": 21000.0, "peak_mem_mb": 270.0},
        {"engine": "z3", "alpha": 0.5, "rollback_rate": 0.0, "versions": 12,
         "n_injected_conflicts": 3, "n_injected_accepted": 3,
         "final_consistent": False, "check_ms_per_version": 12.0,
         "ms_end2end": 3000.0, "peak_mem_mb": 160.0},
    ]
    out = {s["engine"]: s for s in summarize(rows)}
    assert out["hermit"]["n_runs"] == 2
    assert out["hermit"]["detection_rate"] == 1.0          # 注入全部拦下
    assert out["hermit"]["final_consistent_rate"] == 1.0
    assert out["hermit"]["rollback_rate_ci95"] is not None  # 多次运行给 CI
    assert out["z3"]["detection_rate"] == 0.0              # 全部漏检
    assert out["z3"]["final_consistent_rate"] == 0.0
    assert out["z3"]["rollback_rate_ci95"] is None          # 单次运行无 CI


def test_resource_tracker_runs():
    from experiments.run_exp5 import ResourceTracker
    with ResourceTracker(interval=0.01) as rt:
        x = [0] * 200000
        del x
    assert rt.peak_mb >= 0.0
