"""命题 C 评估模块的单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _env_checks import requires_embeddings

from src.eval_qa import (Answer, KCLASSES, NEW, REDEFINED, STABLE,    # noqa: E402
                         answer, axiom_value, build_questions,
                         build_surfaces, concept_surface,
                         label_knowledge_classes, lookup_value,
                         parse_judge, score_one, summarize_scores)
from src.embedding import Embedder                                    # noqa: E402
from src.ontology_model import Axiom, Concept, Ontology               # noqa: E402


def _mk(cid, name, axioms, parent=None, active=True):
    return Concept(cid=cid, name=name, parent=parent,
                   centroid=np.zeros(2, dtype=np.float32),
                   axioms=list(axioms), active=active)


def _onto(concepts):
    return Ontology({c.cid: c for c in concepts})


# ---------------------------------------------------------------------------
# 三类知识标注
# ---------------------------------------------------------------------------
def test_label_classes_from_history():
    onto = _onto([_mk("C0", "a", []), _mk("C1", "b", []), _mk("C2", "c", []),
                  _mk("C3", "d", [], active=False)])
    history = [
        {"op": "emergence", "cid": "C1", "ok": True},
        {"op": "redefinition", "cid": "C2", "ok": True},
        {"op": "redefinition", "cid": "C0", "ok": False},   # 被回滚，不算
    ]
    lab = label_knowledge_classes(onto, history)
    assert lab["C0"] == STABLE      # 回滚的重定义不算知识变化
    assert lab["C1"] == NEW
    assert lab["C2"] == REDEFINED
    assert "C3" not in lab          # 已淘汰，不参与评估


def test_label_redefinition_beats_emergence():
    """先涌现后被重定义的概念算作重定义知识。"""
    onto = _onto([_mk("C1", "b", [])])
    lab = label_knowledge_classes(
        onto, [{"op": "emergence", "cid": "C1", "ok": True},
               {"op": "redefinition", "cid": "C1", "ok": True}])
    assert lab["C1"] == REDEFINED


# ---------------------------------------------------------------------------
# 公理取值
# ---------------------------------------------------------------------------
def test_axiom_value():
    assert axiom_value(Axiom("OP", "C0", pred="p", obj="X")) == "X"
    assert axiom_value(Axiom("CARD", "C0", pred="p", card_min=2, card_max=5)) == "[2, 5]"
    assert axiom_value(Axiom("CARD", "C0", pred="p", card_min=2, card_max=0)) == "[2, ∞]"
    assert axiom_value(Axiom("TRANSITIVE", "C0", pred="p")) is None


def test_lookup_value_parent_and_axiom():
    onto = _onto([_mk("C0", "a", [Axiom("OP", "C0", pred="p", obj="X")],
                      parent="P0")])
    assert lookup_value(onto, "C0", "PARENT") == "P0"
    assert lookup_value(onto, "C0", "OP", "p") == "X"
    assert lookup_value(onto, "C0", "OP", "other") is None
    assert lookup_value(onto, "C9", "OP", "p") is None


# ---------------------------------------------------------------------------
# 出题
# ---------------------------------------------------------------------------
def _two_ontos():
    """O_0 与 O_T：C0 的 p 由 X 改成 Y（重定义），C1 的公理不变（稳定）。"""
    o0 = _onto([_mk("C0", "Alpha", [Axiom("OP", "C0", pred="p", obj="X")],
                    parent="Root"),
                _mk("C1", "Beta", [Axiom("OP", "C1", pred="q", obj="Z")],
                    parent="Root")])
    ot = _onto([_mk("C0", "Alpha", [Axiom("OP", "C0", pred="p", obj="Y")],
                    parent="Root"),
                _mk("C1", "Beta", [Axiom("OP", "C1", pred="q", obj="Z")],
                    parent="Root")])
    return o0, ot


def test_questions_gold_from_final_stale_from_init():
    o0, ot = _two_ontos()
    qs = build_questions(ot, o0, {"C0": REDEFINED, "C1": STABLE},
                         per_class=10, seed=0)
    by_class = {q.kclass: q for q in qs if q.kind == "OP"}
    assert by_class[REDEFINED].gold == "Y"        # 金标 = O_T 语义
    assert by_class[REDEFINED].stale == "X"       # 旧语义 = O_0
    assert by_class[STABLE].gold == "Z"
    assert by_class[STABLE].stale is None         # 未变化 → 无 stale


def test_redefined_questions_only_cover_changed_axioms():
    """重定义类不能出“未被重定义改变”的公理，否则冻结本体也能答对。"""
    o0, ot = _two_ontos()
    qs = build_questions(ot, o0, {"C0": REDEFINED, "C1": REDEFINED},
                         per_class=10, seed=0)
    # C1 的 q->Z 在 O_0 与 O_T 中相同，不应作为重定义题出现
    assert all(not (q.cid == "C1" and q.kind == "OP" and q.pred == "q")
               for q in qs)


def test_two_versions_and_no_name_leak_in_b():
    o0, ot = _two_ontos()
    qs = build_questions(ot, o0, {"C0": REDEFINED, "C1": STABLE},
                         per_class=10, seed=0)
    assert qs, "应至少生成一题"
    for q in qs:
        assert q.q_a and q.q_b
        assert q.q_a != q.q_b
    named = [q for q in qs if "Alpha" in q.q_b or "Beta" in q.q_b]
    assert not named, "B 版不应出现概念名（防泄露）"


def test_per_class_and_per_concept_caps():
    o0, ot = _two_ontos()
    qs = build_questions(ot, o0, {"C1": STABLE}, per_class=2,
                         max_per_concept=1, seed=0)
    assert len(qs) <= 2


# ---------------------------------------------------------------------------
# 作答与评分
# ---------------------------------------------------------------------------
@requires_embeddings
def test_answer_and_score():
    o0, ot = _two_ontos()
    qs = build_questions(ot, o0, {"C0": REDEFINED, "C1": STABLE},
                         per_class=10, seed=0)
    q = [q for q in qs if q.kclass == REDEFINED and q.kind == "OP"][0]
    surf = build_surfaces(ot, Embedder(allow_fallback=True))
    ans = answer(ot, q, Embedder(allow_fallback=True), surf,
                 link_sim=0.0, version="a")
    assert ans.value is not None          # 阈值 0 → 必然作答
    assert score_one(Answer(q.cid, q.gold), q)["correct"] == 1
    assert score_one(Answer(q.cid, q.stale), q)["stale_hit"] == 1   # 语义错位
    assert score_one(Answer(q.cid, q.stale), q)["correct"] == 0
    assert score_one(Answer(None, None, 0.0, True), q)["abstain"] == 1


def test_abstain_below_threshold():
    o0, ot = _two_ontos()
    qs = build_questions(ot, o0, {"C1": STABLE}, per_class=10, seed=0)
    q = qs[0]
    ans = answer(ot, q, Embedder(allow_fallback=True),
                 build_surfaces(ot, Embedder(allow_fallback=True)),
                 link_sim=1.1, version="a")
    assert ans.abstain and ans.value is None


def test_concept_surface_contains_name_and_axiom():
    c = _mk("C0", "Alpha", [Axiom("OP", "C0", pred="p", obj="X")], parent="Root")
    s = concept_surface(c)
    assert "Alpha" in s and "Root" in s and "X" in s


def test_summarize_scores():
    rows = [
        {"kclass": STABLE, "correct": 1, "stale_hit": 0, "abstain": 0},
        {"kclass": STABLE, "correct": 0, "stale_hit": 1, "abstain": 0},
        {"kclass": NEW, "correct": 0, "stale_hit": 0, "abstain": 1},
    ]
    out = summarize_scores(rows)
    assert out[STABLE]["accuracy"] == 0.5
    assert out[STABLE]["stale_rate"] == 0.5
    assert out[NEW]["abstain_rate"] == 1.0
    assert out["all"]["n"] == 3
    assert REDEFINED not in out      # 无样本的类别不出现


def test_parse_judge():
    assert parse_judge("3") == 3
    assert parse_judge("分数：2") == 2
    assert parse_judge("无法判断") is None
    assert parse_judge(None) is None


def test_kclasses_order():
    assert set(KCLASSES) == {STABLE, NEW, REDEFINED}
