"""大模型公理判定器的单元测试（打桩，不访问网络）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import AxiomJudge, _parse_confidence, axiom_key, axiom_text
from src.ontology_model import Axiom


def _judge(tmp_path, **kw):
    """构造判定器并显式打开。

    AxiomJudge 在没配 CHAT_API_URL 时会把自己关掉（生产行为正确），而这些用例
    stubbed 了 _call_api，必须绕过可用性判断，否则测试依赖本机的 .env。
    """
    j = AxiomJudge(cache_path=tmp_path / "j.jsonl", **kw)
    j.enabled = True
    return j


def _axioms() -> list:
    return [Axiom(kind="OP", subj="C0", pred="livesIn", obj="Forest",
                  evidence_count=1, confidence=0.5),
            Axiom(kind="OP", subj="C0", pred="eats", obj="Meat",
                  evidence_count=9, confidence=0.5),
            Axiom(kind="CARD", subj="C0", card_min=1, card_max=3,
                  evidence_count=5, confidence=0.5)]


def test_parse_confidence():
    assert _parse_confidence("0.85") == pytest.approx(0.85)
    assert _parse_confidence("置信度：0.9（成立）") == pytest.approx(0.9)
    assert _parse_confidence("1") == pytest.approx(1.0)
    assert _parse_confidence("无法判断") is None


def test_axiom_text_and_key_stable():
    ax = Axiom(kind="OP", subj="C0", pred="p", obj="A")
    assert ax.obj in axiom_text(ax)
    assert axiom_key(ax, "Dog") == axiom_key(ax, "Dog")
    assert axiom_key(ax, "Dog") != axiom_key(ax, "Cat")


def test_top_k_only_and_evidence_order(tmp_path):
    judge = _judge(tmp_path, top_k=1)
    seen = []

    def fake(prompt: str) -> float:
        seen.append(prompt)
        return 0.9

    judge._call_api = fake
    out = judge.judge_batch(_axioms(), "Dog")
    assert len(seen) == 1                       # top_k=1：只调用一次
    assert judge.stats.calls == 1
    assert "eats" in seen[0]                    # 判的是证据最强的那条
    assert len(out) == 1


def test_cache_avoids_second_call(tmp_path):
    judge = _judge(tmp_path, top_k=5)
    judge._call_api = lambda p: 0.7
    ax = _axioms()[0]
    assert judge.judge(ax, "Dog") == pytest.approx(0.7)
    assert judge.judge(ax, "Dog") == pytest.approx(0.7)
    assert judge.stats.calls == 1 and judge.stats.cache_hits == 1


def test_disabled_returns_none_without_calls(tmp_path):
    judge = AxiomJudge(cache_path=tmp_path / "j.jsonl", enabled=False)
    assert judge.judge(_axioms()[0], "Dog") is None
    assert judge.stats.calls == 0


def test_failure_degrades_gracefully(tmp_path):
    judge = _judge(tmp_path, top_k=5)

    def boom(prompt: str):
        raise RuntimeError("service down")

    judge._call_api = boom
    assert judge.judge(_axioms()[0], "Dog") is None
    assert judge.stats.failures == 1


def test_apply_multiplies_confidence(tmp_path):
    judge = _judge(tmp_path, top_k=5)
    judge._call_api = lambda p: 0.5
    axs = _axioms()
    out = judge.apply(axs, "Dog")
    # 三条都被判定（top_k=5 ≥ 3），置信度 = 0.5 × 0.5
    assert all(a.confidence == pytest.approx(0.25) for a in out)
