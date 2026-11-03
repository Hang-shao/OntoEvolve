"""PMC 语料（轨道 3）的解析与实例流测试。

这些用例**不访问嵌入 API**：解析、词表、切片、槽位都是纯本地逻辑，
可在嵌入服务不可用时照常验证。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pmc import (load_pmc, sentences_of, slice_by_year)          # noqa: E402
from src.pmc_stream import (build_term_vocab, sentence_slots,        # noqa: E402
                            strip_hint, _compile)


@pytest.fixture(scope="module")
def articles():
    try:
        return load_pmc()
    except FileNotFoundError as e:      # 干净克隆里没有 PMC 语料
        pytest.skip(f"PMC 语料缺失：{e}")


def test_corpus_size_and_fields(articles):
    assert len(articles) == 300, "PMC 子集应为 30 个 OAI 文件 × 10 篇"
    for a in articles:
        assert a.year > 0, f"{a.pmid} 缺少年份"
        assert a.paragraphs, f"{a.pmid} 没有可用段落"


def test_year_range(articles):
    ys = [a.year for a in articles]
    assert min(ys) >= 1990 and max(ys) <= 2010


def test_slice_by_year_monotonic_and_complete(articles):
    sl = slice_by_year(articles, 12)
    assert len(sl) == 12
    total = sum(len(s) for s in sl)
    assert total == len(articles), "切片应覆盖全部文章且不重复"
    prev_max = 0
    for s in sl:
        assert s, "不应出现空时间片"
        cur_min = min(a.date_ymd for a in s)
        assert cur_min >= prev_max, "时间片必须按日期递增"
        prev_max = max(a.date_ymd for a in s)


def test_sentences_keep_section_hint(articles):
    sents = sentences_of(articles[0].paragraphs)
    assert sents, "段落应能切出句子"
    for s in sents:
        hint, body = strip_hint(s)
        assert hint and body
        assert not body.startswith("[")


def test_term_vocab_quality(articles):
    vocab = build_term_vocab(articles, top_k=200, min_df=6)
    assert 50 <= len(vocab) <= 400
    # 不应把纯停用词收进词表
    assert not any(v in ("the", "and", "of", "in") for v in vocab)
    # 应包含多词术语（领域术语通常是短语）
    assert any(" " in v for v in vocab), "词表应有多词术语"
    # 覆盖率过高的学术通用词应被过滤
    assert "patients" not in vocab, "patients 属通用词，应被覆盖率上界剔除"


def test_sentence_slots_shape(articles):
    vocab = build_term_vocab(articles, top_k=200, min_df=6)
    matcher = _compile(vocab)
    sents = sentences_of(articles[0].paragraphs)
    hit = 0
    for s in sents:
        leaf, op, dp = sentence_slots(s, matcher)
        assert isinstance(leaf, str) and leaf
        if op:                                  # 命中术语的句子
            hit += 1
            assert len(op[0]) == 2, "op 槽为 (主术语, 共现术语)"
            assert all(len(d) == 2 for d in dp)
            # dp 槽的谓词必须是段落语义类型，否则签名维度会翻倍
            assert all(d[0] in ("background", "method", "result",
                                "conclusion", "other") for d in dp)
    assert hit > 0, "应有句子命中术语"


def test_dp_predicate_keeps_signature_small(articles):
    """dp 槽谓词用语义类型而非术语，保证 M 与 ICEWS 可比（回归保护）。"""
    vocab = build_term_vocab(articles, top_k=200, min_df=6)
    matcher = _compile(vocab)
    sents = sentences_of(articles[1].paragraphs)
    preds = set()
    for s in sents:
        _, _, dp = sentence_slots(s, matcher)
        preds.update(d[0] for d in dp)
    assert len(preds) <= 7, f"dp 谓词集合过大，会撑爆签名维度：{sorted(preds)}"
