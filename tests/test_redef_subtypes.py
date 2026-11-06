# -*- coding: utf-8 -*-
"""重定义子类型的生成器单元测试（不需要嵌入服务）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.controlled_stream import (REDEF_SUBTYPES, subtype_alt_schema)  # noqa: E402
from src.vocab import BASE_TREE, build_vocab  # noqa: E402


def _schema(seed: int = 0):
    v = build_vocab(seed=seed, tree=BASE_TREE)
    return v["Animal.Mammal"]


def test_default_is_identity():
    s = _schema()
    assert subtype_alt_schema(s, "") is s
    assert subtype_alt_schema(s, "default") is s


def test_range_only_keeps_predicates_and_count():
    s = _schema()
    alt = subtype_alt_schema(s, "range_only")
    base_ops = [o for o, _ in s.ops]
    assert [o for o, _ in alt.alt_ops] == base_ops          # 谓词不变
    assert len(alt.alt_ops) == len(s.ops)                   # 槽位数不变
    changed = [i for i, (a, b) in enumerate(zip(s.ops, alt.alt_ops))
               if a[1] != b[1]]
    assert len(changed) == 1                                # 只有一个对象被换
    old_ranges = {r for _, r in s.ops}
    assert alt.alt_ops[changed[0]][1] not in old_ranges      # 换成同类的另一个类
    assert alt.dps == s.dps                                 # 数据属性不动


def test_dp_only_keeps_relations_and_changes_one_value():
    s = _schema()
    alt = subtype_alt_schema(s, "dp_only")
    assert alt.alt_ops == list(s.ops)                        # 关系完全不变
    diff = [i for i, (a, b) in enumerate(zip(s.dps, alt.dps)) if a != b]
    assert len(diff) == 1                                    # 只有一个取值变
    assert alt.dps[diff[0]][0] == s.dps[diff[0]][0]           # 属性名不变
    assert alt.dps[diff[0]][1] != s.dps[diff[0]][1]


def test_card_only_keeps_predicate_set_and_adds_a_slot():
    s = _schema()
    alt = subtype_alt_schema(s, "card_only")
    assert len(alt.alt_ops) == len(s.ops) + 1                 # 基数 2 → 3
    assert {o for o, _ in alt.alt_ops} == {o for o, _ in s.ops}
    assert alt.alt_ops[-1] == s.ops[1]                        # 重复第二个槽位
    assert alt.dps == s.dps


def test_unknown_subtype_rejected():
    s = _schema()
    try:
        subtype_alt_schema(s, "domain_only")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown subtype was accepted")


def test_subtype_list_covers_three_families():
    assert REDEF_SUBTYPES == ("range_only", "dp_only", "card_only")
