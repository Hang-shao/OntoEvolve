"""受控数据流与真值反推的单元测试（不依赖嵌入服务）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C
from src.controlled_stream import (Occurrence, _declare_batch,
                                   build_vocab, instance_text, select_events)
from src.vocab import BASE_TREE, EMERGENCE_TREE


def _occ(leaf: str, variant: str = "base") -> Occurrence:
    return Occurrence(iid=0, leaf=leaf, variant=variant, text="x",
                      vec=np.zeros(4, np.float32), op_slots=[], dp_slots=[],
                      sup="S")


def test_vocab_is_deterministic_and_has_alt_pattern():
    v1 = build_vocab(seed=7, tree=BASE_TREE)
    v2 = build_vocab(seed=7, tree=BASE_TREE)
    assert list(v1.keys()) == list(v2.keys())
    schema = v1[list(v1.keys())[0]]
    assert schema.ops != schema.alt_ops
    assert len(schema.alt_ops) == 3


def test_instance_text_differs_between_variants():
    schema = build_vocab(seed=3, tree=BASE_TREE)["Animal.Mammal"]
    assert instance_text("big", schema) != instance_text("big", schema, variant="alt")


def test_select_events_covers_three_types():
    vb = build_vocab(seed=0, tree=BASE_TREE)
    ve = build_vocab(seed=1000, tree=EMERGENCE_TREE)
    metas = select_events(vb, ve, seed=5, params=C.PARAMS)
    kinds = {m["dtype"] for m in metas}
    assert kinds == {"emergence", "extinction", "redefinition"}
    assert len(metas) == (C.PARAMS.n_emergence + C.PARAMS.n_extinction
                          + C.PARAMS.n_redefinition)


def test_declare_emergence_needs_kappa_consecutive_batches():
    p = C.PARAMS.copy(kappa=3)
    batches = [[_occ("X")], [], [_occ("X")], [_occ("X")], [_occ("X")]]
    assert _declare_batch(batches, "X", "emergence", p, t_start=1) == 5


def test_declare_extinction_uses_window_mean():
    p = C.PARAMS.copy(window_w=3, eps_ext=1.0)
    batches = [[_occ("X") for _ in range(3)], [_occ("X")], [], [], []]
    # 批次计数 [3,1,0,0,0]，W=3：t=4 的窗口 [1,0,0] 均值 0.33 < 1.0
    assert _declare_batch(batches, "X", "extinction", p, t_start=1) == 4


def test_declare_redefinition_needs_half_alt_instances():
    p = C.PARAMS.copy()
    batches = [[_occ("X"), _occ("X")],
               [_occ("X"), _occ("X", "alt")],
               [_occ("X", "alt"), _occ("X", "alt")]]
    assert _declare_batch(batches, "X", "redefinition", p, t_start=1) == 2
