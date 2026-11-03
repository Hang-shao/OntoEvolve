"""命题 B 成本对比实验的单元测试：质量对齐度量 + 小规模冒烟。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _env_checks import requires_legacy
from src import config as C                                   # noqa: E402

from src.ontology_model import Concept, Ontology               # noqa: E402


def _onto(vectors):
    concepts = {f"C{i}": Concept(cid=f"C{i}", name=f"C{i}", parent=None,
                                 centroid=np.asarray(v, dtype=np.float32),
                                 axioms=[], active=True)
                for i, v in enumerate(vectors)}
    return Ontology(concepts, version=1)


def test_ontology_match_identical():
    from experiments.run_exp_cost import ontology_match
    v = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    m = ontology_match(_onto(v), _onto(v))
    assert m["match_rate"] == 1.0
    assert m["mean_sim"] > 0.999
    assert m["n_a"] == m["n_b"] == 3


def test_ontology_match_disjoint():
    """质心互相正交时匹配率应显著下降。"""
    from experiments.run_exp_cost import ontology_match
    a = _onto([[1.0, 0.0], [0.0, 1.0]])
    b = _onto([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    # 维度不同时只比较前两维，仍应低于 0.9 阈值
    m = ontology_match(a, _onto([[0.0, 1.0], [1.0, 0.0]]))
    assert m["match_rate"] == 1.0      # 两个概念互换顺序仍可被最近邻匹配
    m2 = ontology_match(_onto([[1.0, 0.0]]), _onto([[0.0, 1.0]]))
    assert m2["match_rate"] == 0.0     # 完全不相关 → 不匹配


def test_ontology_match_handles_empty():
    from experiments.run_exp_cost import ontology_match
    m = ontology_match(Ontology({}, version=1), _onto([[1.0, 0.0]]))
    assert m["match_rate"] == 0.0 and m["n_a"] == 0


@pytest.mark.slow
@requires_legacy
def test_run_cost_smoke():
    """极小规模冒烟：三条路径都能跑完，且加速比为正。"""
    from experiments.run_exp_cost import run_cost
    from src.embedding import Embedder
    p = C.PARAMS.copy(n_batch=3, n_init_per_leaf=2, pool_per_leaf=3,
                      base_rate=2.0, n_emergence=1, n_extinction=1,
                      n_redefinition=1)
    res = run_cost(0, p, Embedder(), check=False, period=2)
    assert res["T"] == 3
    assert res["incremental"]["ms_total"] > 0
    assert res["full_rebuild"]["ms_total"] > 0
    assert res["speedup_per_batch"] > 0
    assert res["quality_match"]["n_a"] >= 1
    assert len(res["rows"]) >= 6        # 增量 + 全量（+ 周期）
