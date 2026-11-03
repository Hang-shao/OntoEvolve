"""初始本体构建（模块 A/B/C）的单元测试：使用离线合成向量，不调用嵌入服务。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _env_checks import requires_legacy

from src.build_ontology import build_initial_ontology
from src.controlled_stream import PoolItem
from src.embedding import l2_normalize
from src.vocab import build_vocab


def _synthetic_pool(n_leaf: int = 5, per_leaf: int = 20, dim: int = 16,
                    seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    centres = l2_normalize(rng.normal(size=(n_leaf, dim)))
    pool = []
    iid = 0
    vocab = build_vocab(seed=0)
    names = list(vocab.keys())[:n_leaf]
    for li, leaf in enumerate(names):
        schema = vocab[leaf]
        for _ in range(per_leaf):
            v = l2_normalize(centres[li] + rng.normal(scale=0.08, size=dim))
            pool.append(PoolItem(iid=iid, leaf=leaf, sup=schema.parent,
                                 variant="base", text=f"{leaf}-{iid}",
                                 op_slots=list(schema.ops),
                                 dp_slots=list(schema.dps),
                                 vec=v.astype(np.float32)))
            iid += 1
    return pool


@requires_legacy
def test_abc_recovers_synthetic_leaves():
    pool = _synthetic_pool()
    iids = [p.iid for p in pool]
    onto, info = build_initial_ontology(pool, iids, seed=0, k_gt=5,
                                        run_abc_clustering=True)
    assert len(onto.concepts) == 5
    assert len(info["leaf_to_cid"]) == 5
    # 每个概念应只对应一个叶子概念（完美恢复时一一对应）
    assert len(set(info["leaf_to_cid"].values())) == 5


@requires_legacy
def test_concepts_have_centroids_and_axioms():
    pool = _synthetic_pool()
    onto, info = build_initial_ontology(pool, [p.iid for p in pool], seed=0,
                                        k_gt=5, run_abc_clustering=True,
                                        vocab=build_vocab(seed=0))
    for cid, c in onto.concepts.items():
        assert c.centroid.shape[0] == pool[0].vec.shape[0]
        assert abs(float(np.linalg.norm(c.centroid)) - 1.0) < 1e-3
        kinds = {a.kind for a in c.axioms}
        assert {"OP", "DP"} <= kinds
        assert all(a.confidence >= 0.75 for a in c.axioms)
    assert "link_sim_p05" in info


def test_ground_truth_labels_mode_skips_clustering():
    pool = _synthetic_pool()
    onto, info = build_initial_ontology(pool, [p.iid for p in pool], seed=0,
                                        run_abc_clustering=False)
    assert len(onto.concepts) == 5
    assert info.get("ground_truth_labels") is True


@requires_legacy
def test_fixed_k_mode_controls_granularity():
    """关闭模块 B（固定 K）时，概念数应等于指定的 K。"""
    pool = _synthetic_pool()
    iids = [p.iid for p in pool]
    for k in (3, 5, 10):
        onto, info = build_initial_ontology(pool, iids, seed=0, k_gt=k,
                                            run_abc_clustering=True,
                                            k_fixed=k)
        assert len(onto.concepts) == k, f"k_fixed={k} 得到 {len(onto.concepts)}"
        assert info["K_star"] == k
