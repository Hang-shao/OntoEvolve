"""评估口径与基线算法的单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import (ADWIN, BaselineMonitor, EmbeddingDistance,
                           IncrementalClustering, PageHinkley,
                           two_means_separation)
from src.controlled_stream import DriftEvent
from src.drift_detect import Alarm
from src.eval_drift import bootstrap_ci, match_events, wilcoxon_paired


def _ev(dtype: str, leaf: str, t: int) -> DriftEvent:
    return DriftEvent(leaf=leaf, dtype=dtype, t_start=t, t_declare=t)


def test_perfect_match_gives_f1_one():
    events = [_ev("emergence", "Music.Guitar", 5), _ev("extinction", "Animal.Dog", 9)]
    alarms = [Alarm(t=5, cid="NEW0", dtype="emergence", score=1.0),
              Alarm(t=9, cid="C3", dtype="extinction", score=1.0)]
    res = match_events(events, alarms, {"Animal.Dog": "C3"}, delay_tol=2)
    assert res.tp == 2 and res.fp == 0 and res.fn == 0
    assert res.per_type["emergence"]["f1"] == pytest.approx(1.0)
    assert res.macro_f1 == pytest.approx(
        (1.0 + 1.0 + 0.0) / 3)   # redefinition 无事件 → F1 记 0


def test_late_alarm_is_not_matched_and_counts_as_fp():
    events = [_ev("extinction", "Animal.Dog", 9)]
    alarms = [Alarm(t=14, cid="C3", dtype="extinction", score=1.0)]
    res = match_events(events, alarms, {"Animal.Dog": "C3"}, delay_tol=2)
    assert res.fn == 1 and res.fp == 1 and res.tp == 0


def test_untyped_baseline_alarms_use_oracle_typing():
    events = [_ev("extinction", "Animal.Dog", 9)]
    alarms = [Alarm(t=9, cid="C3", dtype="change", score=1.0)]
    res = match_events(events, alarms, {"Animal.Dog": "C3"}, delay_tol=2,
                       typed=False)
    assert res.tp == 1 and res.per_type["extinction"]["r"] == pytest.approx(1.0)


def test_adwin_detects_mean_shift():
    d = ADWIN(delta=0.05)
    fired = False
    rng = np.random.default_rng(0)
    for _ in range(40):
        fired |= d.add(float(rng.normal(0.0, 0.01)))
    for _ in range(80):
        fired |= d.add(float(rng.normal(1.0, 0.01)))
    assert fired


def test_adwin_no_false_alarm_on_constant_stream():
    d = ADWIN(delta=0.05)
    rng = np.random.default_rng(1)
    assert not any(d.add(float(rng.normal(0.5, 0.001))) for _ in range(60))


def test_page_hinkley_detects_shift():
    ph = PageHinkley(delta=0.01, lamb=0.5)
    fired = False
    for _ in range(20):
        fired |= ph.add(0.0)
    for _ in range(60):
        fired |= ph.add(0.2)
    assert fired


def test_embedding_distance_threshold():
    ed = EmbeddingDistance(k_std=2.0, warmup=5, min_obs=3)
    assert not any(ed.add(0.10) for _ in range(6))     # 稳定期不告警
    assert ed.add(0.50)                                # 明显偏离才告警


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(3)
    vals = rng.normal(0.7, 0.05, size=12)
    ci = bootstrap_ci(list(vals), n_boot=500, seed=0)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_wilcoxon_needs_enough_pairs():
    a = [0.8, 0.75, 0.7, 0.72]
    b = [0.4, 0.35, 0.3, 0.32]
    assert np.isnan(wilcoxon_paired(a, b)["p"])     # n<5 不做检验
    a2, b2 = a * 3, b * 3
    res = wilcoxon_paired(a2, b2)
    assert res["p"] < 0.05


# ---------------------------------------------------------------------------
# 新增基线：嵌入 kmeans 与增量聚类
# ---------------------------------------------------------------------------
def test_two_means_separation_splits_two_clusters():
    rng = np.random.default_rng(0)
    one = np.array([1.0, 0.0], dtype=np.float32) + rng.normal(
        0, 0.01, size=(8, 2)).astype(np.float32)
    two = np.vstack([
        np.tile(np.array([1.0, 0.0], dtype=np.float32), (4, 1)),
        np.tile(np.array([0.0, 1.0], dtype=np.float32), (4, 1))])
    sep_one = two_means_separation(one)
    sep_two = two_means_separation(two)
    assert sep_two > 0.9            # 正交两簇 → 分离度接近 1
    assert sep_one < sep_two        # 单簇内部分裂度明显更低
    assert two_means_separation(one[:3]) == 0.0   # 样本不足 → 0


def test_incremental_clustering_flags_new_pattern():
    ic = IncrementalClustering(theta=0.35, warmup=2, stale_window=3)
    base = np.array([1.0, 0.0], dtype=np.float32)
    assert not ic.update(1, "C0", [base])[0]      # warmup 内不告警
    assert not ic.update(2, "C0", [base])[0]
    far = np.array([0.0, 1.0], dtype=np.float32)
    new_pat, extinct = ic.update(3, "C0", [far])  # 已过 warmup
    assert new_pat and not extinct
    # 同一方向的实例不再触发（微簇已吸收）
    assert not ic.update(4, "C0", [np.array([0.0, 1.0], dtype=np.float32)])[0]


def test_incremental_clustering_flags_extinction():
    ic = IncrementalClustering(theta=0.35, warmup=1, stale_window=2)
    b = np.array([1.0, 0.0], dtype=np.float32)
    ic.update(1, "C0", [b])
    assert not ic.update(2, "C0", [])[1]          # 空 1 批还不够
    assert ic.update(3, "C0", [])[1]              # 连续 2 批无新实例


def test_baseline_monitor_supports_new_methods():
    from src.controlled_stream import Occurrence
    from src.ontology_model import Concept, Ontology

    assert "embkmeans" in BaselineMonitor.METHODS
    assert "inc_clustering" in BaselineMonitor.METHODS

    onto = Ontology({"C0": Concept(
        cid="C0", name="a", parent=None,
        centroid=np.array([1.0, 0.0], dtype=np.float32), axioms=[], active=True)})
    occ = Occurrence(iid=0, leaf="a", variant="base", text="x",
                     vec=np.array([1.0, 0.0], dtype=np.float32),
                     op_slots=[], dp_slots=[])
    for m in ("embkmeans", "inc_clustering"):
        mon = BaselineMonitor(onto, m)
        alarms = mon.observe(1, [occ])
        assert isinstance(alarms, list)

    with pytest.raises(ValueError):
        BaselineMonitor(onto, "no_such_method")
