"""标定候选生成的回归测试：joint 模式必须是笛卡尔积，legacy 必须是单键扫描。

不构造真实数据流：`_prep_calib_streams` 与 `_baseline_f1` 都被替换成假件。
（历史上 joint 分支把 `_product(list)` 当成了 `_product(*list)`，
结果每个候选是一个元组而不是完整组合，运行时报 TypeError，
却因为没有任何测试覆盖而进了提交。）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import calibrate as CAL          # noqa: E402
from src import config as C               # noqa: E402

GRID = {"theta": (0.2, 0.3), "stale_window": (2, 3)}


def _patch(monkeypatch):
    seen: list[dict] = []

    monkeypatch.setattr(CAL, "_prep_calib_streams",
                        lambda stream: [("onto", "l2c", "stream")])

    def fake(onto, l2c, st, method, base_params, kw):
        seen.append(dict(kw))
        return float(kw.get("theta", 0.0)) + 0.1 * float(kw.get("stale_window", 0))

    monkeypatch.setattr(CAL, "_baseline_f1", fake)
    return seen


def _run(monkeypatch, mode):
    seen = _patch(monkeypatch)
    out = CAL.calibrate_baseline(
        "stream", "inc_clustering", base_params=C.PARAMS,
        calib_cooldown=99, grid=GRID, mode=mode, verbose=False)
    return seen, out


def test_joint_enumerates_cartesian_product(monkeypatch):
    seen, out = _run(monkeypatch, "joint")
    assert out["n_evals"] == 4, out
    # 最优必须是完整组合，而不是只有某一个键
    assert out["kw"] == {"theta": 0.3, "stale_window": 3}, out["kw"]
    # 标定期抑制窗口必须真的进到监视器参数里
    assert all(kw["cooldown"] == 99 for kw in seen), seen
    # 每个候选都带两个键
    assert all(set(kw) == {"theta", "stale_window", "cooldown"} for kw in seen), seen


def test_legacy_enumerates_single_key(monkeypatch):
    seen, out = _run(monkeypatch, "legacy")
    assert out["n_evals"] == 4, out
    # legacy 每轮只动一个键，另一键不出现（即固定在默认值）
    assert all(len(set(kw) - {"cooldown"}) == 1 for kw in seen), seen
    # 选中项只有单一键（哪一键由分数决定，不在这里断言）
    assert len(out["kw"]) == 1, out["kw"]
    assert set(out["kw"]) <= {"theta", "stale_window"}, out["kw"]


def test_tie_rule_is_first_in_grid_order(monkeypatch):
    monkeypatch.setattr(CAL, "_prep_calib_streams",
                        lambda stream: [("onto", "l2c", "stream")])
    # 所有候选同分 → 取网格顺序第一个
    monkeypatch.setattr(CAL, "_baseline_f1",
                        lambda *a: 0.5)
    out = CAL.calibrate_baseline(
        "stream", "inc_clustering", base_params=C.PARAMS,
        calib_cooldown=99, grid=GRID, mode="joint", verbose=False)
    assert out["kw"] == {"theta": 0.2, "stale_window": 2}, out["kw"]
    assert out["tie_rule"] == "first_in_grid_order"
