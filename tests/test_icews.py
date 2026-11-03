"""ICEWS 加载 / 切片 / 分层抽样的单元测试（使用临时目录里的迷你数据）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.icews import (country_of, load_icews, make_slices, role_of,
                       sample_slice, slice_stat)


def _mini_dataset(tmp_path: Path) -> Path:
    ents = ["Kim Jong-Un", "North Korea", "Media Personnel (Afghanistan)",
            "Police (Afghanistan)", "Citizen (Japan)"]
    rels = ["Make statement", "Demand", "Use conventional military force"]
    (tmp_path / "entity2id.txt").write_text(
        "\n".join(f"{n}\t{i}" for i, n in enumerate(ents)), encoding="utf-8")
    (tmp_path / "relation2id.txt").write_text(
        "\n".join(f"{n}\t{i}" for i, n in enumerate(rels)), encoding="utf-8")
    rng = np.random.default_rng(0)
    lines = []
    for ts in range(0, 96, 24):
        for k in range(10):
            s = int(rng.integers(0, len(ents)))
            o = int(rng.integers(0, len(ents)))
            r = int(rng.integers(0, len(rels)))
            lines.append(f"{s}\t{r}\t{o}\t{ts}\t0")
    # 低频关系保底：只出现一次的关系
    lines.append("0\t2\t1\t72\t0")
    (tmp_path / "train.txt").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "valid.txt").write_text("", encoding="utf-8")
    (tmp_path / "test.txt").write_text("", encoding="utf-8")
    return tmp_path


def test_load_icews_parses_maps_and_events(tmp_path):
    root = _mini_dataset(tmp_path)
    data = load_icews(root)
    assert data.entities[0] == "Kim Jong-Un"
    assert data.relations[2] == "Use conventional military force"
    assert data.n_events == 41
    assert data.timestamps == [0, 24, 48, 72]
    assert np.all(np.diff(data.events[:, 3]) >= 0)      # 按时间排序


def test_make_slices_partitions_all_events(tmp_path):
    data = load_icews(_mini_dataset(tmp_path))
    slices = make_slices(data, 4)
    assert len(slices) == 4
    assert sum(len(s) for s in slices) == data.n_events
    # 时间片之间时间不重叠且递增
    prev_hi = -1
    for s in slices:
        assert int(s[:, 3].min()) > prev_hi
        prev_hi = int(s[:, 3].max())


def test_sample_slice_keeps_every_relation(tmp_path):
    data = load_icews(_mini_dataset(tmp_path))
    sl = make_slices(data, 1)[0]
    sampled = sample_slice(sl, 12, seed=1, min_per_rel=1)
    assert len(sampled) <= 12
    assert set(sampled[:, 1].tolist()) == set(sl[:, 1].tolist())


def test_sample_slice_is_deterministic(tmp_path):
    data = load_icews(_mini_dataset(tmp_path))
    sl = make_slices(data, 1)[0]
    a = sample_slice(sl, 12, seed=7)
    b = sample_slice(sl, 12, seed=7)
    assert np.array_equal(a, b)


def test_sample_slice_noop_when_target_larger(tmp_path):
    data = load_icews(_mini_dataset(tmp_path))
    sl = make_slices(data, 1)[0]
    assert len(sample_slice(sl, 999, seed=0)) == len(sl)


def test_role_and_country_parsing():
    assert role_of("Media Personnel (Afghanistan)") == "Media Personnel"
    assert country_of("Media Personnel (Afghanistan)") == "Afghanistan"
    assert role_of("North Korea") == "North Korea"
    assert country_of("North Korea") == ""


def test_slice_stat_counts(tmp_path):
    data = load_icews(_mini_dataset(tmp_path))
    st = slice_stat(data, make_slices(data, 1)[0])
    assert st["events"] == 41 and st["relations"] == 3
    assert 2 <= st["entities"] <= 5
