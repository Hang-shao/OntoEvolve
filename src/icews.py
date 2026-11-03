"""ontoevolve.src.icews - ICEWS18 真实时序知识图谱的加载、切片与抽样。

数据格式（RE-Net 发布版）：
    entity2id.txt / relation2id.txt : "<名称>\t<id>"
    {train,valid,test}.txt          : "<subj>\t<rel>\t<obj>\t<ts>\t<?>"
实体名自带角色与国别（如 "Media Personnel (Afghanistan)"），可直接作为
概念的域/值域类型使用，无需额外类型标注。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C


@dataclass
class ICEWSData:
    entities: List[str]
    relations: List[str]
    events: np.ndarray                 # (n, 4) int32: subj, rel, obj, ts
    timestamps: List[int] = field(default_factory=list)

    @property
    def n_events(self) -> int:
        return int(self.events.shape[0])

    def text(self, e) -> str:
        s, r, o, _ = int(e[0]), int(e[1]), int(e[2]), int(e[3])
        return (f"{self.entities[s]} {self.relations[r]} {self.entities[o]}")


def _load_map(path: Path) -> List[str]:
    """读取 "<name>\\t<id>" 映射，返回按 id 排序的名称列表。"""
    pairs: List[Tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        name, sid = line.rsplit("\t", 1)
        pairs.append((int(sid.strip()), name.strip()))
    pairs.sort()
    return [n for _, n in pairs]


def load_icews(root: Optional[Path] = None, *,
               splits: Sequence[str] = ("train", "valid", "test")) -> ICEWSData:
    root = Path(root or (C.DATA_ROOT / "01_带时间戳的时序数据"
                         / "TemporalKG_RE-Net" / "data" / "ICEWS18"))
    entities = _load_map(root / "entity2id.txt")
    relations = _load_map(root / "relation2id.txt")
    rows: List[Tuple[int, int, int, int]] = []
    for split in splits:
        f = root / f"{split}.txt"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) < 4:
                continue
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]),
                         int(parts[3])))
    ev = np.asarray(rows, dtype=np.int64)
    ev = ev[np.argsort(ev[:, 3], kind="stable")]
    return ICEWSData(entities=entities, relations=relations, events=ev,
                     timestamps=sorted({int(t) for t in ev[:, 3]}))


def make_slices(data: ICEWSData, n_slices: int = 24) -> List[np.ndarray]:
    """按时间戳等分成 n_slices 个时间片（保持时序顺序）。"""
    ts = np.asarray(data.timestamps, dtype=np.int64)
    if n_slices <= 0 or len(ts) == 0:
        return []
    bounds = np.array_split(ts, min(n_slices, len(ts)))
    slices: List[np.ndarray] = []
    for chunk in bounds:
        lo, hi = int(chunk[0]), int(chunk[-1])
        mask = (data.events[:, 3] >= lo) & (data.events[:, 3] <= hi)
        slices.append(data.events[mask])
    return slices


def sample_slice(slice_events: np.ndarray, n_target: int, *, seed: int = 0,
                 min_per_rel: int = 1) -> np.ndarray:
    """按关系分层的确定性抽样。

    先保证每个关系至少 min_per_rel 条（低频关系保底），剩余名额按关系频次
    比例分配；避免抽样把稀有关系抹掉而人为制造“淘汰”假象。
    """
    n = int(slice_events.shape[0])
    if n_target >= n:
        return slice_events
    rng = np.random.default_rng(seed)
    by_rel: Dict[int, np.ndarray] = {}
    for rel in np.unique(slice_events[:, 1]):
        by_rel[int(rel)] = np.flatnonzero(slice_events[:, 1] == rel)
    picked: List[int] = []
    for rel, idx in by_rel.items():
        take = min(min_per_rel, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    remaining = n_target - len(picked)
    if remaining > 0:
        weights = np.asarray([len(v) for v in by_rel.values()], dtype=float)
        weights = weights / weights.sum()
        counts = rng.multinomial(remaining, weights)
        for (rel, idx), k in zip(by_rel.items(), counts):
            if k <= 0:
                continue
            already = {i for i in picked if slice_events[i, 1] == rel}
            pool = np.array([i for i in idx if i not in already], dtype=int)
            if len(pool) == 0:
                continue
            k = min(int(k), len(pool))
            picked.extend(rng.choice(pool, size=k, replace=False).tolist())
    picked = sorted(set(int(i) for i in picked))
    return slice_events[np.asarray(picked, dtype=int)]


def role_of(entity_name: str) -> str:
    """去掉实体名末尾的国别/地区括号，得到角色类型。"""
    name = entity_name.strip()
    if name.endswith(")") and "(" in name:
        head = name[: name.rfind("(")].strip()
        return head if head else name
    return name


def country_of(entity_name: str) -> str:
    name = entity_name.strip()
    if name.endswith(")") and "(" in name:
        return name[name.rfind("(") + 1: -1].strip()
    return ""


def slice_stat(data: ICEWSData, slice_events: np.ndarray) -> Dict[str, int]:
    return {
        "events": int(slice_events.shape[0]),
        "entities": int(len(set(slice_events[:, 0].tolist())
                            | set(slice_events[:, 2].tolist()))),
        "relations": int(len(set(slice_events[:, 1].tolist()))),
    }
