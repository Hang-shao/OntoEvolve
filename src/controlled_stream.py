"""ontoevolve.src.controlled_stream - 轨道 1 受控数据流。

生成 D_0 + D_1..D_T，并按标注规则从“实际发生的数据”反推三类漂移真值：
    涌现    ：连续 κ 个批次均出现新概念实例
    淘汰    ：连续 W 个批次的平均新增实例数 < ε_ext
    重定义  ：某批中 ≥50% 的实例呈现新的属性/关系模式
困难样本属性：噪声 / 渐变 / 周期 / 低频，用于分组报告。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .embedding import Embedder
from .vocab import (ADJECTIVES, BASE_TREE, EMERGENCE_TREE, OP_VOCAB,
                    RANGE_VOCAB, LeafSchema, build_vocab, instance_text)

NOISE_LEAF = "__noise__"


@dataclass
class PoolItem:
    iid: int
    leaf: str
    sup: str
    variant: str
    text: str
    op_slots: List[Tuple[str, str]]
    dp_slots: List[Tuple[str, str]]
    vec: Optional[np.ndarray] = None


@dataclass
class Occurrence:
    """一次实例出现（批次中的一条记录）。"""
    iid: int
    leaf: str
    variant: str
    text: str
    vec: np.ndarray
    op_slots: List[Tuple[str, str]]
    dp_slots: List[Tuple[str, str]]
    sup: str = ""
    noise: bool = False

    def as_dict(self) -> Dict:
        return {"text": self.text, "op_slots": self.op_slots,
                "dp_slots": self.dp_slots}


@dataclass
class DriftEvent:
    leaf: str
    dtype: str                  # emergence | extinction | redefinition
    t_start: int
    t_declare: int
    attrs: Dict[str, bool] = field(default_factory=dict)

    @property
    def difficulty(self) -> str:
        for k in ("gradual", "periodic", "lowfreq"):
            if self.attrs.get(k):
                return k
        return "abrupt"


@dataclass
class ControlledStream:
    pool: List[PoolItem]
    d0_iids: List[int]
    batches: List[List[Occurrence]]
    events: List[DriftEvent]
    vocab_base: Dict[str, LeafSchema]
    vocab_emer: Dict[str, LeafSchema]
    emergence_leaves: List[str]
    seed: int


# -----------------------------------------------------------------------------
# 事件选取（在构建实例池之前确定，只为重定义目标生成 alt 变体）
# -----------------------------------------------------------------------------
def select_events(vocab_base: Dict[str, LeafSchema],
                  vocab_emer: Dict[str, LeafSchema], *, seed: int = 0,
                  params: Optional[C.DriftParams] = None) -> List[Dict]:
    params = params or C.PARAMS
    rng = np.random.default_rng(seed + 13)
    T = params.n_batch
    base_leaves = list(vocab_base.keys())
    emer_leaves = list(vocab_emer.keys())

    def make_attrs() -> Dict[str, bool]:
        r = float(rng.random())
        g, p, l = params.gradual_ratio, params.periodic_ratio, params.lowfreq_ratio
        if r < g:
            return {"gradual": True}
        if r < g + p:
            return {"periodic": True}
        if r < g + p + l:
            return {"lowfreq": True}
        return {}

    emer_pick = [str(x) for x in rng.choice(
        emer_leaves, size=min(params.n_emergence, len(emer_leaves)),
        replace=False)]
    rest = [x for x in base_leaves]
    ext_idx = rng.choice(len(rest), size=min(params.n_extinction, len(rest)),
                         replace=False)
    ext_leaves = [rest[int(i)] for i in ext_idx]
    cand = [x for x in rest if x not in set(ext_leaves)]
    red_idx = rng.choice(len(cand), size=min(params.n_redefinition, len(cand)),
                         replace=False)
    red_leaves = [cand[int(i)] for i in red_idx]

    metas: List[Dict] = []
    for leaf in emer_pick:
        metas.append({"leaf": leaf, "dtype": "emergence", "ramp": 3,
                      "t_start": int(rng.integers(3, max(4, T // 2))),
                      "attrs": make_attrs()})
    for leaf in ext_leaves:
        metas.append({"leaf": leaf, "dtype": "extinction", "ramp": 3,
                      "t_start": int(rng.integers(max(4, T // 3),
                                                  max(5, 2 * T // 3))),
                      "attrs": make_attrs()})
    for leaf in red_leaves:
        metas.append({"leaf": leaf, "dtype": "redefinition", "ramp": 4,
                      "t_start": int(rng.integers(3, max(4, 2 * T // 3))),
                      "attrs": make_attrs()})
    return metas


# -----------------------------------------------------------------------------
# 实例池
# -----------------------------------------------------------------------------
def build_pool(embedder: Embedder, vocab_base: Dict[str, LeafSchema],
               vocab_emer: Dict[str, LeafSchema], *, seed: int = 0,
               params: Optional[C.DriftParams] = None,
               alt_leaves: Sequence[str] = ()) -> Tuple[
                   List[PoolItem], Dict[str, List[int]]]:
    """构建实例池（一次性嵌入，各重复实验复用向量）。"""
    params = params or C.PARAMS
    alt_set = set(alt_leaves)
    all_vocab = {**vocab_base, **vocab_emer}
    rng = np.random.default_rng(seed + 7)

    pool: List[PoolItem] = []
    iid = 0
    for leaf, schema in all_vocab.items():
        for k in range(params.pool_per_leaf):
            adj = ADJECTIVES[(k + (iid % 5)) % len(ADJECTIVES)]
            pool.append(PoolItem(
                iid=iid, leaf=leaf, sup=schema.parent, variant="base",
                text=instance_text(adj, schema, variant="base"),
                op_slots=list(schema.ops), dp_slots=list(schema.dps)))
            iid += 1
        if leaf in alt_set:
            for k in range(params.pool_per_leaf):
                adj = ADJECTIVES[(k + 3) % len(ADJECTIVES)]
                pool.append(PoolItem(
                    iid=iid, leaf=leaf, sup=schema.parent, variant="alt",
                    text=instance_text(adj, schema, variant="alt"),
                    op_slots=list(schema.alt_ops), dp_slots=list(schema.dps)))
                iid += 1

    # 噪声实例：文本与普通实例同分布（避免制造假的“新域”语义信号），
    # 但关系/属性槽位被随机替换 —— 模拟公理抽取错误，用于检验 τ_conf 过滤。
    base_names = list(vocab_base.keys())
    for k in range(params.pool_per_leaf):
        schema = all_vocab[base_names[int(rng.integers(0, len(base_names)))]]
        adj = ADJECTIVES[(k + 11) % len(ADJECTIVES)]
        ops = [(str(rng.choice(OP_VOCAB)), str(rng.choice(RANGE_VOCAB)))
               for _ in range(2)]
        dps = [(str(d), str(rng.choice(["low", "medium", "high"])))
               for d, _ in schema.dps]
        pool.append(PoolItem(iid=iid, leaf=NOISE_LEAF, sup=schema.parent,
                             variant="base",
                             text=instance_text(adj, schema, variant="base"),
                             op_slots=ops, dp_slots=dps))
        iid += 1

    vecs = embedder.embed([p.text for p in pool])
    for p, v in zip(pool, vecs):
        p.vec = np.asarray(v, dtype=np.float32)

    alt_pool: Dict[str, List[int]] = {}
    for p in pool:
        if p.variant == "alt":
            alt_pool.setdefault(p.leaf, []).append(p.iid)
    return pool, alt_pool


# -----------------------------------------------------------------------------
# 批次生成
# -----------------------------------------------------------------------------
def _rate_multiplier(t: int, attrs: Dict[str, bool]) -> float:
    m = 1.0
    if attrs.get("lowfreq"):
        m *= 0.35
    if attrs.get("periodic"):
        m *= 0.25 + 1.5 * (0.5 + 0.5 * math.sin(2 * math.pi * t / 8.0))
    return m


def generate_stream(pool: Sequence[PoolItem], vocab_base: Dict[str, LeafSchema],
                    vocab_emer: Dict[str, LeafSchema], metas: Sequence[Dict],
                    *, seed: int = 0, params: Optional[C.DriftParams] = None,
                    alt_pool: Optional[Dict[str, List[int]]] = None) -> ControlledStream:
    params = params or C.PARAMS
    alt_pool = alt_pool or {}
    rng = np.random.default_rng(seed + 101)
    T = params.n_batch

    by_leaf: Dict[str, List[int]] = {}
    by_leaf_variant: Dict[Tuple[str, str], List[int]] = {}
    for p in pool:
        by_leaf.setdefault(p.leaf, []).append(p.iid)
        by_leaf_variant.setdefault((p.leaf, p.variant), []).append(p.iid)
    base_leaves = list(vocab_base.keys())
    emer_pick = [m["leaf"] for m in metas if m["dtype"] == "emergence"]
    meta_by_leaf = {m["leaf"]: m for m in metas}

    batches: List[List[Occurrence]] = []
    for t in range(1, T + 1):
        items: List[Occurrence] = []
        for leaf in base_leaves + emer_pick:
            meta = meta_by_leaf.get(leaf)
            attrs = meta["attrs"] if meta else {}
            rate = params.base_rate * _rate_multiplier(t, attrs)
            variant, active = "base", True
            if meta is not None:
                dt, t0, ramp = meta["dtype"], meta["t_start"], meta["ramp"]
                if dt == "emergence":
                    if t < t0:
                        active = False
                    elif attrs.get("gradual"):
                        rate *= min(1.0, (t - t0 + 1) / ramp)
                    else:
                        rate *= 1.2
                elif dt == "extinction" and t >= t0:
                    if attrs.get("gradual"):
                        rate *= max(0.0, 1.0 - (t - t0 + 1) / ramp)
                    else:
                        rate = 0.0
                elif dt == "redefinition" and t >= t0:
                    variant = "alt" if (
                        not attrs.get("gradual")
                        or rng.random() < min(1.0, (t - t0 + 1) / ramp)) else "base"
            if not active or rate <= 0:
                continue
            n = int(rng.poisson(max(rate, 0.0)))
            src = by_leaf_variant.get((leaf, variant), [])
            if n <= 0 or not src:
                continue
            for iid in rng.choice(src, size=min(n, len(src)), replace=False):
                p = pool[int(iid)]
                items.append(Occurrence(iid=p.iid, leaf=leaf, variant=p.variant,
                                        text=p.text, vec=p.vec,
                                        op_slots=list(p.op_slots),
                                        dp_slots=list(p.dp_slots), sup=p.sup))
        if params.noise_ratio > 0 and NOISE_LEAF in by_leaf:
            n_noise = max(1, int(round(params.noise_ratio * params.base_rate
                                       * len(base_leaves) / 4.0)))
            src = by_leaf[NOISE_LEAF]
            for iid in rng.choice(src, size=min(n_noise, len(src)), replace=False):
                p = pool[int(iid)]
                items.append(Occurrence(iid=p.iid, leaf=NOISE_LEAF,
                                        variant="base", text=p.text, vec=p.vec,
                                        op_slots=list(p.op_slots),
                                        dp_slots=list(p.dp_slots), sup=p.sup,
                                        noise=True))
        batches.append(items)

    events: List[DriftEvent] = []
    for m in metas:
        t_dec = _declare_batch(batches, m["leaf"], m["dtype"], params,
                               t_start=int(m["t_start"]))
        if t_dec is None:
            continue
        events.append(DriftEvent(leaf=m["leaf"], dtype=m["dtype"],
                                 t_start=int(m["t_start"]), t_declare=t_dec,
                                 attrs=dict(m["attrs"])))

    d0_iids: List[int] = []
    for leaf in base_leaves:
        src = by_leaf_variant.get((leaf, "base"), [])
        d0_iids.extend(src[: params.n_init_per_leaf])

    return ControlledStream(pool=list(pool), d0_iids=d0_iids, batches=batches,
                            events=events, vocab_base=vocab_base,
                            vocab_emer=vocab_emer, emergence_leaves=emer_pick,
                            seed=seed)


def _declare_batch(batches: Sequence[Sequence[Occurrence]], leaf: str,
                   dtype: str, params: C.DriftParams,
                   *, t_start: int = 1) -> Optional[int]:
    """依据的可操作判定标准，从实际数据中给出真值声明批次。

    扫描从注入起点 t_start 开始，避免低频概念因随机波动被提前判为淘汰
    。
    """
    T = len(batches)
    start = max(0, t_start - 1)
    if dtype == "emergence":
        cnt = [sum(1 for o in b if o.leaf == leaf) for b in batches]
        for t in range(start + params.kappa - 1, T):
            if all(cnt[t - i] >= 1 for i in range(params.kappa)):
                return t + 1
        return None
    if dtype == "extinction":
        cnt = [sum(1 for o in b if o.leaf == leaf) for b in batches]
        W = params.window_w
        for t in range(start + W - 1, T):
            if float(np.mean(cnt[t - W + 1: t + 1])) < params.eps_ext:
                return t + 1
        return None
    if dtype == "redefinition":
        for t in range(start, T):
            items = [o for o in batches[t] if o.leaf == leaf]
            if len(items) >= 2:
                alt = sum(1 for o in items if o.variant == "alt")
                if alt / len(items) >= 0.5:
                    return t + 1
        return None
    raise ValueError(f"unknown drift type: {dtype}")


def make_controlled_stream(embedder: Embedder, *, seed: int = 0,
                           params: Optional[C.DriftParams] = None,
                           vocab_seed: int = 0,
                           alt_all_base: bool = True) -> ControlledStream:
    """一键构建：词表 → 事件 → 实例池 → 批次 → 真值。

    alt_all_base=True 时为所有基类概念生成“重定义”变体，使实例池与随机种子
    无关 —— 多次重复实验可共享同一批嵌入向量（命中缓存，无需重复调用接口）。
    """
    params = params or C.PARAMS
    vocab_base = build_vocab(seed=vocab_seed, tree=BASE_TREE)
    vocab_emer = build_vocab(seed=vocab_seed + 1000, tree=EMERGENCE_TREE)
    metas = select_events(vocab_base, vocab_emer, seed=seed, params=params)
    if alt_all_base:
        alt_leaves = list(vocab_base.keys())
    else:
        alt_leaves = [m["leaf"] for m in metas if m["dtype"] == "redefinition"]
    pool, alt_pool = build_pool(embedder, vocab_base, vocab_emer, seed=vocab_seed,
                                params=params, alt_leaves=alt_leaves)
    return generate_stream(pool, vocab_base, vocab_emer, metas, seed=seed,
                           params=params, alt_pool=alt_pool)
