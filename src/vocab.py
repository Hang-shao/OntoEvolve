"""ontoevolve.src.vocab - 受控数据流的概念词表与实例文本生成。

轨道 1使用受控合成数据：真实概念名 + 真实嵌入 + 受控注入的
三类漂移。概念词表固定，属性/关系槽位由种子确定性生成，便于复现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---- 本体规模常量（公理签名定长 M = m1 + m2 + 6）---------------------------
OP_VOCAB: List[str] = ["livesIn", "eats", "usedFor", "partOf", "madeOf", "locatedNear"]
DP_VOCAB: List[str] = ["size", "color", "lifespan", "weight", "speed", "diet"]
RANGE_VOCAB: List[str] = ["Forest", "Ocean", "City", "Farm", "Factory",
                          "Laboratory", "Market", "Mountain", "Desert", "Hospital"]
DP_VALUES: Dict[str, List[str]] = {
    "size": ["small", "medium", "large"],
    "color": ["dark", "bright", "pale"],
    "lifespan": ["short", "annual", "long"],
    "weight": ["light", "medium", "heavy"],
    "speed": ["slow", "moderate", "fast"],
    "diet": ["plant-based", "mixed", "protein-rich"],
}

MAX_DOMAIN = 4.0
MAX_RANGE = 4.0
CARD_CAP = 8.0
DISJOINT_CAP = 4.0

BASE_TREE: Dict[str, List[str]] = {
    "Animal": ["Mammal", "Bird", "Fish", "Insect", "Reptile", "Amphibian"],
    "Vehicle": ["Car", "Boat", "Aircraft", "Bicycle", "Train", "Motorcycle"],
    "Food": ["Fruit", "Vegetable", "Meat", "Dairy", "Grain", "Dessert"],
    "Tool": ["Hammer", "Saw", "Drill", "Wrench", "Screwdriver", "Pliers"],
    "Plant": ["Tree", "Flower", "Grass", "Fern", "Moss", "Shrub"],
}

# 涌现概念来自与初始本体不同的域，保证与已有锚点语义距离足够大
EMERGENCE_TREE: Dict[str, List[str]] = {
    "Music": ["Guitar", "Drum", "Flute"],
    "Sport": ["Soccer", "Tennis", "Skiing"],
}

ADJECTIVES: List[str] = [
    "large", "small", "ancient", "modern", "common", "rare",
    "tropical", "arctic", "coastal", "mountain", "urban", "rural",
    "bright", "dark", "heavy", "light", "fast", "slow",
    "young", "old", "wild", "domestic", "giant", "tiny",
    "smooth", "rough", "shiny", "dull", "dense", "sparse",
    "round", "flat", "noisy", "quiet", "warm", "cold",
    "solid", "fragile", "simple", "complex", "early", "late",
]


@dataclass
class LeafSchema:
    """一个叶子概念的属性/关系槽位（含“重定义”变体）。"""

    name: str
    parent: str
    dps: List[Tuple[str, str]] = field(default_factory=list)      # [(dp, value)]
    ops: List[Tuple[str, str]] = field(default_factory=list)      # [(op, range)]
    alt_ops: List[Tuple[str, str]] = field(default_factory=list)  # 重定义后的关系
    card_min: int = 1
    card_max: int = 3
    alt_card_min: int = 2
    alt_card_max: int = 6
    transitive: bool = False
    disjoint: List[str] = field(default_factory=list)

    def ops_for(self, variant: str) -> List[Tuple[str, str]]:
        return self.alt_ops if variant == "alt" else self.ops

    def card_for(self, variant: str) -> Tuple[int, int]:
        return (self.alt_card_min, self.alt_card_max) if variant == "alt" \
            else (self.card_min, self.card_max)


def build_vocab(*, seed: int = 0, tree: Dict[str, List[str]] | None = None,
                prefix: str = "") -> Dict[str, LeafSchema]:
    """确定性生成每个叶子概念的槽位模式。"""
    tree = tree or BASE_TREE
    rng = np.random.default_rng(seed)
    schemas: Dict[str, LeafSchema] = {}
    all_leaves = [f"{sup}.{leaf}" for sup, leaves in tree.items() for leaf in leaves]
    for sup, leaves in tree.items():
        for leaf in leaves:
            name = f"{prefix}{sup}.{leaf}"
            dps = [(dp, str(DP_VALUES[dp][int(i)]))
                   for dp, i in zip(rng.choice(DP_VOCAB, size=3, replace=False),
                                    rng.integers(0, 3, size=3))]
            ops = [(op, str(rng.choice(RANGE_VOCAB)))
                   for op in rng.choice(OP_VOCAB, size=2, replace=False)]
            # 重定义变体：保留首个关系，替换其余并新增一个（属性 + 值域 + 基数）
            keep_op, keep_rng = ops[0]
            pool_ops = [o for o in OP_VOCAB if o != keep_op]
            picked = [str(x) for x in rng.choice(pool_ops, size=2, replace=False)]
            alt_ops = [(keep_op, keep_rng)] + [
                (op, str(rng.choice([r for r in RANGE_VOCAB if r != keep_rng])))
                for op in picked]
            others = [x for x in all_leaves if not x.endswith(f".{leaf}")]
            disjoint = [str(x) for x in rng.choice(others, size=2, replace=False)] \
                if len(others) >= 2 else []
            schemas[name] = LeafSchema(
                name=name, parent=f"{prefix}{sup}", dps=dps, ops=ops,
                alt_ops=alt_ops,
                card_min=int(rng.integers(0, 2)),
                card_max=int(rng.integers(2, 5)),
                alt_card_min=int(rng.integers(2, 4)),
                alt_card_max=int(rng.integers(5, 9)),
                transitive=bool(rng.integers(0, 2)),
                disjoint=disjoint,
            )
    return schemas


def instance_text(adj: str, schema: LeafSchema, *, variant: str = "base") -> str:
    """生成实例描述文本。"""
    dp_str = ", ".join(f"{d}={v}" for d, v in schema.dps)
    op_str = ", ".join(f"{o}->{r}" for o, r in schema.ops_for(variant))
    return (f"{adj} {schema.name.split('.')[-1]} is a kind of "
            f"{schema.parent.split('.')[-1]}. Attributes: {dp_str}. "
            f"Relations: {op_str}.")
