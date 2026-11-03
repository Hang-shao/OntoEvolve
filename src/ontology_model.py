"""ontoevolve.src.ontology_model - 轻量本体模型 + 公理签名 + Project 操作。

包含：Project 操作、公理签名（axiom_signature）与 L1 距离_distance、2.3 证据因子。
为避免重型推理依赖，本体以“概念 + 公理列表”表示，可直接导出 Turtle（见
src/serialize.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .vocab import (CARD_CAP, DISJOINT_CAP, DP_VOCAB, MAX_DOMAIN, MAX_RANGE,
                    OP_VOCAB)

AXIOM_KINDS = ("OP", "DP", "DOMAIN", "RANGE", "CARD", "TRANSITIVE", "DISJOINT")


@dataclass
class Axiom:
    """一条公理。kind 决定字段含义。"""

    kind: str
    subj: str
    pred: str = ""
    obj: str = ""
    card_min: int = 0
    card_max: int = 0
    confidence: float = 1.0
    evidence_count: int = 0
    source_doc_count: int = 0
    batch_cover_count: int = 0

    def conflict_key(self) -> Tuple[str, str]:
        """冲突消解键：同一 (kind, pred) 只保留置信度最高的一条。"""
        return (self.kind, self.pred)


def concepts_of_axiom(ax: Axiom) -> Set[str]:
    """公理涉及的概念集合（obj 为概念 id 时计入）。"""
    out = {ax.subj}
    if ax.kind in ("DOMAIN", "RANGE", "DISJOINT") and ax.obj:
        out.add(ax.obj)
    return out


def source_evidence_count_factor(ax: Axiom, *, ref_count: int = 4,
                                 ref_docs: int = 4, ref_batch: int = 3) -> float:
    """：证据支持因子（出现次数 / 文档数 / 批次数，截断归一后加权）。

    参考基准 ref_count / ref_docs 由调用方给出：extract_candidates 取“该批中
    该概念最强候选公理的证据量”作为基准（证据相对强度）。
    这样既保留“只出现一次的噪声应被压低”的绝对计数语义，
    又能在概念规模从 4 变到 70（真实数据）时自动缩放。
    """
    c = min(1.0, ax.evidence_count / max(1, ref_count))
    d = min(1.0, ax.source_doc_count / max(1, ref_docs))
    b = min(1.0, ax.batch_cover_count / max(1, ref_batch))
    return 0.4 * c + 0.3 * d + 0.3 * b


@dataclass
class Concept:
    cid: str
    name: str
    parent: Optional[str] = None
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    axioms: List[Axiom] = field(default_factory=list)
    active: bool = True
    first_batch: int = 0
    last_batch: int = 0
    counts: List[int] = field(default_factory=list)      # 每批新增实例数
    weights: List[float] = field(default_factory=list)   # 内部语义权重
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def in_degree_proxy(self) -> float:
        """内部语义权重分量：关系入度（以 RANGE/OP 公理数近似）。"""
        return float(sum(1 for a in self.axioms if a.kind in ("OP", "RANGE")))

    def internal_weight(self, *, density: float = 0.0, prop_freq: float = 0.0) -> float:
        """淘汰判定②：入度 + 实例密度 + 属性频率。"""
        return float(self.in_degree_proxy + density + prop_freq)


class Ontology:
    """概念集合 + 全局属性词表（保证公理签名定长）。"""

    def __init__(self, concepts: Optional[Dict[str, Concept]] = None,
                 op_vocab: Sequence[str] = OP_VOCAB,
                 dp_vocab: Sequence[str] = DP_VOCAB,
                 version: int = 0):
        self.concepts: Dict[str, Concept] = concepts or {}
        self.op_vocab: List[str] = list(op_vocab)
        self.dp_vocab: List[str] = list(dp_vocab)
        self.version = version
        self.M = len(self.op_vocab) + len(self.dp_vocab) + 6

    # ---- 基本操作 --------------------------------------------------------
    def add(self, concept: Concept) -> None:
        self.concepts[concept.cid] = concept

    def get(self, cid: str) -> Optional[Concept]:
        return self.concepts.get(cid)

    def axioms_of(self, cid: str) -> List[Axiom]:
        c = self.concepts.get(cid)
        return list(c.axioms) if c else []

    def active_ids(self) -> List[str]:
        return [cid for cid, c in self.concepts.items() if c.active]

    # ---- 公理签名-------------------------------------------
    def axiom_signature(self, cid: str) -> np.ndarray:
        """定长 M = m1 + m2 + 6 的公理签名，各分量归一化到 [0,1]。"""
        c = self.concepts.get(cid)
        m1, m2 = len(self.op_vocab), len(self.dp_vocab)
        vec = np.zeros(m1 + m2 + 6, dtype=np.float64)
        if c is None:
            return vec
        ops: Set[str] = set()
        ranges: Set[str] = set()
        domains: Set[str] = set()
        dps: Set[str] = set()
        card_min, card_max, transitive, disjoint = 0, 0, 0.0, 0
        for a in c.axioms:
            if a.kind == "OP":
                ops.add(a.pred)
                if a.obj:
                    ranges.add(a.obj)
            elif a.kind == "DP":
                dps.add(a.pred)
            elif a.kind == "DOMAIN":
                domains.add(a.obj)
            elif a.kind == "RANGE":
                ranges.add(a.obj)
            elif a.kind == "CARD":
                card_min = max(card_min, a.card_min)
                card_max = max(card_max, a.card_max)
            elif a.kind == "TRANSITIVE":
                transitive = 1.0
            elif a.kind == "DISJOINT":
                disjoint += 1
        for i, op in enumerate(self.op_vocab):
            vec[i] = 1.0 if op in ops else 0.0
        for i, dp in enumerate(self.dp_vocab):
            vec[m1 + i] = 1.0 if dp in dps else 0.0
        base = m1 + m2
        vec[base + 0] = min(1.0, len(domains) / MAX_DOMAIN)
        vec[base + 1] = min(1.0, len(ranges) / MAX_RANGE)
        vec[base + 2] = min(1.0, card_min / CARD_CAP)
        vec[base + 3] = min(1.0, (card_max if card_max > 0 else int(CARD_CAP)) / CARD_CAP)
        vec[base + 4] = transitive
        vec[base + 5] = min(1.0, disjoint / DISJOINT_CAP)
        return vec

    # ---- Project--------------------------------------------
    def project(self, delta: Iterable[Axiom], cid: str, *,
                tau_conf: float = 0.0, support_decay: float = 1.0,
                decay_if_new: bool = True) -> "Ontology":
        """只使用数据流抽取的候选公理 delta_A + O_t 中与 c 相关的公理。

        步骤：实体链接 → 相关公理 → 合并 → 冲突消解（保留最高置信度）→
        低置信度过滤 → 构建子本体。

        support_decay < 1 时，若 delta_A 中出现了 O_t 未见过的新谓词，则对
        “本批次不再有证据支持”的旧公理做置信度衰减（重定义：旧内涵失去支撑）。
        """
        delta = list(delta)
        related = [a for a in delta if cid in concepts_of_axiom(a)]
        base = [a for a in self.axioms_of(cid)]
        if support_decay < 1.0:
            base_keys = {(a.kind, a.pred) for a in base}
            delta_keys = {(a.kind, a.pred) for a in delta}
            has_new = bool(delta_keys - base_keys)
            if has_new or not decay_if_new:
                base = [replace(a, confidence=a.confidence * support_decay)
                        if (a.kind, a.pred) not in delta_keys else a
                        for a in base]
        merged = base + related
        resolved = resolve_conflicts(merged)
        resolved = [a for a in resolved if a.confidence >= tau_conf]
        src = self.concepts.get(cid)
        sub = Concept(
            cid=cid,
            name=src.name if src else cid,
            parent=src.parent if src else None,
            centroid=src.centroid if src is not None else np.zeros(1, np.float32),
            axioms=resolved,
            meta={"projected_from": self.version},
        )
        return Ontology({cid: sub}, op_vocab=self.op_vocab,
                        dp_vocab=self.dp_vocab, version=self.version)


def resolve_conflicts(axioms: Iterable[Axiom]) -> List[Axiom]:
    """同一 (kind, pred) 冲突时保留置信度最高的一条（keep_highest_conf）。"""
    best: Dict[Tuple[str, str], Axiom] = {}
    for a in axioms:
        key = a.conflict_key()
        cur = best.get(key)
        if cur is None or a.confidence > cur.confidence:
            best[key] = a
    return list(best.values())


def L1_distance(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """：两个定长归一化向量的 L1（曼哈顿）距离。"""
    a = np.asarray(vec1, dtype=np.float64)
    b = np.asarray(vec2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"signature shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sum(np.abs(a - b)))
