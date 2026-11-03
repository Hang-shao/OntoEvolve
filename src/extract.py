"""ontoevolve.src.extract - 从数据流实例抽取候选公理 delta_A。

轨道 1 使用确定性抽取（不调用 LLM），置信度 = LLM 判定置信度(默认 1.0) ×
证据支持因子；可选的 LLM 判定见 src/llm.py。
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ontology_model import Axiom, source_evidence_count_factor


def extract_candidates(
    cid: str,
    items: Sequence[Dict],
    *,
    batch_idx: int = 0,
    parent: Optional[str] = None,
    transitive_ops: Sequence[str] = (),
    disjoint: Sequence[str] = (),
    ref_ratio: Optional[float] = None,
    ref_batch: int = 3,
) -> List[Axiom]:
    """从一个批次中链接到概念 cid 的实例抽取候选公理。

    items 中每项需包含：{'text': str, 'op_slots': [(op, range)],
    'dp_slots': [(dp, value)]}
    """
    if not items:
        return []
    op_counter: Counter = Counter()
    op_docs: Dict[Tuple[str, str], set] = {}
    dp_counter: Counter = Counter()
    dp_docs: Dict[Tuple[str, str], set] = {}
    n_min, n_max = 10**9, 0
    for it in items:
        ops = it.get("op_slots", [])
        n_min, n_max = min(n_min, len(ops)), max(n_max, len(ops))
        for pair in ops:
            key = (str(pair[0]), str(pair[1]))
            op_counter[key] += 1
            op_docs.setdefault(key, set()).add(it.get("text", ""))
        for pair in it.get("dp_slots", []):
            key = (str(pair[0]), str(pair[1]))
            dp_counter[key] += 1
            dp_docs.setdefault(key, set()).add(it.get("text", ""))
    if n_min > n_max:
        n_min = n_max = 0

    out: List[Axiom] = []
    n_items = len(items)

    def _mk(**kw) -> Axiom:
        return Axiom(**kw)

    for (op, rng), cnt in op_counter.items():
        out.append(_mk(kind="OP", subj=cid, pred=op, obj=rng,
                       evidence_count=cnt,
                       source_doc_count=len(op_docs.get((op, rng), ())),
                       batch_cover_count=1))
    for (dp, val), cnt in dp_counter.items():
        out.append(_mk(kind="DP", subj=cid, pred=dp, obj=val,
                       evidence_count=cnt,
                       source_doc_count=len(dp_docs.get((dp, val), ())),
                       batch_cover_count=1))
    if parent:
        out.append(_mk(kind="DOMAIN", subj=cid, obj=parent,
                       evidence_count=n_items, source_doc_count=n_items,
                       batch_cover_count=1))
    range_support: Counter = Counter()
    for (op, rng), cnt in op_counter.items():
        range_support[rng] += cnt
    for rng, cnt in range_support.items():
        out.append(_mk(kind="RANGE", subj=cid, obj=rng, evidence_count=cnt,
                       source_doc_count=cnt, batch_cover_count=1))
    out.append(_mk(kind="CARD", subj=cid, card_min=int(n_min), card_max=int(n_max),
                   evidence_count=n_items, source_doc_count=n_items,
                   batch_cover_count=1))
    for op in transitive_ops:
        if any(o == op for o, _ in op_counter):
            out.append(_mk(kind="TRANSITIVE", subj=cid, pred=str(op),
                           evidence_count=n_items, source_doc_count=n_items,
                           batch_cover_count=1))
    for other in disjoint:
        out.append(_mk(kind="DISJOINT", subj=cid, obj=str(other),
                       evidence_count=n_items, source_doc_count=n_items,
                       batch_cover_count=1))

    # 参考基准 = 该批该概念候选公理的“平均证据量”，即证据相对强度：
    # 概念规模与模式数量差异很大时（轨道 1 每概念 3 个模式 vs 轨道 2 每概念 30+），
    # 绝对计数基准会把真实数据上的所有公理压到 τ_conf 以下。
    ref_count = int(np.mean([a.evidence_count for a in out])) if out else 1
    ref_docs = int(np.mean([a.source_doc_count for a in out])) if out else 1
    ref_count = max(2, ref_count)
    ref_docs = max(2, ref_docs)
    if ref_ratio is not None:                      # 可选：按实例规模兜底
        ref_count = max(ref_count, int(ref_ratio * n_items))
        ref_docs = max(ref_docs, int(ref_ratio * n_items))
    for ax in out:
        ax.confidence = source_evidence_count_factor(
            ax, ref_count=ref_count, ref_docs=ref_docs, ref_batch=ref_batch)
    return out


def mean_confidence(axioms: Iterable[Axiom], cid: str) -> float:
    """概念 c 相关候选公理的平均置信度。"""
    rel = [a.confidence for a in axioms if a.subj == cid]
    return float(sum(rel) / len(rel)) if rel else 0.0
