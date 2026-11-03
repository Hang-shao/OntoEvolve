"""ontoevolve.src.icews_stream - 把 ICEWS18 的时间片转成检测器可消费的实例流。

映射方式（轨道 2，真实数据）：
    实例   = 一条事件，文本 = "<主体> <关系> <客体>"
    关系槽 = (关系名, 客体角色)        → OP 公理（关系 + 值域）
    属性槽 = (主体角色 / 客体角色, 值)  → DP 公理（域/值域类型）
    概念   = 对事件文本嵌入聚类得到的“事件类型”概念
漂移语义：涌现=新事件类型；淘汰=某事件类型连续多片不再出现；
          重定义=某事件类型的关系/参与者角色模式发生变化。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .controlled_stream import Occurrence
from .embedding import Embedder
from .icews import ICEWSData, country_of, role_of


def slice_occurrences(data: ICEWSData, slice_events: np.ndarray,
                      embedder: Embedder, *, t: int,
                      start_iid: int = 0) -> List[Occurrence]:
    """把一个时间片的事件转成 Occurrence 列表（一次性嵌入）。"""
    texts, metas = [], []
    for e in slice_events:
        s, r, o = int(e[0]), int(e[1]), int(e[2])
        subj, rel, obj = data.entities[s], data.relations[r], data.entities[o]
        texts.append(f"{subj} {rel} {obj}")
        metas.append((subj, rel, obj))
    vecs = embedder.embed(texts)
    out: List[Occurrence] = []
    for i, ((subj, rel, obj), v) in enumerate(zip(metas, vecs)):
        s_role, o_role = role_of(subj), role_of(obj)
        out.append(Occurrence(
            iid=start_iid + i,
            leaf=rel,                                  # 关系名作为分析用标签
            variant="base",
            text=texts[i],
            vec=np.asarray(v, dtype=np.float32),
            op_slots=[(rel, o_role)],
            dp_slots=[("subject_role", s_role), ("object_role", o_role)],
            sup=country_of(subj) or s_role,
        ))
    return out


def build_icews_stream(data: ICEWSData, slices: Sequence[np.ndarray],
                       embedder: Embedder, *, per_slice: Optional[int] = None,
                       seed: int = 0, n_slices: Optional[int] = None,
                       verbose: bool = True) -> List[List[Occurrence]]:
    """抽样 + 嵌入，返回每个时间片的实例列表。"""
    from .icews import sample_slice, slice_stat

    use = list(slices)[: n_slices] if n_slices else list(slices)
    stream: List[List[Occurrence]] = []
    iid = 0
    for si, sl in enumerate(use):
        ev = sl if per_slice is None else sample_slice(sl, per_slice, seed=seed + si)
        occ = slice_occurrences(data, ev, embedder, t=si + 1, start_iid=iid)
        iid += len(occ)
        stream.append(occ)
        if verbose:
            st = slice_stat(data, ev)
            print(f"  slice {si + 1:>2}: events={st['events']:>5} "
                  f"entities={st['entities']:>5} relations={st['relations']:>4}")
    return stream
