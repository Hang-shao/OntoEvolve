"""ontoevolve.src.alarm_pack - 告警级证据导出（供人工/LLM 标注复核）。

只读取检测器与本体状态，不改变检测与演化行为。每条告警整理为：
    track/seed/t/dtype/cid/name/score/signals + 证据文本 + （重定义）公理前后对照

证据文本的含义：
    涌现   —— 触发该候选概念的本批实例文本；
    重定义 —— 本批分配给该概念的实例文本（模式变化即发生在这里）；
    淘汰   —— 本批实例（通常为空），为空时退化为"最近一次非空批次"的文本，
              供标注员核查是"真的不再出现"还是"只是本批没抽到"。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .llm import axiom_text


def _texts(items: Sequence, limit: int = 6, max_len: int = 300) -> List[str]:
    out: List[str] = []
    for o in items[:limit]:
        t = getattr(o, "text", "") or ""
        if t:
            out.append(t[:max_len])
    return out


def record_alarm(*, track: str, seed: int, alarm, detector, onto,
                 batch: Optional[Sequence] = None,
                 emergence_prefix: str = "new_concept_") -> Dict:
    """把一条告警整理为可复核的记录（纯读取）。"""
    cid = alarm.cid
    conc = onto.get(cid)
    detail = alarm.detail or {}
    rec: Dict = {
        "track": track,
        "seed": int(seed),
        "t": int(alarm.t),
        "dtype": alarm.dtype,
        "cid": cid,
        "name": conc.name if conc is not None else f"{emergence_prefix}{cid}",
        "score": round(float(alarm.score), 4),
        "signals": {k: round(float(v), 4)
                    for k, v in (alarm.signals or {}).items()},
    }

    if alarm.dtype == "emergence":
        items = list(detail.get("items") or [])
        rec["n_members"] = len(items)
        rec["min_dist"] = round(float(detail.get("min_dist", 0.0)), 4)
        rec["evidence"] = _texts(items)
        near = detail.get("nearest_cid")
        nc = onto.get(near) if near is not None else None
        rec["nearest_concept"] = nc.name if nc is not None else ""
    else:
        items = list(detector.last_items.get(cid) or [])
        rec["n_members"] = len(items)
        rec["evidence"] = _texts(items)
        if conc is not None:
            rec["definition"] = [axiom_text(a) for a in conc.axioms[:5]]
            parent = onto.hier.parent(cid) if hasattr(onto, "hier") else None
            pc = onto.get(parent) if parent is not None else None
            rec["parent_concept"] = pc.name if pc is not None else ""
        last = detector.last_nonempty.get(cid)
        if last:
            lt, litems = last
            rec["last_seen_t"] = int(lt)
            if not items:
                rec["evidence"] = _texts(litems)
        elif not items:
            # 该概念自建立后从未被匹配到实例：淘汰告警的"历史证据"为空是事实，
            # 标注员需要知道这是"从未出现"而不是"导出遗漏"
            rec["never_mentioned"] = True
        if alarm.dtype == "redefinition":
            rec["axioms_before"] = [
                axiom_text(a) for a in ((conc.axioms if conc else [])[:10])]
            rec["axioms_after"] = [
                axiom_text(a) for a in
                ((detector.last_projected.get(cid) or [])[:10])]
    return rec
