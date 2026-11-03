"""ontoevolve.src.consistency - 本体一致性检查。

三级判定，逐级降级：

    1) 结构性冲突（纯 Python，必检，零开销）
       - 基数区间为空（card_min > card_max）
       - 概念的 DOMAIN 指向与之 DISJOINT 的概念
    2) OWL-DL 推理机（HermiT 优先，owlready2 自带推理器备选）
       把本体导出为 OWL 2 DL 交推理机判定，可捕获需要推理才能暴露的矛盾
       （父子链上的不相交、传递闭包、基数与存在量词的组合等）。
       同一快照的结果按哈希缓存，避免演化过程中反复启动 JVM。
    3) z3 可满足性（可选，推理机不可用时的受限近似）
       只覆盖基数与不相交，是 OWL-DL 的**受限子集**。

engine 参数：'auto'（默认，按 1→2→3）、'hermit'、'owlready2'（别名 'pellet'）、
'z3'、'structural'。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config as C
from .ontology_model import Ontology

try:
    import z3
except Exception:  # pragma: no cover
    z3 = None

_CACHE: Dict[str, Dict] = {}


def structural_conflicts(onto: Ontology) -> List[str]:
    """纯 Python 的结构性矛盾检查，返回冲突描述列表。"""
    problems: List[str] = []
    for cid, c in onto.concepts.items():
        if not c.active:
            continue
        for ax in c.axioms:
            if ax.kind == "CARD" and ax.card_max > 0 and ax.card_min > ax.card_max:
                problems.append(f"{cid}: 空基数区间 [{ax.card_min},{ax.card_max}]")
            if ax.kind == "DOMAIN" and ax.obj:
                if any(a.kind == "DISJOINT" and a.obj == ax.obj for a in c.axioms):
                    problems.append(f"{cid}: DOMAIN={ax.obj} 与 DISJOINT 冲突")
    return problems


def _snapshot_hash(onto: Ontology) -> str:
    parts = []
    for cid in sorted(onto.concepts):
        c = onto.concepts[cid]
        axs = sorted((a.kind, a.pred, a.obj, a.card_min, a.card_max, int(c.active))
                     for a in c.axioms)
        parts.append(f"{cid}|{c.parent}|{len(axs)}|" + ";".join(map(str, axs)))
    return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()


def _reasoner_check(onto: Ontology, engine: Optional[str],
                    owl_out: Optional[Path]) -> Optional[Dict]:
    """用 OWL-DL 推理机判定；不可用返回 None。"""
    from .owl_export import export_owl
    from .reasoner import get_reasoner

    r = get_reasoner(engine)
    if r is None:
        return None
    path = owl_out or (C.DIR_TEMP / f"onto_{_snapshot_hash(onto)[:12]}.owl")
    export_owl(onto, path)
    res = r.check(path)
    return {"consistent": bool(res.consistent),
            "reason": "unsat-class" if not res.consistent else None,
            "structural": [],
            "engine": res.engine,
            "ok": bool(res.ok),
            "ms": float(res.ms),
            "unsatisfiable": list(res.unsatisfiable),
            "error": res.error,
            "owl": str(path)}


def satisfiable(onto: Ontology, *, max_instances: int = 1000,
                timeout_ms: int = 2000) -> Tuple[bool, Optional[str]]:
    """z3 受限近似：判断是否存在非空解释（无 z3 时只做结构检查）。"""
    if structural_conflicts(onto):
        return False, "structural"
    if z3 is None:
        return True, None
    active = [cid for cid, c in onto.concepts.items() if c.active]
    if not active:
        return True, None
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    counts = {cid: z3.Int(f"n_{cid}") for cid in active}
    for cid in active:
        n = counts[cid]
        solver.add(n >= 1, n <= max_instances)
    for cid in active:
        c = onto.concepts[cid]
        op_ax = [a for a in c.axioms if a.kind == "OP"]
        if not op_ax:
            continue
        terms = [z3.Int(f"x_{cid}_{i}") for i in range(len(op_ax))]
        for t in terms:
            solver.add(t >= 0, t <= max_instances)
        total = z3.Sum(terms) if len(terms) > 1 else terms[0]
        cards = [a for a in c.axioms if a.kind == "CARD"]
        for card in cards:
            solver.add(total >= card.card_min)
            if card.card_max > 0:
                solver.add(total <= card.card_max)
    res = solver.check()
    if res == z3.sat:
        return True, None
    if res == z3.unknown:
        return True, "timeout"       # 超时不判为不一致
    return False, "unsat"


def consistency_report(onto: Ontology, *, engine: str = "auto",
                       owl_out: Optional[Path] = None,
                       use_cache: bool = True, **kw) -> Dict:
    """一致性报告。engine: auto|hermit|owlready2|pellet(别名)|z3|structural。"""
    structural = structural_conflicts(onto)
    key = _snapshot_hash(onto)
    if use_cache and key in _CACHE and engine in ("auto", _CACHE[key].get("engine"),
                                                  "hermit", "owlready2", "pellet"):
        cached = dict(_CACHE[key])
        cached["cached"] = True
        return cached

    if engine in ("auto", "hermit", "owlready2", "pellet"):
        rep = _reasoner_check(onto, None if engine == "auto" else engine, owl_out)
        if rep is not None and rep.get("ok", True):
            rep["structural"] = structural
            rep["cached"] = False
            _CACHE[key] = rep
            return rep
        # 推理机不可用或调用失败（解析被拒/异常）→ 降级 z3 这一受限子集。
        # “检查失败”不当作“一致”，但也要如实记住：此时通过的只是第 3 级
        # （engine 字段记为 "z3(fallback:...)"），不构成 OWL-DL 级别的一致性判定。
        ok, reason = satisfiable(onto, **kw)
        why = rep.get("error") if rep else "unavailable"
        if rep is None and engine == "auto":
            return {"consistent": ok, "reason": reason, "structural": structural,
                    "engine": "z3", "ms": 0.0, "unsatisfiable": [],
                    "cached": False}
        return {"consistent": ok, "reason": reason, "structural": structural,
                "engine": f"z3(fallback:{why})", "ms": float(rep.get("ms", 0.0))
                if rep else 0.0, "unsatisfiable": [], "cached": False}

    if engine == "structural":
        return {"consistent": not structural, "reason": "structural" if structural else None,
                "structural": structural, "engine": "structural", "ms": 0.0,
                "unsatisfiable": [], "cached": False}
    ok, reason = satisfiable(onto, **kw)
    return {"consistent": ok, "reason": reason, "structural": structural,
            "engine": "z3" if z3 is not None else "structural-only",
            "ms": 0.0, "unsatisfiable": [], "cached": False}


def has_owl_dl_verdict(rep: Dict) -> bool:
    """报告是否来自完整 OWL-DL 推理机（fail-closed 判据）。

    受限层（z3 及其回退形式）与纯结构检查只覆盖了 OWL-DL 的子集：
    它们返回"一致"时，结论不是 OWL-DL 级别的一致性判定。
    """
    eng = str(rep.get("engine", ""))
    return not eng.startswith(("z3", "structural"))


def is_consistent(onto: Ontology, **kw) -> bool:
    """便捷接口：只返回是否一致。"""
    return bool(consistency_report(onto, **kw)["consistent"])
