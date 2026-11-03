"""ontoevolve.src.evolve - 本体演化算子与回滚。

每类漂移对应一个算子：
    涌现    → 新增概念（含质心与公理）
    重定义  → 用 Project 结果替换该概念的公理集
    淘汰    → 标记概念为非活跃（保留以便溯源）

每个算子执行后做一致性检查；不一致则回滚到执行前快照，并计入回退率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import time

import numpy as np

from .consistency import consistency_report, has_owl_dl_verdict
from .ontology_model import Axiom, Concept, Ontology
from .ontology_model import resolve_conflicts


@dataclass
class EvolveStats:
    applied: Dict[str, int] = field(default_factory=dict)
    rollbacks: Dict[str, int] = field(default_factory=dict)
    inconsistent_versions: int = 0
    versions: int = 0
    ms_check: float = 0.0          # 一致性检查累计耗时（实验 5 的资源指标）
    engines: Dict[str, int] = field(default_factory=dict)
    no_verdict: int = 0            # fail-closed 拒绝：没有任何 OWL-DL 裁决

    @property
    def rollback_rate(self) -> float:
        """回退率 = 回滚次数 / 尝试次数（成功 + 回滚）。"""
        rb = sum(self.rollbacks.values())
        total = sum(self.applied.values()) + rb
        return float(rb / total) if total else 0.0


class OntologyEvolver:
    """带快照/回滚的本体演化器。"""

    def __init__(self, ontology: Ontology, *, check: bool = True,
                 engine: str = "auto", use_cache: bool = True,
                 require_dl_verdict: bool = True):
        self.onto = ontology
        self.check = check
        # fail-closed：只有完整 OWL-DL 推理机的裁决才允许提交（部署规则）。
        # 对照实验（exp5 的 z3/structural 分支）需要让受限层按自身裁决提交，
        # 否则这些分支一次都提交不了，无法比较检出能力，故显式关闭。
        self.require_dl_verdict = require_dl_verdict
        self.engine = engine          # 一致性检查引擎（auto|hermit|owlready2|pellet|z3|structural）
        # 关闭缓存可测到一致性检查的真实成本（实验 5 用）
        self.use_cache = use_cache
        self.stats = EvolveStats()
        self.history: List[Dict] = []

    # ---- 快照 ---------------------------------------------------------------
    def _snapshot(self) -> Dict:
        return {
            cid: Concept(cid=c.cid, name=c.name, parent=c.parent,
                         centroid=c.centroid.copy(), axioms=list(c.axioms),
                         active=c.active, first_batch=c.first_batch,
                         last_batch=c.last_batch, counts=list(c.counts),
                         weights=list(c.weights), meta=dict(c.meta))
            for cid, c in self.onto.concepts.items()}

    def _restore(self, snap: Dict) -> None:
        # 版号只在提交成功时递增（见 _commit），被拒的操作从未改过它，
        # 所以回滚只恢复概念集合。旧实现此处减 1，会把被拒次数从版号里
        # 扣掉：注入矛盾的运行中版号会偏小，甚至从 0 变成负数。
        self.onto.concepts = snap

    def _commit(self, op: str, snap: Dict) -> bool:
        """提交前做一致性检查；不一致则回滚。返回是否成功提交。"""
        self.stats.versions += 1
        if self.check:
            t0 = time.perf_counter()
            rep = consistency_report(self.onto, engine=self.engine,
                                     use_cache=self.use_cache)
            self.stats.ms_check += (time.perf_counter() - t0) * 1000.0
            eng = str(rep.get("engine", "?"))
            self.stats.engines[eng] = self.stats.engines.get(eng, 0) + 1
            if not rep["consistent"]:
                self._restore(snap)
                self.stats.rollbacks[op] = self.stats.rollbacks.get(op, 0) + 1
                self.stats.inconsistent_versions += 1
                return False
            # fail-closed：只有完整 OWL-DL 推理机给出的裁决才能提交。
            # 受限层（z3 及其回退形式）与纯结构检查说"一致"不算裁决，
            # 因为它们的覆盖只是 OWL-DL 的子集。
            if self.require_dl_verdict and not has_owl_dl_verdict(rep):
                self._restore(snap)
                self.stats.rollbacks[op] = self.stats.rollbacks.get(op, 0) + 1
                self.stats.no_verdict += 1
                return False
        self.onto.version += 1
        self.stats.applied[op] = self.stats.applied.get(op, 0) + 1
        return True

    # ---- 算子 ---------------------------------------------------------------
    def add_concept(self, cid: str, name: str, *, parent: Optional[str] = None,
                    centroid: Optional[np.ndarray] = None,
                    axioms: Sequence[Axiom] = ()) -> bool:
        snap = self._snapshot()
        self.onto.add(Concept(cid=cid, name=name, parent=parent,
                              centroid=np.asarray(centroid, dtype=np.float32)
                              if centroid is not None else np.zeros(1, np.float32),
                              axioms=list(axioms), active=True))
        ok = self._commit("emergence", snap)
        self.history.append({"op": "emergence", "cid": cid, "ok": ok})
        return ok

    def redefine(self, cid: str, axioms: Sequence[Axiom]) -> bool:
        snap = self._snapshot()
        c = self.onto.get(cid)
        if c is None:
            return False
        c.axioms = resolve_conflicts(list(axioms))
        ok = self._commit("redefinition", snap)
        self.history.append({"op": "redefinition", "cid": cid, "ok": ok})
        return ok

    def retire(self, cid: str) -> bool:
        snap = self._snapshot()
        c = self.onto.get(cid)
        if c is None:
            return False
        c.active = False
        ok = self._commit("extinction", snap)
        self.history.append({"op": "extinction", "cid": cid, "ok": ok})
        return ok
