"""ontoevolve.src.drift_detect - 模块 D：三信号融合的概念漂移检测。

信号 1 语义：新增实例与概念锚点的余弦距离 / 候选新概念与已有锚点的距离
信号 2 统计：连续 W 批新增实例数低于 ε_ext 且内部语义权重下降
信号 3 公理：axiom_signature(O_t) 与 axiom_signature(Project(O_t, δA_t)) 的
            L1 距离 > θ_red 且平均置信度 ≥ τ_conf
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config as C
from .controlled_stream import Occurrence
from .embedding import l2_normalize
from .extract import extract_candidates
from .ontology_model import L1_distance, Ontology


@dataclass
class Alarm:
    t: int                       # 批次号（从 1 开始）
    cid: str                     # 本体概念 id，涌现为 NEW#k
    dtype: str                   # emergence | extinction | redefinition
    score: float                 # 融合分
    signals: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, object] = field(default_factory=dict)


class DriftDetector:
    def __init__(self, ontology: Ontology, params: Optional[C.DriftParams] = None,
                 *, support_decay: float = 0.5, cooldown: int = 99,
                 cluster_sim: float = 0.70, use_fusion_gate: bool = False,
                 adaptive_link: bool = False, link_k: float = 1.0):
        self.onto = ontology
        self.p = params or C.PARAMS
        self.support_decay = support_decay
        self.cooldown = cooldown
        self.cluster_sim = cluster_sim
        self.use_fusion_gate = use_fusion_gate
        self.link_sim = self.p.link_sim
        # 按概念自适应的链接阈值（可选，默认关闭以保持轨道 1—3 口径）：
        # 概念 c 的阈值取 max(全局 link_sim, μ_c − k·σ_c)，μ_c/σ_c 为该概念成员
        # 到质心余弦的均值与标准差（构建期记录于 meta）。目的：紧致概念不再被迫
        # 接收与自己成员相似度明显偏低的实例，从而不吸收邻近新类型的实例。
        self.adaptive_link = adaptive_link
        self.link_k = link_k
        self.thr: Dict[str, float] = {}
        if adaptive_link:
            for cid, c in ontology.concepts.items():
                mu = c.meta.get("sim_mu")
                sd = c.meta.get("sim_sigma")
                self.thr[cid] = (self.link_sim if mu is None else
                                 max(self.link_sim,
                                     float(mu) - self.link_k * float(sd or 0.0)))
        # 公理签名维度随领域词表规模变化（轨道 1 的 M_ref=18，ICEWS 的 M≈264），
        # 跨领域迁移 θ_red 时按 M/M_ref 线性缩放，保证阈值含义一致。
        self.M_ref = float(getattr(C, "AXIOM_SIGNATURE_M_REF", 18))
        self.theta_red_eff = self.p.theta_red * max(1.0, ontology.M / self.M_ref)
        # 涌现候选簇内的聚合相似度（Qwen3 嵌入的绝对余弦值集中在 0.6~0.7，
        # 需要用相对阈值，见 config.theta_sem）

        self.centroids: Dict[str, np.ndarray] = {
            cid: np.asarray(c.centroid, dtype=np.float32)
            for cid, c in ontology.concepts.items()}
        self.counts: Dict[str, List[int]] = {cid: [] for cid in ontology.concepts}
        self.weights: Dict[str, List[float]] = {cid: [] for cid in ontology.concepts}
        self.candidates: List[Dict] = []
        self.last_alarm: Dict[tuple, int] = {}
        self.log: List[Dict] = []
        self.last_projected: Dict[str, List] = {}   # 供演化算子取用
        # 标注导出用：本批分配给该概念的实例 + 该概念最近一次非空批次
        self.last_items: Dict[str, List] = {}
        self.last_nonempty: Dict[str, tuple] = {}

    # ---- 主循环 -------------------------------------------------------------
    def observe(self, t: int, items: Sequence[Occurrence]) -> List[Alarm]:
        alarms: List[Alarm] = []
        assigned: Dict[str, List[Occurrence]] = {cid: [] for cid in self.centroids}
        unassigned: List[Occurrence] = []

        for o in items:
            v = np.asarray(o.vec, dtype=np.float32)
            best, bs = None, -2.0
            for cid, cen in self.centroids.items():
                s = float(np.dot(v, cen))
                if s > bs:
                    bs, best = s, cid
            need = (self.thr.get(best, self.link_sim) if self.adaptive_link
                    else self.link_sim)
            if best is not None and bs >= need:
                assigned[best].append(o)
            else:
                unassigned.append(o)

        for cid, lst in assigned.items():
            alarm = self._check_concept(t, cid, lst)
            if alarm is not None:
                alarms.append(alarm)

        alarms.extend(self._check_emergence(t, unassigned))
        return alarms

    # ---- 已有概念：重定义 / 淘汰 -------------------------------------------
    def _check_concept(self, t: int, cid: str,
                       lst: List[Occurrence]) -> Optional[Alarm]:
        n = len(lst)
        self.counts[cid].append(n)
        self.last_items[cid] = list(lst)
        if lst:
            self.last_nonempty[cid] = (t, list(lst))
        conc = self.onto.get(cid)
        if conc is None:
            return None

        # 信号 1：语义漂移
        if n:
            batch_cen = l2_normalize(np.mean(np.vstack([o.vec for o in lst]), axis=0))
            shift = 1.0 - float(np.dot(batch_cen, self.centroids[cid]))
        else:
            shift = 0.0

        # 信号 3：公理签名（Project 只使用数据流抽取的 delta_A）
        # 置信度按该批该概念的实例规模归一（见 ontology_model 说明）
        delta = extract_candidates(
            cid, [o.as_dict() for o in lst], parent=conc.parent,
            transitive_ops=[a.pred for a in conc.axioms if a.kind == "TRANSITIVE"],
            disjoint=[a.obj for a in conc.axioms if a.kind == "DISJOINT"])
        sig_old = self.onto.axiom_signature(cid)
        proj = self.onto.project(delta, cid, tau_conf=self.p.tau_conf,
                                 support_decay=self.support_decay)
        sig_new = proj.axiom_signature(cid)
        proj_axioms = list(proj.axioms_of(cid))
        self.last_projected[cid] = proj_axioms
        dist = L1_distance(sig_old, sig_new)
        # 平均置信度取“Project 之后（τ_conf 过滤 + 冲突消解）保留下来的公理”的
        # 均值：真实数据模式空间大时，未过滤候选集里大量弱模式会把均值拉到
        # τ_conf 以下，使重定义永远无法触发。
        mconf = (float(np.mean([a.confidence for a in proj_axioms]))
                 if proj_axioms else 0.0)

        # 信号 2：统计 + 内部语义权重
        W = self.p.window_w
        cnts = self.counts[cid]
        recent = cnts[-W:]
        mean_recent = float(np.mean(recent)) if recent else 0.0
        stat_ready = len(cnts) >= W
        stat_score = (max(0.0, (self.p.eps_ext - mean_recent) / max(self.p.eps_ext, 1e-9))
                      if stat_ready else 0.0)
        density = n / max(self.p.base_rate, 1e-9)
        prop_freq = len({d for o in lst for d, _ in o.dp_slots}) / 6.0
        weight = conc.internal_weight(density=density, prop_freq=prop_freq)
        self.weights[cid].append(weight)
        ws = self.weights[cid]
        # “持续下降”：最近 W 批的均值低于此前所有历史批次的均值
        if len(ws) > W:
            declining = float(np.mean(ws[-W:])) < float(np.mean(ws[:-W]))
        else:
            declining = len(ws) >= W

        sem_norm = min(1.0, shift / max(self.p.theta_shift, 1e-9))
        ax_norm = min(1.0, dist / max(self.theta_red_eff, 1e-9))
        score = (self.p.w_sem * sem_norm + self.p.w_stat * stat_score
                 + self.p.w_ax * ax_norm)
        signals = {"sem": shift, "stat": stat_score, "ax": dist,
                   "mean_conf": mconf, "weight": weight}

        self.log.append({"t": t, "cid": cid, "n": n, "signals": signals,
                         "score": score})

        # ---- 重定义：公理签名显著变化且有足够置信度 -------------------------
        fired = None
        if dist > self.theta_red_eff and mconf >= self.p.tau_conf and n >= 1:
            fired = Alarm(t=t, cid=cid, dtype="redefinition", score=score,
                          signals=signals, detail={"L1": dist, "mconf": mconf})
        # ---- 淘汰：连续 W 批低于 ε_ext 且内部权重下降 -----------------------
        elif stat_ready and stat_score > 0 and declining:
            fired = Alarm(t=t, cid=cid, dtype="extinction", score=score,
                          signals=signals,
                          detail={"mean_recent": mean_recent, "W": W})

        if fired is None:
            return None
        if self.use_fusion_gate and score < self.p.theta_score:
            return None
        key = (cid, fired.dtype)
        last = self.last_alarm.get(key)
        if last is not None and t - last < self.cooldown:
            return None
        self.last_alarm[key] = t
        return fired

    # ---- 涌现：未链接实例形成候选新概念 -----------------------------------
    def _check_emergence(self, t: int, unassigned: List[Occurrence]) -> List[Alarm]:
        for o in unassigned:
            v = np.asarray(o.vec, dtype=np.float32)
            placed = False
            for cand in self.candidates:
                if cand.get("dead"):
                    continue
                if float(np.dot(v, cand["centroid"])) >= self.cluster_sim:
                    cand["items"].append(o)
                    placed = True
                    break
            if not placed:
                self.candidates.append({
                    "id": f"NEW{len(self.candidates)}",
                    "centroid": v.copy(), "items": [o], "batches": set(),
                    "fired": False, "dead": False})

        alarms: List[Alarm] = []
        for cand in self.candidates:
            if cand.get("fired") or cand.get("dead"):
                continue
            cand["batches"].add(t)
            if t - max(cand["batches"]) > self.p.kappa + 1:
                cand["dead"] = True
                continue
            if not all((t - i) in cand["batches"] for i in range(self.p.kappa)):
                continue
            recent_items = [o for o in cand["items"]]
            if len(recent_items) < self.p.min_new_size:
                continue
            cen = l2_normalize(np.mean(np.vstack([o.vec for o in recent_items]),
                                       axis=0))
            cand["centroid"] = cen
            dmin, nearest = min(
                ((1.0 - float(np.dot(cen, c)), cid)
                 for cid, c in self.centroids.items()),
                key=lambda x: x[0])
            if dmin <= self.p.theta_sem:
                continue
            signals = {"sem": dmin, "stat": 0.0, "ax": 0.0,
                       "mean_conf": 1.0, "weight": float(len(recent_items))}
            score = self.p.w_sem + self.p.w_stat * 0.0
            alarms.append(Alarm(t=t, cid=cand["id"], dtype="emergence",
                                score=score, signals=signals,
                                detail={"min_dist": dmin,
                                        "nearest_cid": nearest,
                                        "size": len(recent_items),
                                        "centroid": cen,
                                        "items": list(recent_items)}))
            cand["fired"] = True
        return alarms
