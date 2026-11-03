"""ontoevolve.src.baselines - 漂移检测基线。

    adwin            ADWIN 自适应滑动窗口（Bifet & Gavaldà）
    page_hinkley     Page-Hinkley 累积和变点检测
    embdist          嵌入分布：新增实例到概念锚点的余弦距离超过 μ + kσ
    embkmeans        嵌入 kmeans：概念内 k=2 聚类的**簇间分离度**超过 μ + kσ
    inc_clustering   增量聚类：在线微簇，出现新微簇 / 概念长期无新实例 → 告警

所有基线只输出“发生了变化”的告警（不含漂移类型），评估时按时间窗匹配真值
并采用“最优类型标注”（对基线有利的宽松口径）。

> 关于 ODRT：基线清单里列了 ODRT，但未给出定义与引用，公开检索
> 也查不到对应方法。这里**不冒名实现**：ODRT 一栏暂缺，改用上述两条无歧义的
> 基线（增量聚类、嵌入 kmeans）补齐对照面。若后续拿到 ODRT 的定义或出处，
> 在 `METHODS` 中按同样接口接入即可。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .controlled_stream import Occurrence
from .drift_detect import Alarm
from .ontology_model import Ontology


class ADWIN:
    """精确版 ADWIN（窗口规模小，直接扫描切分点）。"""

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.W: List[float] = []

    def add(self, x: float) -> bool:
        self.W.append(float(x))
        drift = False
        while len(self.W) >= 2:
            n = len(self.W)
            var = float(np.var(self.W, ddof=1)) if n > 1 else 0.0
            cut = None
            for i in range(1, n):
                m = 1.0 / (1.0 / i + 1.0 / (n - i))
                eps = (math.sqrt(2.0 * var / m * math.log(2.0 / self.delta))
                       + 2.0 / (3.0 * m) * math.log(2.0 / self.delta))
                if abs(float(np.mean(self.W[:i])) - float(np.mean(self.W[i:]))) > eps:
                    cut = i
                    break
            if cut is None:
                break
            self.W = self.W[cut:]
            drift = True
        return drift


class PageHinkley:
    """Page-Hinkley 变点检测（同时监测上升与下降）。"""

    def __init__(self, delta: float = 0.05, lamb: float = 1.5):
        self.delta = delta
        self.lamb = lamb
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.m_min = 0.0
        self.m_max = 0.0

    def add(self, x: float) -> bool:
        self.n += 1
        self.mean += (float(x) - self.mean) / self.n
        self.cum += float(x) - self.mean - self.delta
        self.m_min = min(self.m_min, self.cum)
        self.m_max = max(self.m_max, self.cum)
        return ((self.cum - self.m_min) > self.lamb
                or (self.m_max - self.cum) > self.lamb)


class EmbeddingDistance:
    """嵌入分布基线：距离超过 warmup 窗口的 μ + kσ 即告警。"""

    def __init__(self, k_std: float = 3.0, warmup: int = 5, min_obs: int = 3):
        self.k_std = k_std
        self.warmup = warmup
        self.min_obs = min_obs
        self.hist: List[float] = []

    def add(self, x: float) -> bool:
        x = float(x)
        if len(self.hist) < self.min_obs:
            self.hist.append(x)
            return False
        h = self.hist[:max(self.warmup, self.min_obs)]
        mu, sd = float(np.mean(h)), float(np.std(h))
        sd = max(sd, 0.05 * abs(mu), 1e-3)      # 防止方差塌缩导致的病态阈值
        self.hist.append(x)
        return x > mu + self.k_std * sd


class IncrementalClustering:
    """增量聚类基线（leader clustering / 在线微簇）。

    每个概念维护一组微簇中心（L2 归一的均值向量）：
      - 新实例与最近微簇中心的余弦距离 > `theta` → 新建微簇：
        概念内出现了新的语义模式 → “重定义 / 涌现”信号
      - 概念连续 `stale_window` 批没有任何新实例 → “淘汰”信号

    与标量序列类基线的差别：它真的在概念内部维护结构，而不是只看一个汇总量。

    `update(t, cid, vecs)` 返回 `(new_pattern, extinct)` 两个布尔。
    """

    def __init__(self, theta: float = 0.35, warmup: int = 3,
                 stale_window: int = 3):
        self.theta = float(theta)
        self.warmup = int(warmup)
        self.stale_window = int(stale_window)
        self.micro: Dict[str, List[np.ndarray]] = {}
        self.seen_batches: Dict[str, int] = {}        # 该概念已观察的批次数
        self.empty_streak: Dict[str, int] = {}        # 连续无新实例的批次数

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    def update(self, t: int, cid: str,
               vecs: Sequence[np.ndarray]) -> Tuple[bool, bool]:
        cl = self.micro.setdefault(cid, [])
        self.seen_batches[cid] = self.seen_batches.get(cid, 0) + 1
        warm = self.seen_batches[cid] > self.warmup

        if not vecs:
            self.empty_streak[cid] = self.empty_streak.get(cid, 0) + 1
            extinct = warm and self.empty_streak[cid] >= self.stale_window
            return False, extinct
        self.empty_streak[cid] = 0

        new = 0
        for v in vecs:
            v = self._norm(np.asarray(v, dtype=np.float32))
            if cl:
                sims = [float(np.dot(v, c)) for c in cl]
                j = int(np.argmax(sims))
                if 1.0 - sims[j] > self.theta:
                    cl.append(v)
                    new += 1
                else:
                    # 滑动更新微簇中心，让概念内部的新模式逐渐被吸收
                    cl[j] = self._norm(cl[j] * 0.9 + v * 0.1)
            else:
                cl.append(v)
                new += 1
        return (warm and new > 0), False


def two_means_separation(vecs: Sequence[np.ndarray]) -> float:
    """概念内 k=2 聚类的簇间余弦距离（分离度）；样本不足时为 0。

    用于 `embkmeans` 基线：概念内部若真的分化出新语义，两簇中心会明显分开；
    只是整体漂移（离质心变远）则两簇仍会靠在一起——因此它比 embdist 更能区分
    “内部模式分裂” 与 “整体漂移”。
    """
    V = np.asarray(list(vecs), dtype=np.float32)
    if V.ndim != 2 or V.shape[0] < 4:
        return 0.0
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    S = V @ V.T
    i, j = np.unravel_index(int(np.argmin(S)), S.shape)
    c1, c2 = V[i].copy(), V[j].copy()
    for _ in range(10):
        s1, s2 = V @ c1, V @ c2
        m = s1 >= s2
        if not m.any() or m.all():
            break
        c1 = V[m].mean(axis=0)
        c2 = V[~m].mean(axis=0)
        n1, n2 = float(np.linalg.norm(c1)), float(np.linalg.norm(c2))
        if n1 < 1e-9 or n2 < 1e-9:
            break
        c1, c2 = c1 / n1, c2 / n2
    return float(1.0 - float(np.dot(c1, c2)))


class BaselineMonitor:
    """把基线套用到三条标量序列上。

    count 序列  ：每个概念每批的新增实例数（按 base_rate 归一）—— 可对淘汰敏感
    dist  序列  ：每个概念内新增实例到质心的余弦距离（逐实例）—— 可对语义变化敏感
    global序列  ：每个实例到最近已有锚点的距离（逐实例）—— 可对涌现敏感
    """

    METHODS = ("adwin", "page_hinkley", "embdist", "embkmeans",
               "inc_clustering")

    def __init__(self, ontology: Ontology, method: str = "adwin",
                 *, params: Optional[C.DriftParams] = None, **kw):
        if method not in self.METHODS:
            raise ValueError(f"unknown baseline: {method}")
        self.method = method
        self.onto = ontology
        self.p = params or C.PARAMS
        self.kw = kw
        # 增量聚类是结构化方法，不走“标量序列 + 变点检测”的路径
        self.structured: Optional[IncrementalClustering] = (
            IncrementalClustering(theta=kw.get("theta", 0.35),
                                  warmup=kw.get("warmup", 3),
                                  stale_window=kw.get("stale_window", 3))
            if method == "inc_clustering" else None)
        self.centroids = {cid: np.asarray(c.centroid, dtype=np.float32)
                          for cid, c in ontology.concepts.items()}
        self.det_count: Dict[str, object] = {cid: self._mk() for cid in self.centroids}
        self.det_dist: Dict[str, object] = {cid: self._mk() for cid in self.centroids}
        self.det_global = self._mk()
        self.last_alarm: Dict[tuple, int] = {}
        self.cooldown = int(kw.get("cooldown", 3))

    def _mk(self):
        if self.method == "adwin":
            return ADWIN(delta=self.kw.get("delta", 0.1))
        if self.method == "page_hinkley":
            return PageHinkley(delta=self.kw.get("ph_delta", 0.01),
                               lamb=self.kw.get("ph_lambda", 1.2))
        return EmbeddingDistance(k_std=self.kw.get("k_std", 3.0),
                                 warmup=self.kw.get("warmup", 5))

    def _fire(self, key: str, t: int) -> bool:
        """带冷却的告警判定。"""
        last = self.last_alarm.get(key)
        if last is not None and t - last < self.cooldown:
            return False
        self.last_alarm[key] = t
        return True

    def observe(self, t: int, items: Sequence[Occurrence]) -> List[Alarm]:
        per: Dict[str, List[Occurrence]] = {cid: [] for cid in self.centroids}
        for o in items:
            v = np.asarray(o.vec, dtype=np.float32)
            best, bs = None, -2.0
            for cid, cen in self.centroids.items():
                s = float(np.dot(v, cen))
                if s > bs:
                    bs, best = s, cid
            if best is not None:
                per[best].append(o)

        # 每批汇总为一个标量（序列长度 = T，ADWIN/PH 均可承受其 O(n²) 扫描）
        if items:
            gvals = []
            for o in items:
                v = np.asarray(o.vec, dtype=np.float32)
                gvals.append(min(1.0 - float(np.dot(v, c))
                                 for c in self.centroids.values()))
            x_global = float(np.mean(gvals))
        else:
            x_global = 1.0

        if self.structured is not None:
            return self._observe_clustering(t, per)

        alarms: List[Alarm] = []
        for cid in self.centroids:
            lst = per[cid]
            x_cnt = len(lst) / max(self.p.base_rate, 1e-9)
            if self.det_count[cid].add(x_cnt) and self._fire(f"count::{cid}", t):
                alarms.append(Alarm(t=t, cid=cid, dtype="change",
                                    score=float(x_cnt),
                                    signals={"count": float(len(lst))},
                                    detail={"method": self.method,
                                            "series": "count"}))
            if len(lst) >= 4 and self.method == "embkmeans":
                x_dist = two_means_separation([o.vec for o in lst])
            elif lst:
                x_dist = float(np.mean([1.0 - float(np.dot(o.vec,
                                                           self.centroids[cid]))
                                        for o in lst]))
            else:
                x_dist = 1.0        # 无新增实例 → 支撑消失
            if self.det_dist[cid].add(x_dist) and self._fire(f"dist::{cid}", t):
                alarms.append(Alarm(t=t, cid=cid, dtype="change",
                                    score=float(x_dist), signals={"dist": x_dist},
                                    detail={"method": self.method,
                                            "series": "dist"}))
        if self.det_global.add(x_global) and self._fire("global", t):
            target = max(per.items(), key=lambda kv: len(kv[1]))[0] if per \
                else None
            alarms.append(Alarm(t=t, cid=target or "__GLOBAL__",
                                dtype="change", score=float(x_global),
                                signals={"min_dist": x_global},
                                detail={"method": self.method, "series": "global"}))
        return alarms

    def _observe_clustering(self, t: int,
                            per: Dict[str, List[Occurrence]]) -> List[Alarm]:
        """增量聚类路径：概念内出现新微簇 / 概念长期无新实例 → 告警。"""
        alarms: List[Alarm] = []
        for cid, lst in per.items():
            new_pat, extinct = self.structured.update(
                t, cid, [o.vec for o in lst])
            if new_pat and self._fire(f"cluster::{cid}", t):
                alarms.append(Alarm(t=t, cid=cid, dtype="change",
                                    score=float(len(lst)),
                                    signals={"new_micro": 1.0},
                                    detail={"method": self.method,
                                            "series": "cluster"}))
            if extinct and self._fire(f"extinct::{cid}", t):
                alarms.append(Alarm(t=t, cid=cid, dtype="change", score=0.0,
                                    signals={"empty_streak": 1.0},
                                    detail={"method": self.method,
                                            "series": "stale"}))
        return alarms
