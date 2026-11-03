"""ontoevolve.src.eval_drift - 漂移检测评估（macro-F1 / bootstrap CI / Wilcoxon）。

匹配口径（对所有方法一致）：
    真值事件 (leaf, type, t_declare) 与告警 (cid, type, t) 在同一概念上且
    |t - t_declare| <= delay_tol 视为命中（TP），同一告警不重复计数。
    未命中的告警为 FP：按类型归因（有类型告警用自身类型；无类型基线用其所在
    概念承载的事件类型；落在无事件概念上的告警为 fp_other，三类型均摊）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .controlled_stream import DriftEvent
from .drift_detect import Alarm

TYPES = ("emergence", "extinction", "redefinition")


@dataclass
class EvalResult:
    macro_f1: float
    per_type: Dict[str, Dict[str, float]]
    tp: int
    fp: int
    fn: int
    n_alarms: int
    n_events: int
    mean_delay: float
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_row(self, prefix: str = "") -> Dict[str, float]:
        row = {f"{prefix}macro_f1": self.macro_f1, f"{prefix}tp": self.tp,
               f"{prefix}fp": self.fp, f"{prefix}fn": self.fn,
               f"{prefix}n_alarms": self.n_alarms,
               f"{prefix}mean_delay": self.mean_delay}
        for t in TYPES:
            for k in ("p", "r", "f1"):
                row[f"{prefix}{t[:3]}_{k}"] = self.per_type.get(t, {}).get(k, float("nan"))
        return row


def _cid_match(alarm: Alarm, leaf: str, leaf_to_cid: Dict[str, str]) -> bool:
    if leaf not in leaf_to_cid:               # 涌现概念不在 O_0 中
        return alarm.cid.startswith("NEW")
    return alarm.cid == leaf_to_cid.get(leaf)


def match_events(events: Sequence[DriftEvent], alarms: Sequence[Alarm],
                 leaf_to_cid: Dict[str, str], *, delay_tol: int = 2,
                 typed: bool = True) -> EvalResult:
    used = [False] * len(alarms)
    tp_by: Dict[str, int] = {t: 0 for t in TYPES}
    fn_by: Dict[str, int] = {t: 0 for t in TYPES}
    delays: List[int] = []
    event_hit: List[tuple] = []       # [(事件, 是否命中)]

    for ev in events:
        hit: Optional[int] = None
        for j, a in enumerate(alarms):
            if used[j]:
                continue
            if abs(a.t - ev.t_declare) > delay_tol:
                continue
            if typed and a.dtype != ev.dtype:
                continue
            if ev.dtype == "emergence":
                ok = a.cid.startswith("NEW") if typed else True
            else:
                ok = _cid_match(a, ev.leaf, leaf_to_cid)
            if ok:
                hit = j
                break
        if hit is None:
            fn_by[ev.dtype] += 1
            event_hit.append((ev, 0))
        else:
            used[hit] = True
            tp_by[ev.dtype] += 1
            delays.append(int(alarms[hit].t - ev.t_declare))
            event_hit.append((ev, 1))

    # ---- FP 归因 ------------------------------------------------------------
    concept_type: Dict[str, str] = {}
    for ev in events:
        if ev.leaf in leaf_to_cid:
            concept_type[leaf_to_cid[ev.leaf]] = ev.dtype
    fp_by: Dict[str, int] = {t: 0 for t in TYPES}
    fp_other = 0
    for j, a in enumerate(alarms):
        if used[j]:
            continue
        if a.cid.startswith("NEW"):
            fp_by["emergence"] += 1
            continue
        t = concept_type.get(a.cid)
        if t is None:
            fp_other += 1
        else:
            fp_by[t] += 1
    share = fp_other / 3.0

    per_type: Dict[str, Dict[str, float]] = {}
    for t in TYPES:
        tp, fn = tp_by[t], fn_by[t]
        fp = fp_by[t] + share
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_type[t] = {"p": prec, "r": rec, "f1": f1, "tp": float(tp),
                       "fp": float(fp), "fn": float(fn)}
    macro = float(np.mean([per_type[t]["f1"] for t in TYPES]))

    # ---- 困难样本分组（召回）------------------------------------------------
    by_diff: Dict[str, Dict[str, float]] = {}
    for d in ("abrupt", "gradual", "periodic", "lowfreq"):
        flags = [h for ev, h in event_hit if ev.difficulty == d]
        if not flags:
            continue
        by_diff[d] = {"n": float(len(flags)), "recall": float(np.mean(flags))}

    return EvalResult(macro_f1=macro, per_type=per_type,
                      tp=int(sum(tp_by.values())), fp=int(sum(fp_by.values()) + fp_other),
                      fn=int(sum(fn_by.values())), n_alarms=len(alarms),
                      n_events=len(events),
                      mean_delay=float(np.mean(delays)) if delays else float("nan"),
                      by_difficulty=by_diff)


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """百分位 bootstrap 置信区间（重采样单位 = 一次重复实验）。"""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return {"mean": float(v.mean()),
            "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2))}


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """配对 Wilcoxon 符号秩检验（双面）。"""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return {"stat": float("nan"), "p": float("nan")}
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size < 5 or np.allclose(x - y, 0):
        return {"stat": float("nan"), "p": float("nan")}
    try:
        stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return {"stat": float(stat), "p": float(p)}
    except Exception:
        return {"stat": float("nan"), "p": float("nan")}
