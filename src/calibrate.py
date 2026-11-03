"""ontoevolve.src.calibrate - 轨道 1 阈值标定。

在与测试流独立的“标定流”上做随机网格搜索，目标：macro-F1 最大。
标定得到的阈值再用于所有测试重复，避免在同数据上调参导致虚高。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config as C
from .baselines import BaselineMonitor
from .build_ontology import build_initial_ontology
from .controlled_stream import ControlledStream, DriftEvent
from .drift_detect import Alarm, DriftDetector
from .eval_drift import match_events

# ---- OntoEvolve 模块 D 的搜索空间 -------------------------------------------
GRID: Dict[str, Sequence] = {
    "theta_sem": (0.18, 0.22, 0.26, 0.30, 0.34),
    "theta_shift": (0.06, 0.08, 0.10, 0.14),
    "theta_red": (0.75, 1.0, 1.5, 2.0, 2.5),
    "tau_conf": (0.40, 0.50, 0.60, 0.70),
    "eps_ext": (0.4, 0.8, 1.2, 1.6),
    "window_w": (3, 4, 5),
    "kappa": (2, 3),
    "min_new_size": (2, 3, 4),
    "link_offset": (0.0, 0.02, 0.05, 0.10),
    "delay_tol": (2, 3),
}

BASELINE_GRID: Dict[str, Dict[str, Sequence]] = {
    "adwin": {"delta": (0.02, 0.05, 0.1, 0.2, 0.4)},
    "page_hinkley": {"ph_lambda": (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)},
    "embdist": {"k_std": (1.5, 2.0, 2.5, 3.0, 4.0)},
    "embkmeans": {"k_std": (1.0, 1.5, 2.0, 2.5, 3.0)},
    "inc_clustering": {"theta": (0.2, 0.25, 0.3, 0.35, 0.45),
                       "stale_window": (2, 3, 5)},
}


def run_detector(stream: ControlledStream, onto, leaf_to_cid: Dict[str, str],
                 params: C.DriftParams, *, link_offset: float = 0.0,
                 base_p05: float = 0.9, support_decay: float = 0.5,
                 fusion_gate: bool = False,
                 cooldown: Optional[int] = None) -> List[Alarm]:
    p = params.copy(link_sim=max(0.40, base_p05 - link_offset))
    kw = {} if cooldown is None else {"cooldown": int(cooldown)}
    det = DriftDetector(onto, p, support_decay=support_decay,
                        use_fusion_gate=fusion_gate, **kw)
    alarms: List[Alarm] = []
    for t, items in enumerate(stream.batches, start=1):
        alarms.extend(det.observe(t, items))
    return alarms


def evaluate(stream: ControlledStream, onto, leaf_to_cid: Dict[str, str],
             params: C.DriftParams, *, link_offset: float = 0.0,
             base_p05: float = 0.9, support_decay: float = 0.5,
             fusion_gate: bool = False, cooldown: Optional[int] = None):
    alarms = run_detector(stream, onto, leaf_to_cid, params,
                          link_offset=link_offset, base_p05=base_p05,
                          support_decay=support_decay,
                          fusion_gate=fusion_gate, cooldown=cooldown)
    return match_events(stream.events, alarms, leaf_to_cid,
                        delay_tol=params.delay_tol), alarms


def calibrate_ontoevolve(streams: Sequence[ControlledStream], *,
                         base_params: C.DriftParams,
                         n_trials: int = 240, seed: int = 0,
                         verbose: bool = True,
                         cooldown: Optional[int] = None) -> Dict:
    """在标定流（可多条）上搜索模块 D 的阈值组合。

    多条标定流取 macro-F1 均值作为选择准则，降低对单条流过拟合的风险。
    返回 {'params','link_offset','macro_f1'}。
    """
    streams = list(streams)
    prepared = []
    for stream in streams:
        onto, info = build_initial_ontology(
            stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
            vocab={**stream.vocab_base, **stream.vocab_emer})
        prepared.append((stream, onto,
                         float(info.get("link_sim_p05", base_params.link_sim)),
                         info["leaf_to_cid"]))

    rng = np.random.default_rng(seed)
    keys = list(GRID.keys())
    best = {"macro_f1": -1.0, "params": base_params, "link_offset": 0.0}
    trials: List[Dict] = []
    for _ in range(n_trials):
        cand = {k: GRID[k][int(rng.integers(0, len(GRID[k])))] for k in keys}
        link_offset = float(cand.pop("link_offset"))
        params = base_params.copy(**cand)
        f1s = []
        for stream, onto, p05, leaf_to_cid in prepared:
            res, _ = evaluate(stream, onto, leaf_to_cid, params,
                              link_offset=link_offset, base_p05=p05,
                              cooldown=cooldown)
            f1s.append(res.macro_f1)
        score = float(np.mean(f1s))
        trials.append({"macro_f1": score, "per_stream": f1s, **cand,
                       "link_offset": link_offset})
        if score > best["macro_f1"]:
            best = {"macro_f1": score, "params": params,
                    "link_offset": link_offset}
    if verbose:
        print(f"[calib] OntoEvolve best macroF1={best['macro_f1']:.3f} "
              f"link_offset={best['link_offset']} (streams={len(prepared)})")
        print(f"[calib]   {best['params']}")
    return {**best, "n_trials": n_trials, "trials": trials}


def calibrate_baseline(stream: ControlledStream, method: str, *,
                       base_params: C.DriftParams, seed: int = 0,
                       verbose: bool = True,
                       cooldown: Optional[int] = None) -> Dict:
    """在标定流上选择基线的超参数（共享同一个 O_0）。"""
    onto, info = build_initial_ontology(
        stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
        vocab={**stream.vocab_base, **stream.vocab_emer})
    leaf_to_cid = info["leaf_to_cid"]
    grid = BASELINE_GRID[method]
    best = {"macro_f1": -1.0, "kw": {}}
    for key, values in grid.items():
        for v in values:
            mon = BaselineMonitor(onto, method, params=base_params, **{key: v})
            alarms: List[Alarm] = []
            for t, items in enumerate(stream.batches, start=1):
                alarms.extend(mon.observe(t, items))
            res = match_events(stream.events, alarms, leaf_to_cid,
                               delay_tol=base_params.delay_tol, typed=False)
            if res.macro_f1 > best["macro_f1"]:
                best = {"macro_f1": res.macro_f1, "kw": {key: v}}
    if verbose:
        print(f"[calib] {method:<14} best macroF1={best['macro_f1']:.3f} {best['kw']}")
    return best
