"""ontoevolve.src.calibrate - 轨道 1 阈值标定。

在与测试流独立的“标定流”上做随机网格搜索，目标：macro-F1 最大。
标定得到的阈值再用于所有测试重复，避免在同数据上调参导致虚高。
"""
from __future__ import annotations

from itertools import product as _product
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



def _prep_calib_streams(streams):
    """为每条标定流各建一次 O_0（不能跨流复用同一个本体对象）。"""
    if isinstance(streams, ControlledStream):
        streams = [streams]
    prepared = []
    for st in streams:
        onto, info = build_initial_ontology(
            st.pool, st.d0_iids, seed=0, k_gt=len(st.vocab_base),
            vocab={**st.vocab_base, **st.vocab_emer})
        prepared.append((onto, info["leaf_to_cid"], st))
    return prepared


def _baseline_f1(onto, leaf_to_cid, stream, method, base_params, kw) -> float:
    mon = BaselineMonitor(onto, method, params=base_params, **kw)
    alarms: List[Alarm] = []
    for t, items in enumerate(stream.batches, start=1):
        alarms.extend(mon.observe(t, items))
    res = match_events(stream.events, alarms, leaf_to_cid,
                       delay_tol=base_params.delay_tol, typed=False)
    return float(res.macro_f1)


def calibrate_baseline(stream, method: str, *,
                       base_params: C.DriftParams, seed: int = 0,
                       verbose: bool = True,
                       cooldown: Optional[int] = None,
                       calib_cooldown: Optional[int] = None,
                       grid: Optional[Dict[str, Sequence]] = None,
                       mode: str = "legacy",
                       return_trials: bool = False) -> Dict:
    """在标定流上选择基线的超参数。

    参数
    ----
    stream : 单条标定流，或标定流序列。多条时每流独立建 O_0、独立跑监视器，
             取 macro-F1 的算术平均作为选择准则（与 OntoEvolve 的准则一致）。
    cooldown : 保留的历史参数，**不参与计算**（V23 行为）；新增的
             `calib_cooldown` 才真正把告警抑制窗口带进标定。
    grid : 覆盖 BASELINE_GRID[method] 的候选网格。
    mode : "legacy" 逐键分轮搜索（V23 行为，每轮另一键取默认值）；
           "joint" 按键做笛卡尔积联合搜索。
    """
    prepared = _prep_calib_streams(stream)
    grid = grid if grid is not None else BASELINE_GRID[method]
    fixed = {} if calib_cooldown is None else {"cooldown": int(calib_cooldown)}
    trials: List[Dict] = []

    def score(kw: Dict) -> tuple:
        per = [_baseline_f1(onto, l2c, st, method, base_params, {**kw, **fixed})
               for onto, l2c, st in prepared]
        return float(np.mean(per)), per

    best = {"macro_f1": -1.0, "kw": {}}
    candidates: List[Dict] = []
    if mode == "joint":
        keys = list(grid)
        for combo in _product([grid[k] for k in keys]):
            candidates.append(dict(zip(keys, combo)))
    else:
        for key, values in grid.items():
            for v in values:
                candidates.append({key: v})
    for kw in candidates:
        m, per = score(kw)
        trials.append({"kw": dict(kw), "mean": m, "per_stream": per})
        # 并列时保留网格顺序中的第一个候选（显式规则，冻结在协议里）
        if m > best["macro_f1"]:
            best = {"macro_f1": m, "kw": dict(kw)}
    if verbose:
        print(f"[calib] {method:<14} best macroF1={best['macro_f1']:.3f} "
              f"{best['kw']} streams={len(prepared)} mode={mode} "
              f"calib_cooldown={calib_cooldown}")
    out = {**best, "n_evals": len(trials), "n_streams": len(prepared),
           "mode": mode, "calib_cooldown": calib_cooldown,
           "tie_rule": "first_in_grid_order"}
    if return_trials:
        out["trials"] = trials
    return out

