"""受控成本运行器：逐种子跑，前后各做一次 CPU 速度探针。

背景：2026-11-02 上午的一轮里，seed 2 用了 102 分钟、seed 3 只用 25 分钟，而同一配置在
前一晚的干净运行里每个种子只需 27—29 分钟——说明计时窗口内进程曾被系统降速（Windows 11
对后台进程的 Power Throttling / EcoQoS 最可能）。休眠守护只挡住"睡眠"，挡不住 CPU 降速。
所以本运行器做三件事：

1. `SetThreadExecutionState` 压住系统睡眠（含屏幕，避免进入低功耗状态）；
2. `SetProcessInformation(ProcessPowerThrottling)` 关闭执行速度限制，并对每个子进程同样处理；
3. 每个种子跑之前与跑之后各做一次定长 CPU 基准（同一份浮点负载），把探针写进
   `results/exp_cost/speed_probes.json`；两次探针中位数相差超过 10% 即判该种子无效。

1、2 仅 Windows 有效；其他平台只保留探针与"记录但不判死"的流程。

它调用 `experiments/run_cost_multiseed.py` 的单种子模式，逐种子串行执行（计时刻度对
CPU 竞争极敏感，不能并行）。整轮五个种子请优先用 `run_cost_full5.py`；本脚本用于
"只重测个别异常种子"或夜间无人值守时。

用法：
    python experiments/run_cost_guarded.py --seeds 2
    python experiments/run_cost_guarded.py --seeds 2,3,4 --tag t36_allexact_c
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE_LOG = ROOT / "results/exp_cost/speed_probes.json"

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:  # pragma: no cover - 平台相关
    import ctypes
    from ctypes import wintypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    ProcessPowerThrottling = 4
    HIGH_PRIORITY_CLASS = 0x00000080

    kernel32 = ctypes.windll.kernel32
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]

    class ThrottleState(ctypes.Structure):
        _fields_ = [("Version", wintypes.DWORD), ("ControlMask", wintypes.DWORD),
                    ("StateMask", wintypes.DWORD)]


def keep_awake() -> None:
    if not IS_WINDOWS:
        print("[guard] non-Windows platform: sleep guard skipped", flush=True)
        return
    prev = kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED |
                                            ES_DISPLAY_REQUIRED)
    print("[guard] SetThreadExecutionState ->", prev, flush=True)


def release_awake() -> None:
    if IS_WINDOWS:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def no_throttle(handle=None) -> bool:
    """关闭执行速度限制；handle 为 None 时作用于本进程，否则作用于子进程句柄。"""
    if not IS_WINDOWS:
        return False
    h = handle if handle is not None else kernel32.GetCurrentProcess()
    st = ThrottleState(PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                       PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
    ok = kernel32.SetProcessInformation(h, ProcessPowerThrottling,
                                        ctypes.byref(st), ctypes.sizeof(st))
    return bool(ok)


def set_high_priority() -> bool:
    if not IS_WINDOWS:
        return False
    return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(),
                                          HIGH_PRIORITY_CLASS))


def probe(seconds: float = 6.0) -> dict:
    """定长浮点负载：返回迭代次数与单次耗时中位数，用于判断机器是否被降速。"""
    a = np.random.default_rng(0).random((256, 256))
    b = np.random.default_rng(1).random((256, 256))
    times = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        acc = 0.0
        for _ in range(40):
            acc += float((a @ b).sum())
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    return {"n_block": int(times.size), "block_ms_mean": round(float(times.mean() * 1e3), 2),
            "block_ms_med": round(float(np.median(times) * 1e3), 2),
            "block_ms_min": round(float(times.min() * 1e3), 2),
            "checksum": round(acc, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description="带 CPU 探针与反降速的成本运行器")
    ap.add_argument("--seeds", default="2")
    ap.add_argument("--tag", default="t36_allexact_c")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--probe-s", type=float, default=6.0)
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    keep_awake()
    self_ok = no_throttle()
    prio_ok = set_high_priority()
    print("[ctl] self throttle-off:", self_ok, " high-priority:", prio_ok, flush=True)

    records = []
    if PROBE_LOG.exists():
        records = json.loads(PROBE_LOG.read_text(encoding="utf-8"))
    PROBE_LOG.parent.mkdir(parents=True, exist_ok=True)

    try:
        for s in seeds:
            pre = probe(args.probe_s)
            print("[probe] before seed %d: %s" % (s, pre), flush=True)
            cmd = [sys.executable, str(ROOT / "experiments/run_cost_multiseed.py"),
                   "--seeds", str(s), "--full-every", "1",
                   "--period", str(args.period), "--tag", "%s_s%d" % (args.tag, s)]
            print("[run]", " ".join(cmd), flush=True)
            t0 = time.time()
            p = subprocess.Popen(cmd, cwd=str(ROOT))
            child_ok = no_throttle(p._handle) if IS_WINDOWS else False
            print("[ctl] child %d throttle-off: %s" % (p.pid, child_ok), flush=True)
            rc = p.wait()
            wall = round(time.time() - t0, 1)
            post = probe(args.probe_s)
            drift = abs(post["block_ms_med"] - pre["block_ms_med"]) / pre["block_ms_med"]
            rec = {"seed": s, "tag": "%s_s%d" % (args.tag, s), "rc": rc, "wall_s": wall,
                   "pre": pre, "post": post, "probe_drift": round(drift, 4),
                   "valid": bool(rc == 0 and drift < 0.10)}
            records.append(rec)
            PROBE_LOG.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            print("[done] seed %d rc=%d wall=%.1f min probe_drift=%.1f%% valid=%s"
                  % (s, rc, wall / 60, drift * 100, rec["valid"]), flush=True)
    finally:
        release_awake()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
