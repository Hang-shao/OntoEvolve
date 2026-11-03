# -*- coding: utf-8 -*-
"""阶段 2 runner：等成本多 seed 实验结束后，串行补跑 ②③④ 三项实验。

为什么串行：成本实验与阈值敏感性实验都含墙钟计时，并行会互相污染计时结果，
因此用本 watcher 保证严格串行（前一项完全退出后才启动下一项）。

② 重定义检测阈值在真实轨道的敏感性（回应“真实轨道上阈值被推高、无法外推”）
   - 轨道 2（ICEWS18）：z ∈ {1.0,1.5,2.0,3.0} × 标定模式 {stat, transfer}
       · stat     = 用轨道 2 前 6 片热身估计 μ+zσ
       · transfer = 用轨道 1 的尺度线性迁移到轨道 2 的公理签名维度
   - 轨道 3（PMC）    ：z ∈ {1.0,1.5,2.0,3.0}
③ 模块 G 的 OWL 2 DL 合法性通过率与公理增长（完整 T=36）
④ 一致性检查开销对照（T=12，check=True 对照既有 check=False）

用法（后台）：
    python experiments/run_stage2.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
COST_DONE = ROOT / "results" / "exp_cost" / "exp_cost_t36_multiseed.json"
MAX_WAIT_S = 3 * 3600


def run(argv, note):
    print("\n===== %s :: %s =====" % (note, " ".join(argv)), flush=True)
    t0 = time.perf_counter()
    try:
        r = subprocess.run([PY, *argv], cwd=str(ROOT), timeout=5400)
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = "TIMEOUT"
    print("----- %s 结束（退出码 %s，用时 %.1f min）-----"
          % (note, code, (time.perf_counter() - t0) / 60.0), flush=True)


def wait_cost():
    t0 = time.time()
    while not COST_DONE.exists():
        if time.time() - t0 > MAX_WAIT_S:
            print("等待成本实验超时（%.1f h），仍继续阶段 2"
                  % (MAX_WAIT_S / 3600.0), flush=True)
            return
        print("等待成本多 seed 实验…（已等 %.0f min）"
              % ((time.time() - t0) / 60.0), flush=True)
        time.sleep(60)
    print("成本多 seed 实验产物已出现，缓冲 30s 后开始阶段 2", flush=True)
    time.sleep(30)


def main() -> int:
    wait_cost()

    # ② 轨道 2：标定模式 × z（先 stat 后 transfer，便于对照）
    for mode in ("stat", "transfer"):
        for zt in ("1.0", "1.5", "2.0", "3.0"):
            tag = "z%s_%s" % (zt, mode)
            run(["experiments/run_track2.py", "--z", zt,
                 "--theta-red-mode", mode,
                 "--out", "results/track2_%s.csv" % tag,
                 "--summary", "results/track2_%s.json" % tag],
                "② 轨道2 z=%s 标定=%s" % (zt, mode))

    # ② 轨道 3：z 扫描
    for zt in ("1.0", "1.5", "2.0", "3.0"):
        run(["experiments/run_exp_pmc.py", "--z", zt,
             "--out-dir", "results/exp_pmc_z%s" % zt],
            "② 轨道3 z=%s" % zt)

    # ③ 模块 G：OWL 2 DL 合法性通过率与公理增长
    run(["experiments/run_exp_gowl.py", "--seed", "0", "--tag", "main"],
        "③ 模块G OWL合法性/公理增长")

    # ④ 一致性检查开销对照
    run(["experiments/run_exp_cost.py", "--quick", "--tag", "quick_check"],
        "④ T=12 check=True 对照")

    print("\n阶段 2 全部完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
