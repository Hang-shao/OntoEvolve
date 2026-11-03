"""通用重试包装：反复运行一条命令直到成功（用于间歇可用的外部服务）。

嵌入服务会不定期不可用（连接被拒/重置）。实验脚本在严格模式下会因此直接
退出，但**已嵌入的文本会落在缓存里**，重跑时不必重来。所以正确做法不是
放宽回退，而是失败后等待再重试，让进度靠缓存累积。

Run:
    python experiments/retry_until_done.py --max 40 --wait 90 -- \
        experiments/run_exp_pmc.py --per-slice 1200 --n-slices 12
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=40, help="最大尝试次数")
    ap.add_argument("--wait", type=int, default=90, help="失败后等待秒数")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        print("usage: retry_until_done.py --max N --wait S -- <cmd> ...")
        return 2
    argv = [sys.executable] + cmd

    for attempt in range(args.max):
        t0 = time.time()
        r = subprocess.run(argv, cwd=str(ROOT))
        if r.returncode == 0:
            print(f"[retry] 第 {attempt + 1} 次尝试成功 "
                  f"（{time.time() - t0:.0f}s）", flush=True)
            return 0
        print(f"[retry] 第 {attempt + 1}/{args.max} 次失败 "
              f"（rc={r.returncode}，{time.time() - t0:.0f}s），"
              f"{args.wait}s 后重试", flush=True)
        if attempt + 1 < args.max:
            time.sleep(args.wait)
    print(f"[retry] 已达最大尝试次数 {args.max}，仍未成功")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
