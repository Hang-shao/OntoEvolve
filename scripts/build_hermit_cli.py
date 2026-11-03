"""编译 tools/java/HermiTCheck.java（HermiT 一致性检查入口）。

用法：python scripts/build_hermit_cli.py

产物：build/classes/HermiTCheck.class，由 src.reasoner 调用。
需要 JDK（javac）；只有 JRE 时跳过，此时 HermiT 退化为官方 CLI 或 Pellet。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C          # noqa: E402
from src.reasoner import find_java   # noqa: E402


def find_javac() -> str | None:
    java = find_java()
    if java:
        cand = Path(java).with_name("javac.exe")
        if cand.exists():
            return str(cand)
        cand2 = Path(java).with_name("javac")
        if cand2.exists():
            return str(cand2)
    found = shutil.which("javac")
    return found


def main() -> int:
    javac = find_javac()
    if not javac:
        print("!! javac not found; HermiT CLI entry point not built")
        return 1
    src = Path(__file__).resolve().parents[1] / "tools" / "java" / "HermiTCheck.java"
    out = C.DIR_BUILD
    out.mkdir(parents=True, exist_ok=True)
    cp = str(C.DIR_VENDOR / "*")
    cmd = [javac, "-encoding", "UTF-8", "-nowarn", "-cp", cp, "-d", str(out), str(src)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print((r.stdout or b"").decode("utf-8", "ignore")[:2000])
        print((r.stderr or b"").decode("utf-8", "ignore")[:2000])
        return r.returncode
    print("built", out / "HermiTCheck.class")
    return 0


if __name__ == "__main__":
    sys.exit(main())
