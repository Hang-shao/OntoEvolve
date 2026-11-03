"""生成可公开的开源发布包，并检查不该出现的东西。

三种模式：
    python scripts/package_release.py --list     # 只打印将要打包的文件（含体积）
    python scripts/package_release.py --check    # 只做检查：密钥、绝对路径、超大文件
    python scripts/package_release.py            # 生成 dist/ontoevolve-<日期>.zip

打包白名单而不是黑名单：新增目录不会因为忘了改排除规则而把数据或密钥带出去。
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

# 白名单：目录（整棵）与单文件
INCLUDE_DIRS = ["src", "legacy", "experiments", "tests", "tools", "scripts"]
INCLUDE_FILES = [
    "README.md", "requirements.txt",
    "requirements-optional.txt", "pyproject.toml", ".env.example",
    ".gitignore", ".gitattributes", "data/README.md", "LICENSE", "CITATION.cff",
]
# 目录里要跳过的部分（相对仓库根）
SKIP_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "vendor", "logs", "results", "cache",
}
SKIP_SUFFIX = {".pyc", ".pyo", ".jar", ".zip"}

# 检查规则：(说明, 正则)
SECRET_RULES = [
    ("疑似 API 密钥", re.compile(r"\b(sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{20,})\b")),
    ("疑似内网/真实端点", re.compile(r"\b(?:10|127|192\.168|172\.1[6-9]|172\.2\d|172\.3[01])"
                              r"\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+")),
    ("疑似硬编码的 key= 值", re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*"
                                r"['\"][^'\"\s]{12,}['\"]")),
]
PATH_RULES = [
    ("Windows 绝对路径", re.compile(r"[A-Za-z]:[\\/]+(?:code|Users|soft|data)[\\/]", re.I)),
    ("POSIX 家目录路径", re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+/")),
]
BIG_MB = 20


def iter_files():
    for f in INCLUDE_FILES:
        p = ROOT / f
        if p.exists():
            yield p
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT)
            if SKIP_PARTS & set(rel.parts):
                continue
            if p.suffix.lower() in SKIP_SUFFIX:
                continue
            yield p


def show(files) -> None:
    total = 0
    for p in files:
        n = p.stat().st_size
        total += n
        print(f"{n / 1024:9.1f} kB  {p.relative_to(ROOT)}")
    print(f"\n共 {len(files)} 个文件，{total / 1048576:.2f} MB")


def check(files) -> int:
    """扫描密钥、绝对路径与超大文件；返回问题条数。"""
    problems = 0
    for p in files:
        rel = p.relative_to(ROOT)
        if p.name == ".env":
            print(f"[问题] 发布包里出现了 .env：{rel}")
            problems += 1
            continue
        size_mb = p.stat().st_size / 1048576
        if size_mb > BIG_MB:
            print(f"[提示] 大文件 {size_mb:.1f} MB：{rel}")
        if p.suffix.lower() not in {".py", ".md", ".toml", ".txt", ".cfg", ".ini",
                                    ".example", ".gitignore", ".gitattributes", ".json"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for label, rx in SECRET_RULES + PATH_RULES:
                if rx.search(line):
                    print(f"[问题] {label}：{rel}:{line_no}: {line.strip()[:110]}")
                    problems += 1
    print(f"\n检查完成：{problems} 处问题（0 才可直接发布）")
    return problems


def build(files, tag: str) -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / f"ontoevolve-{tag}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path("ontoevolve") / p.relative_to(ROOT))
    print(f"\n写出 {out}（{out.stat().st_size / 1048576:.2f} MB，{len(files)} 个文件）")
    with zipfile.ZipFile(out) as z:
        bad = [n for n in z.namelist() if n.endswith("/.env") or Path(n).name == ".env"]
    print("包内 .env 条目：", bad or "无")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="生成开源发布包并检查泄漏")
    ap.add_argument("--list", action="store_true", help="只打印文件清单")
    ap.add_argument("--check", action="store_true", help="只做密钥/路径检查")
    ap.add_argument("--allow-problems", action="store_true",
                    help="即使检查发现问题也照样打包（不建议）")
    ap.add_argument("--tag", default=dt.date.today().strftime("%Y%m%d"))
    args = ap.parse_args()

    files = list(iter_files())
    if not files:
        print("没有可打包的文件", file=sys.stderr)
        return 1
    if args.list:
        show(files)
        return 0
    if args.check:
        return 1 if check(files) else 0

    problems = check(files)
    if problems and not args.allow_problems:
        print("\n有未解决的问题，已停止打包（确认无误可加 --allow-problems）。")
        return 1
    build(files, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
