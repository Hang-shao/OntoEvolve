r"""把归档结果文件里的本地绝对路径改写为仓库内相对路径。

论文的 Data availability 声明把 `results/` 与代码一起归档；归档前需要去掉含作者本地目录的绝对路径（例如
`<项目根>\ontoevolve\data\x.owl` 这类写法），改为相对仓库根的 `data/x.owl` 形式。
**只改路径字符串，不改任何数值。**

三种模式：
    python scripts/relativize_results.py --check   # 只报告会改哪些文件、几处
    python scripts/relativize_results.py           # 原地改写（先备份，再自检数值未变）

备份写在 `dist/relativize_backup-<时间戳>/` 下，保持相对路径结构。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MARK = "ontoevolve"          # 仓库目录名；其后的部分即仓库内相对路径
WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def _norm(p: str) -> str:
    """把重复转义的反斜杠收敛成单反斜杠，再统一成 '/'。"""
    while "\\\\" in p:
        p = p.replace("\\\\", "\\")
    return p.replace("\\", "/")


def _rel(p: str) -> str:
    """绝对路径 -> 仓库内相对路径；判定失败则原样返回。"""
    if not WIN_ABS.match(_norm(p)):
        return p
    q = _norm(p)
    idx = q.find("/" + MARK + "/")
    if idx < 0:
        return p
    return q[idx + len(MARK) + 2:]


def _walk(node, fn):
    if isinstance(node, dict):
        return {k: _walk(v, fn) for k, v in node.items()}
    if isinstance(node, list):
        return [_walk(v, fn) for v in node]
    if isinstance(node, str):
        return fn(node)
    return node


def _numbers(node, out):
    if isinstance(node, dict):
        for v in node.values():
            _numbers(v, out)
    elif isinstance(node, list):
        for v in node:
            _numbers(v, out)
    elif isinstance(node, bool):
        out.append(repr(node))
    elif isinstance(node, (int, float)):
        out.append(repr(float(node)))
    return out


def iter_targets():
    for p in sorted(RESULTS.rglob("*.json")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text):
            yield p, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只报告，不改文件")
    ap.add_argument("--backup", default="", help="备份目录（默认 dist/relativize_backup-<ts>）")
    args = ap.parse_args()

    targets = list(iter_targets())
    if not targets:
        print("没有含绝对路径的结果文件。")
        return 0

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = Path(args.backup) if args.backup else ROOT / "dist" / f"relativize_backup-{stamp}"

    total_changed = 0
    for p, text in targets:
        obj = json.loads(text)
        before_nums = _numbers(obj, [])
        n_hits = len(re.findall(r"(?<![A-Za-z])[A-Za-z]:[\\/]", text))
        new = _walk(obj, _rel)
        if args.check:
            rel = p.relative_to(ROOT)
            sample = [m.group(0)[:90] for m in
                      re.finditer(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\"\n]{0,80}", text)][:2]
            print(f"[check] {rel}: {n_hits} 处绝对路径, 例：{sample}")
            total_changed += n_hits
            continue
        # 备份 + 写回
        dst = backup / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        indent = 2 if re.search(r'\n  "', text[:400]) else 4
        p.write_text(json.dumps(new, ensure_ascii=False, indent=indent) + "\n",
                     encoding="utf-8")
        # 自检：数值集合未变
        after = json.loads(p.read_text(encoding="utf-8"))
        if _numbers(after, []) != before_nums:
            raise SystemExit(f"数值发生变化，已备份在 {dst}，请人工核对：{p}")
        print(f"[done] {p.relative_to(ROOT)}: {n_hits} 处已改为相对路径")
        total_changed += n_hits

    if args.check:
        print(f"\n共 {len(targets)} 个文件、{total_changed} 处绝对路径待处理。")
    else:
        print(f"\n备份在 {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
