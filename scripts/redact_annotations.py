# -*- coding: utf-8 -*-
"""发布前对 annotations/ 做去原文处理（第三方语料的再分发限制）。

背景：论文 Data availability 明写 Track 2 (ICEWS18) 与 Track 3 (PMC) 不再分发。
标注包里有两类列与这条声明冲突：
    逐字证据  evidence, llm_evidence_quote, raw
    派生内容  definition（ICEWS18 的实体-角色链）, axioms_before, axioms_after
本脚本把这几列清空，保留复现 0.1585 所需的全部字段
（alarm_id、track、t、dtype、concept、特征分、判定标签、confidence），
原文件先备份到 dist/ 下，不做原地覆盖以外任何操作。

Run:
    python scripts/redact_annotations.py --check    # 只报告
    python scripts/redact_annotations.py            # 执行（自动备份）
"""
from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotations"
DIST = ROOT / "dist"

REDACT = ["evidence", "definition", "axioms_before", "axioms_after",
          "llm_evidence_quote", "raw"]
TARGETS = ["alarm_pack.csv", "alarm_pack_llm.csv", "alarm_pack_review.csv",
           "alarm_pack_shuffled.csv", "alarm_pack_wide.csv"]


def plan():
    out = []
    for name in TARGETS:
        p = ANN / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
            fields = list(rows[0].keys()) if rows else []
        hits = {c: sum(1 for r in rows if (r.get(c) or "").strip()) for c in REDACT if c in fields}
        out.append((name, len(rows), hits))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    print(f"{'file':<28}{'rows':>6}  非空的待清空列")
    for name, n, hits in plan():
        print(f"{name:<28}{n:>6}  {hits or '—'}")
    if args.check:
        return

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = DIST / f"annotations_backup-{ts}"
    backup.mkdir(parents=True, exist_ok=True)

    for name in TARGETS:
        p = ANN / name
        if not p.exists():
            continue
        shutil.copy2(p, backup / name)
        with p.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames)
            rows = list(reader)
        for r in rows:
            for c in REDACT:
                if c in r:
                    r[c] = ""
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print("redacted", name)

    print("backup ->", backup)


if __name__ == "__main__":
    main()
