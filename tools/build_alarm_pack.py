"""build_alarm_pack - 把两条真实轨道的告警证据合并为人工/LLM 标注包。

输入（由 run_track2.py / run_exp_pmc.py 的 --dump-alarms 产生）：
    results/_annot/track2_s0.jsonl
    results/_annot/pmc_s0.jsonl
输出：
    annotations/alarm_pack.csv           按 alarm_id 排序的完整表（供核对）
    annotations/alarm_pack_shuffled.csv  打乱顺序的标注表（供标注员使用）

设计要点：
    1. 标注单位是一条告警，键为 (轨道, t, dtype, cid)；
    2. 打乱顺序时固定随机种子，保证两位标注员拿到同一顺序；
    3. 不写入种子编号（两轨道均为种子 0），避免标注员据此联想；
    4. 标注列（is_true_drift / dtype_true / confidence / note）留空。

Run:
    python tools/build_alarm_pack.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "alarm_id", "track", "t", "dtype", "concept", "score", "n_members",
    "last_seen_t", "never_mentioned", "sem", "stat", "ax", "mean_conf",
    "nearest_concept", "parent_concept", "definition",
    "evidence", "axioms_before", "axioms_after",
    "is_true_drift", "dtype_true", "confidence", "note",
]


def _load(path: Path) -> List[Dict]:
    if not path.exists():
        sys.exit(f"[error] 缺少 {path}，请先运行对应实验脚本并加 --dump-alarms")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _row(rec: Dict, alarm_id: str) -> Dict:
    sig = rec.get("signals") or {}
    return {
        "alarm_id": alarm_id,
        "track": rec["track"],
        "t": rec["t"],
        "dtype": rec["dtype"],
        "concept": rec.get("name", ""),
        "score": rec.get("score", ""),
        "n_members": rec.get("n_members", ""),
        "last_seen_t": rec.get("last_seen_t", ""),
        "never_mentioned": "是" if rec.get("never_mentioned") else "",
        "sem": sig.get("sem", ""),
        "stat": sig.get("stat", ""),
        "ax": sig.get("ax", ""),
        "mean_conf": sig.get("mean_conf", ""),
        "nearest_concept": rec.get("nearest_concept", ""),
        "parent_concept": rec.get("parent_concept", ""),
        "definition": " || ".join(rec.get("definition") or []),
        "evidence": " || ".join(rec.get("evidence") or []),
        "axioms_before": " || ".join(rec.get("axioms_before") or []),
        "axioms_after": " || ".join(rec.get("axioms_after") or []),
        "is_true_drift": "", "dtype_true": "", "confidence": "", "note": "",
    }


def main() -> int:
    out_dir = ROOT / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for tag, fname in (("T2", "results/_annot/track2_s0.jsonl"),
                       ("PMC", "results/_annot/pmc_s0.jsonl")):
        recs = _load(ROOT / fname)
        for i, rec in enumerate(sorted(recs, key=lambda r: (r["t"], r["cid"])), 1):
            rows.append(_row(rec, f"{tag}-{i:02d}"))
        print(f"[pack] {tag}: {len(recs)} 条")

    with (out_dir / "alarm_pack.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    shuffled = rows[:]
    random.Random(20261017).shuffle(shuffled)
    with (out_dir / "alarm_pack_shuffled.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(shuffled)

    empty_ev = sum(1 for r in rows if not r["evidence"])
    print(f"[pack] 总 {len(rows)} 条（无证据句 {empty_ev} 条，已用 never_mentioned 标记）")
    print(f"-> {out_dir / 'alarm_pack.csv'}")
    print(f"-> {out_dir / 'alarm_pack_shuffled.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
