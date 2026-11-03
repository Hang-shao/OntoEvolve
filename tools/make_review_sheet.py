"""make_review_sheet - 生成人工审核工作表（LLM 判定可见，人工逐条确认或改正）。

与盲标表的区别：
    盲标表（alarm_pack_shuffled.csv）不含 LLM 结论，人工独立判定，用于计算 κ；
    审核表（alarm_pack_review.csv）并列给出 LLM 结论与理由，人工只需确认或改正，
                            用于快速得到"人工校正后的精确率"（此时不报 κ）。

注意：发布版的 alarm_pack.csv 已去原文（evidence / axioms_* 列为空），重跑本脚本会重建
      一张证据为空的表，并覆盖其中已填的人工判定。因此目标文件存在时默认拒绝写入，
      确需重建时加 --force。

Run:
    python tools/make_review_sheet.py [--force]
输出: annotations/alarm_pack_review.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "annotations"

FIELDS = ["alarm_id", "track", "t", "system_dtype", "concept", "n_members",
          "last_seen_t", "evidence", "axioms_before", "axioms_after",
          "llm_is_true_drift", "llm_dtype", "llm_confidence", "llm_rationale",
          "is_true_drift", "dtype_true", "confidence", "note"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="目标文件已存在时仍然覆盖（会丢失其中已填的人工判定）")
    args = ap.parse_args(argv)
    target = A / "alarm_pack_review.csv"
    if target.exists() and not args.force:
        print(f"{target} 已存在；重建会清空其中的人工判定列。确需重建请加 --force。")
        return 1
    pack = {r["alarm_id"]: r for r in csv.DictReader(
        open(A / "alarm_pack.csv", encoding="utf-8-sig"))}
    llm = {r["alarm_id"]: r for r in csv.DictReader(
        open(A / "alarm_pack_llm.csv", encoding="utf-8-sig"))}
    rows = []
    for aid in sorted(pack, key=lambda s: (s.split("-")[0], int(s.split("-")[1]))):
        p, l = pack[aid], llm.get(aid, {})
        rows.append({
            "alarm_id": aid, "track": p["track"], "t": p["t"],
            "system_dtype": p["dtype"], "concept": p["concept"],
            "n_members": p["n_members"], "last_seen_t": p["last_seen_t"],
            "evidence": p["evidence"],
            "axioms_before": p["axioms_before"], "axioms_after": p["axioms_after"],
            "llm_is_true_drift": l.get("llm_is_true_drift", ""),
            "llm_dtype": l.get("llm_dtype", ""),
            "llm_confidence": l.get("llm_confidence", ""),
            "llm_rationale": l.get("llm_rationale", ""),
            "is_true_drift": "", "dtype_true": "", "confidence": "", "note": "",
        })
    out = A / "alarm_pack_review.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[review] {len(rows)} 条 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
