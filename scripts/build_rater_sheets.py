# -*- coding: utf-8 -*-
r"""从**未脱敏备份**生成双人盲评打分表（供 `V24_MC4_双人盲评流程草案.md` 的流程使用）。

输入只能是 `dist/annotations_backup-*/alarm_pack_review.csv`：
发布版 `annotations/alarm_pack.csv`、`alarm_pack_review.csv` 的 evidence/definition/
axioms 已被 `scripts/redact_annotations.py` 清空，拿它做打分表标注者将看不到证据。

可见列（白名单）：alarm_id, track, t, dtype, concept, n_members, last_seen_t, evidence
不可见列（黑名单，含判定、LLM 结论、特征分、派生内容）：见 FORBIDDEN，脚本会断言它们
没有进入输出。

同一条告警的 evidence 在生成时经常重复并列同一句话（85 条里 46 条如此），默认按 `||`
切开去重：重复句会让标注者误以为"出现多次 = 更重要"。

用法：
    python scripts/build_rater_sheets.py                     # 默认 85 条 -> dist/rater_pack/
    python scripts/build_rater_sheets.py --out-dir <目录> --names R2 R3 --seeds 20261106 20261107
    python scripts/build_rater_sheets.py --check             # 只报告，不写文件

每位标注者拿到的是行序不同的同一批条目（不同随机种子），合并时按 `alarm_id` 对齐。
填入的列只有 `label`（真漂移 / 误报 / 无法判断）、`confidence`（高 / 中 / 低）、
`note`（可选，一句话理由；仲裁时不交给仲裁人，避免按理由的措辞偏袒一方）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

VISIBLE = ["alarm_id", "track", "t", "dtype", "concept", "n_members",
           "last_seen_t", "evidence"]
FORBIDDEN = ["score", "sem", "stat", "ax", "mean_conf", "definition",
             "axioms_before", "axioms_after", "never_mentioned", "nearest_concept",
             "parent_concept", "llm_is_true_drift", "llm_dtype", "llm_confidence",
             "llm_evidence_quote", "llm_rationale", "raw", "is_true_drift",
             "dtype_true", "confidence", "note", "agree"]
BLANK = ["label", "confidence", "note"]


def default_src() -> Path:
    """优先 backup 里的 alarm_pack.csv（判定列本来就空，evidence 完整）。"""
    for pattern in ("annotations_backup-*/alarm_pack.csv",
                    "annotations_backup-*/alarm_pack_review.csv"):
        cands = sorted(DIST.glob(pattern))
        if cands:
            return cands[-1]
    raise SystemExit("找不到 dist/annotations_backup-*/alarm_pack.csv，"
                     "请用 --src 指定未脱敏备份")


def split_evidence(text: str, dedupe: bool) -> tuple[str, int, int]:
    """按 `||` 切句、去空、可选去重；返回 (拼接文本, 原句数, 去掉的重复句数)。"""
    parts = [p.strip() for p in (text or "").split("||")]
    parts = [p for p in parts if p]
    if not dedupe:
        return "\n".join(parts), len(parts), 0
    seen, keep = set(), []
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        keep.append(p)
    return "\n".join(keep), len(parts), len(parts) - len(keep)


def load_rows(src: Path, dedupe: bool):
    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    # review 表把告警类型叫 system_dtype，统一成 dtype
    if "dtype" not in fields and "system_dtype" in fields:
        for r in rows:
            r["dtype"] = r.get("system_dtype", "")
        fields = fields + ["dtype"]
    missing = [c for c in VISIBLE if c not in fields]
    if missing:
        raise SystemExit(f"{src} 缺少可见列 {missing}，确认是否为未脱敏备份")
    leaked = [c for c in FORBIDDEN if c not in fields]
    if leaked:
        print("提示：源文件本就没有这些列 %s" % leaked)

    out, dup_rows, dup_sent = [], 0, 0
    for r in rows:
        rec = {k: (r.get(k) or "").strip() for k in VISIBLE}
        text, n_old, n_dup = split_evidence(rec["evidence"], dedupe)
        rec["evidence"] = text
        if n_dup:
            dup_rows += 1
            dup_sent += n_dup
        out.append(rec)
    return out, fields, dup_rows, dup_sent


def check_only(rows, fields, src: Path, dup_rows, dup_sent):
    empty = [r["alarm_id"] for r in rows if not r["evidence"]]
    print(f"源文件   {src}")
    print(f"条目     {len(rows)}   证据为空 {len(empty)}：{empty}")
    print(f"重复句   {dup_rows} 条受影响，去掉 {dup_sent} 句")
    lens = [len(r["evidence"]) for r in rows]
    print(f"证据长度 中位 {statistics.median(lens):.0f} 字符，最长 {max(lens)}")
    from collections import Counter
    print("按轨道   ", dict(Counter(r["track"] for r in rows)))
    print("按类型   ", dict(Counter(r["dtype"] for r in rows)))
    leak = [c for c in FORBIDDEN if c in VISIBLE]
    print("黑名单泄漏", leak or "无")
    print("可见列   ", VISIBLE + BLANK)


def build(rows, seeds, names, out_dir: Path, dedupe: bool, src: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = VISIBLE + BLANK
    manifest = {"src": str(src), "src_sha256": sha256(src), "rows": len(rows),
                "dedupe_evidence": dedupe, "created": datetime.now().isoformat(timespec="seconds"),
                "sheets": []}
    for name, seed in zip(names, seeds):
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        out = out_dir / f"rater_{name}.csv"
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seq"] + fields)
            w.writeheader()
            for i, idx in enumerate(order, 1):
                rec = {"seq": i}
                rec.update({k: rows[idx][k] for k in VISIBLE})
                rec.update({k: "" for k in BLANK})
                w.writerow(rec)
        manifest["sheets"].append({"rater": name, "seed": seed,
                                   "file": out.name, "sha256": sha256(out)})
        print(f"写出 {out}   seed={seed}   {len(rows)} 行")
    (out_dir / "rater_sheet_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("清单", out_dir / "rater_sheet_manifest.json")
    print("填入用值：label = 真漂移 / 误报 / 无法判断；confidence = 高 / 中 / 低；note 可留空")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None,
                    help="未脱敏标注包（默认取 dist/ 下最新的 annotations_backup-*）")
    ap.add_argument("--out-dir", type=Path, default=DIST / "rater_pack")
    ap.add_argument("--names", nargs="+", default=["R2", "R3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[20261106, 20261107])
    ap.add_argument("--no-dedupe", action="store_true", help="保留重复证据句")
    ap.add_argument("--check", action="store_true", help="只报告统计，不写打分表")
    a = ap.parse_args()

    if len(a.names) != len(a.seeds):
        raise SystemExit("--names 与 --seeds 数量必须一致")
    src = a.src or default_src()
    rows, fields, dup_rows, dup_sent = load_rows(src, dedupe=not a.no_dedupe)
    if sum(1 for r in rows if r["evidence"]) < 0.5 * len(rows):
        raise SystemExit(f"{src} 里 evidence 大面积为空，可能不是未脱敏备份，停止")

    check_only(rows, fields, src, dup_rows, dup_sent)
    if a.check:
        return
    build(rows, a.seeds, a.names, a.out_dir, not a.no_dedupe, src)


if __name__ == "__main__":
    main()
