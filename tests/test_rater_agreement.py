"""双人盲评工具链的单元测试：打分表列白名单、证据去重、κ/α/PABAK/Wilson 数值。

对应脚本 `scripts/build_rater_sheets.py` 与 `scripts/score_rater_agreement.py`，
用合成数据，不读 dist/ 下的真实标注包。
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(mod_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / "scripts" / f"{mod_name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load("build_rater_sheets")
score = _load("score_rater_agreement")


def _src_csv(path: Path, n: int = 8) -> Path:
    """合成未脱敏标注包：含可见列、黑名单列与重复证据句。"""
    head = ["alarm_id", "track", "t", "dtype", "concept", "score", "n_members",
            "last_seen_t", "never_mentioned", "definition", "evidence",
            "axioms_before", "axioms_after", "is_true_drift", "llm_is_true_drift", "note"]
    rows = []
    for i in range(n):
        rows.append({
            "alarm_id": f"A{i:02d}", "track": "T1", "t": str(i), "dtype": "emergence",
            "concept": f"c{i}", "score": "0.5", "n_members": "3", "last_seen_t": "",
            "never_mentioned": "是", "definition": "泄露用定义",
            "evidence": "句子甲 || 句子乙 || 句子甲", "axioms_before": "x",
            "axioms_after": "y", "is_true_drift": "是",
            "llm_is_true_drift": "否", "note": "与LLM一致",
        })
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=head)
        w.writeheader()
        w.writerows(rows)
    return path


def test_kappa_and_alpha_reference_values():
    T, F = score.T, score.F
    perfect = [(T, T)] * 5 + [(F, F)] * 5
    assert score.kappa(perfect) == (1.0, 1.0)
    assert score.kripp_alpha([[a, b] for a, b in perfect]) == pytest.approx(1.0)

    # 4×(T,T) 4×(F,F) 1×(T,F) 1×(F,T)：po=0.8, κ=0.6, α=0.6190
    units = [(T, T)] * 4 + [(F, F)] * 4 + [(T, F), (F, T)]
    k, po = score.kappa(units)
    assert po == pytest.approx(0.8)
    assert k == pytest.approx(0.6)
    assert score.kripp_alpha([[a, b] for a, b in units]) == pytest.approx(0.6190, abs=1e-3)


def test_wilson_reproduces_published_interval():
    """论文 13/82 的 95% CI 写作 [0.095, 0.253]，用同一口径复现。"""
    lo, hi = score.wilson(13, 82)
    assert lo == pytest.approx(0.095, abs=0.002)
    assert hi == pytest.approx(0.253, abs=0.002)


def test_build_sheet_whitelist_and_dedupe(tmp_path):
    src = _src_csv(tmp_path / "src.csv")
    rows, _, dup_rows, dup_sent = build.load_rows(src, dedupe=True)
    assert dup_rows == 8 and dup_sent == 8          # 每条去掉一个重复句
    assert all(r["evidence"] == "句子甲\n句子乙" for r in rows)

    out = tmp_path / "pack"
    build.build(rows, [1, 2], ["R2", "R3"], out, True, src)
    for name in ("rater_R2.csv", "rater_R3.csv"):
        with (out / name).open(encoding="utf-8-sig", newline="") as f:
            got = list(csv.DictReader(f))
        assert len(got) == 8
        assert set(got[0]) == {"seq"} | set(build.VISIBLE) | set(build.BLANK)
        assert not (set(got[0]) & (set(build.FORBIDDEN) - set(build.BLANK)))
        assert all(r["label"] == "" and r["confidence"] == "" for r in got)
    # 两张表行序不同，但条目集合相同
    a = [r["alarm_id"] for r in csv.DictReader((out / "rater_R2.csv").open(encoding="utf-8-sig"))]
    b = [r["alarm_id"] for r in csv.DictReader((out / "rater_R3.csv").open(encoding="utf-8-sig"))]
    assert a != b and set(a) == set(b)
    assert (out / "rater_sheet_manifest.json").exists()


def test_report_end_to_end(tmp_path):
    src = _src_csv(tmp_path / "src.csv", n=10)
    rows, _, _, _ = build.load_rows(src, dedupe=True)
    out = tmp_path / "pack"
    build.build(rows, [1, 2], ["R2", "R3"], out, True, src)

    ids = [f"A{i:02d}" for i in range(10)]
    labels_a = {i: ("真漂移" if k < 3 else "误报") for k, i in enumerate(ids)}
    labels_b = dict(labels_a)
    labels_b["A03"] = "无法判断"          # B 把一条改判为无法判断
    labels_b["A05"] = "真漂移"            # B 与 A 在另一条上分歧
    for name, labels in (("rater_R2.csv", labels_a), ("rater_R3.csv", labels_b)):
        p = out / name
        with p.open(encoding="utf-8-sig", newline="") as f:
            got = list(csv.DictReader(f))
        for r in got:
            r["label"] = labels[r["alarm_id"]]
            r["confidence"] = "高"
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(got[0].keys()))
            w.writeheader()
            w.writerows(got)

    a_sheet, _ = score.load_sheet(out / "rater_R2.csv")
    b_sheet, _ = score.load_sheet(out / "rater_R3.csv")
    text = score.report(a_sheet, b_sheet, "R2", "R3", boot=200, seed=0)

    assert "Cohen's κ" in text and "Krippendorff's α" in text
    # A 有 3 条真漂移；B 把 A03 改判为无法判断、把 A05 改判为真漂移
    # 共同可判定 9 条：一致 8 条，阳性 3/4，阴性 5/6
    assert "8 条" in text and "3/4" in text and "5/6" in text
    assert "3/10" in text                     # R2 自可判定子集精度 3/10
    assert "3/9" in text                      # 共同可判定子集下 R2 的精度 3/9
    # 分歧清单里有两条，且不含理由列
    assert "A03" in text and "A05" in text and "note" not in text
    assert "R1" not in text                   # 未传 r1 时不出对照节


def test_unrecognised_label_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alarm_id", "label", "confidence", "track", "dtype"])
        w.writeheader()
        w.writerow({"alarm_id": "A1", "label": "大概是吧", "confidence": "高",
                    "track": "T1", "dtype": "emergence"})
    with pytest.raises(SystemExit):
        score.load_sheet(p)
