"""experiments/track4_audit.py — 轨道 4（反诈公开材料）数据体检与类别语义核定。

阶段 D1。只做只读分析 + 产出标准化的候选用语料池，不构造批次、不调用任何 API。

输入（均在 <workspace>/data 下，公开数据集）：
    ScamGen/ScamGen.zip         Telescam 中文电诈意图语料（句子级 5 分类）
    HDRIN-main/ZhCCVi_clean.csv 中文电话诈骗转写（二分类，真实采集+平台收集）

输出：
    data/track4/pool/telescam.jsonl   去重后的句子池（id/text/label）
    data/track4/pool/zhccvi.jsonl     去空格还原后的转写池（id/text/label）
    data/track4/audit/d1_audit.json   机器可读体检结果
    data/track4/audit/d1_report.txt   人类可读体检报告（供撰写 docs/track4_audit.md）

用法：
    python experiments/track4_audit.py
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

WORKSPACE = ROOT.parent
DATA = WORKSPACE / "data"
OUT_POOL = ROOT / "data" / "track4" / "pool"
OUT_AUDIT = ROOT / "data" / "track4" / "audit"

ZIP_PATH = DATA / "ScamGen" / "ScamGen.zip"
ZIP_PREFIX = "Telescam-and-fraud-intent-detection/Data/D_Datasest_v5/1.datasetv1/"
TELESCAM_FILES = ["Easy_train.txt", "Easy_dev.txt", "Easy_test.txt"]
ZHCCVI = DATA / "HDRIN-main" / "ZhCCVi_clean.csv"

CJK = re.compile(r"[\u4e00-\u9fff]")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()


def detokenize(s: str) -> str:
    """ZhCCVi 的 transcript 是分词后以空格拼接，还原为连续中文。"""
    toks = s.split(" ")
    out: list[str] = []
    for tok in toks:
        if out and CJK.match(out[-1][-1:]) and CJK.match(tok[:1]):
            out[-1] += tok
        else:
            out.append(tok)
    return "".join(out)


def load_telescam():
    rows, per_file = [], {}
    z = zipfile.ZipFile(ZIP_PATH)
    for fn in TELESCAM_FILES:
        try:
            raw = z.read(ZIP_PREFIX + fn)
        except KeyError:
            per_file[fn] = None
            continue
        lines = raw.decode("utf-8").splitlines()
        n = 0
        for ln, line in enumerate(lines):
            if not line.strip():
                continue
            if "\t" in line:
                text, label = line.rsplit("\t", 1)
            else:
                text, label = line, ""
            text, label = norm_text(text), label.strip()
            if not text or label == "":
                continue
            rows.append({"id": f"{fn}:{ln}", "text": text, "label": label,
                         "src": "telescam", "split": fn})
            n += 1
        per_file[fn] = n
    return rows, per_file


def load_zhccvi():
    rows = []
    with ZHCCVI.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for i, r in enumerate(rd):
            t = detokenize(norm_text(r.get("transcript") or ""))
            if not t:
                continue
            lab = (r.get("label") or "").strip()
            try:
                lab = str(int(float(lab)))
            except Exception:
                pass
            rows.append({"id": f"zh{i}", "text": t, "label": lab,
                         "src": "zhccvi",
                         "summary": norm_text(r.get("zhaiyao") or "")})
    return rows


def dedupe(rows):
    seen, out, dups = set(), [], Counter()
    for r in rows:
        k = r["text"]
        if k in seen:
            dups[r.get("split") or r.get("src")] += 1
            continue
        seen.add(k)
        out.append(r)
    return out, dups


def label_profile(rows, top_k: int = 25):
    """按标签做字符 n-gram TF-IDF 画像（中文无需分词，字符 2—4gram）。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    labels = sorted({r["label"] for r in rows})
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=5,
                          max_features=40000, sublinear_tf=True)
    X = vec.fit_transform([r["text"] for r in rows])
    names = np.asarray(vec.get_feature_names_out())
    y = np.asarray([r["label"] for r in rows])
    prof = {}
    for lab in labels:
        m = (y == lab)
        mean = np.asarray(X[m].mean(axis=0)).ravel()
        # 相对提升：该类均值 / 全体均值，避免高频通用词霸榜
        allmean = np.asarray(X.mean(axis=0)).ravel() + 1e-9
        lift = mean / allmean
        idx = np.argsort(-lift)[:top_k]
        prof[lab] = [(str(names[i]), round(float(lift[i]), 2),
                      round(float(mean[i]), 4)) for i in idx if mean[i] > 0]
    # 类间区分度：卡方
    from sklearn.feature_selection import chi2
    chi, _ = chi2(X, y)
    top_chi = [(str(names[i]), round(float(chi[i]), 1))
               for i in np.argsort(-chi)[:40]]
    return prof, top_chi


def sample_per_label(rows, n: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r["text"])
    out = {}
    for lab, texts in sorted(by.items()):
        idx = rng.permutation(len(texts))[:n]
        out[lab] = [texts[int(i)] for i in idx]
    return out


def len_stats(rows):
    L = np.asarray([len(r["text"]) for r in rows], dtype=float)
    if not len(L):
        return {}
    return {"n": int(len(L)), "mean": round(float(L.mean()), 1),
            "p50": round(float(np.percentile(L, 50)), 1),
            "p95": round(float(np.percentile(L, 95)), 1),
            "max": int(L.max())}


def main() -> int:
    OUT_POOL.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    rep: list[str] = []
    audit: dict = {}

    # ---------- Telescam ----------
    ts_raw, per_file = load_telescam()
    ts, ts_dups = dedupe(ts_raw)
    rep.append("=" * 70)
    rep.append("【1】Telescam（ScamGen.zip / 1.datasetv1，仅 Easy 掩码级别）")
    rep.append(f"文件行数：{per_file}")
    rep.append(f"原始条数 {len(ts_raw)} → 去重后 {len(ts)}（删除重复 {len(ts_raw)-len(ts)}）")
    lab_cnt = Counter(r["label"] for r in ts)
    rep.append(f"标签分布：{dict(sorted(lab_cnt.items()))}")
    rep.append(f"长度统计：{len_stats(ts)}")
    audit["telescam"] = {"files": per_file, "raw": len(ts_raw),
                         "dedup": len(ts), "labels": dict(lab_cnt),
                         "len": len_stats(ts)}

    prof, top_chi = label_profile(ts)
    rep.append("\n-- 每类字符 2—3gram 最高提升词（相对提升，Notch 越高越具类特征）：")
    for lab in sorted(prof):
        toks = "、".join(f"{w}×{l}" for w, l, _ in prof[lab][:18])
        rep.append(f"  label={lab} (n={lab_cnt[lab]}): {toks}")
    rep.append("\n-- 全体卡方最强区分 2—3gram（前 40）：")
    rep.append("  " + "、".join(f"{w}" for w, _ in top_chi))
    rep.append("\n-- 每类随机样本（seed=0）：")
    for lab, texts in sample_per_label(ts).items():
        rep.append(f"  [label={lab}]")
        for t in texts:
            rep.append(f"    · {t[:120]}")
    audit["telescam_profile"] = {k: v[:18] for k, v in prof.items()}

    with (OUT_POOL / "telescam.jsonl").open("w", encoding="utf-8") as f:
        for r in ts:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- ZhCCVi ----------
    zh_raw = load_zhccvi()
    zh, zh_dups = dedupe(zh_raw)
    # 去重后再去“高度相似”（前缀 30 字相同）近重复
    seen_pref, zh2, near = set(), [], 0
    for r in zh:
        k = r["text"][:30]
        if k in seen_pref:
            near += 1
            continue
        seen_pref.add(k)
        zh2.append(r)
    zh = zh2
    rep.append("\n" + "=" * 70)
    rep.append("【2】ZhCCVi（HDRIN-main/ZhCCVi_clean.csv，真实转写，二分类）")
    rep.append(f"原始 {len(zh_raw)} → 完全去重 {len(zh_raw)-len(zh_dups)} → "
               f"前缀近重复再删 {near} → 保留 {len(zh)}")
    zl = Counter(r["label"] for r in zh)
    rep.append(f"标签分布：{dict(sorted(zl.items()))}")
    rep.append(f"长度统计：{len_stats(zh)}")
    rep.append(f"摘要字段非空比例："
               f"{np.mean([bool(r.get('summary')) for r in zh]):.3f}")
    rep.append("样本：")
    for r in zh[:4]:
        rep.append(f"  [{r['label']}] {r['text'][:110]}")
    audit["zhccvi"] = {"raw": len(zh_raw), "dedup": len(zh), "labels": dict(zl),
                       "near_dup_removed": near, "len": len_stats(zh)}
    with (OUT_POOL / "zhccvi.jsonl").open("w", encoding="utf-8") as f:
        for r in zh:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- 交集检查 ----------
    ts_set = {r["text"] for r in ts}
    overlap = sum(1 for r in zh if r["text"] in ts_set)
    rep.append("\n" + "=" * 70)
    rep.append(f"【3】两语料完全重合句数：{overlap}")
    audit["overlap"] = overlap

    (OUT_AUDIT / "d1_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_AUDIT / "d1_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join([r for r in rep if not CJK.search(r)]) or "report written")
    print(f"-> {OUT_POOL}\\telescam.jsonl ({len(ts)})")
    print(f"-> {OUT_POOL}\\zhccvi.jsonl ({len(zh)})")
    print(f"-> {OUT_AUDIT}\\d1_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
