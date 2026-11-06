# -*- coding: utf-8 -*-
r"""双人盲评的一致性指标（配合 `scripts/build_rater_sheets.py`）。

读回两张打分表（行序可以不同，按 `alarm_id` 对齐），输出：
  1) 覆盖与未填、判定分布
  2) 三类口径（真漂移 / 误报 / 无法判断）：一致率、Cohen's κ、Krippendorff's α、
     PABAK，各自 95% bootstrap CI（按条目重抽）与混淆矩阵
  3) 二类口径（双方都可判定的子集）：一致率、κ、阳性/阴性一致率（含计数）
  4) 各自精度与分轨道精度（Wilson 区间），以及"共同可判定子集"的同分母精度
  5) 与首位标注者 R1 的对照（R1 非盲，其一致性只能当参考；会顺手校验 13/82 = 0.1585）
  6) 待仲裁的分歧条目清单（只列 ID 与两方标签，不含理由）

不做的事：不合并两位标注者的结论去改写已发表的 0.1585；仲裁结果只另列。

用法：
    python scripts/score_rater_agreement.py --a dist/rater_pack/rater_R2.csv \
                                            --b dist/rater_pack/rater_R3.csv
    python scripts/score_rater_agreement.py --a ... --b ... --out report.md --boot 2000
    python scripts/score_rater_agreement.py --selftest        # 只跑数学自检

κ 的 CI 与 α 的 CI 按条目重抽；条目数只有 85、其中阳性十余条，区间本来就会很宽，
报告里同时给计数，别只给比例。
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

T, F, U = "true", "false", "undecidable"
NAMES = {T: "真漂移", F: "误报", U: "无法判断"}
ALIASES = {
    "真漂移": T, "true": T, "t": T, "yes": T, "y": T, "1": T, "真": T, "是": T,
    "drift": T, "true_drift": T,
    "误报": F, "false": F, "f": F, "no": F, "n": F, "0": F, "假": F, "否": F,
    "false_alarm": F,
    "无法判断": U, "undecidable": U, "u": U, "无法判定": U, "不确定": U, "?": U,
}


# ---------------------------------------------------------------- 基础数学

def canon(x: str):
    s = (x or "").strip().lower()
    return ALIASES.get(s)


def kappa(pairs) -> tuple[float, float]:
    """Cohen's κ（固定边际）与原始一致率；输入 [(a, b)]。"""
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (1.0 if pe == 1 else (po - pe) / (1 - pe)), po


def pabak(po: float) -> float:
    return 2 * po - 1


def kripp_alpha(units) -> float:
    """Krippendorff's α（nominal，允许缺测）。units: [[label, label], ...]"""
    vals = sorted({v for u in units for v in u if v is not None})
    if len(vals) < 2:
        return float("nan")
    idx = {v: i for i, v in enumerate(vals)}
    k = len(vals)
    o = [[0.0] * k for _ in range(k)]
    n = 0
    for u in units:
        vals_u = [v for v in u if v is not None]
        m = len(vals_u)
        if m < 2:
            continue
        n += m
        for i, a in enumerate(vals_u):
            for j, b in enumerate(vals_u):
                if i == j:
                    continue
                o[idx[a]][idx[b]] += 1.0 / (m - 1)
    if n < 2:
        return float("nan")
    do = sum(o[i][j] for i in range(k) for j in range(k) if i != j) / n
    nc = [sum(o[i]) for i in range(k)]
    de = sum(nc[i] * nc[j] for i in range(k) for j in range(k) if i != j) / (n * (n - 1))
    return float("nan") if de == 0 else 1 - do / de


def boot_ci(fn, units, rounds=2000, seed=0):
    """按条目重抽（units 为单位），返回百分位 95% 区间。"""
    rng = random.Random(seed)
    n = len(units)
    vals = []
    for _ in range(rounds):
        s = [units[rng.randrange(n)] for _ in range(n)]
        with_both = [u for u in s if u[0] is not None and u[1] is not None]
        if not with_both:
            continue
        v = fn(s)
        if v is not None and not math.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return lo, hi


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


# ---------------------------------------------------------------- 数据读写

def load_sheet(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} 是空表")
    need = {"alarm_id", "label"}
    if not need <= set(rows[0].keys()):
        raise SystemExit(f"{path} 缺列 {sorted(need - set(rows[0].keys()))}")
    out, bad = {}, []
    for r in rows:
        aid = (r["alarm_id"] or "").strip()
        lab = canon(r.get("label", ""))
        if (r.get("label") or "").strip() and lab is None:
            bad.append((aid, r.get("label")))
        out[aid] = {"label": lab, "raw": (r.get("label") or "").strip(),
                    "confidence": (r.get("confidence") or "").strip(),
                    "track": (r.get("track") or "").strip(),
                    "dtype": (r.get("dtype") or "").strip()}
    if bad:
        raise SystemExit(f"{path} 有无法识别的 label：{bad}")
    return out, rows


def load_r1(path: Path):
    """首位标注者（非盲）：is_true_drift / dtype_true / confidence。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out[(r["alarm_id"] or "").strip()] = {
            "label": canon(r.get("is_true_drift", "")),
            "track": (r.get("track") or "").strip(),
            "dtype": (r.get("system_dtype") or "").strip(),
        }
    return out


def load_arbitration(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        lab = canon(r.get("ruling", ""))
        if lab:
            out[(r.get("alarm_id") or "").strip()] = lab
    return out


# ---------------------------------------------------------------- 报告

def counts(sub, label_key):
    c = Counter(s[label_key] for s in sub)
    dec = sum(v for k, v in c.items() if k is not None)
    return c, dec


def precision(sheet, ids=None):
    ids = ids if ids is not None else list(sheet)
    c = Counter(sheet[i]["label"] for i in ids if sheet[i]["label"] is not None)
    t, dec = c.get(T, 0), c.get(T, 0) + c.get(F, 0)
    return t, dec, (t / dec if dec else float("nan"))


def fmt(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def report(a_sheet, b_sheet, a_name, b_name, r1=None, arb=None, boot=2000, seed=0):
    ids = sorted(set(a_sheet) & set(b_sheet))
    only_a = sorted(set(a_sheet) - set(b_sheet))
    only_b = sorted(set(b_sheet) - set(a_sheet))
    L = []
    add = L.append

    add("# 双人盲评一致性报告")
    add("")
    add(f"- 标注者 A：`{a_name}`；B：`{b_name}`；条目 {len(ids)} 条"
        f"（A 独有 {len(only_a)}，B 独有 {len(only_b)}）")
    unfilled_a = [i for i in ids if a_sheet[i]["label"] is None]
    unfilled_b = [i for i in ids if b_sheet[i]["label"] is None]
    add(f"- 未填：A {len(unfilled_a)} 条 {unfilled_a[:5]}，B {len(unfilled_b)} 条 {unfilled_b[:5]}")
    if only_a or only_b:
        add(f"- **ID 不一致**，先修表再解读指标：A 独有 {only_a}，B 独有 {only_b}")
    add("")

    # 一、分布
    ca, da = counts([a_sheet[i] for i in ids], "label")
    cb, db = counts([b_sheet[i] for i in ids], "label")
    add("## 一、判定分布")
    add("")
    add("| | 真漂移 | 误报 | 无法判断 | 未填 | 可判定 |")
    add("|---|---|---|---|---|---|")
    for nm, c, dec in ((a_name, ca, da), (b_name, cb, db)):
        add(f"| {nm} | {c.get(T,0)} | {c.get(F,0)} | {c.get(U,0)} | "
            f"{c.get(None,0)} | {dec} |")
    add("")

    # 二、三类一致性
    units = [(a_sheet[i]["label"], b_sheet[i]["label"]) for i in ids]
    add("## 二、三类一致性（含「无法判断」，未填的一对按缺测处理）")
    add("")
    k3, po3 = kappa(units)
    lo3, hi3 = boot_ci(lambda s: kappa(s)[0], units, boot, seed)
    al = kripp_alpha([[a, b] for a, b in units])
    al_lo, al_hi = boot_ci(lambda s: kripp_alpha([[x, y] for x, y in s]), units, boot, seed)
    add("| 指标 | 值 | 95% CI |")
    add("|---|---|---|")
    add(f"| n（两方都有判定） | {len(units)} | — |")
    add(f"| 原始一致率 | {fmt(po3)} | — |")
    add(f"| Cohen's κ | {fmt(k3)} | [{fmt(lo3)}, {fmt(hi3)}] |")
    add(f"| Krippendorff's α | {fmt(al)} | [{fmt(al_lo)}, {fmt(al_hi)}] |")
    add(f"| PABAK（2·po−1，边际极偏时的校正） | {fmt(pabak(po3))} | — |")
    add("")
    full = [u for u in units if u[0] is not None and u[1] is not None]
    add("混淆矩阵（行 = A，列 = B）")
    add("")
    add("| A \\ B | 真漂移 | 误报 | 无法判断 |")
    add("|---|---|---|---|")
    for x in (T, F, U):
        cells = [sum(1 for p in full if p == (x, y)) for y in (T, F, U)]
        add(f"| {NAMES[x]} | " + " | ".join(str(v) for v in cells) + " |")
    add("")

    # 三、二类（共同可判定）
    common = [i for i in ids
              if a_sheet[i]["label"] in (T, F) and b_sheet[i]["label"] in (T, F)]
    pairs2 = [(a_sheet[i]["label"], b_sheet[i]["label"]) for i in common]
    k2, po2 = kappa(pairs2)
    lo2, hi2 = boot_ci(lambda s: kappa(s)[0], pairs2, boot, seed)
    both_t = sum(1 for p in pairs2 if p == (T, T))
    at_least_t = sum(1 for p in pairs2 if p[0] == T or p[1] == T)
    both_f = sum(1 for p in pairs2 if p == (F, F))
    at_least_f = sum(1 for p in pairs2 if p[0] == F or p[1] == F)
    add("## 三、二类口径（双方都判可定的子集，1 − x/82 这类数从这里来）")
    add("")
    add(f"- n = {len(common)}；一致 {len(pairs2) - sum(1 for p in pairs2 if p[0] != p[1])} 条")
    add(f"- 一致率 {fmt(po2)}；Cohen's κ {fmt(k2)} [{fmt(lo2)}, {fmt(hi2)}]；κ 的 PABAK {fmt(pabak(po2))}")
    add(f"- **阳性一致率** {fmt(both_t / at_least_t if at_least_t else float('nan'))}"
        f"（{both_t}/{at_least_t}；两人都判真漂移 / 至少一人判真漂移）")
    add(f"- **阴性一致率** {fmt(both_f / at_least_f if at_least_f else float('nan'))}"
        f"（{both_f}/{at_least_f}）")
    add("")

    # 四、精度
    add("## 四、精度（真漂移 / 可判定）")
    add("")
    add("| 口径 | 分子/分母 | 精度 | 95% CI (Wilson) |")
    add("|---|---|---|---|")
    for nm, sheet in ((a_name, a_sheet), (b_name, b_sheet)):
        t, dec, p = precision(sheet, ids)
        lo, hi = wilson(t, dec)
        add(f"| {nm} 自可判定子集 | {t}/{dec} | {fmt(p)} | [{fmt(lo)}, {fmt(hi)}] |")
    ta, _, pa = precision(a_sheet, common)
    tb, _, pb = precision(b_sheet, common)
    lo, hi = wilson(ta, len(common))
    add(f"| 共同可判定子集 · {a_name} | {ta}/{len(common)} | {fmt(pa)} | [{fmt(lo)}, {fmt(hi)}] |")
    lo, hi = wilson(tb, len(common))
    add(f"| 共同可判定子集 · {b_name} | {tb}/{len(common)} | {fmt(pb)} | [{fmt(lo)}, {fmt(hi)}] |")
    add("")
    tracks = sorted({a_sheet[i]["track"] or b_sheet[i]["track"] for i in ids})
    add(f"| 轨道 | {a_name} 精度 | {b_name} 精度 |")
    add("|---|---|---|")
    for tr in tracks:
        sub = [i for i in ids if (a_sheet[i]["track"] or b_sheet[i]["track"]) == tr]
        cells = []
        for sheet in (a_sheet, b_sheet):
            t, dec, p = precision(sheet, sub)
            cells.append(f"{t}/{dec} = {fmt(p)}")
        add(f"| {tr} | " + " | ".join(cells) + " |")
    add("")

    # 五、与 R1 对照
    if r1 is not None:
        add("## 五、与首位标注者 R1 的对照（R1 非盲，只作参考，不进 κ 结论）")
        add("")
        t, dec, p = precision(r1, sorted(r1))
        add(f"- R1 自身精度 {t}/{dec} = {fmt(p)}（校验：论文值 13/82 = 0.1585）")
        for nm, sheet in ((a_name, a_sheet), (b_name, b_sheet)):
            pairs = [(r1[i]["label"], sheet[i]["label"]) for i in ids
                     if i in r1 and r1[i]["label"] is not None and sheet[i]["label"] is not None]
            k, po = kappa(pairs)
            add(f"- R1 vs {nm}：n={len(pairs)}，一致率 {fmt(po)}，κ {fmt(k)}")
        add("")

    # 六、置信
    add("## 六、置信等级分布（单独报告，不并入 κ）")
    add("")
    for nm, sheet in ((a_name, a_sheet), (b_name, b_sheet)):
        c = Counter(sheet[i]["confidence"] for i in ids)
        add(f"- {nm}：" + "，".join(f"{k or '空'} {v}" for k, v in c.most_common()))
    add("")

    # 七、分歧
    dis = [i for i in ids if a_sheet[i]["label"] != b_sheet[i]["label"]]
    add(f"## 七、待仲裁分歧条目（{len(dis)} 条，只给 ID 与两方标签）")
    add("")
    if dis:
        add("| alarm_id | track | dtype | " + a_name + " | " + b_name + " |")
        add("|---|---|---|---|---|")
        for i in dis:
            add(f"| {i} | {a_sheet[i]['track']} | {a_sheet[i]['dtype']} | "
                f"{NAMES.get(a_sheet[i]['label'],'—')} | {NAMES.get(b_sheet[i]['label'],'—')} |")
        add("")
    else:
        add("无分歧。")
        add("")

    if arb:
        add("## 八、仲裁后（仅另列，不替换上表的 κ 与已发表的 0.1585）")
        add("")
        t = sum(1 for v in arb.values() if v == T)
        dec = sum(1 for v in arb.values() if v in (T, F))
        add(f"- 已裁定 {len(arb)}/{len(dis)} 条；判真漂移 {t}/{dec} = {fmt(t/dec if dec else float('nan'))}")
        add("")

    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- 自检

def selftest():
    # 1) 完全一致 -> κ = 1；α = 1
    units = [(T, T)] * 5 + [(F, F)] * 5
    assert kappa(units)[0] == 1.0
    assert abs(kripp_alpha([[a, b] for a, b in units]) - 1.0) < 1e-12
    # 2) 手算例：4×(T,T) 4×(F,F) 1×(T,F) 1×(F,T) -> po=0.8, κ=0.6, α=0.6190
    units = [(T, T)] * 4 + [(F, F)] * 4 + [(T, F), (F, T)]
    k, po = kappa(units)
    assert abs(po - 0.8) < 1e-12 and abs(k - 0.6) < 1e-9, (k, po)
    al = kripp_alpha([[a, b] for a, b in units])
    assert abs(al - 0.6190) < 1e-3, al
    # 3) 与 sklearn 对拍
    try:
        from sklearn.metrics import cohen_kappa_score
        rng = random.Random(7)
        xs = [rng.choice([T, F, U]) for _ in range(200)]
        ys = [x if rng.random() < 0.7 else rng.choice([T, F, U]) for x in xs]
        assert abs(kappa(list(zip(xs, ys)))[0] - cohen_kappa_score(xs, ys)) < 1e-12
    except ImportError:
        pass
    # 4) bootstrap 区间包含点估计
    units = [(T, T)] * 3 + [(F, F)] * 40 + [(F, T)] * 2
    lo, hi = boot_ci(lambda s: kappa(s)[0], units, 500, 0)
    k = kappa(units)[0]
    assert lo <= k <= hi, (lo, k, hi)
    # 5) Wilson 对称性
    lo, hi = wilson(5, 10)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9
    print("selftest ok：κ/α/PABAK/bootstrap/Wilson 全部通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, help="打分表 A（如 rater_R2.csv）")
    ap.add_argument("--b", type=Path, help="打分表 B（如 rater_R3.csv）")
    ap.add_argument("--a-name", default="R2")
    ap.add_argument("--b-name", default="R3")
    ap.add_argument("--r1", type=Path, default=None,
                    help="首位标注者的未脱敏 review 表（默认取 dist/ 下最新备份）")
    ap.add_argument("--no-r1", action="store_true")
    ap.add_argument("--arbitration", type=Path, default=None, help="仲裁表：alarm_id,ruling")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="报告写到文件（默认只打印）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not (a.a and a.b):
        raise SystemExit("需要 --a 与 --b（或用 --selftest）")

    a_sheet, _ = load_sheet(a.a)
    b_sheet, _ = load_sheet(a.b)
    r1 = None
    if not a.no_r1:
        p = a.r1
        if p is None:
            cands = sorted(DIST.glob("annotations_backup-*/alarm_pack_review.csv"))
            p = cands[-1] if cands else None
        if p and p.exists():
            r1 = load_r1(p)
    arb = load_arbitration(a.arbitration) if a.arbitration and a.arbitration.exists() else None

    text = report(a_sheet, b_sheet, a.a_name, a.b_name, r1, arb, a.boot, a.seed)
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
        print("报告 ->", a.out)


if __name__ == "__main__":
    main()
