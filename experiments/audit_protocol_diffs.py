# -*- coding: utf-8 -*-
"""G0 / G6 及各项协议修正的逐项差异审计。

只读已有结果文件 `results/baseline_protocol.json` 与
`results/baseline_protocol_detail_G0G6.json`，计算并写出一张可复核的审计表。
不改任何实验逻辑、不改任何评估口径。

产出
  results/protocol_audit.md   人读审计表
  results/protocol_audit.json 机器读（含每条断言）

退出码：任一条断言失败即 1。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.stats import wilcoxon  # noqa: E402

from src.eval_drift import bootstrap_ci  # noqa: E402

BASELINES = ("adwin", "page_hinkley", "embdist", "embkmeans", "inc_clustering")
ORDER = ("adwin", "inc_clustering", "page_hinkley", "embdist", "embkmeans")
LABEL = {
    "ontoevolve": "OntoEvolve (ours)",
    "adwin": "ADWIN",
    "inc_clustering": "Incremental clustering",
    "page_hinkley": "Page-Hinkley",
    "embdist": "Embedding distribution distance",
    "embkmeans": "Embedding k-means",
}
TYPES = ("emergence", "extinction", "redefinition")
SEED_CI = 42          # 与 C.PARAMS.seed 一致；已用 G0 复现 Table 3 的 CI 验证
N_BOOT = 1000

# V24 Table 3（执行口径）的公开数字，用于校验 G0 是否仍是同一张表
T3 = {
    "ontoevolve": (0.817, 0.716, 0.802, 0.778, 0.699, 0.845),
    "adwin": (0.557, 0.582, 0.050, 0.396, 0.356, 0.435),
    "inc_clustering": (0.689, 0.351, 0.000, 0.347, 0.304, 0.393),
    "page_hinkley": (0.440, 0.311, 0.114, 0.289, 0.260, 0.318),
    "embdist": (0.410, 0.082, 0.163, 0.218, 0.188, 0.250),
    "embkmeans": (0.354, 0.055, 0.067, 0.159, 0.138, 0.179),
}

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


def load() -> tuple[dict, dict]:
    main = json.loads((ROOT / "results/baseline_protocol.json").read_text("utf-8"))
    det_p = ROOT / "results/baseline_protocol_detail_G0G6.json"
    det = json.loads(det_p.read_text("utf-8")) if det_p.exists() else {}
    return main, det


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def paired(method: list[float], base: list[float]) -> dict:
    pairs = [(a, b) for a, b in zip(method, base)]
    diffs = [a - b for a, b in pairs]
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    wins = sum(1 for d in diffs if d > 1e-12)
    if not nonzero:
        return {"wins": wins, "nonzero": 0, "p": 1.0, "method": "identical"}
    if len(nonzero) < len(diffs):
        # 有零差值：精确分布不再适用，按 scipy 行为记录并标注
        r = wilcoxon(method, base)
        return {"wins": wins, "nonzero": len(nonzero), "p": float(r.pvalue),
                "method": "normal_approx (zeros present)"}
    ranks = _rankdata([abs(d) for d in nonzero])
    ties = len(set(ranks)) != len(ranks)
    r = wilcoxon(method, base, method="exact")
    return {"wins": wins, "nonzero": len(nonzero), "p": float(r.pvalue),
            "method": "exact" + (" (ties, exact may be invalid)" if ties else "")}


def _rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(xs[order[j + 1]] - xs[order[i]]) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def kwstr(kw: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in kw.items()) or "—"


def profile(cell: dict, detail: dict | None = None) -> dict:
    ms = cell["methods"]
    per = {m: [float(x) for x in ms[m]["per_repeat"]] for m in ms}
    ci = {m: bootstrap_ci(v, n_boot=N_BOOT, seed=SEED_CI) for m, v in per.items()}
    pt: dict[str, dict[str, float]] = {}
    if detail:
        for m in ms:
            d = detail["methods"][m].get("per_type")
            if d:
                pt[m] = {t: sum(d[t]) / len(d[t]) for t in d}
    return {"per": per, "ci": ci, "pt": pt,
            "mean": {m: sum(v) / len(v) for m, v in per.items()}}


def main() -> int:
    main_res, det = load()
    cells = list(main_res.keys())
    g0 = main_res["G0_v23_repro"]
    g6 = main_res["G6_main_candidate"]

    g0d = det.get("G0_v23_repro") if det else None
    g6d = det.get("G6_main_candidate") if det else None
    p0, p6 = profile(g0, g0d), profile(g6, g6d)

    # ---------- 断言 1：G0 仍是 Table 3 那张表 ----------
    worst = 0.0
    for m, (e, x, r, mac, lo, hi) in T3.items():
        pt = g0["methods"][m].get("per_type")
        if pt is None and m in det:
            pt = det[m]["methods"][m]["per_type"]
        d_macro = abs(p0["mean"][m] - mac)
        d_lo = abs(p0["ci"][m]["lo"] - lo)
        d_hi = abs(p0["ci"][m]["hi"] - hi)
        worst = max(worst, d_macro, d_lo, d_hi)
        if pt:
            for t, want in zip(TYPES, (e, x, r)):
                got = sum(pt[t]) / len(pt[t])
                worst = max(worst, abs(got - want))
    check("G0 复现 V24 Table 3（macro/CI/逐类型，容差 1e-3）", worst < 1e-3,
          f"最大偏差 {worst:.2e}")

    # ---------- 断言 2：本方法行十格恒定 ----------
    mvals = {c: main_res[c]["methods"]["ontoevolve"]["mean"] for c in cells}
    spread = max(mvals.values()) - min(mvals.values())
    check("本方法行在十格上恒定（审计探针）", spread < 1e-12,
          f"spread={spread:.2e}")

    # ---------- 断言 3：标定分数归属（本轮 0.5678 / 0.6120 混淆的根源）----------
    g1 = main_res["G1_cooldown_only"]["methods"]["inc_clustering"]
    g3 = main_res["G3_documented"]["methods"]["inc_clustering"]
    g4 = main_res["G4_ext_documented"]["methods"]["inc_clustering"]
    g6i = g6["methods"]["inc_clustering"]
    check("0.5678 = 仅修 cooldown（G1）的 inc_clustering 标定分",
          abs(g1["calib_f1"] - 0.5678) < 5e-5 and g1["kw"] == {"stale_window": 2},
          f"calib={g1['calib_f1']:.4f} kw={g1['kw']}")
    check("0.6120 = 两处都修（G3/G4）的 inc_clustering 标定分",
          abs(g3["calib_f1"] - 0.6120) < 5e-5 and abs(g4["calib_f1"] - 0.6120) < 5e-5
          and g3["kw"] == {"theta": 0.2},
          f"G3 calib={g3['calib_f1']:.4f} kw={g3['kw']}")
    check("G6 的 inc_clustering 既不是 0.5678 也不是 0.6120",
          abs(g6i["calib_f1"] - 0.4904) < 5e-5 and g6i["kw"] == {"theta": 0.35},
          f"calib={g6i['calib_f1']:.4f} kw={g6i['kw']} test={g6i['mean']:.4f}")
    hit5678 = [c for c in cells
               if any(abs(main_res[c]["methods"][m]["calib_f1"] - 0.5678) < 5e-5
                      for m in BASELINES if main_res[c]["methods"][m]["calib_f1"])]
    hit6120 = [c for c in cells
               if any(abs(main_res[c]["methods"][m]["calib_f1"] - 0.6120) < 5e-5
                      for m in BASELINES if main_res[c]["methods"][m]["calib_f1"])]
    check("0.5678 只出现在 G1；0.6120 只出现在 G3/G4",
          hit5678 == ["G1_cooldown_only"] and hit6120 == ["G3_documented", "G4_ext_documented"],
          f"0.5678@{hit5678} 0.6120@{hit6120}")

    # ---------- 断言 4：重跑确定性（若 detail 与本轮一致） ----------
    if det:
        same = all(det["G0_v23_repro"]["methods"][m]["per_repeat"]
                   == g0["methods"][m]["per_repeat"] for m in g0["methods"])
        same &= all(det["G6_main_candidate"]["methods"][m]["per_repeat"]
                    == g6["methods"][m]["per_repeat"] for m in g6["methods"])
    else:
        same = False
    check("G0/G6 重跑与归档逐重复逐位一致", same,
          "detail 已落盘" if det else "缺 baseline_protocol_detail_G0G6.json")

    # ---------- 相对增益与配对检验 ----------
    def gains(prof: dict) -> tuple[dict, dict]:
        raw = {b: paired(prof["per"]["ontoevolve"], prof["per"][b]) for b in ORDER}
        hol = holm({b: raw[b]["p"] for b in ORDER})
        return raw, hol

    gain0, hol0 = gains(p0)
    gain6, hol6 = gains(p6)
    check("G6 下五个基线全部 10/10 落败",
          all(gain6[b]["wins"] == 10 for b in ORDER),
          ",".join(f"{b}:{gain6[b]['wins']}" for b in ORDER))

    # ---------- ADWIN 与 inc_clustering 在 G6 下是否可分 ----------
    tie = paired(p6["per"]["inc_clustering"], p6["per"]["adwin"])
    check("G6 下 ADWIN 与 incremental clustering 统计上不可分（p>0.05）",
          tie["p"] > 0.05,
          f"inc={p6['mean']['inc_clustering']:.4f} adwin={p6['mean']['adwin']:.4f} "
          f"p={tie['p']:.3g}（{tie['method']}）")

    # ---------- 单因子对照 ----------
    F = {c: main_res[c]["config"] for c in cells}
    CONTRASTS = [
        ("G0_v23_repro", "G1_cooldown_only", "只修 cooldown（3→99）"),
        ("G0_v23_repro", "G2_tolerance_only", "只修 delay_tol（2→3）"),
        ("G0_v23_repro", "G3_documented", "两处同时修"),
        ("G3_documented", "G4_ext_documented", "仅扩网格（单流、声明口径）"),
        ("G3_documented", "G5_two_streams_documented", "仅双流（原网格、声明口径）"),
        ("G4_ext_documented", "G6_main_candidate", "仅双流（扩展网格、声明口径）"),
        ("G5_two_streams_documented", "G6_main_candidate", "仅扩网格（双流、声明口径）"),
        ("G8_ext_two_streams_legacy", "G6_main_candidate", "扩展+双流下：执行口径→声明口径"),
    ]
    delta_rows = []
    for a, b, lab in CONTRASTS:
        per = {}
        for m in BASELINES:
            per[m] = (main_res[b]["methods"][m]["mean"]
                      - main_res[a]["methods"][m]["mean"])
        delta_rows.append((a, b, lab, per))

    # ---------- 写 markdown ----------
    L: list[str] = []
    L.append("# 基线校准协议差异审计（G0–G9 + G0/G6 全量对比）")
    L.append("")
    L.append("由 `experiments/audit_protocol_diffs.py` 从 "
             "`results/baseline_protocol.json`、"
             "`results/baseline_protocol_detail_G0G6.json` 生成；不含手工数字。")
    L.append("")
    L.append("## A. 十格因子矩阵")
    L.append("")
    L.append("| 格 | 网格 | 标定流 | cooldown | delay_tol | " +
             " | ".join(LABEL[b] for b in ORDER) + " | 本方法 |")
    L.append("|---|---|---|---|---|" + "---|" * (len(ORDER) + 1))
    for c in cells:
        cfg = F[c]
        row = [c, "扩展" if cfg["grid"] == "ext" else "原", str(cfg["n_streams"]),
               str(cfg["calib_cooldown"]), str(cfg["calib_tolerance"])]
        row += [fmt(main_res[c]["methods"][m]["mean"]) for m in ORDER]
        row.append(fmt(main_res[c]["methods"]["ontoevolve"]["mean"]))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("标定阶段选中的参数与标定分：")
    L.append("")
    L.append("| 格 | " + " | ".join(LABEL[b] for b in ORDER) + " |")
    L.append("|---|" + "---|" * len(ORDER))
    for c in cells:
        vals = [f"{kwstr(main_res[c]['methods'][m]['kw'])}<br>calib "
                f"{fmt(main_res[c]['methods'][m]['calib_f1'], 4)}" for m in ORDER]
        L.append(f"| {c} | " + " | ".join(vals) + " |")
    L.append("")
    L.append("## B. 单因子对照（测试 macro-F1 的 Δ，正数 = 后一格更高）")
    L.append("")
    L.append("| 对照 | 改动 | " + " | ".join(LABEL[b] for b in ORDER) + " |")
    L.append("|---|---|" + "---|" * len(ORDER))
    for a, b, lab, per in delta_rows:
        L.append(f"| {a} → {b} | {lab} | " +
                 " | ".join(f"{per[m]:+.3f}" for m in ORDER) + " |")
    L.append("")
    L.append("## C. G0 与 G6 全量对比")
    L.append("")
    L.append("| 方法 | G0 macro-F1 [95% CI] | G6 macro-F1 [95% CI] | "
             "G6 相对增益 | G6 配对 p（精确） | Holm(n=5) | 10 对胜场 |")
    L.append("|---|---|---|---|---|---|---|")
    for m in ("ontoevolve",) + ORDER:
        c0, c6 = p0["ci"][m], p6["ci"][m]
        if m == "ontoevolve":
            L.append(f"| {LABEL[m]} | {fmt(p0['mean'][m])} "
                     f"[{fmt(c0['lo'])}, {fmt(c0['hi'])}] | {fmt(p6['mean'][m])} "
                     f"[{fmt(c6['lo'])}, {fmt(c6['hi'])}] | 1.00× | — | — | — |")
        else:
            g = p6["mean"]["ontoevolve"] / p6["mean"][m]
            L.append(f"| {LABEL[m]} | {fmt(p0['mean'][m])} "
                     f"[{fmt(c0['lo'])}, {fmt(c0['hi'])}] | {fmt(p6['mean'][m])} "
                     f"[{fmt(c6['lo'])}, {fmt(c6['hi'])}] | {g:.2f}× | "
                     f"{gain6[m]['p']:.3g}（{gain6[m]['method']}） | "
                     f"{hol6[m]:.3g} | {gain6[m]['wins']}/10 |")
    L.append("")
    L.append(f"ADWIN 与 incremental clustering 在 G6 下不可分："
             f"{p6['mean']['adwin']:.4f} 对 {p6['mean']['inc_clustering']:.4f}，"
             f"配对 p = {tie['p']:.3g}（{tie['method']}）。"
             f"因此“最强基线是 ADWIN”这一表述在 G6 下只能读作"
             f"“ADWIN 与 incremental clustering 并列最强”。")
    L.append("")
    L.append("逐类型 F1（G0 → G6，只列发生变化的单元；其余逐位不变）：")
    L.append("")
    L.append("| 方法 | Emergence | Extinction | Redefinition |")
    L.append("|---|---|---|---|")
    changed_any = False
    for m in ("ontoevolve",) + ORDER:
        cells_row = []
        for t in TYPES:
            v0 = p0.get("pt", {}).get(m, {}).get(t)
            v6 = p6.get("pt", {}).get(m, {}).get(t)
            if v0 is None or v6 is None:
                cells_row.append("n/a")
            elif abs(v6 - v0) < 5e-4:
                cells_row.append(f"{fmt(v0)} → 不变")
            else:
                changed_any = True
                cells_row.append(f"{fmt(v0)} → **{fmt(v6)}**（{v6 - v0:+.3f}）")
        L.append(f"| {LABEL[m]} | " + " | ".join(cells_row) + " |")
    L.append("")
    if det:
        L.append("告警计数（十次重复的均值）：")
        L.append("")
        L.append("| 方法 | G0 tp/fp/fn/告警 | G6 tp/fp/fn/告警 |")
        L.append("|---|---|---|")
        for m in ("ontoevolve",) + ORDER:
            def cs(cell):
                ct = det[cell]["methods"][m]["counts"]
                return "/".join(f"{sum(ct[k]) / len(ct[k]):.1f}"
                                for k in ("tp", "fp", "fn", "n_alarms"))
            L.append(f"| {LABEL[m]} | {cs('G0_v23_repro')} | {cs('G6_main_candidate')} |")
        L.append("")
    L.append("## D. 断言")
    L.append("")
    L.append("| # | 断言 | 结果 | 证据 |")
    L.append("|---|---|---|---|")
    for i, (n, ok, d) in enumerate(CHECKS, 1):
        L.append(f"| {i} | {n} | {'✅' if ok else '❌'} | {d} |")
    L.append("")
    L.append("## E. 本审计未覆盖")
    L.append("")
    L.append("- 十格中除 G0/G6 外没有落盘 per_type/counts，逐类型对照只在 G0↔G6 上成立。")
    L.append("- `mode=\"legacy\"`（inc_clustering 分轮搜索）是十格的共同设置；联合搜索"
             "（声明口径最优 θ=0.2、stale=3、标定 0.612）属另一次单独运行，未并入本表。")
    L.append("- 逐难度分层（by_difficulty）、警戒延迟、真实轨道指标未纳入。")
    L.append("")

    (ROOT / "results/protocol_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    payload = {
        "cells": {c: {"config": F[c],
                      "methods": {m: {"kw": main_res[c]["methods"][m]["kw"],
                                      "calib_f1": main_res[c]["methods"][m]["calib_f1"],
                                      "test_f1": main_res[c]["methods"][m]["mean"]}
                                  for m in main_res[c]["methods"]}}
                  for c in cells},
        "contrasts": [{"from": a, "to": b, "label": lab, "delta": per}
                      for a, b, lab, per in delta_rows],
        "g0": {"mean": p0["mean"], "ci": p0["ci"]},
        "g6": {"mean": p6["mean"], "ci": p6["ci"],
               "paired": {b: gain6[b] for b in ORDER},
               "holm": hol6,
               "relative_gain": {b: p6["mean"]["ontoevolve"] / p6["mean"][b]
                                 for b in ORDER},
               "adwin_vs_inc": tie},
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
    }
    (ROOT / "results/protocol_audit.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    bad = [n for n, ok, _ in CHECKS if not ok]
    print(f"[audit] {len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
    for n in bad:
        print("  FAIL:", n)
    print(f"[audit] -> results/protocol_audit.md / .json "
          f"({'逐类型有变化' if changed_any else '逐类型无变化'})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
