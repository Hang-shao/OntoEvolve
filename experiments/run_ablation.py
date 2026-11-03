"""实验 4（消融）：逐个移除模块 D 的组件，观察 macro-F1 与各类 F1 的下降。

变体：
    full                三信号融合 + Project 证据衰减 + 模块 B 自动 K
    no_semantic         去掉语义信号（θ_sem/θ_shift 失效）
    no_stat             去掉统计信号（W/ε_ext 失效）
    no_axiom            去掉公理信号（θ_red 失效）
    no_support_decay    Project 不做“旧公理失去支撑则衰减”（support_decay=1）
    fixed_k_gt          关闭模块 B，固定 K=K_gt
    granularity_half    固定 K = K_gt/2（粒度偏粗）
    granularity_double  固定 K = 2·K_gt（粒度偏细）

阈值直接复用实验 1 的标定结果（不再重新调参），保证“只动被消融的组件”。

Run:
    python experiments/run_ablation.py [--repeats 10]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config as C
from src.build_ontology import build_initial_ontology
from src.calibrate import evaluate
from src.controlled_stream import make_controlled_stream
from src.embedding import Embedder
from src.eval_drift import TYPES, bootstrap_ci, wilcoxon_paired

INF = 1e6

VARIANTS: List[tuple] = [
    ("full", {}),
    ("no_semantic", {"theta_sem": INF, "theta_shift": INF, "w_sem": 0.0}),
    ("no_stat", {"window_w": 10 ** 6, "eps_ext": 0.0, "w_stat": 0.0}),
    ("no_axiom", {"theta_red": INF, "w_ax": 0.0}),
    ("no_support_decay", {"_support_decay": 1.0}),
    ("fixed_k_gt", {"_k": "gt"}),
    ("granularity_half", {"_k": "half"}),
    ("granularity_double", {"_k": "double"}),
]


def load_calibrated_params(path: Path) -> tuple:
    """读取实验 1 的标定阈值与链接阈值偏移；缺失时退回默认参数。"""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return (C.DriftParams(**data["calibration"]["params"]),
                float(data["calibration"].get("link_offset", 0.0)))
    print(f"[warn] 未找到 {path}，使用默认阈值")
    return C.PARAMS, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=C.PARAMS.n_repeats)
    ap.add_argument("--seed", type=int, default=C.PARAMS.seed)
    ap.add_argument("--summary-in", default=str(C.DIR_RESULTS / "exp1_summary.json"))
    ap.add_argument("--out", default=str(C.DIR_RESULTS / "exp4_ablation.csv"))
    ap.add_argument("--json-out", default=str(C.DIR_RESULTS / "exp4_ablation.json"))
    args = ap.parse_args()

    base_params, link_offset = load_calibrated_params(Path(args.summary_in))
    gen = C.PARAMS            # 数据流生成参数与实验 1 保持一致
    t_start = time.time()
    emb = Embedder(allow_fallback=False, verbose=True)

    macro: Dict[str, List[float]] = {v: [] for v, _ in VARIANTS}
    per_type: Dict[str, Dict[str, List[float]]] = {
        v: {t: [] for t in TYPES} for v, _ in VARIANTS}
    concepts_n: Dict[str, List[int]] = {v: [] for v, _ in VARIANTS}
    rows: List[Dict] = []

    for r in range(args.repeats):
        stream = make_controlled_stream(emb, seed=args.seed + r, params=gen)
        k_gt = len(stream.vocab_base)
        cache: Dict[str, tuple] = {}
        for name, spec in VARIANTS:
            k_fixed = None
            if spec.get("_k") == "gt":
                k_fixed = k_gt
            elif spec.get("_k") == "half":
                k_fixed = max(2, k_gt // 2)
            elif spec.get("_k") == "double":
                k_fixed = k_gt * 2
            key = f"k={k_fixed}"
            if key not in cache:
                cache[key] = build_initial_ontology(
                    stream.pool, stream.d0_iids, seed=0, k_gt=k_gt,
                    vocab={**stream.vocab_base, **stream.vocab_emer},
                    k_fixed=k_fixed)
            onto, info = cache[key]
            p = base_params.copy(**{k: v for k, v in spec.items()
                                    if not k.startswith("_")})
            res, _ = evaluate(stream, onto, info["leaf_to_cid"], p,
                              link_offset=link_offset,
                              base_p05=float(info.get("link_sim_p05", p.link_sim)),
                              support_decay=float(spec.get("_support_decay", 0.5)))
            macro[name].append(res.macro_f1)
            concepts_n[name].append(len(onto.concepts))
            for t in TYPES:
                per_type[name][t].append(res.per_type[t]["f1"])
            rows.append({"repeat": r, "variant": name, **res.as_row(),
                         "n_concepts": len(onto.concepts)})
        print(f"[rep {r:>2}] " + " ".join(
            f"{n}={macro[n][-1]:.3f}" for n, _ in VARIANTS))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)

    summary = {"n_repeats": args.repeats, "variants": {}}
    full_vals = macro["full"]
    for name, _ in VARIANTS:
        ci = bootstrap_ci(macro[name], n_boot=C.PARAMS.n_bootstrap, seed=args.seed)
        entry = {
            "macro_f1_mean": ci["mean"],
            "macro_f1_ci95": [ci["lo"], ci["hi"]],
            "per_type_mean": {t: float(np.mean(per_type[name][t])) for t in TYPES},
            "n_concepts": float(np.mean(concepts_n[name])),
            "delta_vs_full": ci["mean"] - float(np.mean(full_vals)),
        }
        if name != "full":
            entry["wilcoxon_vs_full"] = wilcoxon_paired(full_vals, macro[name])
        summary["variants"][name] = entry
    Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    print("\n==== 实验 4 消融（macro-F1，bootstrap 95% CI） ====")
    print(f"{'variant':<20}{'macroF1':>9}{'95%CI':>18}{'Δ vs full':>11}"
          f"{'eme':>7}{'ext':>7}{'red':>7}{'K':>6}{'p':>9}")
    for name, _ in VARIANTS:
        e = summary["variants"][name]
        p = e.get("wilcoxon_vs_full", {}).get("p", float("nan"))
        lo, hi = e["macro_f1_ci95"]
        print(f"{name:<20}{e['macro_f1_mean']:>9.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{e['delta_vs_full']:>+11.3f}"
              f"{e['per_type_mean']['emergence']:>7.2f}"
              f"{e['per_type_mean']['extinction']:>7.2f}"
              f"{e['per_type_mean']['redefinition']:>7.2f}"
              f"{e['n_concepts']:>6.0f}{p:>9.4f}")
    print(f"\n-> {out}\n-> {args.json_out}  ({time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
