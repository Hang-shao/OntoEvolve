"""命题 B 的“快”：增量演化 vs 全量重构 的成本对比。

命题 B 要证明“增量演化较全量重构降低一个数量级以上成本”。
本实验在同一条数据流上跑三条路径，比较每批耗时、累计耗时、内存峰值，
并给出质量对齐指标（增量结果是否达到全量重建的本体质量）：

    incremental  增量演化：检测 → 演化算子 → 一致性检查（我们的方法）
    full_rebuild 全量重构：每批都用“截止当前”的全部数据从头聚类/抽取/构建
    periodic     周期性全量重构：每 P 批重建一次（工程上常见的折中）

成本差异的来源：全量重构每批都要处理 n_t = |D_0| + Σ|D_i| 条数据（累计
O(T²)），而增量只处理本批新数据（累计 O(T)）。因此加速比随 T 单调上升。

用法：
    python experiments/run_exp_cost.py --quick
    python experiments/run_exp_cost.py                # 全量（T=36）
    python experiments/run_exp_cost.py --no-check     # 增量不做一致性检查（理论最快）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                     # noqa: E402
from src.build_ontology import (build_from_occurrences,         # noqa: E402
                                items_to_occurrences)
from src.controlled_stream import make_controlled_stream        # noqa: E402
from src.drift_detect import DriftDetector                      # noqa: E402
from src.embedding import Embedder                              # noqa: E402
from src.evolve import OntologyEvolver                          # noqa: E402
from src.extract import extract_candidates                      # noqa: E402
from src.owl_export import export_owl                           # noqa: E402

try:                                                            # 复用资源采样器
    from run_exp5 import ResourceTracker
except Exception:                                               # pragma: no cover
    ResourceTracker = None                                      # type: ignore


def _l2(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n == 0, 1.0, n)


def ontology_match(onto_a, onto_b, thr: float = 0.9) -> Dict:
    """两个本体活跃概念质心的最近邻匹配：增量结果是否达到全量重建的质量。"""
    A = [c for c in onto_a.concepts.values() if c.active]
    B = [c for c in onto_b.concepts.values() if c.active]
    if not A or not B:
        return {"n_a": len(A), "n_b": len(B), "match_rate": 0.0, "mean_sim": 0.0}
    Ma = _l2(np.vstack([np.asarray(c.centroid, dtype=np.float64) for c in A]))
    Mb = _l2(np.vstack([np.asarray(c.centroid, dtype=np.float64) for c in B]))
    S = Ma @ Mb.T
    best = S.max(axis=1)
    return {"n_a": len(A), "n_b": len(B),
            "match_rate": round(float(np.mean(best >= thr)), 3),
            "mean_sim": round(float(best.mean()), 3)}


def run_cost(seed: int, params: C.DriftParams, emb: Embedder, *,
             check: bool = True, period: int = 0, full_every: int = 1,
             outdir: Path | None = None) -> Dict:
    stream = make_controlled_stream(emb, seed=seed, params=params)
    vocab = {**stream.vocab_base, **stream.vocab_emer}
    k_gt = len(stream.vocab_base)
    d0_occ = items_to_occurrences(stream.pool, stream.d0_iids)

    rows: List[Dict] = []

    # ---------- 共同的初始构建 O_0 ----------
    t0 = time.perf_counter()
    onto0, info0 = build_from_occurrences(d0_occ, seed=0, k_gt=k_gt, vocab=vocab)
    setup_ms = (time.perf_counter() - t0) * 1000.0
    op_vocab, dp_vocab = list(onto0.op_vocab), list(onto0.dp_vocab)

    # ---------- 1) 增量演化 ----------
    onto = onto0
    p = params.copy(link_sim=max(0.50, float(info0.get("link_sim_p05",
                                                        params.link_sim)) - 0.02))
    det = DriftDetector(onto, p)
    evo = OntologyEvolver(onto, check=check)
    new_cid = len(onto.concepts)
    inc_ms: List[float] = []
    inc_mem: List[float] = []
    n_ops = 0

    for t, batch in enumerate(stream.batches, start=1):
        s = time.perf_counter()
        with (ResourceTracker() if ResourceTracker else _null()) as rt:
            alarms = det.observe(t, batch)
            for a in alarms:
                if a.dtype == "emergence":
                    new_cid += 1
                    cid = f"C{new_cid}"
                    items = list(a.detail.get("items", []))
                    if not items:
                        continue
                    axs = extract_candidates(cid, [o.as_dict() for o in items])
                    cen = a.detail.get("centroid")
                    ok = evo.add_concept(
                        cid, a.cid,
                        centroid=np.asarray(cen, dtype=np.float32)
                        if cen is not None else np.mean([o.vec for o in items],
                                                        axis=0),
                        axioms=axs)
                    if ok and cid in onto.concepts:
                        det.centroids[cid] = np.asarray(
                            onto.concepts[cid].centroid, dtype=np.float32)
                        det.counts[cid] = []
                        det.weights[cid] = []
                elif a.dtype == "redefinition":
                    axs = det.last_projected.get(a.cid) or []
                    if axs:
                        evo.redefine(a.cid, axs)
                elif a.dtype == "extinction":
                    evo.retire(a.cid)
                n_ops += 1
        ms = (time.perf_counter() - s) * 1000.0
        inc_ms.append(ms)
        inc_mem.append(getattr(rt, "peak_mb", 0.0))
        rows.append({"t": t, "track": "incremental", "ms": round(ms, 1),
                     "kind": "measured", "n_data": len(batch),
                     "n_total": len(d0_occ) + sum(len(b) for b in
                                                  stream.batches[:t]),
                     "n_concepts": len([c for c in onto.concepts.values()
                                        if c.active]),
                     "peak_mem_mb": round(getattr(rt, "peak_mb", 0.0), 1)})
    onto_inc = onto

    # ---------- 2) 全量重构（每批） ----------
    # 全量重构在大规模上很慢（聚类随 n 超线性增长），默认每 full_every 批
    # 实测一次，其余用拟合模型补全；哪些是实测、哪些是估计都会写进结果。
    occ_so_far = list(d0_occ)
    full_ms: List[float] = []
    full_mem: List[float] = []
    measured: Dict[int, float] = {}
    n_of_t: Dict[int, int] = {}
    onto_full = None
    T = len(stream.batches)
    for t, batch in enumerate(stream.batches, start=1):
        occ_so_far.extend(batch)
        n_of_t[t] = len(occ_so_far)
        do_measure = (full_every <= 1 or t % full_every == 0 or t == T)
        if not do_measure:
            full_ms.append(float("nan"))
            continue
        print(f"[full] t={t}/{T} n={len(occ_so_far)} ...", flush=True)
        s = time.perf_counter()
        with (ResourceTracker() if ResourceTracker else _null()) as rt:
            onto_full, _ = build_from_occurrences(
                occ_so_far, seed=0, k_gt=k_gt, vocab=vocab,
                op_vocab=op_vocab, dp_vocab=dp_vocab)
        ms = (time.perf_counter() - s) * 1000.0
        measured[t] = ms
        full_ms.append(ms)
        full_mem.append(getattr(rt, "peak_mb", 0.0))
        print(f"[full] t={t} done {ms/1000:.1f}s", flush=True)
        rows.append({"t": t, "track": "full_rebuild", "ms": round(ms, 1),
                     "kind": "measured", "n_data": len(batch),
                     "n_total": len(occ_so_far),
                     "n_concepts": len([c for c in onto_full.concepts.values()
                                        if c.active]),
                     "peak_mem_mb": round(getattr(rt, "peak_mb", 0.0), 1)})

    fit = None
    if full_every > 1 and len(measured) >= 2:
        xs = np.array([n_of_t[t] for t in sorted(measured)], dtype=float)
        ys = np.array([measured[t] for t in sorted(measured)], dtype=float)
        A = np.vstack([xs ** 2, xs]).T          # 聚类近似 O(n²)：ms ≈ a·n² + b·n
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        pred = A @ coef
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        fit = {"a_n2": a, "b_n": b, "r2": round(r2, 4),
               "n_measured": len(measured)}
        for i, t in enumerate(range(1, T + 1)):
            if np.isnan(full_ms[i]):
                n = n_of_t[t]
                est = a * n * n + b * n
                full_ms[i] = float(est)
                rows.append({"t": t, "track": "full_rebuild",
                             "ms": round(float(est), 1), "kind": "estimated",
                             "n_data": len(stream.batches[t - 1]),
                             "n_total": n, "n_concepts": None,
                             "peak_mem_mb": None})


    # ---------- 3) 周期性全量重构（每 P 批） ----------
    per_ms: List[float] = []
    if period and period > 0:
        occ_p: List = list(d0_occ)
        for t, batch in enumerate(stream.batches, start=1):
            occ_p.extend(batch)
            s = time.perf_counter()
            if t % period == 0 or t == len(stream.batches):
                build_from_occurrences(occ_p, seed=0, k_gt=k_gt, vocab=vocab,
                                       op_vocab=op_vocab, dp_vocab=dp_vocab)
                per_ms.append((time.perf_counter() - s) * 1000.0)
            else:
                per_ms.append(0.0)
        for t, ms in enumerate(per_ms, start=1):
            rows.append({"t": t, "track": f"periodic_P{period}",
                         "ms": round(ms, 1), "kind": "measured",
                         "n_data": len(
                             stream.batches[t - 1]),
                         "n_total": len(d0_occ) + sum(
                             len(b) for b in stream.batches[:t]),
                         "n_concepts": None, "peak_mem_mb": None})

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        export_owl(onto_inc, outdir / f"incremental_seed{seed}.owl")
        if onto_full is not None:
            export_owl(onto_full, outdir / f"full_rebuild_seed{seed}.owl")

    cum_inc = float(np.sum(inc_ms)) + setup_ms
    cum_full = float(np.sum(full_ms)) + setup_ms
    cum_per = float(np.sum(per_ms)) + setup_ms if per_ms else None
    match = ontology_match(onto_inc, onto_full) if onto_full is not None else {}

    return {
        "seed": seed, "T": len(stream.batches),
        "n_d0": len(d0_occ),
        "check": check, "period": period,
        "setup_ms": round(setup_ms, 1),
        "incremental": {
            "ms_mean": round(float(np.mean(inc_ms)), 1),
            "ms_total": round(float(np.sum(inc_ms)), 1),
            "cum_ms": round(cum_inc, 1),
            "peak_mem_mb": round(float(np.max(inc_mem)), 1),
            "n_ops": n_ops,
            "n_concepts_final": len([c for c in onto_inc.concepts.values()
                                     if c.active]),
            "rollback_rate": round(evo.stats.rollback_rate, 4),
        },
        "full_rebuild": {
            "ms_mean": round(float(np.mean(full_ms)), 1),
            "ms_total": round(float(np.sum(full_ms)), 1),
            "cum_ms": round(cum_full, 1),
            "peak_mem_mb": round(float(np.max(full_mem)), 1),
            "n_concepts_final": len([c for c in onto_full.concepts.values()
                                     if c.active]) if onto_full else None,
        },
        "periodic": {
            "ms_total": round(float(np.sum(per_ms)), 1) if per_ms else None,
            "cum_ms": round(cum_per, 1) if cum_per is not None else None,
        } if per_ms else None,
        "fit_full_rebuild": fit,
        "speedup_cumulative": round(cum_full / cum_inc, 2) if cum_inc else None,
        "speedup_per_batch": round(float(np.mean(full_ms)) /
                                   max(float(np.mean(inc_ms)), 1e-9), 2),
        "quality_match": match,
        "rows": rows,
    }


class _null:
    """ResourceTracker 不可用时的占位上下文管理器。"""
    peak_mb = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true",
                    help="增量不做一致性检查（理论最快，用作成本下界）")
    ap.add_argument("--period", type=int, default=0,
                    help="周期性全量重构的周期 P（0=不做）")
    ap.add_argument("--full-every", type=int, default=3,
                    help="全量重构每多少批实测一次（1=每批都实测，很慢）")
    ap.add_argument("--tag", default="main",
                    help="结果文件标签，避免不同规模的运行互相覆盖")
    args = ap.parse_args()

    base = C.PARAMS
    if args.quick:
        base = base.copy(n_batch=12, n_init_per_leaf=6, pool_per_leaf=12,
                         base_rate=3.0, n_emergence=1, n_extinction=1,
                         n_redefinition=1)
    emb = Embedder(allow_fallback=False, )
    outdir = C.DIR_ONTOLOGY / "exp_cost"

    res = run_cost(args.seed, base, emb, check=not args.no_check,
                   period=args.period, full_every=args.full_every,
                   outdir=outdir)

    print("== 命题 B：增量演化 vs 全量重构 ==")
    inc, full = res["incremental"], res["full_rebuild"]
    print(f"T={res['T']}  n(D_0)={res['n_d0']}  "
          f"初始构建 O_0 = {res['setup_ms']/1000:.2f}s")
    print(f"增量演化   每批 {inc['ms_mean']:8.1f} ms  "
          f"合计 {inc['ms_total']/1000:7.2f}s  "
          f"累计(含O_0) {inc['cum_ms']/1000:7.2f}s  "
          f"内存 {inc['peak_mem_mb']:.0f}MB  概念 {inc['n_concepts_final']}")
    print(f"全量重构   每批 {full['ms_mean']:8.1f} ms  "
          f"合计 {full['ms_total']/1000:7.2f}s  "
          f"累计(含O_0) {full['cum_ms']/1000:7.2f}s  "
          f"内存 {full['peak_mem_mb']:.0f}MB  概念 {full['n_concepts_final']}")
    if res.get("periodic"):
        print(f"周期重构P{args.period}  合计 "
              f"{res['periodic']['ms_total']/1000:.2f}s  累计 "
              f"{res['periodic']['cum_ms']/1000:.2f}s")
    print(f"加速比：累计 {res['speedup_cumulative']}×  "
          f"单批均值 {res['speedup_per_batch']}×")
    if res.get("fit_full_rebuild"):
        f = res["fit_full_rebuild"]
        print(f"（全量重构仅实测 {f['n_measured']} 个批次，其余按 "
              f"ms≈a·n²+b·n 拟合补全，R²={f['r2']}）")
    q = res["quality_match"]
    if q:
        print(f"质量对齐：增量 {q['n_a']} 概念 vs 全量 {q['n_b']} 概念；"
              f"质心匹配率(cos≥0.9) {q['match_rate']:.2f}，平均相似度 "
              f"{q['mean_sim']:.3f}")

    out = C.DIR_RESULTS / "exp_cost"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"exp_cost_{args.tag}"
    (out / f"{stem}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
        w.writeheader()
        w.writerows(res["rows"])
    print(f"\n结果写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
