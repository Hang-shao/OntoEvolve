"""实验：轨道 3（PMC，生物医学全文）——跨领域泛化性验证。

与轨道 2（ICEWS18，政治事件时序知识图谱）使用**完全相同**的检测—演化—检查
流程，唯一差别是数据源与实例映射方式（见 src/pmc_stream.py 顶部对照表）。
目的是回答"方法是否只在时序知识图谱上成立"。

协议与轨道 2 一致：真实数据无注入真值，重定义阈值用无监督 μ+zσ（热身片
统计），因此本脚本报告的是**演化行为指标**（告警构成、概念/公理增长、
一致率、回退率、资源），而非 F1。

Run:
    python experiments/run_exp_pmc.py [--per-slice 600] [--n-slices 12]
                                      [--quick]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config as C
from src.alarm_pack import record_alarm
from src.build_ontology import build_from_occurrences
from src.consistency import consistency_report
from src.drift_detect import DriftDetector
from src.embedding import Embedder
from src.evolve import OntologyEvolver
from src.extract import extract_candidates
from src.pmc import load_pmc, slice_by_year
from src.pmc_stream import build_pmc_stream


def load_calibrated(path: Path):
    """沿用轨道 1 的标定参数（真实数据只做迁移，不重新标定）。"""
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        cal = d["calibration"]
        return C.DriftParams(**cal["params"]), float(cal.get("link_offset", 0.0))
    print(f"[warn] 未找到 {path}，使用默认阈值")
    return C.PARAMS, 0.0


def term_shift(stream: List[List]) -> List[Dict]:
    """每片相对上一片的术语分布变化（Jaccard），描述语料自身的漂移强度。"""
    rows: List[Dict] = []
    prev: Counter = Counter()
    for t, batch in enumerate(stream, start=1):
        cur = Counter(o.leaf for o in batch if o.leaf != "misc")
        union = set(cur) | set(prev)
        inter = set(cur) & set(prev)
        jac = len(inter) / len(union) if union else 0.0
        rows.append({"slice": t, "n_terms": len(cur),
                     "jaccard_prev": round(jac, 4),
                     "top": cur.most_common(3)})
        prev = cur
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-slice", type=int, default=600)
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--k-gt", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=300)
    ap.add_argument("--min-df", type=int, default=6)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--quick", action="store_true",
                    help="小规模试跑（每片 120 句、6 片）")
    ap.add_argument("--out-dir", default=str(C.DIR_RESULTS / "exp_pmc"))
    ap.add_argument("--dump-alarms", default="",
                    help="把每条告警的证据上下文导出为 JSONL，供人工/LLM 标注")
    args = ap.parse_args()
    if args.quick:
        args.per_slice, args.n_slices = 120, 6
        args.warmup = 2

    t_all = time.time()
    params, link_offset = load_calibrated(C.DIR_RESULTS / "exp1_summary.json")

    print("== 轨道 3：PMC 生物医学全文（跨领域验证）==")
    arts = load_pmc()
    print(f"[data] 文章 {len(arts)}  段落 {sum(len(a.paragraphs) for a in arts)}")

    # 严格模式：嵌入 API 不可用时直接失败，而不是静默退回哈希伪向量
    # （哈希向量不含语义，会让"跨领域验证"产出看似正常实则随机的结果）
    emb = Embedder(verbose=True, allow_fallback=False)
    t1 = time.time()
    stream, vocab = build_pmc_stream(
        arts, args.n_slices, emb, per_slice=args.per_slice, seed=args.seed,
        top_k=args.top_k, min_df=args.min_df)
    t_embed = time.time() - t1
    print(f"[embed] 句子 {sum(len(b) for b in stream)}  "
          f"api_calls={emb.n_api_calls}  {t_embed:.1f}s")

    op_vocab = list(vocab)
    dp_vocab = sorted({str(p) for b in stream for o in b for p, _ in o.dp_slots})
    onto, info = build_from_occurrences(stream[0], seed=args.seed,
                                        k_gt=args.k_gt,
                                        op_vocab=op_vocab, dp_vocab=dp_vocab)
    base_rate = len(stream[0]) / max(1, len(onto.concepts))
    p = params.copy(base_rate=base_rate,
                    link_sim=max(0.40, float(info.get("link_sim_p05", 0.9))
                                 - link_offset))
    # 必须在演化循环**之前**记录：onto 会被后续演化算子就地修改，
    # 循环后再取 len(onto.concepts) 得到的是最终概念数而非 O_0 的
    n_concepts_0 = len(onto.concepts)
    print(f"[O_0] M={onto.M}（术语 {len(op_vocab)} + dp {len(dp_vocab)}）"
          f"  concepts={n_concepts_0} K*={info.get('K_star')}  "
          f"base_rate={base_rate:.1f} link_sim={p.link_sim:.3f}")

    det = DriftDetector(onto, p)
    evolver = OntologyEvolver(onto)
    det.theta_red_eff = float("inf")            # 热身期不发重定义告警
    warm: List[float] = []

    FIELDS = ["slice", "sentences", "alarms", "n_emergence", "n_extinction",
              "n_redefinition", "concepts_active", "concepts_total", "axioms",
              "consistent", "version", "seconds"]
    rows: List[Dict] = []
    total: Counter = Counter()
    alarm_recs: List[Dict] = []
    for t, batch in enumerate(stream[1:], start=2):
        t0 = time.time()
        alarms = det.observe(t, batch)
        warm.extend(e["signals"]["ax"] for e in det.log if e["t"] == t)
        if t == args.warmup + 1 and warm:
            mu, sd = float(np.mean(warm)), float(np.std(warm))
            det.theta_red_eff = mu + args.z * max(sd, 1e-6)
            print(f"[calib] 无监督 θ_red = {mu:.2f}+{args.z}×{sd:.2f} "
                  f"= {det.theta_red_eff:.2f}（热身 {args.warmup} 片）")
        by: Counter = Counter()
        for a in alarms:
            by[a.dtype] += 1
            total[a.dtype] += 1
            if args.dump_alarms:
                alarm_recs.append(record_alarm(
                    track="PMC", seed=args.seed, alarm=a, detector=det,
                    onto=onto, emergence_prefix="topic_"))
            if a.dtype == "redefinition":
                evolver.redefine(a.cid, det.last_projected.get(a.cid, []))
            elif a.dtype == "extinction":
                evolver.retire(a.cid)
            elif a.dtype == "emergence":
                items = a.detail.get("items", [])
                axioms = extract_candidates(
                    a.cid, [o.as_dict() for o in items], parent=None)
                evolver.add_concept(a.cid, name=f"topic_{a.cid}",
                                    centroid=a.detail.get("centroid"),
                                    axioms=axioms)
        rep = consistency_report(onto)
        n_ax = sum(len(c.axioms) for c in onto.concepts.values())
        rows.append({"slice": t, "sentences": len(batch), "alarms": len(alarms),
                     "n_emergence": by.get("emergence", 0),
                     "n_extinction": by.get("extinction", 0),
                     "n_redefinition": by.get("redefinition", 0),
                     "concepts_active": sum(1 for c in onto.concepts.values()
                                            if c.active),
                     "concepts_total": len(onto.concepts), "axioms": n_ax,
                     "consistent": int(rep["consistent"]),
                     "version": onto.version,
                     "seconds": round(time.time() - t0, 2)})
        print(f"[slice {t:>2}] 句={len(batch):>4} 告警={len(alarms):>3} "
              f"{dict(by)} 活跃={rows[-1]['concepts_active']:>3} "
              f"公理={n_ax:>5} 一致={int(rep['consistent'])} "
              f"({time.time()-t0:.1f}s)")

    if args.dump_alarms and alarm_recs:
        dp = Path(args.dump_alarms)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with dp.open("w", encoding="utf-8") as f:
            for rec in alarm_recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[alarms] {len(alarm_recs)} 条已导出 -> {dp}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "exp_pmc.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    cons_rate = float(np.mean([r["consistent"] for r in rows])) if rows else 0.0
    shift = term_shift(stream)
    summary = {
        "dataset": {"n_articles": len(arts), "n_slices": args.n_slices,
                    "per_slice": args.per_slice, "vocab": len(vocab),
                    "year_range": [min(a.year for a in arts),
                                   max(a.year for a in arts)]},
        "O_0": {"M": onto.M, "concepts": n_concepts_0,
                "K_star": info.get("K_star"), "base_rate": base_rate,
                "link_sim": p.link_sim},
        "theta_red_unsupervised": (None if det.theta_red_eff == float("inf")
                                   else det.theta_red_eff),
        "alarms_total": dict(total),
        "consistency_rate": cons_rate,
        "rollback_rate": evolver.stats.rollback_rate,
        "rollback_detail": {"applied": evolver.stats.applied,
                            "rollbacks": evolver.stats.rollbacks},
        "engines": dict(evolver.stats.engines),
        "final": {"concepts": len(onto.concepts),
                  "active": sum(1 for c in onto.concepts.values() if c.active),
                  "axioms": sum(len(c.axioms) for c in onto.concepts.values())},
        "embed": {"sentences": sum(len(b) for b in stream),
                  "api_calls": emb.n_api_calls, "seconds": round(t_embed, 1)},
        "term_shift": shift,
        "total_seconds": round(time.time() - t_all, 1),
    }
    (out / "exp_pmc.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== 轨道 3 汇总 ====")
    print(f"  告警 {dict(total)}  一致率 {cons_rate:.3f}  "
          f"回退率 {summary['rollback_rate']}")
    print(f"  概念 {summary['final']['concepts']}"
          f"（活跃 {summary['final']['active']}）  "
          f"公理 {summary['final']['axioms']}  "
          f"耗时 {summary['total_seconds']}s")
    print(f"  术语分布 Jaccard（相邻片）："
          f"{[r['jaccard_prev'] for r in shift]}")
    print(f"-> {out/'exp_pmc.csv'}\n-> {out/'exp_pmc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
