"""命题 C 实验：三类知识问答准确率（动态演化 vs 冻结本体）。

要证明的三条断言：
    H1 稳定知识：动态不比冻结差（演化没改坏已有知识）
    H2 新知识：  动态显著优于冻结（冻结本体没有这个知识）
    H3 重定义：  动态显著优于冻结（冻结本体保留旧语义 → 语义错位）

流程：
    1. 受控数据流上跑演化，每批后保存本体快照（供衰减曲线）
    2. 依据演化历史自动标注三类知识，按公理派生题目（金标 = O_T 语义，
       重定义类额外记录 O_0 的旧语义 stale）
    3. 用 {最终本体 O_T, 初始本体 O_0, 各批快照 O_t} 分别作答
    4. 输出三类准确率（含 bootstrap 95% CI 与配对检验）与衰减曲线

用法：
    python experiments/run_exp_qa.py --quick
    python experiments/run_exp_qa.py --per-class 100 --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                    # noqa: E402
from src.build_ontology import (build_from_occurrences,        # noqa: E402
                                build_initial_ontology,
                                items_to_occurrences)
from src.controlled_stream import make_controlled_stream       # noqa: E402
from src.drift_detect import DriftDetector                     # noqa: E402
from src.embedding import Embedder                             # noqa: E402
from src.eval_drift import bootstrap_ci, wilcoxon_paired       # noqa: E402
from src.eval_qa import (KCLASSES, Answer, answer, build_questions,   # noqa: E402
                         build_surfaces, judge_prompt,
                         label_knowledge_classes, lookup_value,
                         parse_judge, score_one)
from src.evolve import OntologyEvolver                         # noqa: E402
from src.extract import extract_candidates                     # noqa: E402

SYSTEMS = ("dynamic", "frozen")


# ---------------------------------------------------------------------------
# 跑一次演化，返回最终本体 / 初始本体 / 各批快照 / 事件批次
# ---------------------------------------------------------------------------
def evolve(seed: int, params: C.DriftParams, emb: Embedder, *,
           check: bool = True, engine: str = "hermit",
           keep_extinct: bool = True, period: int = 0
           ) -> Tuple[object, object, List[object], Dict[str, int], Dict,
                      Optional[object]]:
    """返回 (O_T, O_0, 每批后的快照列表, cid→事件批次, 统计, O_periodic)。

    `keep_extinct=True`（默认）时不执行淘汰：命题 C 评估的是**三类知识是否被
    维持**，而被主动淘汰的知识已经不在本体里、无从回答；若执行淘汰，重定义类
    的概念会被顺带移除，导致该类无题可出。

    `period > 0` 时额外构造**周期全量重构臂**（第三臂）：累积截止当前批的全部
    occurrence，在 `t % P == 0` 及末批用 `build_from_occurrences` 从零重建，
    返回最后一次重建的本体 `O_periodic`；`period = 0` 时该臂为 None。
    """
    stream = make_controlled_stream(emb, seed=seed, params=params)
    onto, info = build_initial_ontology(
        stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
        vocab={**stream.vocab_base, **stream.vocab_emer})
    onto_init = copy.deepcopy(onto)          # 冻结本体：永不更新

    # 第三臂准备：累积 occurrence，词表与 O_0 对齐（与 run_exp_cost 口径一致）
    occ_p: List = list(items_to_occurrences(stream.pool, stream.d0_iids))
    onto_periodic: Optional[object] = None
    n_rebuild = 0
    ms_rebuild = 0.0
    if period > 0:
        vocab_all = {**stream.vocab_base, **stream.vocab_emer}
        k_gt_p = len(stream.vocab_base)
        op_vocab_p = list(getattr(onto, "op_vocab", []) or [])
        dp_vocab_p = list(getattr(onto, "dp_vocab", []) or [])

    p = params.copy(link_sim=max(0.50, float(info.get("link_sim_p05",
                                                       params.link_sim)) - 0.02))
    det = DriftDetector(onto, p)
    evo = OntologyEvolver(onto, check=check, engine=engine, use_cache=False)

    snapshots: List[object] = []
    event_t: Dict[str, int] = {}
    new_cid = len(onto.concepts)
    n_ops = {"emergence": 0, "redefinition": 0, "extinction": 0}

    for t, batch in enumerate(stream.batches, start=1):
        # 周期全量重构：到点即用截止当前批的全部数据从零重建（末批必建）
        occ_p.extend(batch)
        if period > 0 and (t % period == 0 or t == len(stream.batches)):
            _t0 = time.time()
            onto_periodic, _ = build_from_occurrences(
                occ_p, seed=0, k_gt=k_gt_p, vocab=vocab_all,
                op_vocab=op_vocab_p, dp_vocab=dp_vocab_p)
            ms_rebuild += (time.time() - _t0) * 1000.0
            n_rebuild += 1
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
                    if cen is not None
                    else np.mean([o.vec for o in items], axis=0),
                    axioms=axs)
                if ok and cid in onto.concepts:
                    det.centroids[cid] = np.asarray(
                        onto.concepts[cid].centroid, dtype=np.float32)
                    det.counts[cid] = []
                    det.weights[cid] = []
                    event_t.setdefault(cid, t)
                    n_ops["emergence"] += 1
            elif a.dtype == "redefinition":
                axs = det.last_projected.get(a.cid) or []
                if not axs:
                    continue
                if evo.redefine(a.cid, axs):
                    event_t.setdefault(a.cid, t)
                    n_ops["redefinition"] += 1
            elif a.dtype == "extinction" and not keep_extinct:
                if evo.retire(a.cid):
                    event_t.setdefault(a.cid, t)
                    n_ops["extinction"] += 1
        snapshots.append(copy.deepcopy(onto))

    stats = {"n_ops": n_ops,
             "rollback_rate": evo.stats.rollback_rate,
             "n_concepts_final": sum(1 for c in onto.concepts.values() if c.active),
             "link_sim": p.link_sim,
             "history": list(evo.history),
             "periodic": {
                 "period": period,
                 "n_rebuild": n_rebuild,
                 "ms_rebuild": round(ms_rebuild, 1),
                 "n_concepts_final": (None if onto_periodic is None else
                                      sum(1 for c in onto_periodic.concepts.values()
                                          if c.active))}}
    return onto, onto_init, snapshots, event_t, stats, onto_periodic


# ---------------------------------------------------------------------------
# 单个 seed：出题 + 各系统作答
# ---------------------------------------------------------------------------
def run_case(seed: int, params: C.DriftParams, emb: Embedder, *,
             per_class: int = 100, max_per_concept: int = 10,
             versions: Sequence[str] = ("a", "b"),
             link_sim: float = 0.5, keep_extinct: bool = True,
             period: int = 0, judge: Optional[object] = None) -> Dict:
    onto, onto_init, snaps, event_t, stats, onto_periodic = evolve(
        seed, params, emb, keep_extinct=keep_extinct, period=period)
    labels = label_knowledge_classes(onto, stats["history"])
    questions = build_questions(onto, onto_init, labels, per_class=per_class,
                                max_per_concept=max_per_concept, seed=seed,
                                t_event_by_cid=event_t)

    # 各系统的检索面：最终本体 / 冻结本体 / 周期重构本体 / 每批快照
    surf_dyn = build_surfaces(onto, emb)
    surf_frz = build_surfaces(onto_init, emb)
    surf_per = build_surfaces(onto_periodic, emb) if onto_periodic is not None \
        else None
    surf_snaps = [build_surfaces(o, emb) for o in snaps]

    sys_pairs = [("dynamic", (onto, surf_dyn)),
                 ("frozen", (onto_init, surf_frz))]
    if onto_periodic is not None and surf_per is not None:
        sys_pairs.append(("periodic", (onto_periodic, surf_per)))

    rows: List[Dict] = []
    curve: List[Dict] = []
    # 检索阈值：题干→概念表层，与“实例→概念”的 link_sim 标定口径不同，
    # 单独给一个较低值，避免两系统全部弃答、丢失区分度与“语义错位”证据
    link_sim = float(link_sim)

    for ver in versions:
        qv = emb.embed([q.text(ver) for q in questions])
        vecs = {q.qid: np.asarray(v, dtype=np.float32)
                for q, v in zip(questions, qv)}

        def _ans(onto_x, surf, q):
            """用预先算好的题干向量作答（避免重复嵌入）。"""
            cids, sv = surf
            if not cids:
                return answer(onto_x, q, emb, surf, link_sim=link_sim,
                              version=ver)
            sims = sv @ vecs[q.qid]
            j = int(np.argmax(sims))
            if float(sims[j]) < link_sim:
                return Answer(None, None, float(sims[j]), True)
            cid = cids[j]
            val = lookup_value(onto_x, cid, q.kind, q.pred)
            return Answer(cid, val, float(sims[j]), val is None)

        for q in questions:
            for sysname, (onto_x, surf) in sys_pairs:
                a = _ans(onto_x, surf, q)
                sc = score_one(a, q)
                row = {"seed": seed, "qid": q.qid, "kclass": q.kclass,
                       "system": sysname, "version": ver, "cid": q.cid,
                       "kind": q.kind, "gold": q.gold, "stale": q.stale or "",
                       "pred_value": a.value or "", "sim": round(a.sim, 4),
                       **sc}
                if judge is not None:
                    jr = parse_judge(judge.ask(judge_prompt(q, a, ver)))
                    row["judge"] = -1 if jr is None else jr
                rows.append(row)

        # 衰减曲线：用 O_t 作答（题干向量同上，复用）
        for t, (onto_t, surf) in enumerate(zip(snaps, surf_snaps), start=1):
            accs: Dict[str, List[int]] = {k: [] for k in KCLASSES}
            for q in questions:
                a = _ans(onto_t, surf, q)
                accs[q.kclass].append(score_one(a, q)["correct"])
            for k, v in accs.items():
                if v:
                    curve.append({"seed": seed, "t": t, "system": "dynamic",
                                  "kclass": k, "version": ver,
                                  "accuracy": round(float(np.mean(v)), 4)})
            # 冻结本体不随时间变化，曲线是常数（也写出来便于作图对比）
            frz_accs: Dict[str, List[int]] = {k: [] for k in KCLASSES}
            for q in questions:
                a = _ans(onto_init, surf_frz, q)
                frz_accs[q.kclass].append(score_one(a, q)["correct"])
            for k, v in frz_accs.items():
                if v:
                    curve.append({"seed": seed, "t": t, "system": "frozen",
                                  "kclass": k, "version": ver,
                                  "accuracy": round(float(np.mean(v)), 4)})

    return {"seed": seed, "questions": [q.__dict__ for q in questions],
            "rows": rows, "curve": curve, "stats": {k: v for k, v in
                                                    stats.items()
                                                    if k != "history"},
            "label_counts": {k: sum(1 for v in labels.values() if v == k)
                             for k in KCLASSES}}


# ---------------------------------------------------------------------------
# 汇总：跨 seed 的准确率 CI + 动态 vs 冻结配对检验
# ---------------------------------------------------------------------------
def summarize(rows: List[Dict], *, n_boot: int = 1000) -> List[Dict]:
    """按 (kclass, system, version) 聚合；每个 seed 一个准确率样本点。"""
    out: List[Dict] = []
    groups: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in rows:
        groups.setdefault((r["kclass"], r["system"], r["version"]), []).append(r)
    for (kc, sysname, ver), rs in sorted(groups.items()):
        by_seed: Dict[int, List[Dict]] = {}
        for r in rs:
            by_seed.setdefault(r["seed"], []).append(r)
        accs = [float(np.mean([x["correct"] for x in v]))
                for v in by_seed.values()]
        stale = [float(np.mean([x["stale_hit"] for x in v]))
                 for v in by_seed.values()]
        abst = [float(np.mean([x["abstain"] for x in v]))
                for v in by_seed.values()]
        ci = bootstrap_ci(accs, n_boot=n_boot, seed=0) if len(accs) > 1 else None
        out.append({"kclass": kc, "system": sysname, "version": ver,
                    "n_runs": len(accs), "n_questions": len(rs) // max(1, len(accs)),
                    "accuracy": round(float(np.mean(accs)), 4),
                    "accuracy_ci95": None if ci is None else
                    [round(ci["lo"], 4), round(ci["hi"], 4)],
                    "stale_rate": round(float(np.mean(stale)), 4),
                    "abstain_rate": round(float(np.mean(abst)), 4)})
    return out


def paired_tests(rows: List[Dict]) -> List[Dict]:
    """每个 (kclass, version) 上，动态 vs 其余各臂（冻结 / 周期重构）的配对检验。"""
    out: List[Dict] = []
    others = sorted({r["system"] for r in rows} - {"dynamic"})
    for ver in sorted({r["version"] for r in rows}):
        for kc in KCLASSES:
            sub = [r for r in rows if r["kclass"] == kc and r["version"] == ver]
            dyn = {(r["seed"], r["qid"]): r["correct"]
                   for r in sub if r["system"] == "dynamic"}
            if not dyn:
                continue
            for sysname in others:
                oth = {(r["seed"], r["qid"]): r["correct"]
                       for r in sub if r["system"] == sysname}
                keys = sorted(set(dyn) & set(oth))
                if not keys:
                    continue
                a = [dyn[k] for k in keys]
                b = [oth[k] for k in keys]
                try:
                    res = wilcoxon_paired(a, b)
                    p = res.get("p")
                except Exception:
                    p = None
                out.append({"kclass": kc, "version": ver, "system": sysname,
                            "n_pairs": len(keys),
                            "acc_dynamic": round(float(np.mean(a)), 4),
                            "acc_other": round(float(np.mean(b)), 4),
                            "diff": round(float(np.mean(a)) - float(np.mean(b)), 4),
                            "p": (None if p is None else round(float(p), 6))})
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--max-per-concept", type=int, default=10)
    ap.add_argument("--versions", default="a,b")
    ap.add_argument("--link-sim", type=float, default=0.5,
                    help="题干→概念表层的检索阈值（低于此值回答“不知道”）")
    ap.add_argument("--allow-extinct", action="store_true",
                    help="执行淘汰（默认保留，否则重定义类概念会被移除）")
    ap.add_argument("--judge", action="store_true",
                    help="额外用 LLM-Judge 打 0-3 分（需 CHAT_API_URL）")
    ap.add_argument("--period", type=int, default=0,
                    help="周期全量重构臂的周期 P（0=不做该臂；末批必重建）")
    ap.add_argument("--tag", default="exp_qa",
                    help="结果输出目录名（results/<tag>），默认 exp_qa")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    versions = tuple(v.strip() for v in args.versions.split(",") if v.strip())

    # 重定义事件多注入一些：每次重定义只改变少数公理，重定义类题量才够
    base = C.PARAMS.copy(n_emergence=8, n_redefinition=20, n_extinction=3)
    if args.quick:
        base = base.copy(n_batch=12, n_init_per_leaf=6, pool_per_leaf=12,
                         base_rate=3.0, n_emergence=4, n_redefinition=10,
                         n_extinction=2)

    emb = Embedder(allow_fallback=False, )
    judge = None
    if args.judge:
        from src.llm import AxiomJudge
        judge = AxiomJudge()

    res = C.DIR_RESULTS / args.tag
    res.mkdir(parents=True, exist_ok=True)

    print("== 命题 C：三类知识问答准确率（动态 vs 冻结）==")
    all_rows: List[Dict] = []
    all_curve: List[Dict] = []
    all_q: List[Dict] = []
    t0 = time.time()
    for seed in seeds:
        out = run_case(seed, base, emb, per_class=args.per_class,
                       max_per_concept=args.max_per_concept,
                       versions=versions, link_sim=args.link_sim,
                       keep_extinct=not args.allow_extinct,
                       period=args.period, judge=judge)
        all_rows.extend(out["rows"])
        all_curve.extend(out["curve"])
        all_q.extend(out["questions"])
        st = out["stats"]
        print(f"seed={seed}  概念标注 {out['label_counts']}  "
              f"演化 {st['n_ops']}  回退率 {st['rollback_rate']:.3f}  "
              f"题目 {len(out['questions'])}  "
              f"({time.time() - t0:.1f}s)")

    summ = summarize(all_rows)
    tests = paired_tests(all_rows)

    print(f"\n{'类别':<8}{'系统':<9}{'版':<4}{'准确率[95%CI]':<24}"
          f"{'错位率':<8}{'弃答率':<8}")
    for s in summ:
        ci = s["accuracy_ci95"]
        acc = f"{s['accuracy']:.3f}" if ci is None else \
            f"{s['accuracy']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        print(f"{s['kclass']:<8}{s['system']:<9}{s['version']:<4}"
              f"{acc:<24}{s['stale_rate']:<8.3f}{s['abstain_rate']:<8.3f}")

    print("\n配对检验（动态 - 其余各臂）：")
    for t in tests:
        if t["p"] is None and abs(t["diff"]) < 1e-9:
            p_s = "两系统逐题完全一致（无需检验）"
        elif t["p"] is None:
            p_s = "样本不足/无法计算"
        else:
            p_s = f"p={t['p']}"
        print(f"  {t['kclass']:<10} 版{t['version']}  "
              f"动态 {t['acc_dynamic']:.3f} vs {t['system']:<8} "
              f"{t['acc_other']:.3f}  "
              f"差 {t['diff']:+.3f}  {p_s}  (n={t['n_pairs']})")

    if all_rows:
        sv = np.array([r["sim"] for r in all_rows])
        print(f"\n检索相似度：min {sv.min():.3f}  p50 {np.median(sv):.3f}  "
              f"max {sv.max():.3f}（弃答阈值 {args.link_sim}）")

    (res / "questions.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in all_q),
        encoding="utf-8")
    (res / "answers.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows),
        encoding="utf-8")
    if all_rows:
        with (res / "scores.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    if all_curve:
        with (res / "curve.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_curve[0].keys()))
            w.writeheader()
            w.writerows(all_curve)
    (res / "summary.json").write_text(
        json.dumps({"seeds": seeds, "versions": list(versions),
                    "per_class": args.per_class, "summary": summ,
                    "tests": tests, "quick": args.quick},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
