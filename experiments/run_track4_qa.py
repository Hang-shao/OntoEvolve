"""experiments/run_track4_qa.py — 阶段 D7：反诈轨道的应用层问答（三组对照）。

设计要点（论文需写明）：
  * 题集**外置人工编写**（data/track4/questions/qa_set.jsonl），不使用
    src.eval_qa.build_questions() 的 O_T 差分生成，避免"按演化结果出题"的泄漏；
  * 三种本体方案接入同一检索骨架：题干嵌入 → 与概念表层描述取余弦 → 最近概念 →
    按题型返回（类型名 / 该概念支撑度最高的步骤 / 是否已弃用）；相似度低于
    link_sim 判为拒答；
  * 对照：冻结 O_0、增量演化 O_T（run_track4 落盘）、周期全量重构（末 P 批重建）。

用法：
    python experiments/run_track4_qa.py --seed 0 [--periodic-p 4]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import config as C                                    # noqa: E402
from src.build_ontology import build_from_occurrences           # noqa: E402
from src.fraud_ontology import build_skeleton_ontology
from src.fraud_stream import read_materialized, to_occurrences   # noqa: E402
from src.fraud_vocab import DP_VOCAB, OP_VOCAB, PARENT_EVENT    # noqa: E402
from src.onto_io import dump_onto, load_onto                     # noqa: E402

RES = ROOT / "results" / "track4"
QD = ROOT / "data" / "track4" / "questions" / "qa_set.jsonl"
RETIRED = "已弃用"
STILL_OK = "仍适用"


def make_embedder(kind: str = "local", verbose: bool = False):
    if kind == "api":
        from src.embedding import Embedder
        return Embedder(cache_dir=C.DIR_CACHE_API, allow_fallback=False,
                        verbose=verbose)
    from src.local_embed import LocalEmbedder
    return LocalEmbedder(cache_dir=C.DIR_CACHE_LOCAL, verbose=verbose)


def surface(c) -> str:
    """概念表层描述（检索文本），与 eval_qa.concept_surface 同构。"""
    bits = [c.name]
    if c.parent:
        bits.append(f"是{c.parent}的子类")
    for ax in c.axioms[:8]:
        if ax.kind == "OP":
            bits.append(f"包含步骤{ax.pred}")
        elif ax.kind == "DP":
            bits.append(f"{ax.pred}为{ax.obj}")
    return "；".join(bits)


def build_surfaces(onto, emb) -> Tuple[List[str], np.ndarray]:
    # 已弃用概念也保留在检索面上：淘汰型题目正是要检索到"已弃用"的概念
    cids = list(onto.concepts.keys())
    texts = [surface(onto.concepts[cid]) for cid in cids]
    return cids, np.asarray(emb.embed(texts), dtype=np.float32)


def top_step(c) -> Optional[str]:
    """该概念支撑度最高的步骤（OP 公理按 evidence_count 排序）。"""
    ops = [(a.evidence_count, a.confidence, a.pred) for a in c.axioms
           if a.kind == "OP" and a.pred]
    if not ops:
        return None
    ops.sort(key=lambda x: (-x[0], -x[1]))
    return ops[0][2]


def type_of_name(name: str) -> str:
    return name.replace("新增·", "").split("·")[0]


def answer_one(onto, q: Dict, emb, surfaces, *, link_sim: float = 0.5,
               version: str = "a") -> Dict:
    cids, vecs = surfaces
    qv = np.asarray(emb.embed([q["q_" + version]])[0], dtype=np.float32)
    sims = vecs @ qv
    j = int(np.argmax(sims))
    sim = float(sims[j])
    cid = cids[j]
    c = onto.get(cid)
    out = {"qid": q["qid"], "cid": cid, "sim": round(sim, 4),
           "concept": c.name if c else None}
    if sim < link_sim or c is None:
        return {**out, "answer": None, "abstain": True}
    if not c.active:
        return {**out, "answer": RETIRED, "abstain": False}
    ask = q["ask"]
    if ask == "type":
        return {**out, "answer": type_of_name(c.name), "abstain": False}
    if ask == "step":
        s = top_step(c)
        return {**out, "answer": s, "abstain": s is None}
    if ask == "usability":
        return {**out, "answer": STILL_OK, "abstain": False}
    raise ValueError(ask)


def score(q: Dict, ans: Dict) -> Dict:
    a = ans["answer"]
    if ans.get("abstain"):
        return {"correct": 0, "abstain": 1, "stale": 0, "wrong": 0}
    if a == q["gold"]:
        return {"correct": 1, "abstain": 0, "stale": 0, "wrong": 0}
    if q.get("stale") and a == q["stale"]:
        return {"correct": 0, "abstain": 0, "stale": 1, "wrong": 0}
    return {"correct": 0, "abstain": 0, "stale": 0, "wrong": 1}


def mcnemar(b: int, c: int) -> float:
    """精确 McNemar（双侧）。b/c 为两种方案在配对题上"仅一方正确"的计数。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_questions(path: Path = QD) -> List[Dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", choices=("local", "api"), default="local")
    ap.add_argument("--periodic-p", type=int, default=4)
    ap.add_argument("--link-sim", type=float, default=0.5)
    ap.add_argument("--min-members", type=int, default=10)
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    batches, truth, manifest = read_materialized()
    qs = load_questions()
    print(f"[questions] {len(qs)} 题，类型分布 "
          f"{dict(Counter(q['kclass'] for q in qs))}")

    emb = make_embedder(args.embedder, verbose=True)
    streams = []
    for items in batches:
        streams.append(to_occurrences(items, emb))

    # ---- 三个本体 ----
    # 三组方案必须用与主实验相同的 O₀ 构造（类型级骨架），否则"冻结 vs 增量"
    # 的差异会混入构建方式的差异（聚类 K=30 与骨架 3 概念的基线不可比）。
    onto_frozen, info0 = build_skeleton_ontology(
        streams[0], min_members=args.min_members,
        op_vocab=OP_VOCAB, dp_vocab=DP_VOCAB, granularity="type")
    P = args.periodic_p
    onto_periodic, infoP = build_skeleton_ontology(
        [o for b in streams[-P:] for o in b], min_members=args.min_members,
        op_vocab=OP_VOCAB, dp_vocab=DP_VOCAB, granularity="type")
    dyn_path = RES / f"onto_s{args.seed}.json"
    if dyn_path.exists():
        onto_dyn = load_onto(dyn_path)
    else:
        print(f"[err] 缺少 {dyn_path}，请先运行 experiments/run_track4.py")
        return 2
    dump_onto(onto_periodic, RES / f"onto_periodic_s{args.seed}.json")
    dump_onto(onto_frozen, RES / f"onto_frozen_s{args.seed}.json")
    print(f"[onto] frozen={len(onto_frozen.concepts)} 概念；"
          f"dynamic={len(onto_dyn.concepts)}（活跃 "
          f"{sum(1 for c in onto_dyn.concepts.values() if c.active)}）；"
          f"periodic(P={P})={len(onto_periodic.concepts)} 概念")

    schemes = {"frozen": onto_frozen, "dynamic": onto_dyn,
               "periodic": onto_periodic}
    rows: List[Dict] = []
    for name, onto in schemes.items():
        cids, vecs = build_surfaces(onto, emb)
        for q in qs:
            for ver in ("a", "b"):
                ans = answer_one(onto, q, emb, (cids, vecs),
                                 link_sim=args.link_sim, version=ver)
                sc = score(q, ans)
                rows.append({"scheme": name, "qid": q["qid"],
                             "kclass": q["kclass"], "target": q["target"],
                             "ask": q["ask"], "version": ver,
                             "gold": q["gold"], "stale": q.get("stale"),
                             "answer": ans["answer"], "cid": ans["cid"],
                             "concept": ans.get("concept"),
                             "sim": ans["sim"], **sc})

    # ---- 汇总 ----
    summ: Dict[str, Dict] = {}
    for sch in schemes:
        summ[sch] = {}
        for kc in ("stable", "new", "redefined", "extinct"):
            sub = [r for r in rows if r["scheme"] == sch and r["kclass"] == kc]
            if not sub:
                continue
            n = len(sub)
            summ[sch][kc] = {
                "n": n,
                "acc": round(sum(r["correct"] for r in sub) / n, 4),
                "abstain": round(sum(r["abstain"] for r in sub) / n, 4),
                "stale": round(sum(r["stale"] for r in sub) / n, 4),
                "wrong": round(sum(r["wrong"] for r in sub) / n, 4),
            }

    # ---- 配对检验（dyn vs frozen / dyn vs periodic，题对级）----
    def paired(ref: str) -> Dict:
        key = lambda r: (r["qid"], r["version"])
        a = {key(r): r["correct"] for r in rows if r["scheme"] == "dynamic"}
        b = {key(r): r["correct"] for r in rows if r["scheme"] == ref}
        out = {}
        for kc in ("stable", "new", "redefined", "extinct"):
            ks = [k for k in a if k in b and next(
                r["kclass"] for r in rows
                if r["scheme"] == "dynamic" and (r["qid"], r["version"]) == k) == kc]
            if not ks:
                continue
            b01 = sum(1 for k in ks if a[k] == 1 and b[k] == 0)
            c10 = sum(1 for k in ks if a[k] == 0 and b[k] == 1)
            diff = (b01 - c10) / len(ks)
            out[kc] = {"n_pairs": len(ks), "dyn_only_correct": b01,
                       "ref_only_correct": c10, "mean_diff": round(diff, 4),
                       "mcnemar_p": round(mcnemar(b01, c10), 6)}
        return out

    tests = {"vs_frozen": paired("frozen"), "vs_periodic": paired("periodic")}

    out = {
        "seed": args.seed, "embedder": args.embedder,
        "link_sim": args.link_sim, "periodic_p": P,
        "n_questions": len(qs),
        "n_items_per_scheme": {s: sum(1 for r in rows if r["scheme"] == s)
                               for s in schemes},
        "by_kclass": summ, "tests": tests,
        "onto_sizes": {s: {"concepts": len(o.concepts),
                           "active": sum(1 for c in o.concepts.values() if c.active),
                           "axioms": sum(len(c.axioms) for c in o.concepts.values())}
                       for s, o in schemes.items()},
        "note": "题集人工编写（不取自候选公理）；答案由同一检索骨架给出；"
                "重定义为案例级（公检法 R2），不作精度级主张",
    }
    (RES / f"qa_summary_s{args.seed}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RES / f"qa_items_s{args.seed}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n==== 应用层问答（seed %d）====" % args.seed)
    for kc in ("stable", "new", "redefined", "extinct"):
        line = f"  {kc:<10}"
        for sch in ("frozen", "dynamic", "periodic"):
            v = summ[sch].get(kc)
            line += f" | {sch}: acc={v['acc']:.3f} 拒答={v['abstain']:.3f}" if v else ""
        print(line)
    for ref, t in tests.items():
        print(f"  dynamic {ref}: " + " ; ".join(
            f"{kc} Δ={v['mean_diff']:+.3f} p={v['mcnemar_p']:.3g}"
            for kc, v in t.items()))
    print(f"-> {RES / f'qa_summary_s{args.seed}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
