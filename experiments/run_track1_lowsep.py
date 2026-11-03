"""轨道 1 可分性消融（B）：只压缩类内相似度，不改任何判据。

目的：把"检测精度依赖域内可分性"从反诈轨道的单个观察点，变成一条受控曲线。

做法：
    在轨道 1 的实例嵌入上施加**各向同性扰动**（σ 递增），使 D₀ 成员相似度的
    5 分位 p05 单调下降，从而 1−link_sim 从 0.16 被推到 0.46（跨过 θ_sem=0.30）。
    文本、公理槽位、真值事件、聚类流程、检测阈值**一律不动**——被改动的只有
    嵌入的几何尺度。

    另设"长尾"模式：只扰动 ρ 比例的实例（σ 固定），模拟"少数实例风格迥异"
    （更接近反诈轨道"均值高、5 分位低"的形态）。

判据口径与 run_exp1.py 完全一致：阈值取自 results/exp1_summary.json 的标定结果，
link_sim = max(0.40, p05 − link_offset)，θ_sem 等不重新标定（避免对真值过拟合）。

Run:
    python experiments/run_track1_lowsep.py [--repeats 10] [--seed 42]
    python experiments/run_track1_lowsep.py --mode tail
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src import config as C
from src.build_ontology import build_initial_ontology
from src.controlled_stream import (PoolItem, build_pool, generate_stream,
                                   select_events)
from src.drift_detect import DriftDetector
from src.embedding import Embedder
from src.eval_drift import TYPES, bootstrap_ci, match_events
from src.vocab import BASE_TREE, EMERGENCE_TREE, build_vocab

try:
    from sklearn.metrics import normalized_mutual_info_score as _nmi
except Exception:                                   # pragma: no cover
    _nmi = None

GRID_UNIFORM = (0.0, 0.3, 0.5, 0.7, 0.9, 1.2)
GRID_TAIL = (0.0, 0.1, 0.2, 0.4)     # 被扰动的实例比例（σ 固定为 TAIL_SIGMA）
TAIL_SIGMA = 1.2


class ProbeDetector(DriftDetector):
    """记录本批未链接实例的最近概念距离，用于"判据是否恒成立"的诊断。

    只读取检测器自己的链接口径（与 observe() 完全一致），不改变任何判定。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unlinked: List[Dict] = []

    def observe(self, t: int, items: Sequence) -> List:
        alarms = super().observe(t, items)
        for o in items:
            v = np.asarray(o.vec, dtype=np.float32)
            cid, bs = None, -2.0
            for c, cen in self.centroids.items():
                s = float(np.dot(v, cen))
                if s > bs:
                    bs, cid = s, c
            need = (self.thr.get(cid, self.link_sim) if self.adaptive_link
                    else self.link_sim)
            if bs < need:
                self.unlinked.append({"t": t, "leaf": o.leaf, "cid": cid,
                                      "d_near": 1.0 - bs})
        return alarms


def perturb(pool: Sequence[PoolItem], sigma: float, *, seed: int = 0,
            frac: float = 1.0) -> List[PoolItem]:
    """各向同性扰动：v' = normalize(v + σ·u)，u ~ N(0, I/d)。

    当 frac < 1 时只扰动按 iid 稳定抽取的一部分实例（长尾模式）。
    文本与槽位原样保留，因此公理与真值不受影响。
    """
    if sigma <= 0:
        return [PoolItem(iid=p.iid, leaf=p.leaf, sup=p.sup, variant=p.variant,
                         text=p.text, op_slots=list(p.op_slots),
                         dp_slots=list(p.dp_slots), vec=p.vec) for p in pool]
    rng = np.random.default_rng(seed + 777)
    picked = rng.random(len(pool)) < frac
    out = []
    for p, hit in zip(pool, picked):
        v = np.asarray(p.vec, dtype=np.float32)
        if hit:
            u = rng.normal(size=v.shape).astype(np.float32) / np.sqrt(v.size)
            v = v + sigma * u
            v = v / max(float(np.linalg.norm(v)), 1e-12)
        out.append(PoolItem(iid=p.iid, leaf=p.leaf, sup=p.sup,
                            variant=p.variant, text=p.text,
                            op_slots=list(p.op_slots),
                            dp_slots=list(p.dp_slots), vec=v.astype(np.float32)))
    return out


def load_calibrated(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    cal = d["calibration"]
    return C.DriftParams(**cal["params"]), float(cal.get("link_offset", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=("uniform", "tail"), default="uniform")
    ap.add_argument("--outdir", default=str(C.DIR_RESULTS / "track1_lowsep"))
    args = ap.parse_args()

    params, link_offset = load_calibrated(C.DIR_RESULTS / "exp1_summary.json")
    # 数据流必须用 C.PARAMS 生成、用标定参数检测——这是 run_exp1.py 的口径：
    # kappa / window_w 影响真值声明的批次（数据侧），检测侧才用标定值。
    ps = C.PARAMS
    grid = GRID_UNIFORM if args.mode == "uniform" else GRID_TAIL
    labels = ([f"sigma={s}" for s in grid] if args.mode == "uniform"
              else [f"frac={f}" for f in grid])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    emb = Embedder(allow_fallback=False, verbose=False)
    vb = build_vocab(seed=0, tree=BASE_TREE)
    ve = build_vocab(seed=1000, tree=EMERGENCE_TREE)
    pool, alt_pool = build_pool(emb, vb, ve, seed=0, params=ps,
                                alt_leaves=list(vb.keys()))
    print(f"[pool] texts={len(pool)} api_calls={emb.n_api_calls} "
          f"θ_sem={params.theta_sem} link_offset={link_offset}")

    rows: List[Dict] = []
    diag: Dict[str, Dict] = {}
    summary: Dict[str, Dict] = {}
    t_all = time.time()

    for cond, val in zip(labels, grid):
        sigma = TAIL_SIGMA if args.mode == "tail" else float(val)
        frac = 1.0 if args.mode == "uniform" else float(val)
        t0 = time.time()
        pool_c = perturb(pool, sigma, seed=args.seed, frac=frac)

        # O_0 与 repeats 无关（D_0 由词表种子决定），每个条件只建一次
        s0 = generate_stream(pool_c, vb, ve,
                             select_events(vb, ve, seed=args.seed, params=ps),
                             seed=args.seed, params=ps, alt_pool=alt_pool)
        onto, info = build_initial_ontology(s0.pool, s0.d0_iids, seed=0,
                                            k_gt=len(vb), vocab={**vb, **ve})
        p05 = float(info["link_sim_p05"])
        p50 = float(info["link_sim_p50"])
        p = params.copy(link_sim=max(0.40, p05 - link_offset))
        # 聚类结构是否仍完好（用于证明数据流的语义结构未被扰动破坏）
        d0 = [x for x in s0.pool if x.iid in set(s0.d0_iids)]
        lab = np.full(len(d0), -1, dtype=int)      # D_0 顺序 = build 时的 occ 顺序
        for cid, conc in onto.concepts.items():
            for pos in conc.meta.get("members", []):
                lab[int(pos)] = int(cid[1:])
        nmi = (float(_nmi([x.leaf for x in d0], lab))
               if _nmi is not None else None)
        print(f"[{cond:<10}] p05={p05:.3f} p50={p50:.3f} "
              f"link_sim={p.link_sim:.3f} 1-link={1-p.link_sim:.3f} "
              f"O0 concepts={len(onto.concepts)} nmi={nmi}")

        f1s: List[float] = []
        linked: List[Dict] = []
        unlinked_frac: List[float] = []
        for r in range(args.repeats):
            seed_r = args.seed + r
            metas = select_events(vb, ve, seed=seed_r, params=ps)
            stream = generate_stream(pool_c, vb, ve, metas, seed=seed_r,
                                     params=ps, alt_pool=alt_pool)
            det = ProbeDetector(onto, p)
            alarms = []
            n_items = 0
            for t, items in enumerate(stream.batches, start=1):
                alarms.extend(det.observe(t, items))
                n_items += len(items)
            res = match_events(stream.events, alarms, info["leaf_to_cid"],
                               delay_tol=params.delay_tol, typed=True)
            f1s.append(res.macro_f1)
            unlinked_frac.append(len(det.unlinked) / max(1, n_items))
            emer_leaves = {e.leaf for e in stream.events
                           if e.dtype == "emergence"}
            for u in det.unlinked:
                linked.append({"t": u["t"], "leaf": u["leaf"],
                               "d_near": u["d_near"],
                               "novel": int(u["leaf"] in emer_leaves)})
            rows.append({
                "mode": args.mode, "cond": cond, "sigma": sigma, "frac": frac,
                "repeat": r, "p05": p05, "p50": p50,
                "link_sim": p.link_sim, "one_minus_link": 1 - p.link_sim,
                "macro_f1": res.macro_f1, "tp": res.tp, "fp": res.fp,
                "fn": res.fn, "n_alarms": res.n_alarms,
                "precision_upper": res.tp / max(1, res.n_alarms),
                "mean_delay": res.mean_delay,
                "unlinked_rate": len(det.unlinked) / max(1, n_items),
                **{f"f1_{t[:3]}": res.per_type[t]["f1"] for t in TYPES},
            })
        ci = bootstrap_ci(f1s, n_boot=1000, seed=args.seed)
        nov = [u["d_near"] for u in linked if u["novel"]]
        res_ = [u["d_near"] for u in linked if not u["novel"]]
        diag[cond] = {
            "n_unlinked": len(linked),
            "novel_d_median": float(np.median(nov)) if nov else None,
            "residual_d_median": float(np.median(res_) ) if res_ else None,
            "residual_share_above_theta": (float(np.mean(
                np.asarray(res_) > params.theta_sem)) if res_ else None),
            "novel_share_above_theta": (float(np.mean(
                np.asarray(nov) > params.theta_sem)) if nov else None),
            "novel_share_above_one_minus_link": (float(np.mean(
                np.asarray(nov) > 1 - p.link_sim)) if nov else None),
            "residual_share_above_one_minus_link": (float(np.mean(
                np.asarray(res_) > 1 - p.link_sim)) if res_ else None),
        }
        summary[cond] = {
            "sigma": sigma, "frac": frac, "p05": p05, "p50": p50,
            "link_sim": p.link_sim, "one_minus_link": 1 - p.link_sim,
            "macro_f1": ci["mean"], "macro_f1_ci95": [ci["lo"], ci["hi"]],
            "per_type_f1": {t: float(np.mean([r[f"f1_{t[:3]}"] for r in rows
                                              if r["cond"] == cond]))
                            for t in TYPES},
            "tp": float(np.mean([r["tp"] for r in rows if r["cond"] == cond])),
            "n_alarms": float(np.mean([r["n_alarms"] for r in rows
                                       if r["cond"] == cond])),
            "precision_upper": float(np.mean([r["precision_upper"] for r in rows
                                              if r["cond"] == cond])),
            "mean_delay": float(np.nanmean([r["mean_delay"] for r in rows
                                            if r["cond"] == cond])),
            "unlinked_rate": float(np.mean(unlinked_frac)),
            "o0_concepts": len(onto.concepts), "o0_nmi": nmi,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"    macroF1={ci['mean']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}] "
              f"tp={summary[cond]['tp']:.1f} alarms={summary[cond]['n_alarms']:.1f} "
              f"prec_ub={summary[cond]['precision_upper']:.3f} "
              f"unlinked={summary[cond]['unlinked_rate']:.3f} "
              f"({time.time()-t0:.0f}s)")

    tag = args.mode
    with (outdir / f"lowsep_rows_{tag}.csv").open("w", newline="",
                                                  encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    payload = {"mode": args.mode, "repeats": args.repeats, "seed": args.seed,
               "link_offset": link_offset,
               "theta_sem": params.theta_sem,
               "params": {k: v for k, v in vars(params).items()},
               "summary": summary, "diagnostics": diag}
    (outdir / f"lowsep_summary_{tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== 轨道 1 可分性消融 ====")
    print(f"{'cond':<12}{'1-link':>8}{'macroF1':>9}{'tp':>5}{'alarms':>7}"
          f"{'prec_ub':>9}{'unlinked':>10}")
    for cond in labels:
        s = summary[cond]
        print(f"{cond:<12}{s['one_minus_link']:>8.3f}{s['macro_f1']:>9.3f}"
              f"{s['tp']:>5.1f}{s['n_alarms']:>7.1f}"
              f"{s['precision_upper']:>9.3f}{s['unlinked_rate']:>10.3f}")
    print(f"\n-> {outdir}  ({time.time()-t_all:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
