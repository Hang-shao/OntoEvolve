"""实验 5：资源消耗与回退率。

两部分：
  A. 资源消耗：嵌入 API 调用数/缓存命中、各阶段耗时（构建 O_0、漂移检测、
     一致性检查）、内存峰值（含推理机 JVM 子进程）、端到端耗时。
  B. 回退率：同一批演化事件在不同一致性策略下的回退率与最终一致性
     - hermit   完整 OWL-DL 推理
     - z3       受限子集近似
     - structural 仅结构冲突
     - none     不做检查（对照：回退率为 0 但不一致版本会留下）
     并以对抗注入比例 alpha ∈ {0, 0.2, 0.5} 人为制造矛盾公理，观察回退率
     如何随矛盾密度上升，以及各策略的检出/漏检差异。

用法：
    python experiments/run_exp5.py                # 全量
    python experiments/run_exp5.py --quick        # 小规模
    python experiments/run_exp5.py --alphas 0,0.5 --engines hermit,z3
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C                                    # noqa: E402
from src.build_ontology import build_initial_ontology          # noqa: E402
from src.consistency import consistency_report                 # noqa: E402
from src.controlled_stream import make_controlled_stream       # noqa: E402
from src.drift_detect import DriftDetector                     # noqa: E402
from src.embedding import Embedder                             # noqa: E402
from src.evolve import OntologyEvolver                         # noqa: E402
from src.extract import extract_candidates                     # noqa: E402
from src.ontology_model import Axiom                           # noqa: E402
from src.owl_export import export_owl                          # noqa: E402


# ---------------------------------------------------------------------------
# 资源采样
# ---------------------------------------------------------------------------
class ResourceTracker:
    """采样当前进程及其子进程（推理机 JVM）的 RSS 峰值。"""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def __enter__(self):
        try:
            import psutil
        except Exception:                          # pragma: no cover
            return self
        self._proc = psutil.Process()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
        return False

    def _loop(self):
        import psutil
        while not self._stop.is_set():
            total = 0
            try:
                total += self._proc.memory_info().rss
                for ch in self._proc.children(recursive=True):
                    try:
                        total += ch.memory_info().rss
                    except Exception:
                        pass
            except Exception:
                pass
            self.peak_mb = max(self.peak_mb, total / (1024.0 * 1024.0))
            time.sleep(self.interval)


def _parent_of(onto, cid: str) -> Optional[str]:
    c = onto.concepts.get(cid)
    if c is None or not c.active:
        return None
    return c.parent if c.parent in onto.concepts else None


def _inject_conflict(onto, cid: str, mode: int, rng) -> List[Axiom]:
    """人为矛盾，返回一个用于 redefine 的公理列表（可能同时改动父概念）。

    mode 0  空基数区间          → 结构性检查即可发现
    mode 1  DOMAIN 与 DISJOINT  → 结构性检查即可发现
    mode 2  父子链上的不相交    → 只有 OWL-DL 推理能发现
    """
    others = [c for c in onto.concepts.values()
              if c.cid != cid and c.active]
    if mode == 0:
        return [Axiom("CARD", cid, pred="rel", card_min=9, card_max=1)]
    if mode == 1:
        tgt = others[int(rng.integers(0, len(others)))] if others else None
        if tgt is None:
            return [Axiom("CARD", cid, pred="rel", card_min=9, card_max=1)]
        return [Axiom("DOMAIN", cid, obj=tgt.cid),
                Axiom("DISJOINT", cid, obj=tgt.cid)]
    # mode 2 由调用方以“对父概念的一次独立 redefine”提交（保证可回滚），
    # 这里退化为 mode 0，避免直接改本体导致矛盾残留
    return [Axiom("CARD", cid, pred="rel", card_min=9, card_max=1)]


# ---------------------------------------------------------------------------
# 单次演化（可指定引擎与对抗注入比例）
# ---------------------------------------------------------------------------
def run_case(seed: int, engine: str, alpha: float, params: C.DriftParams,
             emb: Embedder, outdir: Optional[Path] = None,
             debug: bool = False) -> Dict:
    rng = np.random.default_rng(seed * 977 + int(alpha * 100))
    t_start = time.perf_counter()

    n_api0 = emb.n_api_calls
    t0 = time.perf_counter()
    stream = make_controlled_stream(emb, seed=seed, params=params)
    t_stream = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    onto, info = build_initial_ontology(
        stream.pool, stream.d0_iids, seed=0, k_gt=len(stream.vocab_base),
        vocab={**stream.vocab_base, **stream.vocab_emer})
    t_build = (time.perf_counter() - t0) * 1000.0

    p = params.copy(link_sim=max(0.50, float(info.get("link_sim_p05",
                                                       params.link_sim)) - 0.02))
    det = DriftDetector(onto, p)
    check = engine != "none"
    # 对照分支（z3/structural）关闭 fail-closed 提交规则，让该策略按自身裁决
    # 提交，从而比较各检查策略的检出能力；主链路的部署规则不受影响。
    evo = OntologyEvolver(onto, check=check, engine=engine, use_cache=False,
                          require_dl_verdict=(engine in ("auto", "hermit",
                                                         "owlready2", "pellet")))

    new_cid = len(onto.concepts)
    injected = 0
    injected_applied = 0
    modes: Dict[str, int] = {}
    modes_ok: Dict[str, int] = {}        # 按注入类型统计“被接受（=漏检）”数
    k_target = int(round(alpha * params.n_redefinition))
    t_detect = 0.0
    t_evolve = 0.0

    with ResourceTracker() as rt:
        for t, batch in enumerate(stream.batches, start=1):
            s = time.perf_counter()
            alarms = det.observe(t, batch)
            t_detect += (time.perf_counter() - s) * 1000.0
            s = time.perf_counter()
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
                    # 新增概念可能因不一致被回滚，此时不应登记到检测器
                    if ok and cid in onto.concepts:
                        det.centroids[cid] = np.asarray(
                            onto.concepts[cid].centroid, dtype=np.float32)
                        det.counts[cid] = []
                        det.weights[cid] = []
                elif a.dtype == "redefinition":
                    axs = det.last_projected.get(a.cid) or []
                    if not axs:
                        continue
                    # 确定性注入：前 k 个重定义事件各注入一次（k = alpha × 事件数），
                    # 保证同一 alpha 下各引擎面对完全相同的注入序列
                    c_now = onto.concepts.get(a.cid)
                    if injected < k_target and c_now is not None and c_now.active:
                        mode = int(rng.integers(0, 3))
                        modes[f"mode{mode}"] = modes.get(f"mode{mode}", 0) + 1
                        injected += 1
                        if mode == 2 and _parent_of(onto, a.cid):
                            # 父子链上的不相交：作为对父概念的一次独立提交，
                            # 走完整的快照/回滚，避免矛盾残留
                            p_cid = _parent_of(onto, a.cid)
                            p_axs = list(onto.concepts[p_cid].axioms) + [
                                Axiom("DISJOINT", p_cid, obj=a.cid)]
                            ok = evo.redefine(p_cid, p_axs)
                            injected_applied += int(ok)   # 接受 = 漏检
                            evo.redefine(a.cid, axs)
                        elif mode == 2:
                            # 无父概念时构造等价情形：先 C ⊑ D，再 D ⊓ C ⊑ ⊥
                            # （矛盾分散在两个概念上，结构性检查看不见）
                            others = [c for c in onto.concepts.values()
                                      if c.active and c.cid != a.cid]
                            if others:
                                d = others[int(rng.integers(0, len(others)))]
                                evo.redefine(a.cid,
                                             [Axiom("DOMAIN", a.cid, obj=d.cid)]
                                             + list(axs))
                                ok = evo.redefine(
                                    d.cid, list(d.axioms)
                                    + [Axiom("DISJOINT", d.cid, obj=a.cid)])
                                injected_applied += int(ok)
                            else:
                                ok = evo.redefine(
                                    a.cid,
                                    _inject_conflict(onto, a.cid, 0, rng))
                                injected_applied += int(ok)
                        else:
                            axs = _inject_conflict(onto, a.cid, mode, rng)
                            ok = evo.redefine(a.cid, axs)
                            injected_applied += int(ok)   # 接受 = 漏检
                        modes_ok[f"mode{mode}"] = (modes_ok.get(f"mode{mode}", 0)
                                                   + int(ok))
                        if debug:
                            rep2 = consistency_report(
                                onto, engine=engine, use_cache=False,
                                owl_out=C.DIR_TEMP /
                                f"exp5_dbg_{engine}_{injected}.owl")
                            print(f"  [inject] cid={a.cid} mode={mode} "
                                  f"ok={ok} consistent={rep2['consistent']} "
                                  f"engine={rep2['engine']} "
                                  f"struct={rep2['structural']} "
                                  f"axs={[ (x.kind,x.pred,x.card_min,x.card_max) for x in onto.concepts[a.cid].axioms ]} "
                                  f"active={onto.concepts[a.cid].active}")
                        continue
                    evo.redefine(a.cid, axs)
                elif a.dtype == "extinction":
                    evo.retire(a.cid)
            t_evolve += (time.perf_counter() - s) * 1000.0

    st = evo.stats
    path = None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        path = export_owl(onto, outdir / f"seed{seed}_{engine}_a{alpha}.owl")
    # 最终一致性用金标准（HermiT，不可用则 auto）判定，口径与策略无关；
    # 同时记录该策略自己的判定，便于看“自认为一致 / 实际不一致”
    rep = consistency_report(onto, engine="auto", use_cache=False,
                             owl_out=path or (C.DIR_TEMP / "exp5_final.owl"))
    rep_self = consistency_report(
        onto, engine=("structural" if engine == "none" else engine),
        use_cache=False)

    return {
        "seed": seed, "engine": engine, "alpha": alpha,
        "n_batches": len(stream.batches), "n_events": len(stream.events),
        "n_concepts_final": len([c for c in onto.concepts.values() if c.active]),
        "versions": st.versions,
        "applied": dict(st.applied),
        "rollbacks": dict(st.rollbacks),
        "rollback_rate": round(st.rollback_rate, 4),
        "inconsistent_versions": st.inconsistent_versions,
        "n_injected_conflicts": injected,
        "n_injected_accepted": injected_applied,   # 注入的矛盾被漏检（未回滚）
        "injected_modes": dict(modes),
        "injected_modes_accepted": dict(modes_ok),  # 按类型的漏检数
        "final_consistent": bool(rep["consistent"]),
        "final_consistent_self": bool(rep_self["consistent"]),
        "final_engine_gold": rep.get("engine"),
        "check_ms_total": round(st.ms_check, 1),
        "check_ms_per_version": round(st.ms_check / max(st.versions, 1), 1),
        "engines": dict(st.engines),
        "embed_api_calls": int(emb.n_api_calls - n_api0),
        "ms_stream": round(t_stream, 1),
        "ms_build_o0": round(t_build, 1),
        "ms_detect": round(t_detect, 1),
        "ms_evolve": round(t_evolve, 1),
        "ms_end2end": round((time.perf_counter() - t_start) * 1000.0, 1),
        "peak_mem_mb": round(rt.peak_mb, 1),
        "owl": str(path) if path else None,
    }


def summarize(rows: List[Dict], n_boot: int = 1000) -> List[Dict]:
    """按 (engine, alpha) 聚合：回退率均值 + bootstrap 95% CI、矛盾检出率、
    资源均值。"""
    from src.eval_drift import bootstrap_ci

    groups: Dict[tuple, List[Dict]] = {}
    for r in rows:
        groups.setdefault((r["engine"], r["alpha"]), []).append(r)
    out: List[Dict] = []
    for (eng, a), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rb = [float(r["rollback_rate"]) for r in rs]
        inj = sum(int(r["n_injected_conflicts"]) for r in rs)
        acc = sum(int(r["n_injected_accepted"]) for r in rs)
        ci = bootstrap_ci(rb, n_boot=n_boot, seed=0) if len(rb) > 1 else None
        out.append({
            "engine": eng, "alpha": a, "n_runs": len(rs),
            "rollback_rate_mean": round(float(np.mean(rb)), 4),
            "rollback_rate_ci95": None if ci is None else [ci["lo"], ci["hi"]],
            "n_injected": inj,
            "n_injected_accepted": acc,
            "detection_rate": None if inj == 0 else round(1.0 - acc / inj, 4),
            "final_consistent_rate": round(float(np.mean(
                [int(bool(r["final_consistent"])) for r in rs])), 4),
            "versions_mean": round(float(np.mean([r["versions"] for r in rs])), 1),
            "check_ms_per_version_mean": round(float(np.mean(
                [r["check_ms_per_version"] for r in rs])), 1),
            "e2e_s_mean": round(float(np.mean(
                [r["ms_end2end"] for r in rs])) / 1000.0, 1),
            "peak_mem_mb_mean": round(float(np.mean(
                [r["peak_mem_mb"] for r in rs])), 1),
        })
    return out


def mode_breakdown(rows: List[Dict]) -> List[Dict]:
    """按 (engine, alpha) 汇总各注入类型的注入数 / 漏检（被接受）数 / 检出率。

    mode0 = 空基数区间（结构性可见）；mode1 = DOMAIN 与 DISJOINT 同指（结构性
    可见）；mode2 = 父子链上的不相交（需 subsumption 推理，结构性不可见）。
    """
    g: Dict[tuple, Dict[str, Dict[str, int]]] = {}
    for r in rows:
        inj = r.get("injected_modes") or {}
        acc = r.get("injected_modes_accepted") or {}
        d = g.setdefault((r["engine"], r["alpha"]), {})
        for m in set(list(inj) + list(acc)):
            e = d.setdefault(m, {"injected": 0, "accepted": 0})
            e["injected"] += int(inj.get(m, 0))
            e["accepted"] += int(acc.get(m, 0))
    out: List[Dict] = []
    for (eng, a), d in sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row: Dict = {"engine": eng, "alpha": a}
        for m in sorted(d):
            n, k = d[m]["injected"], d[m]["accepted"]
            row[m] = {"injected": n, "accepted": k,
                      "detection_rate": (None if n == 0 else
                                         round(1.0 - k / n, 4))}
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="",
                    help="多个随机种子（逗号分隔），用于给回退率/资源出置信区间")
    ap.add_argument("--engines", default="hermit,z3,structural,none")
    ap.add_argument("--alphas", default="0,0.2,0.5")
    ap.add_argument("--tag", default="exp5",
                    help="结果输出目录名（results/<tag>），默认 exp5")
    args = ap.parse_args()

    base = C.PARAMS.copy(n_redefinition=6)      # 重定义事件多一些，注入才有统计意义
    if args.quick:
        base = base.copy(n_batch=12, n_init_per_leaf=6, pool_per_leaf=12,
                         base_rate=3.0, n_emergence=1, n_extinction=1,
                         n_redefinition=6)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or [args.seed]

    emb = Embedder(allow_fallback=False, )
    outdir = C.DIR_ONTOLOGY / "exp5"
    rows: List[Dict] = []

    print("== 实验 5：资源消耗与回退率 ==")
    for seed in seeds:
        for alpha in alphas:
            for eng in engines:
                r = run_case(seed, eng, alpha, base, emb, outdir)
                rows.append(r)
                print(f"seed={seed} alpha={alpha:<4} engine={eng:<10} "
                      f"versions={r['versions']:<3} "
                      f"inj={r['n_injected_conflicts']:<2} "
                      f"rollbacks={sum(r['rollbacks'].values()):<3} "
                      f"rollback_rate={r['rollback_rate']:.3f} "
                      f"final_gold={str(r['final_consistent']):<5} "
                      f"self={str(r['final_consistent_self']):<5} "
                      f"check_ms/ver={r['check_ms_per_version']:.0f} "
                      f"e2e={r['ms_end2end']/1000:.1f}s "
                      f"mem={r['peak_mem_mb']:.0f}MB")

    summary = summarize(rows)
    print("\n==== 汇总（按 alpha / 引擎，均值与 bootstrap 95% CI）====")
    print(f"{'alpha':<6}{'engine':<12}{'回退率':<18}{'检出率':<9}"
          f"{'最终一致率':<10}{'检查ms/版本':<12}{'端到端s':<9}{'内存MB':<8}")
    for s in summary:
        rb = s["rollback_rate_mean"]
        ci = s["rollback_rate_ci95"]
        rb_s = f"{rb:.3f}" if ci is None else f"{rb:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
        det = "-" if s["detection_rate"] is None else f"{s['detection_rate']:.2f}"
        print(f"{s['alpha']:<6}{s['engine']:<12}{rb_s:<18}{det:<9}"
              f"{s['final_consistent_rate']:.2f}{'':<8}"
              f"{s['check_ms_per_version_mean']:<12.1f}"
              f"{s['e2e_s_mean']:<9.1f}{s['peak_mem_mb_mean']:<8.1f}")

    res = C.DIR_RESULTS / args.tag
    res.mkdir(parents=True, exist_ok=True)

    modes_bd = mode_breakdown(rows)
    print("\n==== 注入类型分解（injected=注入数，accepted=漏检数）====")
    for m in modes_bd:
        cells = []
        for k, v in m.items():
            if not k.startswith("mode"):
                continue
            det = ("-" if v["detection_rate"] is None
                   else f"{v['detection_rate']:.2f}")
            cells.append(f"{k} {v['injected']}/{v['accepted']} 检出{det}")
        print(f"alpha={m['alpha']:<5} engine={m['engine']:<12} " + "  ".join(cells))
    (res / "exp5_modes.json").write_text(
        json.dumps(modes_bd, ensure_ascii=False, indent=2), encoding="utf-8")
    (res / "exp5.json").write_text(
        json.dumps({"rows": rows, "summary": summary, "engines": engines,
                    "alphas": alphas, "seeds": seeds, "quick": args.quick},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    (res / "exp5_summary.json").write_text(
        json.dumps({"seeds": seeds, "alphas": alphas, "engines": engines,
                    "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    if rows:
        import csv
        with (res / "exp5.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n结果写入 {res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
