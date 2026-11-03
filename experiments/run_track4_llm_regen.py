"""experiments/run_track4_llm_regen.py — 第四基线：LLM 直接重生成整份本体。

定位：与"冻结 / 增量演化 / 周期全量重构"并列的第四组对照。同一数据窗口、同一
受控词表（5 类诈骗类型 + 14 个话术步骤），把"整份本体的生成"交给大模型一次完成，
不做增量维护、不做一致性回滚；随后用与另三组完全相同的检索骨架与题集评测。

公平性约定（写论文时要交代）：
  1) 只给模型 sentences 的 text 与 type 两个字段，**不给** data/track4/batches 里
     流水线自己抽出的 steps 字段（避免把确定性抽取器的结果泄漏给 LLM 基线）；
  2) 词表与另三组一致（类型名、14 步步骤名），因此差异来自"生成方式"而非先验知识；
  3) 概念名允许子串匹配计分（严格计分同时报告），避免因为命名差异（如"冒充公检法类
     诈骗"）把知识判错；
  4) 步骤的典型程度由模型自报顺序给出，按顺序赋递减 evidence_count（另一个模型无从
     获得频次，故不使用流水线词表去数频次）。

用法：
    python experiments/run_track4_llm_regen.py --window p4 --per-type 12
    python experiments/run_track4_llm_regen.py --window all --per-type 12
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from src import config as C                                    # noqa: E402
from src.consistency import consistency_report                  # noqa: E402
from src.fraud_stream import read_materialized                   # noqa: E402
from src.fraud_vocab import OP_VOCAB, PARENT_EVENT, TYPE_BY_LABEL  # noqa: E402
from src.ontology_model import Axiom, Concept, Ontology          # noqa: E402

import run_track4_qa as Q                                        # noqa: E402

OUT = ROOT / "results" / "track4_llm_regen"
LABELS = ["冒充公检法", "利诱代办", "冒充客服", "冒充熟人", "其他干扰"]
IGNORE_TYPES = ("非诈骗转写", "复合话术")

PROMPT_HEAD = """你是本体工程师。下面是一批已标注的电信网络诈骗话术句子（格式：类型|句子）。
请据此生成一份完整的领域本体：概念集合 + 每个概念所包含的话术步骤。

硬性要求：
1. 概念名只能从这五类中选，不要自造新名：{labels}；
2. 每个概念的 steps 只能从下列步骤词表中选，按"该类话术的典型程度"从高到低排序：
{steps}；
3. 某类型若在这批数据中已不再出现，status 填 "deprecated"，仍在出现的填 "active"；
4. 若某类型的步骤构成相对早期已发生变化，只保留当前数据中仍在用的步骤，不要保留已消失的步骤；
5. "非诈骗转写""复合话术"不是独立类型，忽略，不要为它们建概念；
6. 只输出 JSON，不要解释、不要 markdown 代码块，格式示例：
{{"concepts":[{{"name":"冒充公检法","parent":"诈骗事件","status":"active","steps":["身份被冒用","要求报案自证"]}}]}}

数据如下：
"""


def call_llm(prompt: str, *, model: str, timeout: int = 180,
             max_tokens: int = 1500) -> Dict:
    """直接调用 OpenAI 兼容接口，返回文本与用量（便于在日志里记成本）。"""
    import requests
    headers = {"Content-Type": "application/json"}
    if C.CHAT_API_KEY:
        headers["Authorization"] = f"Bearer {C.CHAT_API_KEY}"
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    r = requests.post(C.CHAT_API_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    msg = body["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if not text:
        text = (msg.get("reasoning_content") or "").strip()
    return {"text": text, "usage": body.get("usage", {}),
            "seconds": round(time.time() - t0, 2), "model": body.get("model", model)}


def parse_json(text: str) -> Optional[Dict]:
    t = re.sub(r"```(?:json)?", "", text or "").strip("` \n")
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S | re.I)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except Exception:
        return None


def sample_rows(batches: List[List[Dict]], window: str, per_type: int) -> List[Dict]:
    """按 (批, 类型) 分层抽样；每条只带 text 与 type，不带流水线抽出的 steps。"""
    idx = range(len(batches) - 4, len(batches)) if window == "p4" else range(len(batches))
    picked: List[Dict] = []
    for bi in idx:
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for it in batches[bi]:
            # FraudItem：只用 text 与真实类型，不把流水线抽出的 steps 给模型
            if it.ftype in IGNORE_TYPES:
                continue
            by_type[it.ftype].append({"text": it.text})
        for t, items in by_type.items():
            for it in items[:per_type]:
                picked.append({"batch": bi + 1, "type": t, "text": it["text"]})
    return picked


def to_ontology(data: Dict) -> Ontology:
    concepts: Dict[str, Concept] = {}
    for i, c in enumerate(data.get("concepts", [])):
        name = str(c.get("name", "")).strip()
        cid = f"llm{i:02d}"
        steps = [s for s in (c.get("steps") or []) if s in OP_VOCAB]
        seen, uniq = set(), []
        for s in steps:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        axioms = [Axiom(kind="OP", subj=cid, pred=s, obj=PARENT_EVENT,
                        evidence_count=len(uniq) - k)
                  for k, s in enumerate(uniq)]
        concepts[cid] = Concept(cid=cid, name=name,
                                parent=c.get("parent") or PARENT_EVENT,
                                axioms=axioms,
                                active=(str(c.get("status", "active")).lower()
                                        != "deprecated"),
                                meta={"llm_steps": uniq})
    return Ontology(concepts, op_vocab=OP_VOCAB)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=("p4", "all"), default="p4")
    ap.add_argument("--per-type", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", choices=("local", "api"), default="local")
    ap.add_argument("--link-sim", type=float, default=0.5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--dry-run", action="store_true",
                    help="只构建 prompt 并打印长度，不调用模型")
    args = ap.parse_args()

    model = args.model or C.CHAT_MODEL
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.window

    batches, truth, manifest = read_materialized()
    rows = sample_rows(batches, args.window, args.per_type)
    body = "\n".join(f"{r['type']}|{r['text']}" for r in rows)
    prompt = PROMPT_HEAD.format(labels="、".join(LABELS),
                                steps="、".join(OP_VOCAB)) + body
    (OUT / f"prompt_{tag}.txt").write_text(prompt, encoding="utf-8")
    print(f"[prompt] 句数={len(rows)} 类型分布={dict(Counter(r['type'] for r in rows))} "
          f"字符数={len(prompt)}")
    if args.dry_run:
        return 0

    print(f"[llm] model={model} url=set:{bool(C.CHAT_API_URL)}")
    resp = call_llm(prompt, model=model, timeout=args.timeout,
                    max_tokens=args.max_tokens)
    (OUT / f"response_{tag}.txt").write_text(resp["text"], encoding="utf-8")
    print(f"[llm] seconds={resp['seconds']} usage={resp['usage']} "
          f"chars={len(resp['text'])}")

    data = parse_json(resp["text"])
    if not data or not data.get("concepts"):
        print("[err] 未能解析出 concepts，原始输出已落盘：", OUT / f"response_{tag}.txt")
        return 2

    onto = to_ontology(data)
    rep = consistency_report(onto)
    n_ax = sum(len(c.axioms) for c in onto.concepts.values())
    print(f"[onto] concepts={len(onto.concepts)} active="
          f"{sum(1 for c in onto.concepts.values() if c.active)} axioms={n_ax} "
          f"consistent={rep['consistent']} engine={rep['engine']}")

    # ---- 问答评测（与另三组同一骨架）----
    qs = Q.load_questions()
    emb = Q.make_embedder(args.embedder, verbose=True)
    cids, vecs = Q.build_surfaces(onto, emb)
    items: List[Dict] = []
    for q in qs:
        for ver in ("a", "b"):
            ans = Q.answer_one(onto, q, emb, (cids, vecs),
                               link_sim=args.link_sim, version=ver)
            sc = Q.score(q, ans)
            gold, a = q["gold"], ans.get("answer") or ""
            lenient = sc["correct"] or (q["ask"] == "type" and gold in a)
            items.append({"scheme": "llm_regen", "qid": q["qid"],
                          "kclass": q["kclass"], "ask": q["ask"], "version": ver,
                          "gold": gold, "answer": ans.get("answer"),
                          "concept": ans.get("concept"), "sim": ans["sim"],
                          "correct": sc["correct"], "correct_lenient": int(bool(lenient)),
                          "abstain": sc["abstain"], "stale": sc["stale"],
                          "wrong": sc["wrong"]})

    def acc(sel, key="correct"):
        n = len(sel)
        return round(sum(r[key] for r in sel) / n, 4) if n else None

    by_kclass = {}
    for kc in ("stable", "new", "redefined", "extinct"):
        sub = [r for r in items if r["kclass"] == kc]
        by_kclass[kc] = {"n": len(sub), "acc": acc(sub),
                         "acc_lenient": acc(sub, "correct_lenient"),
                         "abstain": acc(sub, "abstain")}
    overall = {"acc": acc(items), "acc_lenient": acc(items, "correct_lenient"),
               "n": len(items)}

    # ---- 与已有三组配对比较（读现有 qa_items_s{seed}.jsonl）----
    tests = {}
    ref_path = ROOT / "results" / "track4" / f"qa_items_s{args.seed}.jsonl"
    if ref_path.exists():
        ref_rows = [json.loads(l) for l in ref_path.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
        key = lambda r: (r["qid"], r["version"])
        for scheme in ("dynamic", "periodic", "frozen"):
            base = {key(r): r for r in ref_rows if r["scheme"] == scheme}
            mine = {key(r): r for r in items}
            common = [k for k in mine if k in base]
            per = {}
            for kc in ("stable", "new", "redefined", "extinct"):
                ks = [k for k in common if mine[k]["kclass"] == kc]
                if not ks:
                    continue
                b01 = sum(1 for k in ks if mine[k]["correct_lenient"] == 1
                          and base[k]["correct"] == 0)
                c10 = sum(1 for k in ks if mine[k]["correct_lenient"] == 0
                          and base[k]["correct"] == 1)
                per[kc] = {"n_pairs": len(ks), "llm_only": b01, "ref_only": c10,
                           "mcnemar_p": round(Q.mcnemar(b01, c10), 6),
                           "acc_llm": acc([mine[k] for k in ks], "correct_lenient"),
                           "acc_ref": round(sum(base[k]["correct"] for k in ks) / len(ks), 4)}
            tests[f"vs_{scheme}"] = per
    else:
        print(f"[warn] 缺少 {ref_path}，跳过配对比较")

    summary = {
        "scheme": "llm_regen", "window": tag, "model": resp.get("model"),
        "n_sentences": len(rows),
        "type_dist": dict(Counter(r["type"] for r in rows)),
        "usage": resp["usage"], "wall_seconds": resp["seconds"],
        "link_sim": args.link_sim, "embedder": args.embedder,
        "ontology": {
            "concepts": len(onto.concepts),
            "active": sum(1 for c in onto.concepts.values() if c.active),
            "axioms": n_ax,
            "consistent": rep["consistent"], "engine": rep["engine"],
            "concept_table": [{"cid": c.cid, "name": c.name, "active": c.active,
                               "steps": c.meta.get("llm_steps", [])}
                              for c in onto.concepts.values()],
            "raw": data,
        },
        "qa": {"overall": overall, "by_kclass": by_kclass},
        "tests": tests,
    }
    (OUT / f"summary_{tag}_s{args.seed}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / f"qa_items_{tag}_s{args.seed}.jsonl").open("w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n==== LLM 直接重生成（window=%s）====" % tag)
    for kc, v in by_kclass.items():
        print(f"  {kc:<10} n={v['n']:<3} acc={v['acc']:.3f} "
              f"（含子串计分 {v['acc_lenient']:.3f}）拒答={v['abstain']:.3f}")
    print(f"  合计       acc={overall['acc']:.3f}（含子串 {overall['acc_lenient']:.3f}）")
    for ref, per in tests.items():
        print(f"  {ref}: " + " ; ".join(
            f"{kc} llm={v['acc_llm']:.3f} ref={v['acc_ref']:.3f} p={v['mcnemar_p']:.3g}"
            for kc, v in per.items()))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
