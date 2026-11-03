"""ontoevolve.src.eval_qa - 命题 C：三类知识问答评估。

设计要点：

1. **三类知识自动标注**：依据 `OntologyEvolver.history`
   - `emergence` → new（新知识）
   - `redefinition` → redefined（重定义知识，优先于 new）
   - 无操作 → stable（稳定知识）
   - `extinction` → 已不活跃，不参与评估

2. **题目由一条公理派生**，金标即该公理的取值，因此**不需要人工标注答案**。
   重定义类额外记录初始本体 O_0 中的旧取值 `stale`：被测系统若答出 stale，
   判为**语义错位**（冻结本体“答得出来但答的是过时语义”）。

3. **每题两版**（防泄露）：A 版含概念名，B 版为描述式（用上位概念与同概念的
   其它公理作线索，不出现概念名）。两版都跑，差值用于说明是否靠字面匹配。

4. **金标固定为最终本体 O_T 的语义**；用不同时刻 / 不同策略的本体作答，
   得到三类准确率，以及准确率随批次的衰减曲线。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ontology_model import Ontology, Concept, Axiom        # noqa: F401

STABLE, NEW, REDEFINED = "stable", "new", "redefined"
KCLASSES = (STABLE, NEW, REDEFINED)

# 可出题的公理类型（TRANSITIVE 等不适合出成问答，跳过）
QUESTION_KINDS = ("PARENT", "DOMAIN", "RANGE", "OP", "DP", "DISJOINT", "CARD")


# ---------------------------------------------------------------------------
# 三类知识标注
# ---------------------------------------------------------------------------
def label_knowledge_classes(onto: Ontology,
                            history: Sequence[Dict]) -> Dict[str, str]:
    """cid -> 'stable' | 'new' | 'redefined'（不活跃概念不出现在结果里）。"""
    ops: Dict[str, set] = {}
    for h in history or []:
        if not h.get("ok", True):
            continue                       # 被回滚的操作不算知识变化
        ops.setdefault(str(h.get("cid")), set()).add(str(h.get("op")))
    out: Dict[str, str] = {}
    for cid, c in onto.concepts.items():
        if not c.active:
            continue
        s = ops.get(cid, set())
        if "redefinition" in s:
            out[cid] = REDEFINED
        elif "emergence" in s:
            out[cid] = NEW
        else:
            out[cid] = STABLE
    return out


# ---------------------------------------------------------------------------
# 公理取值 / 题干
# ---------------------------------------------------------------------------
def axiom_value(ax: Axiom) -> Optional[str]:
    """公理的“答案值”；无取值（如纯 TRANSITIVE）返回 None。"""
    if ax.kind == "CARD":
        hi = ax.card_max if ax.card_max else "∞"
        return f"[{ax.card_min}, {hi}]"
    return ax.obj or None


def lookup_value(onto: Ontology, cid: str, kind: str,
                 pred: str = "") -> Optional[str]:
    """在给定本体中查 (cid, kind, pred) 对应的取值（PARENT 查上位概念）。"""
    c = onto.concepts.get(cid)
    if c is None or not c.active:
        return None
    if kind == "PARENT":
        return c.parent or None
    for ax in c.axioms:
        if ax.kind == kind and (ax.pred or "") == (pred or ""):
            v = axiom_value(ax)
            if v:
                return v
    return None


def _readable(kind: str, pred: str) -> str:
    """把公理类型说成人话，用于题干。"""
    if kind == "PARENT":
        return "上位概念"
    if kind == "CARD":
        return f"“{pred}”关系的数量区间"
    if kind == "DOMAIN":
        return f"“{pred}”关系的定义域"
    if kind == "RANGE":
        return f"“{pred}”关系的值域"
    if kind == "DISJOINT":
        return "与之不相交的概念"
    return f"“{pred}”的取值"


@dataclass
class Question:
    qid: str
    kclass: str                 # stable | new | redefined
    cid: str
    kind: str                   # 公理类型（PARENT 表示父子关系）
    pred: str
    q_a: str                    # A 版：含概念名
    q_b: str                    # B 版：描述式（不出现概念名）
    gold: str                   # 金标（最终本体 O_T 的语义）
    stale: Optional[str] = None # 旧语义（O_0），仅重定义类通常非空
    t_event: int = 0            # 事件批次（稳定类为 0）
    degraded_b: bool = False    # B 版无可用线索，退化为无信息题干

    def text(self, version: str = "a") -> str:
        return self.q_a if version == "a" else self.q_b


def _candidates(onto: Ontology, cid: str) -> List[Tuple[str, str, str]]:
    """某概念可出的题：(kind, pred, value)。"""
    c = onto.concepts[cid]
    items: List[Tuple[str, str, str]] = []
    if c.parent:
        items.append(("PARENT", "", c.parent))
    seen = set()
    for ax in c.axioms:
        if ax.kind not in QUESTION_KINDS:
            continue
        # CARD/OP/DP/DOMAIN/RANGE 必须有谓词才能成题（DISJOINT 以 obj 为主）
        if not (ax.pred or "") and ax.kind in ("CARD", "OP", "DP",
                                               "DOMAIN", "RANGE"):
            continue
        v = axiom_value(ax)
        if not v:
            continue
        key = (ax.kind, ax.pred)
        if key in seen:
            continue
        seen.add(key)
        items.append((ax.kind, ax.pred or "", v))
    return items


def _cues(onto: Ontology, cid: str, kind: str,
          pred: str) -> Tuple[str, bool]:
    """B 版线索：上位概念 + 同概念的另一条公理（不含答案本身）。

    返回 (线索文本, 是否退化)。退化时题干没有有效线索，只标记，不出题。
    """
    c = onto.concepts[cid]
    bits: List[str] = []
    if c.parent:
        bits.append(f"它是 {c.parent} 的子类")
    for ax in c.axioms:
        if (ax.kind, ax.pred or "") == (kind, pred):
            continue
        v = axiom_value(ax)
        if v:
            bits.append(f"其{_readable(ax.kind, ax.pred or '')}为 {v}")
        if len(bits) >= 2:
            break
    if not bits:
        return "", True
    return "，" + "、".join(bits), False


def build_questions(onto_final: Ontology, onto_init: Ontology,
                    labels: Dict[str, str], *, per_class: int = 100,
                    max_per_concept: int = 10, seed: int = 0,
                    t_event_by_cid: Optional[Dict[str, int]] = None) -> List[Question]:
    """按三类知识生成题目。

    - 金标取自 `onto_final`（O_T 语义）
    - `stale` 取自 `onto_init`（O_0 语义），用于识别语义错位
    - 每类最多 `per_class` 题，每个概念最多 `max_per_concept` 题
    """
    rng = np.random.default_rng(seed)
    t_event_by_cid = t_event_by_cid or {}
    out: List[Question] = []

    for kc in KCLASSES:
        cids = [cid for cid, v in labels.items() if v == kc
                and cid in onto_final.concepts
                and onto_final.concepts[cid].active]
        pool: List[Tuple[str, str, str, str]] = []      # (cid, kind, pred, value)
        for cid in cids:
            for kind, pred, val in _candidates(onto_final, cid):
                if kc == REDEFINED:
                    # 只出“被重定义真正改变”的公理：O_0 无此公理，或取值不同。
                    # 否则抽到的是该概念未变动的公理，冻结本体也能答对，
                    # 会把重定义类稀释成稳定类（H3 就测不出来了）。
                    if lookup_value(onto_init, cid, kind, pred) == val:
                        continue
                pool.append((cid, kind, pred, val))
        if not pool:
            continue
        order = rng.permutation(len(pool))
        per_concept: Dict[str, int] = {}
        picked: List[Tuple[str, str, str, str]] = []
        for i in order:
            cid, kind, pred, val = pool[int(i)]
            if per_concept.get(cid, 0) >= max_per_concept:
                continue
            per_concept[cid] = per_concept.get(cid, 0) + 1
            picked.append((cid, kind, pred, val))
            if len(picked) >= per_class:
                break

        for cid, kind, pred, val in picked:
            name = onto_final.concepts[cid].name
            ask = _readable(kind, pred)
            cue, degraded = _cues(onto_final, cid, kind, pred)
            q_a = f"{name} 的{ask}是什么？"
            q_b = f"某个概念{cue}，它的{ask}是什么？"
            stale = lookup_value(onto_init, cid, kind, pred)
            out.append(Question(
                qid=f"{kc[:3]}-{len(out):04d}", kclass=kc, cid=cid,
                kind=kind, pred=pred, q_a=q_a, q_b=q_b, gold=val,
                stale=(None if stale == val else stale),
                t_event=int(t_event_by_cid.get(cid, 0)),
                degraded_b=degraded))
    return out


# ---------------------------------------------------------------------------
# 检索与作答
# ---------------------------------------------------------------------------
@dataclass
class Answer:
    cid: Optional[str]
    value: Optional[str]
    sim: float = 0.0
    abstain: bool = False       # 检索相似度低于阈值 → 回答“不知道”


def concept_surface(c: Concept) -> str:
    """概念的表层描述（用于检索）。"""
    bits = [f"{c.name}"]
    if c.parent:
        bits.append(f"是 {c.parent} 的子类")
    for ax in c.axioms[:8]:
        v = axiom_value(ax)
        if v:
            bits.append(f"{_readable(ax.kind, ax.pred or '')}为 {v}")
    return "；".join(bits)


def build_surfaces(onto: Ontology, emb) -> Tuple[List[str], np.ndarray]:
    """活跃概念的 (cid 列表, 表层描述嵌入矩阵)。"""
    cids = [cid for cid, c in onto.concepts.items() if c.active]
    if not cids:
        return [], np.zeros((0, 1), dtype=np.float32)
    vecs = emb.embed([concept_surface(onto.concepts[cid]) for cid in cids])
    return cids, np.asarray(vecs, dtype=np.float32)


def answer(onto: Ontology, q: Question, emb, surfaces: Tuple[List[str], np.ndarray],
           *, link_sim: float = 0.5, version: str = "a") -> Answer:
    """用给定本体回答一题：题干嵌入 → 最近概念 → 取该 (kind, pred) 的取值。"""
    cids, vecs = surfaces
    if not cids:
        return Answer(None, None, 0.0, True)
    qv = np.asarray(emb.embed([q.text(version)])[0], dtype=np.float32)
    sims = vecs @ qv
    j = int(np.argmax(sims))
    sim = float(sims[j])
    if sim < link_sim:
        return Answer(None, None, sim, True)
    cid = cids[j]
    val = lookup_value(onto, cid, q.kind, q.pred)
    return Answer(cid, val, sim, val is None)


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------
def score_one(ans: Answer, q: Question) -> Dict[str, int]:
    """结构化评分：0/1 正确 + 是否语义错位（答出旧语义）。"""
    hit = bool(ans.value) and ans.value == q.gold
    stale_hit = bool(q.stale) and (not hit) and ans.value == q.stale
    return {"correct": int(hit), "stale_hit": int(stale_hit),
            "abstain": int(ans.value is None)}


JUDGE_TMPL = (
    "请评判下列问答，只输出一个数字。\n"
    "3=完全正确且关键要素齐全；2=核心正确但缺次要细节；\n"
    "1=答错但相关；0=错误、幻觉或回答不知道。\n\n"
    "问题：{q}\n参考答案：{gold}\n系统回答：{ans}\n\n分数：")


def judge_prompt(q: Question, ans: Answer, version: str = "a") -> str:
    return JUDGE_TMPL.format(q=q.text(version), gold=q.gold,
                             ans=(ans.value or "不知道"))


def parse_judge(text: Optional[str]) -> Optional[int]:
    """从 LLM 输出里解析 0~3 的分数。"""
    if not text:
        return None
    m = re.search(r"[0-3]", text)
    return int(m.group(0)) if m else None


def summarize_scores(rows: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """按 kclass 聚合：准确率、语义错位率、弃答率、题量。"""
    out: Dict[str, Dict[str, float]] = {}
    for kc in KCLASSES:
        sub = [r for r in rows if r.get("kclass") == kc]
        if not sub:
            continue
        n = len(sub)
        out[kc] = {
            "n": n,
            "accuracy": round(sum(r["correct"] for r in sub) / n, 4),
            "stale_rate": round(sum(r["stale_hit"] for r in sub) / n, 4),
            "abstain_rate": round(sum(r["abstain"] for r in sub) / n, 4),
        }
    if rows:
        n = len(rows)
        out["all"] = {
            "n": n,
            "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
            "stale_rate": round(sum(r["stale_hit"] for r in rows) / n, 4),
            "abstain_rate": round(sum(r["abstain"] for r in rows) / n, 4),
        }
    return out
