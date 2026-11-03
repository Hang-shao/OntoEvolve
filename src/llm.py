"""ontoevolve.src.llm - 候选公理的大模型判定。

设计要点：
    1) 只对“新出现/证据最强”的 top-K 条候选公理调用大模型，其余沿用规则置信度；
    2) 判定结果按公理签名落盘缓存，可断点续跑；
    3) 服务不可用或超时时返回 None，调用方保持规则置信度——实验永不卡在网络上。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import config as C
from .ontology_model import Axiom

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


PROMPT_TMPL = (
    "你是本体工程专家。给定概念与从数据流中抽取的候选公理，判断该公理是否成立。\n"
    "概念：{concept}\n公理类型：{kind}\n内容：{content}\n"
    "只输出一个 0 到 1 之间的数字表示你的置信度，不要解释。"
)


def axiom_text(ax: Axiom) -> str:
    if ax.kind in ("OP", "DP", "TRANSITIVE"):
        return f"{ax.pred} -> {ax.obj}" if ax.obj else ax.pred
    if ax.kind == "CARD":
        return f"关系数量区间 [{ax.card_min}, {ax.card_max if ax.card_max else '∞'}]"
    return ax.obj


def axiom_key(ax: Axiom, concept: str) -> str:
    raw = f"{concept}|{ax.kind}|{ax.pred}|{ax.obj}|{ax.card_min}|{ax.card_max}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@dataclass
class JudgeStats:
    calls: int = 0
    cache_hits: int = 0
    failures: int = 0
    seconds: float = 0.0


class AxiomJudge:
    """大模型公理判定器（带缓存与降级）。"""

    def __init__(self, *, cache_path: Optional[Path] = None,
                 top_k: int = 30, timeout: int = 60, enabled: bool = True,
                 model: Optional[str] = None, verbose: bool = False):
        self.cache_path = Path(cache_path or (C.DIR_CACHE / "llm_axiom_judgements.jsonl"))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.top_k = int(top_k)
        self.timeout = timeout
        self.enabled = bool(enabled) and bool(C.CHAT_API_URL)
        self.model = model or C.CHAT_MODEL
        self.verbose = verbose
        self.stats = JudgeStats()
        self._cache: Dict[str, float] = self._load_cache()

    # ---- 缓存 ---------------------------------------------------------------
    def _load_cache(self) -> Dict[str, float]:
        cache: Dict[str, float] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    cache[d["key"]] = float(d["conf"])
                except Exception:
                    continue
        return cache

    def _append_cache(self, key: str, conf: float) -> None:
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "conf": conf}, ensure_ascii=False) + "\n")

    # ---- 单条判定 -----------------------------------------------------------
    def _request(self, prompt: str, *, max_tokens: int = 16) -> Optional[str]:
        """调用一次大模型并返回原始文本；不可用时返回 None。"""
        if requests is None:
            return None
        headers = {"Content-Type": "application/json"}
        if C.CHAT_API_KEY:
            headers["Authorization"] = f"Bearer {C.CHAT_API_KEY}"
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "max_tokens": max_tokens,
                   # 对 thinking 类模型关闭思维链，保证输出可被解析器直接读取
                   "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = requests.post(C.CHAT_API_URL, headers=headers, json=payload,
                              timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
            msg = body["choices"][0]["message"]
            return _strip_thinking(msg.get("content") or "",
                                   msg.get("reasoning_content") or "")
        except Exception as e:
            if self.verbose:
                print(f"  [warn] LLM judge failed: {e}")
            return None

    def _call_api(self, prompt: str) -> Optional[float]:
        return _parse_confidence(self._request(prompt) or "")

    def ask(self, prompt: str, *, max_tokens: int = 256) -> Optional[str]:
        """通用文本接口（命题 C 的 LLM-Judge 用）；LLM 不可用时返回 None。"""
        if not self.enabled:
            return None
        t0 = time.time()
        out = self._request(prompt, max_tokens=max_tokens)
        self.stats.calls += 1
        self.stats.seconds += time.time() - t0
        if out is None:
            self.stats.failures += 1
        return out

    def judge(self, ax: Axiom, concept: str) -> Optional[float]:
        """返回大模型置信度；不可用时返回 None。"""
        key = axiom_key(ax, concept)
        if key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        if not self.enabled:
            return None
        t0 = time.time()
        prompt = PROMPT_TMPL.format(concept=concept, kind=ax.kind,
                                    content=axiom_text(ax))
        try:
            conf = self._call_api(prompt)
        except Exception as e:      # 任何异常都降级为“无判定”
            if self.verbose:
                print(f"  [warn] LLM judge error: {e}")
            conf = None
        self.stats.calls += 1
        self.stats.seconds += time.time() - t0
        if conf is None:
            self.stats.failures += 1
            return None
        self._cache[key] = conf
        self._append_cache(key, conf)
        return conf

    # ---- 批量：只判 top-K ---------------------------------------------------
    def judge_batch(self, axioms: Sequence[Axiom], concept: str) -> Dict[int, float]:
        """对证据最强的 top_k 条公理调用大模型，返回 {下标: 置信度}。"""
        if not axioms or self.top_k <= 0:
            return {}
        order = sorted(range(len(axioms)),
                       key=lambda i: -axioms[i].evidence_count)[: self.top_k]
        out: Dict[int, float] = {}
        for i in order:
            conf = self.judge(axioms[i], concept)
            if conf is not None:
                out[i] = conf
        return out

    def apply(self, axioms: Sequence[Axiom], concept: str) -> List[Axiom]:
        """把大模型置信度乘到规则置信度上（LLM 不可用时保持不变）。"""
        confs = self.judge_batch(axioms, concept)
        result = []
        for i, ax in enumerate(axioms):
            if i in confs:
                ax.confidence = float(ax.confidence) * float(confs[i])
            result.append(ax)
        return result


def _strip_thinking(content: str, reasoning: str = "") -> str:
    """去掉 <think>…</think> 思维链，只保留最终回答。

    部分推理服务忽略 enable_thinking 参数；此时模型会在 content 前附带
    思维链文本，其中的数字会污染下游解析器，必须剥离。
    """
    import re
    if content:
        content = re.sub(r"<think>.*?(?:</think>|$)", "", content,
                         flags=re.S | re.I).strip()
    if content:
        return content
    if reasoning:
        return re.sub(r"<think>.*?(?:</think>|$)", "", reasoning,
                      flags=re.S | re.I).strip()
    return ""


def _parse_confidence(text: str) -> Optional[float]:
    """从大模型输出中解析第一个 0~1 的数字。"""
    import re
    m = re.search(r"([01](?:\.\d+)?|\.\d+)", text or "")
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return min(1.0, max(0.0, v))
