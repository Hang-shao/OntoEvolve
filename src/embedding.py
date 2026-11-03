"""ontoevolve.src.embedding - 文本向量化（Qwen3-Embedding + 磁盘缓存 + 回退）。

所有向量返回前做 L2 归一化。
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config as C

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


def l2_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / max(n, eps)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _hash_vec(text: str, dim: int) -> np.ndarray:
    """确定性哈希伪嵌入：无网络时的可复现回退。"""
    seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(np.float32)


class Embedder:
    """带缓存的批量嵌入器，缓存键为文本 sha1。"""

    def __init__(self, *, cache_dir: Optional[Path] = None, batch_size: int = 32,
                 dim: Optional[int] = None, allow_fallback: bool = True,
                 max_retries: int = 4, retry_backoff: float = 1.5,
                 verbose: bool = False):
        self.cache_dir = Path(cache_dir or C.DIR_CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.dim = dim or C.EMBED_DIM
        self.allow_fallback = allow_fallback
        self.verbose = verbose
        self._mem: Dict[str, np.ndarray] = {}
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.n_api_calls = 0
        self.n_fallback = 0
        self.online = bool(C.EMBED_API_URL)

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.npy"

    def _api(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        """调用一次嵌入 API；偶发网络错误重试，持续失败才返回 None。

        ⚠ 这里必须重试：长实验要嵌入上万条文本，服务端偶发的
        ConnectionReset 若直接判失败，会让几小时的实验前功尽弃。
        只有连续 max_retries 次都失败才认定服务不可用，交回上层
        （严格模式下抛错，避免静默退回哈希向量）。
        """
        if not (self.online and requests is not None):
            return None
        payload = {"input": list(texts), "model": C.EMBED_MODEL}
        headers = {"Content-Type": "application/json"}
        if C.EMBED_API_KEY:
            headers["Authorization"] = f"Bearer {C.EMBED_API_KEY}"
        last: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(C.EMBED_API_URL, headers=headers,
                                  json=payload, timeout=120)
                r.raise_for_status()
                body = r.json()
                items = body.get("data", body)
                if (isinstance(items, list) and items
                        and isinstance(items[0], dict) and "index" in items[0]):
                    items = sorted(items, key=lambda d: d["index"])
                arr = np.asarray([d["embedding"] for d in items],
                                 dtype=np.float32)
                if arr.shape[0] != len(texts):
                    raise ValueError(
                        f"embedding count {arr.shape[0]} != {len(texts)}")
                self.n_api_calls += 1
                return arr
            except Exception as e:
                last = str(e)
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff * (2 ** attempt))
                    continue
        if self.verbose:
            print(f"  [warn] embedding API failed after "
                  f"{self.max_retries} attempts: {last}")
        return None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = [t if t is not None else "" for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        todo_idx: List[int] = []
        todo_txt: List[str] = []
        for i, t in enumerate(texts):
            if t in self._mem:
                out[i] = self._mem[t]
                continue
            p = self._cache_path(t)
            if p.exists():
                try:
                    v = np.load(p)
                    if v.shape[0] == self.dim:
                        out[i] = v
                        self._mem[t] = v
                        continue
                except Exception:
                    pass
            todo_idx.append(i)
            todo_txt.append(t)

        if todo_txt:
            t0 = time.time()
            for s in range(0, len(todo_txt), self.batch_size):
                chunk = todo_txt[s:s + self.batch_size]
                arr = self._api(chunk)
                from_api = arr is not None
                if arr is None:
                    if not self.allow_fallback:
                        raise RuntimeError(
                            "embedding API unavailable and fallback disabled")
                    arr = np.vstack([_hash_vec(t, self.dim) for t in chunk])
                    self.n_fallback += len(chunk)
                for j, t in enumerate(chunk):
                    v = l2_normalize(arr[j])
                    out[todo_idx[s + j]] = v
                    self._mem[t] = v
                    if from_api:
                        # 哈希回退向量不含语义，一旦写入缓存就会在后续实验中
                        # 被当作真实嵌入命中，永久污染结果——因此只缓存真值
                        p = self._cache_path(t)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        np.save(p, v)
            if self.verbose:
                print(f"  embedded {len(todo_txt)} texts in {time.time()-t0:.1f}s "
                      f"(api={self.n_api_calls}, fallback={self.n_fallback})")
            if self.n_fallback:
                print(f"  [WARN] 有 {self.n_fallback} 条向量来自哈希回退，"
                      f"不含语义——本次结果不能用于实验结论")
        return out
