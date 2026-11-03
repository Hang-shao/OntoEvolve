"""ontoevolve.src.local_embed - 本地 Qwen3-Embedding 嵌入器（与 API 同模型、同维度）。

背景：轨道 1—3 的嵌入经 API（Qwen3-Embedding-0.6B, d=1024）得到。轨道 4 启动时
该 API 端点返回 401，为避免"换模型导致跨轨道不可比"，这里用同一模型权重的本地
推理替代：last-token pooling + L2 归一化，输出维度 1024，接口与 src.embedding.Embedder
保持一致（embed(list)->np.ndarray），并复用同一套磁盘缓存键（文本 sha1）。

缓存目录与 Embedder 相同（data/cache/<sha1[:2]>/<sha1>.npy），因此一旦 API 恢复，
同一批文本可直接命中缓存或互相替换，不影响结果可比性。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config as C
from .embedding import l2_normalize

_CANDIDATES = [
    os.getenv("ONTOEVOLVE_LOCAL_EMBED_DIR", ""),
    str(Path(C.WORKSPACE) / "shared_env" / "models" / "Qwen3-Embedding-0.6B"),
    str(Path.home() / ".cache" / "ontoevolve" / "Qwen3-Embedding-0.6B"),
]
DEFAULT_MODEL_DIR = next(
    (Path(p) for p in _CANDIDATES if p and (Path(p) / "config.json").exists()),
    Path(_CANDIDATES[1]))


class LocalEmbedder:
    """本地 Qwen3-Embedding（CPU 推理）。"""

    def __init__(self, model_dir: Optional[str] = None, *,
                 cache_dir: Optional[Path] = None, batch_size: int = 32,
                 dim: int = 1024, verbose: bool = False, max_len: int = 256):
        model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        if not (model_dir / "config.json").exists():
            raise RuntimeError(f"本地嵌入模型缺失：{model_dir}")
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(max(1, min(16, os.cpu_count() or 8)))
        self.torch = torch
        self.model_dir = model_dir
        self.cache_dir = Path(cache_dir or C.DIR_CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.dim = dim
        self.verbose = verbose
        self.max_len = max_len
        self.n_api_calls = 0          # 与 Embedder 接口对齐（本地推理不计 API 调用）
        self.n_local_calls = 0
        self.online = False
        self._mem: Dict[str, np.ndarray] = {}

        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.tok.padding_side = "left"
        try:
            self.model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32)
        except TypeError:            # 旧版 transformers 的形参名
            self.model = AutoModel.from_pretrained(str(model_dir),
                                                   torch_dtype=torch.float32)
        self.model.eval()
        if self.verbose:
            n = sum(p.numel() for p in self.model.parameters())
            print(f"[local-embed] {model_dir.name} params={n/1e6:.1f}M dim={dim}")

    # ---- 缓存 ---------------------------------------------------------------
    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.npy"

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        if text in self._mem:
            return self._mem[text]
        p = self._cache_path(text)
        if p.exists():
            try:
                v = np.load(p).astype(np.float32)
                if v.shape[-1] == self.dim:
                    self._mem[text] = v
                    return v
            except Exception:
                pass
        return None

    def _put_cached(self, text: str, v: np.ndarray) -> None:
        p = self._cache_path(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, v.astype(np.float32))
        self._mem[text] = v

    # ---- 推理 ---------------------------------------------------------------
    def _forward(self, texts: Sequence[str]) -> np.ndarray:
        torch = self.torch
        enc = self.tok(list(texts), padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        mask = enc["attention_mask"]
        # left padding 下必须显式给出 position_ids：Qwen3 的 RoPE 默认按序列物理位置
        # 编号，若直接传 input_ids，短句的真实 token 会被 padding 整体平移，末 token
        # 向量随同批最长句的长度变化（实测同一句单独编码与混批编码余弦仅 0.308，
        # 与远程嵌入服务相差 0.31）。此处按 mask 累积得到"从 0 开始"的位置编号。
        pos = (mask.cumsum(dim=1) - 1).clamp(min=0)
        with torch.no_grad():
            out = self.model(**enc, position_ids=pos)
        last = out.last_hidden_state                     # [B, L, H]
        # 左填充下"末 token"的物理下标恒为序列长度 −1；不能用 mask.sum()-1，
        # 那是右填充的写法。原实现用 mask.sum()-1，在混批（本批最长的句子与其
        # 余句子长度不同）时取到的是倒数第几个真实 token，同一句子的向量会随
        # 同批伙伴的长度漂移（实测单独编码与混批编码余弦仅 0.308）。
        idx = mask.shape[1] - 1 - mask.flip(dims=[1]).argmax(dim=1)
        pooled = last[torch.arange(last.shape[0]), idx]
        return l2_normalize(pooled.numpy().astype(np.float32))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        todo: List[int] = []
        for i, t in enumerate(texts):
            v = self._get_cached(t)
            if v is None:
                todo.append(i)
            else:
                out[i] = v
        # 按字符长度排序后再分批：把长噪声转写（ZhCCVi 可达 2000+ 字）集中到最后，
        # 避免短句批次被长文本的 padding 拖慢（实测吞吐差一个数量级）。
        todo.sort(key=lambda i: len(texts[i]))
        for s in range(0, len(todo), self.batch_size):
            chunk = todo[s:s + self.batch_size]
            vecs = self._forward([texts[i] for i in chunk])
            self.n_local_calls += len(chunk)
            for j, i in enumerate(chunk):
                out[i] = vecs[j]
                self._put_cached(texts[i], vecs[j])
            if self.verbose and (s // self.batch_size) % 10 == 0:
                print(f"  [local-embed] {s + len(chunk)}/{len(todo)}", flush=True)
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
