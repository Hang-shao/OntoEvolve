"""测试的环境探测：干净克隆里往往缺语料、推理机、嵌入服务或构建基座模块。

缺什么就跳过对应的用例，而不是让它失败——这样 `pytest` 在没有 data/、vendor/、
.env 的机器上仍然是绿的，跳过原因也写在报告里。PMC 语料由
`tests/test_pmc.py` 的 fixture 自行捕获并跳过（能报出缺失的目录）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import config as C


def has_legacy() -> bool:
    """模块 A/B/C（仓库内 legacy/）是否可用。"""
    base = Path(C.LEGACY_EXP)
    return (base / "__init__.py").exists() or (base / "src" / "__init__.py").exists()


def has_reasoner() -> bool:
    """JDK + HermiT jar 是否就位。"""
    try:
        from src.reasoner import hermit_available
        return bool(hermit_available())
    except Exception:                                            # noqa: BLE001
        return False


def has_embeddings() -> bool:
    """是否有嵌入服务或本地模型（否则只能拿到哈希回退向量）。"""
    if C.EMBED_API_URL:
        return True
    try:
        from src.local_embed import DEFAULT_MODEL_DIR
        return (Path(DEFAULT_MODEL_DIR) / "config.json").exists()
    except Exception:                                            # noqa: BLE001
        return False


requires_legacy = pytest.mark.skipif(
    not has_legacy(), reason="模块 A/B/C 缺失（设 ONTOEVOLVE_LEGACY_EXP 指向其目录）")
requires_reasoner = pytest.mark.skipif(
    not has_reasoner(), reason="未找到 JDK / HermiT（vendor/ 或 ONTOEVOLVE_JAVA_EXE）")
requires_embeddings = pytest.mark.skipif(
    not has_embeddings(), reason="未配置嵌入服务或本地模型（见 .env.example）")
