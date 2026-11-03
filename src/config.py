"""ontoevolve.src.config - 全局配置与超参数。

凭据通过环境变量或仓库根目录的 .env 读取，不写入代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析：KEY=VALUE，不覆盖已有环境变量。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env_path(key: str, default: Path) -> Path:
    """读路径型环境变量；空值按未设置处理（留空的项不该变成当前目录）。"""
    raw = os.getenv(key, "").strip()
    return Path(raw) if raw else default


DATA_ROOT = _env_path("ONTOEVOLVE_DATA_ROOT", WORKSPACE / "data")
LEGACY_EXP = _env_path("ONTOEVOLVE_LEGACY_EXP", ROOT / "legacy")

# ---- 服务端点 ---------------------------------------------------------------
EMBED_API_URL = os.getenv("ONTOEVOLVE_EMBED_API_URL", "").strip()
EMBED_API_KEY = os.getenv("ONTOEVOLVE_EMBED_API_KEY", "").strip()
EMBED_MODEL = os.getenv("ONTOEVOLVE_EMBED_MODEL", "Qwen3-Embedding-0.6B")
EMBED_DIM = int(os.getenv("ONTOEVOLVE_EMBED_DIM", "1024"))

CHAT_API_URL = os.getenv("ONTOEVOLVE_CHAT_API_URL", "").strip()
CHAT_API_KEY = os.getenv("ONTOEVOLVE_CHAT_API_KEY", EMBED_API_KEY).strip()
CHAT_MODEL = os.getenv("ONTOEVOLVE_CHAT_MODEL", "Qwen3-32B")

# ---- Java / OWL-DL 推理机（实验 3）----------------------------------------
DIR_VENDOR = ROOT / "vendor"
# 留空时自动探测：PATH → JAVA_HOME → 常见 JDK 安装目录
JAVA_EXE = os.getenv("ONTOEVOLVE_JAVA_EXE", "").strip()
HERMIT_JAR = _env_path("ONTOEVOLVE_HERMIT_JAR",
                        DIR_VENDOR / "org.semanticweb.hermit-1.4.5.519.jar")
OWLAPI_JAR = _env_path("ONTOEVOLVE_OWLAPI_JAR",
                       DIR_VENDOR / "owlapi-distribution-4.5.25.jar")
GETOPT_JAR = _env_path("ONTOEVOLVE_GETOPT_JAR",
                       DIR_VENDOR / "java-getopt-1.0.13.jar")
# 推理机选择：hermit（默认，OWL 2 DL 完整判定）| pellet（owlready2 内置）| none
REASONER = os.getenv("ONTOEVOLVE_REASONER", "hermit").strip().lower()
REASONER_TIMEOUT_S = float(os.getenv("ONTOEVOLVE_REASONER_TIMEOUT_S", "120"))
# 导出 OWL 时使用的 IRI 基址
IRI_BASE = os.getenv("ONTOEVOLVE_IRI_BASE", "http://ontoevolve.local/ontology#")

# ---- 目录 -------------------------------------------------------------------
DIR_DATA = ROOT / "data"
DIR_RAW = DIR_DATA / "raw"
DIR_PROCESSED = DIR_DATA / "processed"
DIR_ONTOLOGY = DIR_DATA / "ontology"
DIR_TEMP = DIR_DATA / "temp"
DIR_ANNOTATION = DIR_DATA / "annotation"
DIR_CACHE = DIR_DATA / "cache"
# 本地 Qwen3-Embedding 的向量缓存与 API 嵌入器分开：两者维度同为 1024，
# 共用一个目录会把不同模型的向量静默复用（实测共用目录时 240 条批内只有
# 62 个不同向量，同类型实例余弦被拉到 0.98 的常数）。
DIR_CACHE_LOCAL = DIR_DATA / "cache_qwen3_local_v2"
DIR_CACHE_API = DIR_DATA / "cache_api_track4"
DIR_RESULTS = ROOT / "results"
DIR_LOGS = ROOT / "logs"
DIR_BUILD = ROOT / "build" / "classes"

for _d in (DIR_RAW, DIR_PROCESSED, DIR_ONTOLOGY, DIR_TEMP, DIR_ANNOTATION,
           DIR_CACHE, DIR_CACHE_LOCAL, DIR_CACHE_API, DIR_RESULTS, DIR_LOGS, DIR_BUILD):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DriftParams:
    """模块 D 阈值 + 受控数据流规模参数（初值取自参考配置，随后网格标定）。"""

    # 信号 1：语义
    theta_sem: float = 0.25      # 与最近已有锚点的最小余弦距离 > theta_sem
    theta_shift: float = 0.10    # 已有概念质心漂移阈值
    min_new_size: int = 3        # 候选新概念在 κ 个批次内的最少实例数
    kappa: int = 2               # 涌现需连续出现的批次数
    link_sim: float = 0.75       # 实例链接到概念的相似度下限（可用 D_0 自动标定）

    # 信号 2：统计
    window_w: int = 4            # W：淘汰判定的连续窗口
    eps_ext: float = 1.5         # 窗口内平均新增实例数下限
    theta_stat: float = 0.60     # 统计信号告警阈值
    decline_tol: float = 0.0     # 内部语义权重需持续下降

    # 信号 3：公理
    theta_red: float = 1.00      # 公理签名 L1 距离阈值（M=18 维，量级 ~1-3）
    tau_conf: float = 0.60       # 公理置信度下限

    # 融合
    w_sem: float = 0.40
    w_stat: float = 0.25
    w_ax: float = 0.35
    theta_score: float = 0.45
    delay_tol: int = 2           # 允许的检测延迟（批次）

    # 受控数据流
    n_batch: int = 36
    base_rate: float = 4.0
    n_init_per_leaf: int = 20
    pool_per_leaf: int = 40
    n_emergence: int = 3
    n_extinction: int = 3
    n_redefinition: int = 3
    noise_ratio: float = 0.15
    gradual_ratio: float = 0.34
    periodic_ratio: float = 0.34
    lowfreq_ratio: float = 0.34

    # 评估
    # 参考配置用 5 次重复；配对 Wilcoxon 在 n=5 时最小双尾 p=0.0625 无法达到
    # p<0.05，因此默认取 10 次（同时满足“至少 5 次”的要求）。
    n_repeats: int = 10
    seed: int = 42
    n_bootstrap: int = 1000

    def copy(self, **kw) -> "DriftParams":
        d = asdict(self)
        d.update(kw)
        return DriftParams(**d)


PARAMS = DriftParams()

# 模块 B 粒度项的目标簇规模下限（s*，见 build_ontology.s_star_dendrogram_gap）
HP_S_STAR_MIN = 5.0

# 公理签名维度参考值（轨道 1 合成词表：6 关系 + 6 属性 + 6 结构 = 18）。
# 跨领域迁移 θ_red 时按 M/M_ref 缩放。
AXIOM_SIGNATURE_M_REF = 18
