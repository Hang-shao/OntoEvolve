"""ontoevolve.src.reasoner - OWL-DL 推理机封装（HermiT 优先，owlready2 自带推理器备选）。

HermiT 以命令行方式调用（每次一个 action，避免多动作语义歧义）：

    java -cp <hermit.jar:owlapi.jar:getopt.jar> \
         org.semanticweb.HermiT.cli.CommandLine -k <file:///...owl>

`-k` 检查 owl:Thing 的可满足性（= 本体一致性）；`-U` 列出不可满足的类。
仅在 `-k` 判定不一致时才额外跑一次 `-U` 取证据，稳定本体只需一次 JVM 启动。

HermiT jar 不可用时，退到 owlready2 0.51 自带的推理器（其内部也是 HermiT jar，
独立于 vendor/）。这一层只作为“换一个调用路径再问一次”，不改变判定强度；
调用失败一律返回 ok=False（无裁决），绝不读成“一致”。
owlready2 也带 Pellet jar，但该 jar 需要比本仓库所用 JDK 8 更新的运行时
（实测 UnsupportedClassVersionError），故不作为默认路径。
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from . import config as C

HERMIT_MAIN = "org.semanticweb.HermiT.cli.CommandLine"
# 自编译入口（见 tools/java/HermiTCheck.java），绕过 OWLAPI 的 parser SPI
HERMIT_ENTRY = "HermiTCheck"


@dataclass
class ReasonResult:
    """一次推理调用的结果。"""

    engine: str
    ok: bool                       # 调用本身是否成功（False 表示应降级）
    consistent: bool
    unsatisfiable: List[str] = field(default_factory=list)
    ms: float = 0.0
    raw: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Java / jar 探测
# ---------------------------------------------------------------------------
def _candidate_javas() -> List[str]:
    out: List[str] = []
    if C.JAVA_EXE:
        out.append(C.JAVA_EXE)
    env_home = os.getenv("JAVA_HOME", "").strip()
    if env_home:
        out.append(str(Path(env_home) / "bin" / "java.exe"))
        out.append(str(Path(env_home) / "bin" / "java"))
    found = shutil.which("java")
    if found:
        out.append(found)
    if os.name == "nt":
        pats = [
            r"C:\Program Files\Eclipse Adoptium\jdk-*\bin\java.exe",
            r"C:\Program Files\Eclipse Adoptium\jdk-*\*\bin\java.exe",
            r"C:\Program Files\Microsoft\jdk-*\bin\java.exe",
            r"C:\Program Files\Java\jdk*\bin\java.exe",
            r"C:\Program Files\Java\jre*\bin\java.exe",
            r"C:\Program Files\Zulu\zulu-*\bin\java.exe",
            r"C:\Program Files\BellSoft\LibericaJDK-*\bin\java.exe",
            r"C:\Program Files\Amazon Corretto\jdk*\bin\java.exe",
        ]
        for p in pats:
            out.extend(sorted(glob.glob(p)))
    else:
        out.extend(sorted(glob.glob("/usr/lib/jvm/*/bin/java")))
        out.append("/usr/bin/java")
    return out


def find_java() -> Optional[str]:
    """返回可用的 java 可执行文件路径（未找到返回 None）。"""
    for cand in _candidate_javas():
        if not cand:
            continue
        p = Path(cand)
        if not p.exists():
            # PATH 里的裸命令
            if shutil.which(cand):
                return cand
            continue
        try:
            r = subprocess.run([str(p), "-version"], capture_output=True,
                               timeout=20)
        except Exception:
            continue
        if r.returncode == 0 or b"version" in (r.stdout + r.stderr):
            return str(p)
    return None


def java_version(java: Optional[str] = None) -> str:
    j = java or find_java()
    if not j:
        return ""
    try:
        r = subprocess.run([j, "-version"], capture_output=True, timeout=20)
        txt = (r.stderr or r.stdout).decode("utf-8", "ignore")
        return txt.strip().splitlines()[0] if txt.strip() else ""
    except Exception:
        return ""


def _resolve_jar(pattern: str, default) -> Optional[Path]:
    """优先用配置路径，否则在 vendor/ 里按模式取版本号最大的一个。"""
    d = Path(default)
    if d.exists():
        return d
    hits = sorted(C.DIR_VENDOR.glob(pattern))
    return hits[-1] if hits else None


def hermit_jars() -> Optional[List[str]]:
    """HermiT 运行所需 jar；核心三个缺失时返回 None。

    vendor/ 下其余 jar（caffeine、guava、slf4j、jsr305 等 OWLAPI 运行时依赖）
    自动追加到 classpath，避免逐个登记。
    """
    hermit = _resolve_jar("org.semanticweb.hermit-*.jar", C.HERMIT_JAR)
    owlapi = _resolve_jar("owlapi-distribution-*.jar", C.OWLAPI_JAR)
    getopt = _resolve_jar("java-getopt-*.jar", C.GETOPT_JAR)
    if not (hermit and owlapi and getopt):
        return None
    core = [str(hermit), str(owlapi), str(getopt)]
    extra = sorted(str(p) for p in C.DIR_VENDOR.glob("*.jar")
                   if str(p) not in core)
    return core + extra


def hermit_available() -> bool:
    return find_java() is not None and hermit_jars() is not None


# ---------------------------------------------------------------------------
# HermiT
# ---------------------------------------------------------------------------
class HermiTReasoner:
    """HermiT（OWL 2 DL）命令行封装。"""

    name = "hermit"

    def __init__(self, java: Optional[str] = None,
                 jars: Optional[Sequence[str]] = None,
                 timeout_s: float = C.REASONER_TIMEOUT_S):
        self.java = java or find_java()
        self.jars = list(jars) if jars else hermit_jars()
        self.timeout_s = float(timeout_s)

    @property
    def available(self) -> bool:
        return bool(self.java) and bool(self.jars)

    @property
    def entry_class(self) -> Optional[Path]:
        """自编译入口 class（存在则优先使用）。"""
        p = C.DIR_BUILD / f"{HERMIT_ENTRY}.class"
        return p if p.exists() else None

    def _classpath(self) -> str:
        parts = [str(C.DIR_BUILD)] + list(self.jars)
        return os.pathsep.join(parts)

    def _run(self, owl_path: Path, action: str = "") -> ReasonResult:
        t0 = time.perf_counter()
        path = Path(owl_path).resolve()
        if self.entry_class is not None:
            args = [self.java, "-cp", self._classpath(), HERMIT_ENTRY, str(path)]
        else:
            args = [self.java, "-cp", os.pathsep.join(self.jars), HERMIT_MAIN,
                    action or "-k", path.as_uri()]
        try:
            r = subprocess.run(args, capture_output=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return ReasonResult(engine=self.name, ok=False, consistent=True,
                                ms=(time.perf_counter() - t0) * 1000.0,
                                error="timeout")
        except Exception as exc:                       # pragma: no cover
            return ReasonResult(engine=self.name, ok=False, consistent=True,
                                ms=(time.perf_counter() - t0) * 1000.0,
                                error=f"{type(exc).__name__}: {exc}")
        out = (r.stdout or b"").decode("utf-8", "ignore")
        err = (r.stderr or b"").decode("utf-8", "ignore")
        return ReasonResult(engine=self.name, ok=True, consistent=True,
                            ms=(time.perf_counter() - t0) * 1000.0,
                            raw=(out + "\n" + err).strip())

    @staticmethod
    def _parse(res: ReasonResult) -> None:
        """解析输出：CONSISTENT= / UNSAT: / ERROR= 或官方 CLI 的措辞。"""
        txt = res.raw or ""
        low = txt.lower()
        if "consistent=false" in low:
            res.consistent = False
        elif "consistent=true" in low:
            res.consistent = True
        elif "unsatisfiable" in low or "inconsistent" in low:
            res.consistent = False
        elif "satisfiable" in low:
            res.consistent = True
        else:
            # 推理机没有给出判定（解析异常/被拒）时不能当作“一致”
            res.consistent = True
            res.ok = False
            res.error = res.error or "no-verdict"
        res.unsatisfiable = [ln.strip()[len("UNSAT:"):]
                             for ln in txt.splitlines() if ln.strip().startswith("UNSAT:")]
        if "error=" in low:
            res.ok = False
            res.error = next((ln.strip() for ln in txt.splitlines()
                              if ln.strip().lower().startswith("error=")), "error")

    def check(self, owl_path) -> ReasonResult:
        """一致性检查（自编译入口一次调用即可拿到不可满足类）。"""
        res = self._run(owl_path, "-k")
        if not res.ok:
            return res
        self._parse(res)
        # 官方 CLI 的 -k 不列出不可满足类，补一次 -U
        if not res.consistent and self.entry_class is None:
            u = self._run(owl_path, "-U")
            res.ms += u.ms
            res.unsatisfiable = self._parse_unsat(u.raw)
            res.raw = res.raw + "\n" + u.raw
        return res

    @staticmethod
    def _parse_unsat(raw: str) -> List[str]:
        out: List[str] = []
        for line in (raw or "").splitlines():
            s = line.strip()
            if not s:
                continue
            low = s.lower()
            if low.startswith("unsatisfiable") or low.startswith("consistent"):
                continue
            if "owl:nothing" in low:
                continue
            out.append(s)
        return out


# ---------------------------------------------------------------------------
# owlready2 自带推理器（HermiT jar 不可用时的备选路径）
# ---------------------------------------------------------------------------
class Owlready2Reasoner:
    """owlready2 0.51 内置推理器；HermiT 命令行路径不可用时作为备选。

    owlready2 的 sync_reasoner 在执行前把本体导成临时文件再调它自带的 jar，
    所以这是“换一个调用路径”，不是换一种判定强度。
    """

    name = "owlready2"

    def __init__(self, java: Optional[str] = None,
                 timeout_s: float = C.REASONER_TIMEOUT_S):
        self.java = java or find_java()
        self.timeout_s = float(timeout_s)

    @property
    def available(self) -> bool:
        return self.java is not None

    def check(self, owl_path) -> ReasonResult:
        t0 = time.perf_counter()
        try:
            import owlready2
        except Exception as exc:                       # pragma: no cover
            return ReasonResult(engine=self.name, ok=False, consistent=True,
                                error=f"owlready2 missing: {exc}")
        if self.java:
            owlready2.JAVA_EXE = self.java
        try:
            world = owlready2.World()
            # owlready2 0.51 会把 file:///D:/... 误解析为 /D:/...，这里直接传本地路径
            onto = world.get_ontology(str(Path(owl_path).resolve())).load()
            with onto:
                # owlready2 0.51 只接受 x=本体；传 world= 会抛 TypeError
                owlready2.sync_reasoner(onto, infer_property_values=True)
            unsat = [c.name for c in onto.inconsistent_classes()]
            return ReasonResult(engine=self.name, ok=True, consistent=True,
                                unsatisfiable=unsat,
                                ms=(time.perf_counter() - t0) * 1000.0,
                                raw="consistent")
        except getattr(owlready2, "OwlReadyInconsistentOntologyError", ()):
            # 不一致是有效裁决（该异常的 message 为空，只能按类型判）
            return ReasonResult(engine=self.name, ok=True, consistent=False,
                                ms=(time.perf_counter() - t0) * 1000.0,
                                raw="inconsistent")
        except Exception as exc:
            # 调用本身失败（缺 jar、参数不兼容、超时、JVM 版本…）不是判定：
            # 返回 ok=False 交给下一级，“检查失败”永远不等于“一致”。
            msg = f"{type(exc).__name__}: {exc}"
            return ReasonResult(engine=self.name, ok=False, consistent=True,
                                ms=(time.perf_counter() - t0) * 1000.0,
                                raw=msg, error=msg)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
# 兼容旧参数名：engine="pellet" 仍可用，指向 owlready2 自带推理器
_ENGINE_ALIAS = {"pellet": "owlready2"}


def get_reasoner(prefer: Optional[str] = None, **kw):
    """按优先级取得可用推理机；都不可用返回 None。"""
    first = _ENGINE_ALIAS.get(prefer or C.REASONER, prefer or C.REASONER)
    order = [first]
    for alt in ("hermit", "owlready2"):
        if alt not in order:
            order.append(alt)
    for pref in order:
        if pref in ("none", "off", "z3"):
            continue
        if pref == "hermit":
            r = HermiTReasoner(**kw)
            if r.available:
                return r
        elif pref == "owlready2":
            r = Owlready2Reasoner(**kw)
            if r.available:
                return r
    return None
