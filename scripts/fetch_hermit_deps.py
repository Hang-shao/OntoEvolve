"""下载 OWL-DL 推理机所需的 Java 依赖到 vendor/（可重复执行）。

用法：python scripts/fetch_hermit_deps.py [--owlapi 5.5.0] [--hermit 1.4.5.519]

HermiT 与 OWLAPI 都不发布带全部依赖的 fat jar，且**版本必须严格匹配**
（HermiT 1.4.5.519 编译期用的是 OWLAPI 5.x 的 `java.util.Optional` 签名；
搭配 4.5.x 会 NoSuchMethodError，搭配错误的 rdf4j 会让 OWLAPI 的
Injector 拿到 null provider 而 NPE）。
因此这里直接读 OWLAPI 的 pom，按其声明的版本逐个下载，避免手工拼版本。
"""
from __future__ import annotations

import argparse
import re
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
VENDOR.mkdir(parents=True, exist_ok=True)

M2 = "https://repo1.maven.org/maven2/"
UA = {"User-Agent": "Mozilla/5.0"}
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
HERMIT_GROUP = "net/sourceforge/owlapi/org.semanticweb.hermit/"


def _get(url: str) -> str:
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                      timeout=60, context=CTX).read().decode(
            "utf-8", "ignore")
    except Exception as exc:
        print("!! fetch failed:", url, exc)
        return ""


def _download(url: str, dst: Path) -> int:
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                      timeout=240, context=CTX).read()
    except Exception as exc:
        print("!! download failed:", url, exc)
        return 0
    dst.write_bytes(data)
    return len(data)


def _latest(group_path: str, version_filter=None) -> tuple[str, str]:
    html = _get(M2 + group_path)
    vers = [v for v in re.findall(r'href="([^"/]+)/"', html) if not v.startswith(".")]
    if version_filter:
        vers = [v for v in vers if version_filter(v)]
    for v in reversed(sorted(vers)):
        files = _get(M2 + group_path + v + "/")
        jars = [j for j in re.findall(r'href="([^"]+\.jar)"', files)
                if not any(k in j for k in ("sources", "javadoc", "tests"))]
        if jars:
            return v, jars[0]
    return "", ""


def parse_pom(text: str) -> list[tuple[str, str, str]]:
    """返回 [(group, artifact, version|'')]，跳过 test/provided 作用域。"""
    out = []
    for m in re.finditer(r"<dependency>(.*?)</dependency>", text, re.S):
        blk = m.group(1)
        g = re.search(r"<groupId>(.*?)</groupId>", blk)
        a = re.search(r"<artifactId>(.*?)</artifactId>", blk)
        v = re.search(r"<version>(.*?)</version>", blk)
        s = re.search(r"<scope>(.*?)</scope>", blk)
        if not g or not a:
            continue
        if s and s.group(1) in ("test", "provided", "system"):
            continue
        out.append((g.group(1), a.group(1), v.group(1) if v else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owlapi", default="5.5.0")
    ap.add_argument("--hermit", default="1.4.5.519")
    args = ap.parse_args()

    # 每次全量重建，避免旧版本残留造成 classpath 冲突
    for old in list(VENDOR.glob("*.jar")) + list(VENDOR.glob("*.bak")):
        old.unlink()
        print("removed", old.name)

    pom_url = (f"{M2}net/sourceforge/owlapi/owlapi-distribution/{args.owlapi}/"
               f"owlapi-distribution-{args.owlapi}.pom")
    deps = parse_pom(_get(pom_url))
    deps.append(("net.sourceforge.owlapi", "owlapi-distribution", args.owlapi))
    deps.append(("net.sourceforge.owlapi", "org.semanticweb.hermit", args.hermit))
    deps.append(("gnu.getopt", "java-getopt", "1.0.13"))
    deps.append(("javax.inject", "javax.inject", "1"))
    deps.append(("commons-io", "commons-io", "2.11.0"))
    deps.append(("org.slf4j", "slf4j-simple", "1.7.30"))   # 让 Injector 的报错可见

    for group, art, ver in deps:
        gpath = group.replace(".", "/") + "/" + art + "/"
        if not ver:
            ver, _ = _latest(gpath)
            if not ver:
                print("!! cannot resolve version:", group, art)
                continue
        jar = f"{art}-{ver}.jar"
        url = f"{M2}{gpath}{ver}/{jar}"
        n = _download(url, VENDOR / jar)
        print(f"{'saved' if n else 'MISS '} {jar} {n}")

    print("\nvendor/:", len(list(VENDOR.glob('*.jar'))), "jars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
