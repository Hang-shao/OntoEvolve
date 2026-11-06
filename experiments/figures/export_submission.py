"""把 figures/ 里的 8 张图导出成期刊上传用的独立文件（Fig1…Fig8）。

Springer 的图件要求：矢量优先 EPS/PDF，位图用 TIFF，命名 `Fig<编号>`；组合图（含线条与文字）
最低 600 dpi。仓库里的 png 已按 600 dpi 渲染，这里只做重命名与位图转 TIFF（LZW 无损）。

两条防错：
* TIFF 里写的分辨率是图片**实际**分辨率（按 174 mm 印宽折算），不做无条件 600 dpi 标注；
* PDF 比 PNG 旧说明底色/内容没跟上最新一版图，跳过并告警，避免把旧矢量图一起上传。

用法：
    python experiments/figures/export_submission.py                  # 默认 figures/ → figures/submission/
    python experiments/figures/export_submission.py --out path/to/dir
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"

# 图号 -> 仓库里的文件名（不含扩展名）
FIGS = {
    1: "fig1_architecture",
    2: "fig2_lifecycle",
    3: "fig2_drift_f1",
    4: "fig3_component_ablation",
    5: "fig4_alert_budget",
    6: "fig5_cost_speedup",
    7: "fig6_qa_comparison",
    8: "fig7_antifraud_comparison",
}
MIN_DPI = 600
PRINT_W_MM = 174.0          # Springer 双栏正文宽，用来把像素折算成实际 dpi
STALE_PDF_S = 60            # PDF 比 PNG 旧超过 60 s 视为未同步


def main() -> int:
    ap = argparse.ArgumentParser(description="导出期刊上传用的 Fig1…FigN 文件")
    ap.add_argument("--src", default=str(FIGDIR))
    ap.add_argument("--out", default=str(FIGDIR / "submission"))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    missing, low_dpi, stale_pdf = [], [], []
    for n, stem in FIGS.items():
        png, pdf = src / f"{stem}.png", src / f"{stem}.pdf"
        if not png.exists():
            missing.append(str(png))
            continue
        im = Image.open(png)
        dpi = im.info.get("dpi")
        native = float(min(dpi)) if dpi else im.size[0] / (PRINT_W_MM / 25.4)
        if native < MIN_DPI - 1:
            low_dpi.append((png.name, round(native), "%dx%d px" % im.size))
        write_dpi = MIN_DPI if native >= MIN_DPI - 1 else int(round(native))
        # 位图：TIFF（LZW 无损），分辨率如实写入
        im.convert("RGB").save(out / f"Fig{n}.tif", dpi=(write_dpi, write_dpi),
                               compression="tiff_lzw")
        copied = False
        if pdf.exists():
            if pdf.stat().st_mtime < png.stat().st_mtime - STALE_PDF_S:
                stale_pdf.append((n, pdf.name))
            else:
                shutil.copyfile(pdf, out / f"Fig{n}.pdf")
                copied = True
        print("Fig%d: %s.png -> Fig%d.tif" % (n, stem, n)
              + ("  (+ Fig%d.pdf)" % n if copied else ""))

    if missing:
        print("[warn] 缺文件:", ", ".join(missing))
    if low_dpi:
        print("[warn] 分辨率低于 %d dpi（TIFF 已按实际值标注）:" % MIN_DPI, low_dpi)
    if stale_pdf:
        print("[warn] PDF 比 PNG 旧、已跳过（需重新导出矢量版）:", stale_pdf)
    print("输出目录:", out)
    return 0 if not missing and not low_dpi and not stale_pdf else 1


if __name__ == "__main__":
    raise SystemExit(main())
