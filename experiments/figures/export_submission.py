"""把 figures/ 里的 8 张图导出成期刊上传用的独立文件（Fig1…Fig8）。

Springer 的图件要求：矢量优先 EPS/PDF，位图用 TIFF，命名 `Fig<编号>`；组合图（含线条与文字）
最低 600 dpi。仓库里的 png 已按 600 dpi 渲染，这里只做重命名与位图转 TIFF（LZW 无损）。

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


def main() -> int:
    ap = argparse.ArgumentParser(description="导出期刊上传用的 Fig1…FigN 文件")
    ap.add_argument("--src", default=str(FIGDIR))
    ap.add_argument("--out", default=str(FIGDIR / "submission"))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    missing, low_dpi = [], []
    for n, stem in FIGS.items():
        png, pdf = src / f"{stem}.png", src / f"{stem}.pdf"
        if not png.exists():
            missing.append(str(png))
            continue
        im = Image.open(png)
        dpi = im.info.get("dpi") or (0, 0)
        if min(dpi) < MIN_DPI - 1:
            low_dpi.append((png.name, dpi))
        # 位图：TIFF（LZW 无损，写入 600 dpi 元数据）
        im.convert("RGB").save(out / f"Fig{n}.tif", dpi=(MIN_DPI, MIN_DPI),
                               compression="tiff_lzw")
        if pdf.exists():
            shutil.copyfile(pdf, out / f"Fig{n}.pdf")
        print(f"Fig{n}: {stem}.png -> Fig{n}.tif" + ("  (+ Fig%d.pdf)" % n if pdf.exists() else ""))

    if missing:
        print("[warn] 缺文件:", ", ".join(missing))
    if low_dpi:
        print("[warn] 分辨率低于 %d dpi:" % MIN_DPI, low_dpi)
    print("输出目录:", out)
    return 0 if not missing and not low_dpi else 1


if __name__ == "__main__":
    raise SystemExit(main())
