"""Generate the report's figures from our own measurements.

Everything plotted here is data this project measured; nothing is copied from the
paper's figures.  Output is PDF (vector) so it drops straight into LaTeX.

Three figures:
  fig_rate_comparison  -- measured compression rate by method, per modality
  fig_reproduction     -- measured vs published, with the agreement gap
  fig_image_accounting -- how the image denominator is built up
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "report", "figures")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})

# ---------------------------------------------------------------- measured data
# Every number below is a measurement from this project.
DATA = {
    "Text (MeDAL)": {
        "ours": [("LMCompress* (ours)", 9.0410), ("+ LoRA (ours)", 9.3529)],
        "published": [("LMCompress*", 9.08), ("+ LoRA", 9.06)],
    },
    "Audio (LJSpeech sample)": {
        "ours": [("Our port", 19.82), ("Reference impl.", 19.94)],
        "published": [("LMCompress (LJSpeech)", 18.26)],
    },
    "Image (1 PNG)": {
        "ours": [("iGPT, official path", 15.5144)],
        "published": [("LMCompress-igpt (CLIC)", 14.62)],
    },
    "Video (4K clip)": {
        "ours": [("iGPT per frame", 0.0042)],
        "published": [("LMCompress-igpt (high-res)", 15.31)],
    },
}


def fig_reproduction() -> None:
    """Measured against published, per modality, on a log scale."""
    labels, ours, published = [], [], []
    for modality, d in DATA.items():
        # use the primary (first) entry of each
        labels.append(modality)
        ours.append(d["ours"][0][1])
        published.append(d["published"][0][1])

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    b1 = ax.bar(x - w / 2, published, w, label="Published", color="#9ecae1",
                edgecolor="#3182bd", linewidth=0.6)
    b2 = ax.bar(x + w / 2, ours, w, label="This reproduction", color="#fdae6b",
                edgecolor="#e6550d", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Compression rate (%)  — lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.legend(loc="upper right", frameon=False)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.2f}" if h >= 1 else f"{h:.4f}",
                        (r.get_x() + r.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 2.5),
                        ha="center", fontsize=7)
    ax.set_title("Measured versus published compression rate "
                 "(log scale; image/video differ in denominator convention)")
    fig.savefig(os.path.join(FIGDIR, "fig_reproduction.pdf"))
    plt.close(fig)
    print("  wrote fig_reproduction.pdf")


def fig_methods_per_modality() -> None:
    """Method-by-method comparison within each modality, in four panels."""
    panels = [
        ("Text (MeDAL)", [
            ("LMCompress*", 9.0410, "#e6550d"),
            ("+ LoRA", 9.3529, "#e6550d"),
            ("brotli q11", 28.918, "#3182bd"),
            ("lzma 9e", 31.850, "#3182bd"),
            ("zstd -22", 32.119, "#3182bd"),
            ("zlib 9", 35.155, "#3182bd"),
        ]),
        ("Audio (one clip)", [
            ("Our port", 19.82, "#e6550d"),
            ("Reference impl.", 19.94, "#e6550d"),
            ("OptimFROG", 27.311, "#3182bd"),
            ("WavPack", 35.303, "#3182bd"),
            ("FLAC", 45.334, "#3182bd"),
            ("zlib 9", 62.627, "#3182bd"),
        ]),
        ("Image (one PNG)", [
            ("iGPT (official)", 15.5144, "#e6550d"),
            ("WebP", 48.279, "#3182bd"),
            ("JPEG-2000", 53.252, "#3182bd"),
            ("PNG", 84.381, "#3182bd"),
        ]),
        ("Video (60 frames of 4K)", [
            ("iGPT per frame", 0.0042, "#e6550d"),
            ("H.265 crf 0", 22.862, "#3182bd"),
            ("H.264 crf 0", 36.866, "#3182bd"),
            ("FFV1 lvl 3", 40.167, "#3182bd"),
            ("zstd -22", 51.035, "#3182bd"),
        ]),
    ]
    # 2x2 so the panel titles have room; the top row of each panel is the model.
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
    for ax, (title, rows) in zip(axes.ravel(), panels):
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        cols = [r[2] for r in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.4, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("compression rate (%), lower is better", fontsize=7.5)
        ax.set_title(title, fontsize=8.5)
        for yi, v in zip(y, vals):
            ax.annotate(f"{v:.2f}" if v >= 1 else f"{v:.4f}",
                        (v, yi), textcoords="offset points", xytext=(2.5, -1.5),
                        fontsize=6.5, va="center")
        ax.set_xlim(0, max(vals) * 1.34)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Orange = model under reproduction; blue = classical baselines.  "
                 "WARNING: the image and video bars are not like-for-like with the "
                 "baselines (different denominator convention).",
                 fontsize=7.5, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_methods_per_modality.pdf"))
    plt.close(fig)
    print("  wrote fig_methods_per_modality.pdf")


def fig_image_accounting() -> None:
    """Where the image denominator comes from, and what the numerator codes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))

    # left: the three candidate denominators
    names = ["source\npixels", "source RGB\nchannel samples", "denominator\nused"]
    vals = [1957388, 5872164, 5870592]
    cols = ["#c6dbef", "#9ecae1", "#3182bd"]
    ax1.bar(range(3), vals, color=cols, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(names, fontsize=7)
    ax1.set_ylabel("samples", fontsize=8)
    ax1.set_title("The denominator is source channel samples\n"
                  "(5 872 164 minus 1 572 lost to floor division)", fontsize=8)
    for i, v in enumerate(vals):
        ax1.annotate(f"{v:,}", (i, v), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=7)

    # right: numerator object vs denominator object
    ax2.axis("off")
    ax2.set_title("Numerator and denominator describe\ndifferent objects", fontsize=8)
    ax2.text(0.02, 0.86, "Numerator", fontsize=8, fontweight="bold")
    ax2.text(0.02, 0.74,
             "5 733 chunks x 1 024 tokens\n= 5 870 592 palette indices\n"
             "(a 32x32, 512-colour surrogate)",
             fontsize=7.5, va="top")
    ax2.text(0.02, 0.44, "Denominator", fontsize=8, fontweight="bold")
    ax2.text(0.02, 0.32,
             "full-resolution source\nchannel samples\n"
             "(the actual image)",
             fontsize=7.5, va="top")
    ax2.annotate("", xy=(0.5, 0.18), xytext=(0.5, 0.60),
                 arrowprops=dict(arrowstyle="<->", color="#e6550d", lw=1.2))
    ax2.text(0.53, 0.36, "not the\nsame object", fontsize=7.5,
             color="#e6550d", va="center")
    fig.savefig(os.path.join(FIGDIR, "fig_image_accounting.pdf"))
    plt.close(fig)
    print("  wrote fig_image_accounting.pdf")


def main() -> int:
    os.makedirs(FIGDIR, exist_ok=True)
    print(f"writing figures to {FIGDIR}")
    fig_reproduction()
    fig_methods_per_modality()
    fig_image_accounting()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
