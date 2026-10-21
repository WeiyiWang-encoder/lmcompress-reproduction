"""Finalise the video numbers across the documents.

Video is now fully measured: 60 frames, all four classical baselines plus the paper's
own iGPT-per-frame method.  This script replaces the provisional figures in the audit
and the summary table in the Chinese report with the final ones, and asserts each
replacement matched so a silent no-op cannot slip through.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

EDITS = [
    # ---- audit: provisional -> final iGPT row ------------------------------- #
    (
        AUDIT,
        "| frames coded | 25–60 (see `results/video_igpt.jsonl`) |",
        "| frames coded | **60 (all)** |",
    ),
    (
        AUDIT,
        "| distinct palette ids used per frame | 102–106 |",
        "| distinct palette ids used per frame | 102–109 |",
    ),
    (
        AUDIT,
        "| model code length | **≈371 bytes/frame** |",
        "| model code length | **22 293.9 bytes total, ≈371 bytes/frame** |",
    ),
    (
        AUDIT,
        "| rate, paper convention (÷ frame pixels) | **0.0042 %** |",
        "| rate, paper convention (÷ frame pixels) | **0.004200 %** |",
    ),
    (
        AUDIT,
        "| **exact-reconstruction completion (on RGB)** | **≈73 %** |",
        "| **exact-reconstruction completion (on RGB)** | "
        "**73.0546 %** (residual 1 163 389 756 B) |",
    ),
    # ---- report §8: video row ----------------------------------------------- #
    (
        REPORT,
        "| 视频 | ImageGPT | **未测** | Y4M 解析器已验证（4096×2160，60 帧） |",
        "| 视频基线 | h.265/h.264/FFV1/zstd | **是（吻合度最高）** | 22.862 / 36.866 / 40.167 / 51.035% vs 论文 21.69 / 37.42 / 40.16 / 51.04%；顺序完全一致，最大偏差 1.17 pp，FFV1 与 zstd 吻合 <0.01 pp |\n"
        "| 视频 | ImageGPT（iGPT 逐帧） | accounting 口径接近；**非源视频无损** | 0.0042% vs 论文 15.31%；每帧 4096×2160→1024 token（约 8640× 缩减）；精确重建补全 **73.05%**，比所有经典编码器都差 |",
    ),
    # ---- report §0: video no longer unmeasured ------------------------------ #
    (
        REPORT,
        "**已测三种模态（文本/音频/图像）的经典基线也复现成功**\n（音频 OptimFROG/WavPack/FLAC 顺序与量级吻合；**视频未测**）。",
        "**四种模态的经典基线全部复现成功**（音频 OptimFROG/WavPack/FLAC 顺序与量级吻合；\n视频 h.265/h.264/FFV1/zstd 与论文偏差最大仅 1.17 pp，其中 FFV1 与 zstd 吻合好于 0.01 pp——\n**视频是吻合度最高的模态**）。",
    ),
]


def main() -> int:
    applied = skipped = 0
    for path, old, new in EDITS:
        text = open(path, encoding="utf-8").read()
        if text.count(old) != 1:
            print(f"  !! SKIP (found {text.count(old)}): {old[:70]!r}")
            skipped += 1
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text.replace(old, new))
        applied += 1
        print(f"  ok  {os.path.basename(path)}: {old[:60]!r}")
    print(f"\n{applied} applied, {skipped} skipped")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
