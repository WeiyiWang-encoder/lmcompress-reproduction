"""Replace the withdrawn audio numbers in the remaining documents.

After retracting C9, four documents still state 13.12 / 14.18 / 15.78 bpb and the
"6.29x misalignment" as findings.  This rewrites each occurrence with the verified
position: the reference's scheme is coherent and causal, its figure is a genuine code
length, and the reason the paper's headline is not reproduced is unknown.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BANNER = (
    "> **WITHDRAWN / 已撤回**: the numbers 13.12, 13.1233, 14.1804 and 15.7844 bits/byte,\n"
    "> and any \"misalignment\" or \"6.29x\" explanation, came from an index error in **our**\n"
    "> probe (we used flat row `16+k` where the grid has 17 slots per patch, so the correct\n"
    "> row is `17*(k//16) + (k%16)`). They are **void**. The verified position: the\n"
    "> reference's pairing is the byte the model is most confident about (0.3052 at offset\n"
    "> 16 vs 0.0804 at offset 1), the scheme is causal (Δ = 0 under a future-byte\n"
    "> perturbation), and 1.9376 bpb is its genuine code length. See\n"
    "> `docs/REVIEW_GUIDE_FOR_AGENT.md` §5.2.\n"
)

EDITS = [
    # ---- REPRODUCTION_NOTES ------------------------------------------------- #
    (
        os.path.join(HERE, "docs", "REPRODUCTION_NOTES.md"),
        "The checkpoint predicts the byte **at** a slot far better than the byte **after**\n"
        "it. Score against the next byte — i.e. honest next-byte prediction, which is what\n"
        "the paper's method requires — and the full file gives **15.7844 bits/byte**\n"
        "(197.31 %, 0.5068×); score against the byte at the slot and it gives\n"
        "**2.26 bits/byte**, which is the regime the reference reports.",
        "**The checkpoint's coding scheme is not plain next-byte coding.** Measured, the\n"
        "model's row `(p, j)` is most confident about the byte **16 positions on** from its\n"
        "own slot (mean probability **0.3052**), and much less confident about the byte\n"
        "immediately after it (**0.0804**). The reference pairs each row with exactly the\n"
        "byte it is most confident about, so its pairing is the coherent one for this\n"
        "checkpoint, and the scheme is causal: perturbing bytes after a target leaves its\n"
        "score unchanged (`max |Δ log-prob| = 0`), while perturbing bytes before it moves\n"
        "the score by 9.44. Its reported figure is realized coder output.\n"
        "\n"
        + BANNER,
    ),
    (
        os.path.join(HERE, "docs", "REPRODUCTION_NOTES.md"),
        "**Consequence for the report:** on this sample the reference's 4.13× is not\n"
        "reproducible as a next-byte compression result. The honest current number for this\n"
        "checkpoint/file pair is **15.7844 bits/byte = 197.31 % = 0.5068× (expansion)**\n"
        "(`tools/remeasure_audio_nextbyte.py`). Two things could still rescue the paper's\n"
        "headline and are worth testing before concluding: (a) the checkpoint may need the\n"
        "paper's exact preprocessing, which it *is* already in (16 kHz, mono, 8-bit), and\n"
        "(b) the released `weights-audio.pth` may simply be a different training run from\n"
        "the one that produced 18.26 %. Both are testable on more files from the same corpus.",
        "**Consequence for the report:** the released code plus released weights report\n"
        "**1.9376 bpb (≈4.13×)** on the bundled sample, which is a **genuine code length for\n"
        "the scheme the checkpoint implements** — but the paper reports **18.26 % (≈1.46\n"
        "bpb)** on LJSpeech, and this reproduction does **not** reconcile the two.\n"
        "\n"
        "**We do not know why.** The most likely explanation is that the released\n"
        "`weights-audio.pth` is a **different training run** from the one behind 18.26 %.\n"
        "Testing that needs either the authors' configuration or more of their data, and is\n"
        "not something this reproduction establishes. Any claim of a *mechanism* is\n"
        "withdrawn — see below.",
    ),
    # ---- COMPARISON TABLES -------------------------------------------------- #
    (
        os.path.join(HERE, "results", "comparison_tables.md"),
        "one it was computed for. Scoring honest next-byte prediction with the current\n"
        "code gives **15.7844 bits/byte = 197.31 % = 0.5068×** "
        "(`tools/remeasure_audio_nextbyte.py`).\n\n"
        "> An earlier revision quoted 13.12 bits/byte here. That figure came from the\n"
        "> offset-sweep probe, run *before* the alignment fix, and does not correspond to\n"
        "> 197.3 % (13.1233 bpb is 164.0 % / 0.610×). The current value is above.",
        "one it was computed for.\n\n"
        + BANNER,
    ),
]

# Simple line-level removals for the report's residual claims.
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")
REPORT_EDITS = [
    (
        "| 诚实 next-byte 打分，同一 sample | **15.7844 bpb = 197.31 % = 0.5068×** | checkpoint 实际达到的水平 |",
        "| 参考代码报出 | **1.9376 bpb ≈ 4.13×** | 已确认为该方案下**真实的码长**（算术编码器实际输出 900 016 bit） |",
    ),
    (
        "每个窗口长度上，模型对**同位字节**的预测都明显好于对**下一个字节**——即它在**重建**。",
        "模型对**偏移 16 处**那个字节的预测（平均概率 0.3052）明显好于对**紧邻的下一字节**（0.0804）"
        "——**这是它所实现的编码方案本身，不是缺陷**。",
    ),
    (
        "按**真正的 next-byte 打分**（当前代码实测，`tools/remeasure_audio_nextbyte.py`）：",
        "**关于编码器输出（已核实）**：其算术编码器确实输出了 900 016 bit = 112 502 字节，"
        "所以报出的数字是**真实码流长度**，不是打分假象。",
    ),
]


def apply(path: str, edits) -> int:
    if not os.path.exists(path):
        print(f"  !! missing {path}")
        return 0
    text = open(path, encoding="utf-8").read()
    n = 0
    for old, new in edits:
        if old in text:
            text = text.replace(old, new, 1)
            n += 1
            print(f"  ok  {os.path.basename(path)}: {old[:52]!r}")
        else:
            print(f"  --  {os.path.basename(path)}: no match {old[:52]!r}")
    if n:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return n


def main() -> int:
    total = 0
    by_file: dict[str, list] = {}
    for path, old, new in EDITS:
        by_file.setdefault(path, []).append((old, new))
    for path, edits in by_file.items():
        total += apply(path, edits)
    total += apply(REPORT, REPORT_EDITS)
    print(f"\n{total} replacement(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
