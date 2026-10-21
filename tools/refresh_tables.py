"""Bring results/comparison_tables.md up to the canonical facts.

Same discipline as the other refresh scripts: every replacement must match exactly
the expected number of times, so a silent no-match is reported rather than ignored.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(HERE, "results", "comparison_tables.md")

EDITS = [
    # ---- text table + per-file table -------------------------------------- #
    (
        "| LMCompress\\* (raw Llama-3-8B, no fine-tuning) | **9.033 %** | 11.07× | 9.08 % | **−0.05** |",
        "| LMCompress\\* (raw Llama-3-8B, no fine-tuning) | **9.041 %** | 11.06× | 9.08 % | **−0.04** |",
    ),
    (
        "| LMCompress (Llama-3-8B + shipped LoRA) | **9.346 %** | 10.70× | 9.06 % | **+0.29** |",
        "| LMCompress (Llama-3-8B + shipped LoRA) | **9.353 %** | 10.69× | 9.06 % | **+0.29** |",
    ),
    (
        "| **weighted mean** | **9.033 %** | **9.346 %** | **+0.313** |",
        "| **weighted mean** | **9.041 %** | **9.353 %** | **+0.312** |",
    ),
    (
        "   lands **0.05 pp** from the published 9.08 %. That is essentially a",
        "   lands **0.04 pp** from the published 9.08 %. That is essentially a",
    ),
    (
        "| text | **yes, closely** | raw 9.033 % vs 9.08 %; LoRA 9.346 % vs 9.06 % |",
        "| text | **yes, closely** | raw 9.041 % vs 9.08 %; LoRA 9.353 % vs 9.06 % |",
    ),
    # ---- audio row --------------------------------------------------------- #
    (
        "| **bGPT (this port, honest next-byte scoring)** | **197.3 %** | **0.51×** | 18.26 % |",
        "| **bGPT (this port, honest next-byte scoring)** | **197.31 %** | **0.5068×** | 18.26 % |",
    ),
    (
        "one it was computed for. Scoring honest next-byte prediction gives 0.51×.",
        "one it was computed for. Scoring honest next-byte prediction with the current\n"
        "code gives **15.7844 bits/byte = 197.31 % = 0.5068×** "
        "(`tools/remeasure_audio_nextbyte.py`).\n\n"
        "> An earlier revision quoted 13.12 bits/byte here. That figure came from the\n"
        "> offset-sweep probe, run *before* the alignment fix, and does not correspond to\n"
        "> 197.3 % (13.1233 bpb is 164.0 % / 0.610×). The current value is above.",
    ),
    # ---- summary table: audio evidence wording ----------------------------- #
    (
        "| audio | **no** | reference's own number violates the source entropy |",
        "| audio | **no** | the released checkpoint predicts the *current* byte better than the *next* one (argmax 0.415–0.756 vs 0.111–0.551), so its 1.94 bpb is not a next-byte code length |",
    ),
]

# Per-file text table rows (the three files).
PER_FILE = [
    ("| medal_102285.csv | 8.853 % | 9.118 % | **+0.265** |", "| medal_102285.csv | 8.863 % | 9.127 % | **+0.264** |"),
    ("| medal_103851.csv | 9.233 % | 9.548 % | **+0.315** |", "| medal_103851.csv | 9.237 % | 9.551 % | **+0.314** |"),
    ("| medal_108122.csv | 9.012 % | 9.370 % | **+0.358** |", "| medal_108122.csv | 9.022 % | 9.379 % | **+0.357** |"),
]

# LoRA penalty sentence.
SENTENCE = (
    "9.08 % → 9.06 % (0.2 % relative, i.e. nothing) on MeDAL. We measure the\n"
    "   shipped adapter as **worse than the base model by 0.31 pp**",
    "9.08 % → 9.06 % (0.02 pp, i.e. nothing) on MeDAL. We measure the\n"
    "   shipped adapter as **worse than the base model by 0.312 pp**",
)


def main() -> int:
    text = open(TABLES, encoding="utf-8").read()
    applied = skipped = 0
    for old, new in EDITS + PER_FILE + [SENTENCE]:
        n = text.count(old)
        if n != 1:
            print(f"  !! SKIP (found {n}, expected 1): {old[:80]!r}")
            skipped += 1
            continue
        text = text.replace(old, new)
        applied += 1
        print(f"  ok  replaced: {old[:70]!r}")

    if applied:
        with open(TABLES, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"  -> wrote comparison_tables.md")
    print(f"\n{applied} applied, {skipped} skipped")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
