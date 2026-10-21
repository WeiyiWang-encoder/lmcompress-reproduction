"""Why do our text baselines look better than the paper's? Check the protocol.

The paper's Supplementary Table 11 gives traditional text compressors at a "context
length" of 8192 / 16384 / 32768 / unbounded:

    7z    42.02 / 38.21 / 35.26 / 23.89 %
    zstd  39.54 / 37.05 / 35.26(?) / ...

i.e. the same compressor improves sharply as the window grows. Our baseline table reports
brotli at 28.918 %, which is far better than the paper's 8192-column figures -- so either
we used a different protocol (whole-file instead of 8192-byte windows), or a different
corpus, or a different denominator.

This script inspects the recorded baseline rows and reports, per codec, what was actually
compressed, so the protocol question is answered from data rather than guessed.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def show(path: str, title: str) -> None:
    if not os.path.exists(path):
        print(f"  (missing) {path}")
        return
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        print(f"  (empty) {path}")
        return
    print(f"\n=== {title} ===")
    print(f"  file: {os.path.basename(path)}   rows: {len(rows)}")
    keys = sorted(rows[0])
    print(f"  keys: {keys}")

    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        codec = r.get("codec") or r.get("method") or "?"
        raw = r.get("raw_bytes") or r.get("original_bytes") or 0
        comp = r.get("compressed_bytes") or 0
        agg[codec][0] += comp
        agg[codec][1] += raw
    print(f"  {'codec':<22}{'rate %':>9}{'raw bytes':>12}{'comp bytes':>12}")
    for codec, (comp, raw) in sorted(agg.items(),
                                     key=lambda kv: kv[1][0] / max(kv[1][1], 1)):
        rate = 100 * comp / raw if raw else 0
        print(f"  {codec:<22}{rate:>9.3f}{raw:>12,}{comp:>12,}")

    # does any row mention a window/chunk size?
    for k in ("window", "chunk", "context", "block"):
        vals = {r.get(k) for r in rows if k in r}
        if vals:
            print(f"  field '{k}': {sorted(v for v in vals if v is not None)}")


def main() -> int:
    r = os.path.join(HERE, "results")
    show(os.path.join(r, "baselines_text_medal_python.jsonl"),
         "text baselines, python (the table quoted in the report)")
    show(os.path.join(r, "baselines_text_medal_3.jsonl"),
         "text baselines, 3 files")
    show(os.path.join(r, "baselines_text_medal_raw_3.jsonl"),
         "text baselines, raw 3 files")

    print("\n=== what the paper reports (SI Table 11) ===")
    print("  7z    ctx 8192  42.02 %   |  16384  38.21 %  |  32768  35.26 %  |  inf  23.89 %")
    print("  zstd  ctx 8192  39.54 %   (further columns present but garbled in the text dump)")

    print("\n=== the arithmetic of the gap ===")
    for ours, paper, name in ((28.918, 42.02, "brotli vs 7z @8192"),
                              (28.918, 23.89, "brotli vs 7z @inf"),
                              (28.918, 39.54, "brotli vs zstd @8192")):
        print(f"  {name:<24} ours {ours:6.3f} vs paper {paper:6.2f}  "
              f"delta {ours-paper:+6.2f} pp")
    print("\n  Note the paper's own 7z row spans 42.02 -> 23.89 % as the window grows, a")
    print("  18 pp swing from protocol alone. Any comparison that does not fix the window")
    print("  is meaningless, and a whole-file figure sits near the 'inf' column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
