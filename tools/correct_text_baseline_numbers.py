"""Correct two errors of our own that this round of checking exposed.

1. **The text baseline figure 28.918 % is wrong.** The bytes-weighted payload figure in
   `results/baselines_text_medal_python.jsonl` is **28.6187 %** over 10 units and
   **28.6717 %** over the 3 raw files. 28.918 appears in no result row. Quoting a larger
   baseline rate makes the baseline look worse and therefore our model look better, so
   the error leans in our favour and must go.

2. **The "0.13 %" denominator statement conflates two things.** The container and the
   payload differ in size by **0.389 %** (313 272 vs 312 053 bytes), while the *rate*
   moves by **0.133 pp** between the two denominators (brotli 28.8050 -> 28.6717). The
   documents said the denominators differ by 0.13 %, which is the rate shift, not the
   size difference.
"""

from __future__ import annotations

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = [os.path.join(HERE, "docs", n) for n in
        ("AUDIT_FOR_REVIEW.md", "REPORT_CN.md", "RESULTS_SUMMARY.md",
         "REVIEW_GUIDE_FOR_AGENT.md")]

OLD_RATE = "28.918"
NEW_RATE_NOTE = (
    "28.6187 (brotli q11, 10 units, bytes-weighted; **an earlier version of this document "
    "said 28.918, which appears in no result row and is corrected here**)"
)


def main() -> int:
    touched = 0
    for path in DOCS:
        if not os.path.exists(path):
            continue
        t = io.open(path, encoding="utf-8").read()
        if OLD_RATE not in t:
            continue
        n = t.count(OLD_RATE)
        t = t.replace(OLD_RATE, "28.6187")
        io.open(path, "w", encoding="utf-8", newline="\n").write(t)
        print(f"  {os.path.basename(path)}: replaced {n} occurrence(s) of 28.918")
        touched += 1

    # append a correction block to the audit so the change is traceable
    audit = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
    t = io.open(audit, encoding="utf-8").read()
    if "baseline figure was 28.918" not in t:
        block = """
### 4.2.1 Two corrections to our own text numbers [M]

Prompted by a reader asking why the text baselines looked so good relative to the paper.
Two errors on our side, both leaning in our own favour:

**The brotli figure was 28.918 % and is wrong.** The bytes-weighted payload value in
`results/baselines_text_medal_python.jsonl` is **28.6187 %** over 10 units and **28.6717 %**
over the 3 raw files. 28.918 appears in no result row. Corrected throughout. The other
text baselines, bytes-weighted over 10 units, are lzma 9e **31.4479 %**, zstd -22
**31.6755 %**, zlib 9 **34.7582 %**.

**The denominator sentence conflated a size with a rate.** The CSV container and the
extracted payload differ in size by **0.389 %** (313 272 vs 312 053 bytes), whereas the
*measured rate* moves by **0.133 pp** between the two denominators (brotli 28.8050 ->
28.6717). The earlier wording said the denominators differ by 0.13 %, which is the rate
shift, not the size difference.

**And the comparison with the paper's text baselines is not like-for-like.** SI Table 11
runs each compressor at a fixed window of 8192 / 16384 / 32768 bytes and then unbounded;
7z alone spans **42.02 -> 23.89 %** across those columns. Our baselines compress the whole
file in one stream, which corresponds to their *unbounded* column -- against which our
7z-equivalent is **31.45 %**, i.e. **worse** than the paper's 23.89 %, exactly as expected
from 313 KB of input against their ~180 MB. Comparing our whole-file 28.6 % with their
8192-byte window figures would flatter us by more than ten points and is not done here.
"""
        anchor = "### 4.3 Audio"
        if anchor in t:
            t = t.replace(anchor, block + "\n" + anchor, 1)
        else:
            t = t.rstrip() + "\n\n" + block + "\n"
        io.open(audit, "w", encoding="utf-8", newline="\n").write(t)
        print("  audit: added 4.2.1 recording both corrections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
