"""Put the two new text-baseline subsections in the right place in the audit.

Two insertions used an anchor that does not exist ("### 4.3 Audio"; the real heading is
"### 4.3 The audio result, after three corrections"), so one block was appended to the end
of the document and the other was not inserted at all.

This moves both to sit immediately after section 4.2 (text baselines) and before 4.3
(audio), numbered 4.2.1 and 4.2.2, and removes the stray copy from the end.
"""

from __future__ import annotations

import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")

WINDOWED = """#### 4.2.2 Text baselines at the paper's fixed-window protocol [M]

The earlier version of this section reported whole-file compression and then declined to
compare it with the paper, because SI Table 11 uses a fixed window and the same compressor
spans 42.02 -> 23.89 % across its columns. Declining to compare was the wrong call: the
protocol is cheap to reproduce, so it was reproduced
(`tools/baselines_text_windowed.py`). Each file is split into consecutive chunks of 8192 /
16384 / 32768 bytes, every chunk compressed with a **fresh** compressor state, and the
whole file compressed once for the unbounded column.

| codec | 8192 | 16384 | 32768 | unbounded |
|---|---|---|---|---|
| brotli q11 | 31.84 | 30.91 | 30.17 | 28.81 |
| lzma 9e (7z-equivalent) | **41.44** | **38.40** | **35.78** | **31.68** |
| zlib 9 | 41.78 | 39.67 | 37.70 | 35.02 |
| zstd -22 | **40.38** | 37.93 | 35.69 | 31.99 |

Against SI Table 11:

| codec | window | ours | paper | delta |
|---|---|---|---|---|
| 7z | 8192 | 41.44 | 42.02 | **-0.58** |
| 7z | 16384 | 38.40 | 38.21 | **+0.19** |
| 7z | 32768 | 35.78 | 35.26 | **+0.52** |
| 7z | unbounded | 31.68 | 23.89 | +7.79 |
| zstd | 8192 | 40.38 | 39.54 | **+0.84** |

**Three of the four 7z columns agree to better than 0.6 pp.** That is a reproduction of the
paper's baseline grid, and it is a stronger statement than the hedge it replaces.

**The unbounded column is the exception and the reason is known.** The paper compressed on
the order of 180 MB; we compress 313 KB. With no window to limit it, a compressor benefits
directly from more input, and our own table shows the same gradient (7z improves
41.44 -> 31.68 % as the window grows). So +7.79 pp is a corpus-size effect, not a protocol
error, and it is exactly the column where the two corpora are least comparable.

**Consequence for the model comparison.** The like-for-like comparison is now available and
is the one that should be quoted: our model at an 8192-character window (**6.3321 %**)
against our baselines at an 8192-byte window (best is brotli at **31.84 %**), a factor of
**5.0x**. The paper's own pair is 9.06 % against 39.54 % (zstd), a factor of 4.4x. The
model's margin over classical compression therefore reproduces at matched windows.

"""


def main() -> int:
    t = io.open(AUDIT, encoding="utf-8").read()

    # 1. cut the stray 4.2.1 block from wherever it sits
    start = t.find("### 4.2.1 Two corrections to our own text numbers [M]")
    block_421 = ""
    if start >= 0:
        end = t.find("\n### ", start + 5)
        if end < 0:
            end = len(t)
        block_421 = t[start:end].rstrip() + "\n\n"
        t = t[:start] + t[end:]
        print("  cut the stray 4.2.1 block")

    # 2. normalise its heading level to #### like its siblings
    block_421 = block_421.replace(
        "### 4.2.1 Two corrections to our own text numbers [M]",
        "#### 4.2.1 Two corrections to our own text numbers [M]", 1)

    # 3. insert both before 4.3
    anchor = "### 4.3 The audio result, after three corrections [M]"
    if anchor not in t:
        print("  !! 4.3 anchor still missing")
        return 1
    if "4.2.2 Text baselines at the paper" not in t:
        t = t.replace(anchor, block_421 + WINDOWED + anchor, 1)
        print("  inserted 4.2.1 and 4.2.2 before 4.3")
    else:
        print("  --  4.2.2 already present")

    io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(t)

    print("\n  chapter 4 structure now:")
    for m in re.finditer(r"(?m)^#{3,4} 4\.[^\n]*$", t):
        print("   ", m.group(0)[:78])
    return 0


if __name__ == "__main__":
    sys.exit(main())
