"""MAJOR CORRECTION: a log-base error invalidates several of our image, video and text rates.

`torch.log_softmax` returns NATURAL logarithms. Every rate in this project that was
computed as `-log_softmax(logits).gather(target)` and then treated as bits is therefore in
**nats**, and must be multiplied by log2(e) = 1.4427 to become bits. Verified directly:

    -log_softmax(x)      = [2.40761, 1.40761, 0.40761]
    -log2(softmax(x))    = [3.47344, 2.03075, 0.58805]
    ratio                = 0.693147 = ln(2)   for every element

Which of our numbers are affected, and which are not:

  AFFECTED (nats, computed with torch.log_softmax)
    video iGPT 18.1512 %          -> 26.1866 %
    image CLIC weighted 14.3779   -> 20.7429
    image CLIC weighted 17.3212   -> 24.9892
    image CLIC median 14.8875     -> 21.4781
    text A@8192 6.3321            -> 9.1353
    text B@4096 6.1657            -> 8.8952

  UNAFFECTED (true bits)
    the port's engine, which uses math.log2
    every classical baseline, which emits real bytes
    the official iGPT rate 15.5144 %, from the reference's real arithmetic coder

Consequences, in order of how much they change:

1. **The paper's text figure reproduces.** Port 9.0410 % against the paper's 9.08 % is
   -0.04 pp, and the A100 paired run corrected is 9.1353 % against SI Table 9's 9.06 %,
   i.e. +0.08 pp. The "retraction" in section 5.7 was itself the error: the port agrees with
   the model's own loss to **2.5 %** (3.797 vs 3.705 bits/token), not 48 %.
2. **The writing agent's 26.19 % was CORRECT.** Our 18.1512 % was the wrong figure. The
   correction notice sent earlier told the agent to change a right number into a wrong one.
3. **Video is worse than the paper, not better.** 26.19 % against 15.31 % is +10.88 pp, and
   against the same-frame H.265 baseline (26.467 %) the model is only 1.01x better, not the
   1.46x claimed in section 4.5. The "1.417 vs 1.458 ratio agreement" argument was built on
   the wrong figure and must be withdrawn.
4. **Image is worse than the paper.** CLIC weighted 20.74 % against 14.62 % is +6.12 pp.
   The earlier "reproduces within 0.24 pp" was an artefact of the unit error.
5. **The 1.48x "inflation" that motivated the whole precision investigation does not exist.**
   It was 1.4427 (the unit error) times about 1.025 (window aggregation). The authors'
   harness, our port and the paper all agree once units are consistent. The precision test is
   therefore no longer motivated by a discrepancy -- it remains a reasonable sanity check but
   there is nothing for it to explain.

This is the single most consequential error of the project, and it is the same class of
mistake this audit has now caught four times in other people's work: a value that was
correct in one frame of reference being reused in another without the conversion.
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
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")

SECTION = """## 10. MAJOR CORRECTION: a log-base error in our own image, video and text rates [M]

**Read this before any number in chapters 4, 5 or 7.**

`torch.log_softmax` returns **natural** logarithms. Every rate in this project computed as
`-log_softmax(logits).gather(target)` and then treated as bits is in **nats**, and must be
multiplied by `log2(e) = 1.4427` to become bits. Verified directly:

| quantity | values |
|---|---|
| `-log_softmax(x)` | 2.40761, 1.40761, 0.40761 |
| `-log2(softmax(x))` | 3.47344, 2.03075, 0.58805 |
| ratio, every element | **0.693147 = ln(2)** |

### Which numbers move

| quantity | as reported (nats) | corrected (bits) | paper | delta |
|---|---|---|---|---|
| video iGPT | 18.1512 | **26.1866** | 15.31 | **+10.88** |
| image CLIC, weighted excl outlier | 14.3779 | **20.7429** | 14.62 | **+6.12** |
| image CLIC, weighted incl outlier | 17.3212 | **24.9892** | 14.62 | +10.37 |
| image CLIC, median | 14.8875 | **21.4781** | 14.62 | +6.86 |
| text A@8192 chars | 6.3321 | **9.1353** | 9.06 | **+0.08** |
| text B@4096 tokens | 6.1657 | **8.8952** | 9.36 | -0.46 |

### Which numbers do not move

* the port's `engine.measure_stream`, which uses `math.log2` -- **true bits**;
* every classical baseline, which emits real bytes;
* the official iGPT rate **15.5144 %**, produced by the reference's real arithmetic coder.

### Consequences

1. **The paper's text figure reproduces.** The port's 9.0410 % against the paper's 9.08 % is
   **-0.04 pp**, and the A100 paired run corrected is 9.1353 % against SI Table 9's 9.06 %
   (**+0.08 pp**). Section 5.7's retraction was itself the error: the port agrees with the
   model's own loss to **2.5 %** (3.797 vs 3.705 bits/token), not 48 %.
2. **The writing agent's 26.19 % was correct**; our 18.1512 % was wrong. The correction
   notice sent earlier asked for a right number to be replaced by a wrong one and is
   withdrawn.
3. **Video is worse than the paper, not better.** 26.19 % against 15.31 % is **+10.88 pp**,
   and against the same-frame H.265 baseline (26.467 %) the model is **1.01x** better, not
   1.46x. The "1.417 vs 1.458 ratio agreement" argument in section 4.5 was built on the
   wrong figure and is withdrawn.
4. **Image is worse than the paper**: 20.74 % against 14.62 % is **+6.12 pp**. The earlier
   "reproduces within 0.24 pp" was an artefact.
5. **The 1.48x "inflation" does not exist.** It was 1.4427 (this unit error) times about
   1.025 (window aggregation). The authors' harness, our port and the paper all agree once
   units are consistent, so the precision test in
   `colab/Precision_Test_Colab.ipynb` has nothing left to explain; it is retained only as a
   sanity check.

### Why this happened, and what it says about the audit

Section 5.7 named the failure mode "we searched for an artefact, searched in the wrong place,
and reported the absence of evidence as evidence". This is a different but related failure:
**a value correct in one frame of reference reused in another without conversion.** The audit
had already flagged exactly this pattern four times in external work -- the writing agent's
`x log2(e)` scalings, the character-vs-token window units, the container-vs-payload
denominators, and the row-`t` convention -- and then committed it itself.

The one thing that would have caught it earlier: every scorer should assert against
`output.loss` **in the same base as the reported number**. The check that existed compared
nats with nats and so passed trivially, which is why the error survived a cross-check
designed to catch exactly this.
"""


def main() -> int:
    t = io.open(AUDIT, encoding="utf-8").read()
    if "MAJOR CORRECTION: a log-base error" in t:
        print("  --  already recorded")
    else:
        anchor = "## 6. Threats to validity"
        if anchor in t:
            t = t.replace(anchor, SECTION + "\n" + anchor, 1)
        else:
            t = t.rstrip() + "\n\n" + SECTION + "\n"
        io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(t)
        print("  ok  audit: chapter 10 inserted before the threats chapter")

    # withdraw the wrong correction notice
    notice = os.path.join(HERE, "results", "数字更正单_给写作agent.md")
    if os.path.exists(notice):
        n = io.open(notice, encoding="utf-8").read()
        banner = ("> **本文件已作废（见 AUDIT 第 10 章）。** 经复核，`torch.log_softmax`\n"
                  "> 返回自然对数，因此 docx 中的视频 **26.19 %** 是**正确**的，本文档原先\n"
                  "> 要求改回 18.1512 % 是错的。图像与文本的同类修正也一并作废。\n"
                  "> 请以 AUDIT 第 10 章的表为准。\n\n")
        if "本文件已作废" not in n:
            io.open(notice, "w", encoding="utf-8", newline="\n").write(banner + n)
            print("  ok  correction notice marked withdrawn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
