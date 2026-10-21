"""Refresh chapter 6 (threats) so it reflects what has since been measured.

Three items were written against the old state of the audit:

  * item 1 counted "1 image" -- the image work is now 12 CLIC images of baselines plus 5
    of the model;
  * item 2 still said "we used 4-bit" (already retracted: bitsandbytes has no CUDA binary
    here), still called context a caveat when chapter 5.6 measures it, and kept a
    mitigation that has now been carried out;
  * item 4 still offered "CLIC2019 images" as future mitigation.

Each is rewritten to say what is actually true now, keeping the threats that remain.
"""

from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")

ITEM1_OLD = """1. **Sample size (all modalities).** 3 text files, 1 audio file, 1 image. Absolute"""
ITEM1_NEW = """1. **Sample size (all modalities).** 3 text files, 1 audio file, 5 CLIC images, 9 video
   frames. Absolute"""

ITEM2_NEW = """2. **Quantisation, context length and window unit differ from the paper (text).**
   Three separate differences, each now resolved or bounded:

   * **Quantisation.** We did **not** use 4-bit: bitsandbytes has no CUDA binary on this
     machine, so no text run here was quantised (see the retraction in section 4.1). The
     base model ran in bf16 where it fit and with CPU offload where it did not. The paper
     used 8-bit. The paired run of section 5.6 removes this variable.
   * **Context length.** Resolved rather than caveated: section 5.6 measures the adapter
     penalty at **five** contexts on two axes and finds it positive at every one
     (+0.3987 to +0.4728 pp, range 0.0741 pp), while the context effect over the same span
     is at most 0.2001 pp. Context cannot account for the penalty.
   * **Window unit.** Still a genuine comparability limit: the reference windows the text
     by **characters**, our port by **tokens**, so "context 4096" is 739 tokens in one and
     3836 in the other (section 5.5). The absolute rates are therefore not
     interchangeable even though the penalty is.
   * **Unresolved, and larger than any of the above.** Every cell of the paired run is
     2.6-3.0 pp better than the paper, the authors' harness and our own port. Section 5.6
     states the contradiction and the single test that discriminates its two
     explanations. Until that test is run, **no absolute text rate from this audit should
     be quoted**; the penalty figure may be.

"""

ITEM4_OLD = """4. **The image conclusion rests on one image.** The *structural* claim"""
ITEM4_NEW = """4. **The image magnitude rests on five images.** The *structural* claim"""

ITEM4_TAIL_OLD = """(15.51 % measured vs 14.62 % published) is not. *Mitigation:* CLIC2019 images."""
ITEM4_TAIL_NEW = """(14.38 % excluding one outlier, 17.32 % including it, against 14.62 % published) is
   not: with n = 5 the corpus mean is not pinned down. *Mitigation partly taken:* the
   baselines were re-measured on 12 CLIC images and now close to 1.6-3.8 pp (section
   4.4.6), so the earlier corpus confound is gone; the model side still needs more images."""


def main() -> int:
    t = io.open(AUDIT, encoding="utf-8").read()

    if ITEM1_OLD in t:
        t = t.replace(ITEM1_OLD, ITEM1_NEW, 1)
        print("  ok  item 1: sample counts updated")
    else:
        print("  --  item 1 already current")

    start = t.find("2. **Quantisation and context length differ from the paper (text).**")
    if start < 0:
        start = t.find("2. **Quantisation, context length and window unit differ")
    if start >= 0:
        end = t.find("\n3. **", start)
        if end > start:
            t = t[:start] + ITEM2_NEW + t[end + 1:]
            print("  ok  item 2: rewritten")
        else:
            print("  !! item 2 end not found")
    else:
        print("  !! item 2 not found")

    if ITEM4_OLD in t:
        t = t.replace(ITEM4_OLD, ITEM4_NEW, 1)
        print("  ok  item 4: heading updated")
    if ITEM4_TAIL_OLD in t:
        t = t.replace(ITEM4_TAIL_OLD, ITEM4_TAIL_NEW, 1)
        print("  ok  item 4: mitigation updated")

    io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(t)

    print("\n  chapter 6 items now:")
    i = t.find("## 6. Threats")
    j = t.find("\n## 7.", i)
    seg = t[i:j] if j > i else t[i:i + 6000]
    for line in seg.splitlines():
        if line[:3].strip().rstrip(".").isdigit() and "**" in line:
            print("   ", line[:88])
    return 0


if __name__ == "__main__":
    sys.exit(main())
