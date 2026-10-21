"""Add the confirmed alignment mechanism to the audit, replacing the weaker claim.

The mechanism is now established by controlled search rather than argued:

  reference rows  = patch p, slots 0..15   (patch-major, last slot dropped)
  reference targets = payload[k] = grid[16 + k]
  => row k = (patch p, slot j) is paired with a byte from a LATER patch, at the
     same raster offset, instead of with the byte immediately following its own slot.

Measured on the same logits, same targets:
  reference pairing  2.2561 bits/byte
  correct pairing   14.1804 bits/byte
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

ANCHOR = """**C9 [D], from the enumeration in §3.2 plus the table above.** The reference's
index arithmetic selects a target set (grid 1…8160) that is offset from its
prediction set (grid 16…8175), and the resulting pairing is the one the model
scores well on. The two facts — misaligned targets, and better agreement with the
"wrong" byte — are consistent and mutually reinforcing."""

NEW = """**C9 [M] — the mechanism, identified by controlled search.** The earlier statement
of this claim ("the target set is offset from the prediction set") was too vague,
and was written before the mechanism had been pinned down. It has now been
established directly (`tools/identify_reference_rows.py`,
`tools/prove_alignment_mechanism.py`).

What the reference actually does:

* `logits[:-1, :-1, :].reshape(1, -1, 257)` yields 8 160 rows, and searching the
  structured candidates shows these are exactly **patch-major, slots 0…15**
  (the last slot of each patch row is dropped);
* its targets are `patches[:, 16:-16]`, which is **exactly the payload**
  (`ref_ids == grid[16:8176]`, verified element-wise — so the *denominator* is
  correct, and the mask covers 100 % of positions);
* it then pairs row `k` with `payload[k]`.

Row `k = 16p + j` is the model's prediction vector for patch `p`, slot `j`. Its
natural target is `grid[16p + j + 1]` — the byte immediately after slot `j` **in the
same patch**. But `payload[k] = grid[16 + k] = grid[16p + j + 16]`, i.e. slot `j+1`
of a patch **16 bytes later**. So every distribution is paired with a byte from a
different patch rather than with the byte it was computed to predict.

**Measured consequence on identical logits and identical targets:**

| pairing | bits/byte |
|---|---|
| the reference's (row `k` ↔ `payload[k]`) | **2.2561** |
| correct (row predicts the byte after its own slot) | **14.1804** |

The reference's pairing is **6.29× cheaper** for the same model on the same data.
That is the entire "compression" being reported. A per-row illustration:

```
row (p=14, j=3):  p(byte the reference scores it against) = 0.6762
                  p(byte immediately following its slot)  = 0.3235
row (p=14, j=6):  p(reference target) = 0.9269
                  p(own next byte)    = 0.0724
```

**This also explains a fact that previously seemed to contradict it.** The reference's
number *is* sensitive to the data — 2.17 bpb on real audio versus 8.35 bpb on uniform
random bytes, and 5.26 bpb when the bytes are shuffled but their frequencies kept
(`tools/decide_reference_audio_metric.py`). So the figure is not a constant artefact;
it tracks real structure. What it tracks, however, is **cross-patch similarity at a
fixed raster offset** in highly autocorrelated audio, which is 6.29× easier than
next-byte prediction — not the quantity the paper's method requires.

**Why this reconciles with an independent reproduction.** Anyone running the shipped
code on shipped weights obtains this pairing *by construction*, so the number
reproduces deterministically — which is precisely what one would expect from an
index-arithmetic defect and not from a modelling result. Concretely: the published
figure for LJSpeech is **18.26 %** = 1.46 bits/byte; the reference code on the bundled
sample reports **1.9376 bits/byte**; honest scoring of the payload gives
**15.7844 bits/byte**. The first two sit in the same (misaligned) regime; the third
does not. This also strengthens the answer to §4.3.4: of the two explanations offered
there, the scoring-defect one now has independent support."""


def main() -> int:
    text = open(AUDIT, encoding="utf-8").read()
    if ANCHOR not in text:
        print("!! anchor not found")
        return 1
    text = text.replace(ANCHOR, NEW, 1)
    with open(AUDIT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("ok  C9 mechanism replaced in AUDIT_FOR_REVIEW.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
