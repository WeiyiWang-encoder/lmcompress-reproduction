"""Retract Claim C9 (bGPT audio scoring misalignment).

C9 is wrong.  It was the headline audio finding for several revisions and it must be
removed as a claim, not merely softened, because a capstone thesis will cite it.

What the verification actually established, in order:

 1. The reference pairs row k = 16p + j with payload[k], i.e. with grid[16p+j+16].
 2. Measured, the model's row (p, j) IS most confident about exactly that byte:
    mean probability 0.3052 at offset L=16, versus 0.0804 at L=1 (the immediate next
    byte).  So the reference's pairing is the natural one for this checkpoint.
 3. A causality probe confirms the score for a byte does not depend on any later byte
    (max |delta log-prob| = 0 over the whole vocabulary), with a working control
    (perturbing earlier bytes moves it by 9.44).  So it is a legitimate causal code.
 4. The arithmetic coder really emits 900 016 bits = 112 502 bytes for an 82 220-byte
    payload.  The reported number is realized output, not a scoring artefact.

The error's origin: one patch row carries **17** slot predictions, so the flat row for
payload index k is 17*(k//16) + (k%16), NOT 16 + k.  Our earlier "honest" probes used
16 + k, i.e. they scored each row against a byte roughly p positions away from the one
it predicts.  That manufactured the "misalignment" and the 6.29x ratio.

This script rewrites C9 as a retraction, updates the claim table, and removes the
downstream claims (the 6.29x figure, the "rebuild rather than predict" conclusion).
"""

from __future__ import annotations

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")

RETRACTION = """**C9 — RETRACTED. There is no audio scoring misalignment.**

This claim asserted that the reference bGPT-audio code scores each byte against a
prediction for a different byte, inflating its apparent compression by 6.29x. **That
is false.** It rested on an arithmetic error of ours, and it was destroyed by the
review harness written to test it.

**What is actually true, each step measured rather than derived:**

1. The reference pairs row `k = 16p + j` with `payload[k]`, i.e. with `grid[16p+j+16]`.
2. Measured over sampled rows, the model's row `(p, j)` **is** most confident about
   exactly that byte: mean probability **0.3052** at offset `L = 16`, versus **0.0804**
   at `L = 1` (the byte immediately after its own slot), and ≤ 0.19 at every other
   offset in −2…19. The reference pairs each prediction with the byte that prediction
   is *for*.
3. A causality probe (`tools/test_audio_causality.py`) perturbs bytes strictly **after**
   a target and observes `max |Δ log-prob| = 0.000000e+00` across the whole vocabulary,
   while the control (perturbing bytes *before* the target) moves it by **9.44**. So
   the scheme is a legitimate **causal** code: no future information is used.
4. The arithmetic coder really emits **900 016 bits = 112 502 bytes** for an 82 220-byte
   payload. The reported figure is realized output, not a scoring artefact.

**The origin of our error.** A patch row carries **17** slot predictions (16 for its own
bytes plus one carried for the next patch), so the flat row index for payload position
`k` is

```
17 * (k // 16) + (k % 16)          <-- correct
16 + k                             <-- what our earlier probes used
```

The two differ by `k // 16`, i.e. they drift apart by one per patch. Every
"honest next-byte" figure this project published for audio — 13.1219, 14.1804,
15.7844 bpb — came from that incorrect row index and is **withdrawn**. The correct
next-byte-equivalent figure for this scheme is the one the reference reports,
**2.2561 bpb** on the 8 160-byte window.

**What survives, and what does not:**

| Statement | Status |
|---|---|
| The reference's audio scoring is internally coherent and causal | **established** |
| Its reported bpb is a realized code length for its own scheme | **established** |
| The released checkpoint's scheme differs from plain next-byte coding | **established** (it is most confident about the byte 16 positions on, not the next one) |
| The released code + weights fail to reproduce the paper's 18.26 % | **still open** — untouched by this retraction, but no longer supported by a mechanism |
| Any "misalignment" / "6.29x artefact" claim | **RETRACTED — do not cite** |

The remaining open question is therefore the *original* one, with no mechanism
attached: the paper reports 18.26 % on LJSpeech, and this checkpoint under the authors'
own scheme reports 1.9376 bpb on the bundled sample. Whether those differ because of a
different training run, a different sample, or something else is **unknown**, and this
reproduction does not establish an answer.

**Process note.** This claim survived three review rounds and was already written into
the audit as "high confidence" before the harness caught it. It is recorded here in
full rather than deleted, because a thesis that cites it would be wrong, and because
the failure mode — an off-by-`p` index in a *verification* script, dressed up as a
finding about someone else's code — is exactly what the reviewer should watch for
elsewhere in this document.
"""


def main() -> int:
    text = open(AUDIT, encoding="utf-8").read()

    # 1. replace the whole C9 mechanism block
    pattern = re.compile(r"\*\*C9 \[M\].*?(?=\n\*\*C(?:10|\d)|\n###|\n##)", re.S)
    if not pattern.search(text):
        print("!! C9 claim block not found")
        return 1
    text = pattern.sub(RETRACTION, text, count=1)
    print("  ok  C9 claim block replaced with a retraction")

    # 2. fix the summary claim table row
    old_row = ("| C9 | The bGPT reference code scores each byte against the "
               "distribution for the *previous* byte | [D] from [M] | high |")
    if old_row in text:
        text = text.replace(
            old_row,
            "| ~~C9~~ | ~~The bGPT reference code misaligns its audio scoring~~ "
            "**RETRACTED** — the reference's pairing is the one the checkpoint is most "
            "confident about (0.3052 at offset 16 vs 0.0804 at offset 1), the scheme is "
            "causal (Δ = 0 under a future-byte perturbation), and its number is realized "
            "coder output. Our earlier probes used row `16+k` instead of `17(k//16)+(k%16)` | — | **withdrawn** |",
            1)
        print("  ok  claim table row C9 marked retracted")
    else:
        print("  -- claim table row for C9 not found (check manually)")

    # 3. C6 (checkpoint semantics) relied on the same probe family; re-scope it.
    old_c6 = ("**C6 [M].** The released `weights-audio.pth` predicts the byte at a slot "
              "better than the *next* byte.")
    if old_c6 in text:
        text = text.replace(
            old_c6,
            "**C6 — re-scoped, and the earlier wording is withdrawn.** The raw "
            "observation stands: this checkpoint assigns higher probability to the byte "
            "**16 positions after** a given slot than to the byte immediately after it. "
            "But that is simply the scheme it implements, not a defect, and the earlier "
            "framing (\"it reconstructs rather than predicts\") imported the withdrawn C9. "
            "The correct reading is that this checkpoint's coding scheme is not plain "
            "next-byte coding, which is a statement about the method, not about validity.",
            1)
        print("  ok  C6 re-scoped")
    else:
        print("  -- C6 wording not found (check manually)")

    with open(AUDIT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("  -> wrote AUDIT_FOR_REVIEW.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
