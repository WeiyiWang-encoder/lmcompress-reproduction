"""Withdraw the "missing samplings dependency" claim from the documents.

Both blockers reported in the last hand-off are withdrawn:
  * samplings is an installed PyPI package -- nothing is missing;
  * the reference script completes the full file, so the predicted tail crash does not
    occur.

What the reference actually does, measured on the UNMODIFIED capsule, is recorded
instead.  The audio result is unchanged; only the explanation of "why it could not be
run" was wrong.
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
GUIDE = os.path.join(HERE, "docs", "REVIEW_GUIDE_FOR_AGENT.md")

CORRECTION = """**Withdrawn: the "missing `samplings` dependency" claim.**

An earlier revision of this audit asserted that `data/model/bgpt/utils.py:9` imports a
`samplings` module the capsule fails to ship, so that the reference code "fails at
import time", and that this blocked reproducing the paper. **That is false.**

`samplings` is a real PyPI package (by Sander Wood) and it is **already installed** in
the capsule's virtualenv:

```
<capsule>/.venv\\Lib\\site-packages\\samplings\\__init__.py   (21 316 bytes)
```

It is correctly *not* vendored inside the capsule source tree — it is a dependency, and
a virtualenv is where dependencies live; it is also absent from `requirements.txt`,
which is a separate (minor) packaging gap. Our search walked the capsule directory while
excluding `.venv`, found nothing, and drew the wrong conclusion.

**The reference code runs, unmodified, on the original capsule.** Verified twice:

```
import compress_with_bgpt_or_llama   ->  IMPORT OK (exit 0)

python compress_with_bgpt_or_llama.py \\
  --data_path data/test_audios --data_type wav \\
  --model_path data/model/bgpt/weights-audio.pth \\
  --context_window 8160 --batch_size 1

Theoretical compression ratio: 5.016113
Total size: 82264
compressed size: 16399.949935913086
```

**The tail-block claim is also withdrawn.** The arithmetic was right — `split(..., 8160)`
yields 10 full windows plus a 620-byte tail, and `byte_collate` pads that to 652 bytes,
which is not a multiple of `PATCH_SIZE` — but the reference run **completes all 11
blocks** and reports `Total size: 82264`. The predicted crash does not happen.

**Consequence for this audit.** Nothing about the audio *result* changes: the reference
reproduces the paper's audio regime, and so does our port (§4.3). What changes is that
the reproduction is **easier** than we claimed — no repair, no patched copy, no
reconstruction was needed. Anyone can reproduce it with the two commands above.

**Process note.** This is the third claim in this section to be withdrawn, and the
failure mode is the same each time: we searched for an artifact, searched in the wrong
place (excluding `.venv`, or indexing a 17-slot row grid as 16), and reported the
absence of evidence as evidence. We had also run the import check, seen it succeed, and
explained it away rather than following it. The reviewer should treat any remaining
"the reference cannot do X" statement in this document as suspect until re-run.
"""


GUIDE_CORRECTION = """| 7 | `bits2bytes` does not exist; the text failure is an fp32 OOM | ~32 GB vs 5.71 GB | [M] | low |
| ~~10~~ | ~~The capsule omits a `samplings` module, blocking the reference code~~ **— RETRACTED**: `samplings` is an installed PyPI package in `.venv`; the unmodified capsule imports and runs | — | **withdrawn** | — |"""


def main() -> int:
    # ---- audit: insert the correction where the old claim lives ----------- #
    text = open(AUDIT, encoding="utf-8").read()
    marker = "**Reproduction procedure** (`tools/confirm_tail_and_our_window.sh`)"
    if marker in text:
        text = text.replace(marker, CORRECTION + "\n" + marker, 1)
        print("  ok  audit: correction inserted before the procedure block")
    else:
        print("  --  audit marker not found; appending at section end")
        text = text.rstrip() + "\n\n" + CORRECTION + "\n"

    # and rewrite the erroneous bullet 3 inside the three-errors list
    old3 = text.find("**3. \"`samplings` blocks the run\" — correct, and now fixed.**")
    if old3 >= 0:
        end = text.find("\n\n", old3)
        text = text[:old3] + (
            "**3. \"`samplings` blocks the run\" — RETRACTED.** The dependency is "
            "installed; the capsule imports and runs unmodified. See the withdrawal "
            "below.") + text[end:]
        print("  ok  audit: error-3 bullet corrected")

    open(AUDIT, "w", encoding="utf-8", newline="\n").write(text)

    # ---- guide: claim table ------------------------------------------------ #
    g = open(GUIDE, encoding="utf-8").read()
    anchor = "| 7 | `bits2bytes` does not exist; the text failure is an fp32 OOM | ~32 GB vs 5.71 GB | [M] | low |"
    if anchor in g and "~~10~~" not in g:
        g = g.replace(anchor, GUIDE_CORRECTION, 1)
        print("  ok  guide: withdrawn claim 10 added to the table")
    open(GUIDE, "w", encoding="utf-8", newline="\n").write(g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
