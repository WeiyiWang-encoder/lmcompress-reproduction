"""Record the channel-wise mismatch found in our own image/video port.

FINDING (confirmed by external review, then verified here)

The official pipeline is channel-wise.  ``get_imagenet_iterator(is_channel_wised=True)``
splits the image into single-channel 32x32 patches, so each yielded chunk has last
axis 1.  ``language_model.compress`` then sees ``shape[-1] == 1``, takes the
``np.repeat(..., 3, axis=2)`` branch, and hands the processor a GREYSCALE TRIPLE
``(v, v, v)`` built from one channel value.  Verified live:

    yielded patch shapes : (32, 32, 1)
    after repeat         : (32, 32, 3) with R == G == B everywhere
    tokens               : 1024, 9 distinct

Our port's ``uic.data.image_units`` instead calls the processor once on the whole
RGB image:

    out = processor(Image.fromarray(rgb), ...)      # a real colour patch
    -> (1, 1024) tokens, 101 distinct

These are different tokenisations, and they produce different tokens for the same
pixels.  So the reviewer's claim is correct about our port.

WHAT THIS DOES AND DOES NOT AFFECT

* The reported image accounting (15.5144 %, chunks 5733, raw_length 5870592) came
  from ``tools/probe_official_image_accounting.py``, which drives the OFFICIAL
  channel-wise iterator.  It is therefore a faithful reproduction of the official
  path and is unaffected.
* The image non-injectivity finding (256 -> 48 ids, 585 colliding pairs) is a
  property of the official channel-wise path, and is likewise unaffected.
* The video iGPT number (0.0042 %, exact-completion 73.05 %) came from
  ``tools/run_video_igpt.py`` -> ``image_units`` -> RGB path.  It is NOT the
  official path and must be re-measured channel-wise.
* A correction previously applied in this project --- "use the channel-sample
  denominator for the video rate, i.e. divide by 3" --- is WRONG for our port,
  because our port codes ONE channel's worth of tokens while the official path
  codes three.  The per-frame rate in a pixel denominator is the only defensible
  figure for the port as written; properly, the port should be fixed.

REQUIRED FIX

``image_units`` should split into channels, replicate each to a greyscale triple,
and emit one unit per (patch, channel), mirroring the official loader.  Until that
is done, no image or video number from our port should be quoted.
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

SECTION = """### 4.4.4 A channel-wise mismatch in **our own** image and video port [M]

**Found by external review, then verified here. The reviewer is correct.**

The official pipeline is *channel-wise*, and it is worth stating exactly how, because
our port does something different.

`get_imagenet_iterator(is_channel_wised=True)` splits the image into single-channel
patches, so every chunk it yields has last axis 1. `language_model.compress` then takes
its `np.repeat(..., 3, axis=2)` branch and hands the processor a **greyscale triple**
`(v, v, v)` built from one channel value. Verified live
(`tools/check_channel_wise_mismatch.py`, `results/_channel_wise_check.txt`):

| path | processor input | tokens | distinct ids |
|---|---|---|---|
| official, chunk 0 | `(32,32,1)` → repeat → `(32,32,3)`, R = G = B everywhere | 1024 | **9** |
| our port, whole image | `(32,32,3)` real RGB patch | 1024 | **101** |

The two token streams differ from the first token: official
`[71, 71, 71, 71, 71, 71, 96, ...]` against our port
`[51, 51, 51, 51, 51, 51, 51, ...]`.

**What this does and does not affect.**

| Item | Status |
|---|---|
| Image accounting 15.5144 % / chunks 5733 / raw_length 5870592 | **unaffected** — produced by `tools/probe_official_image_accounting.py`, which drives the *official* channel-wise iterator |
| Image non-injectivity (256 → 48 ids, 585 pairs) | **unaffected** — a property of the official path, established through the actual processor on `(v,v,v)` inputs |
| Video iGPT 0.0042 % and exact-completion 73.05 % | **WITHDRAWN for now** — produced by `tools/run_video_igpt.py` via `image_units`, i.e. the RGB path, not the official one |
| A correction previously applied here — "use the channel-sample denominator, i.e. divide the video rate by 3" | **WRONG and withdrawn** — our port codes one channel's worth of tokens while the official path codes three, so scaling the denominator by 3 without also scaling the numerator is not a valid conversion |

**Required fix.** `uic.data.image_units` must mirror the official loader: split the
image into channels, replicate each channel to a greyscale triple, and emit one unit per
(patch, channel). Until that is done, no image or video number produced by our own port
should be quoted; the image figures in this audit stand because they come from the
official code, not from our port.

**Why this is recorded rather than quietly patched.** It is the fourth instance in this
project of the same failure mode — a plausible reconstruction of someone else's pipeline
substituted for the pipeline itself. The difference here is that it was caught by an
external reviewer rather than by our own tooling, which had no check that would have
noticed.
"""


def main() -> int:
    text = io.open(AUDIT, encoding="utf-8").read()
    if "channel-wise mismatch" in text:
        print("  -- already recorded")
        return 0
    anchor = "### 4.5 Video"
    if anchor not in text:
        print("  !! anchor '### 4.5 Video' not found")
        return 1
    text = text.replace(anchor, SECTION + "\n" + anchor, 1)
    io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(text)
    print("  ok  added section 4.4.4 to AUDIT_FOR_REVIEW.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
