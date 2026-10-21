"""SUPERSEDED — retained for review history; NOT evidence for C7.

This script implements a **whole-image** ImageGPT path of our own construction: one
image is run through the processor once (resized to 32x32, quantised to 512
colours) and the resulting single token grid is analysed.  That is *not* the
reference implementation's image path, which is patch-wise and channel-wise.

The figures it prints — including the "67.43 % true lossless rate" — were quoted in
the first revision of the audit and have since been **withdrawn**:
see ``docs/AUDIT_FOR_REVIEW.md`` §4.4.3.  In particular the 330-byte numerator came
from this path, not from the official one (whose final total is 910 788 bytes).

The current evidence for the image claim is:

* ``tools/test_igpt_injectivity.py`` (+ ``results/igpt_grayscale_collisions.json``)
  — the exhaustive non-injectivity proof;
* ``tools/probe_official_image_accounting.py``
  (+ ``results/official_image_accounting.json``) — the official totals;
* ``tools/probe_official_image_path.py`` / ``probe_official_image_preproc.py``
  — the official path's shape chain and single-code-stream fact.

Original docstring follows.

---

Ground-truth controls for the image result.

The model reports 0.0169 % on a single CLIC-style image, versus the paper's
14.62 %.  Before claiming anything about that gap, establish what is actually
being measured:

* how much information is in the 1024-token stream at all (order-0 entropy)?
* how much is left in the original image after the 32x32 quantised reconstruction
  (i.e. what a *lossless* codec would still have to transmit)?
* how do those compare with the classical result for the same file?

If the model's code length already sits at or below the token stream's own
entropy, then the number is dominated by the pipeline's downsampling rather than
by the model's modelling ability -- which is the point worth reporting.
"""

from __future__ import annotations

import glob
import math
import os
import sys
import zlib
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
IMG_DIR = os.path.join(REPO, "data", "test_images")
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")


def order0(values) -> float:
    counts = Counter(values)
    n = sum(counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def order1(values) -> float:
    pairs = Counter(zip(values[:-1], values[1:]))
    singles = Counter(values[:-1])
    total = sum(pairs.values())
    return -sum((c / total) * math.log2(c / singles[a]) for (a, b), c in pairs.items())


def main() -> int:
    from transformers import ImageGPTImageProcessor

    from uic.data import igpt_palette

    proc = ImageGPTImageProcessor.from_pretrained(MODEL_DIR)
    palette = igpt_palette(proc)

    print(f"{'image':<22} {'pixels':>10} {'tokens':>7} {'uniq':>5} "
          f"{'H0(tok)':>8} {'model b/tok':>12} {'model bpp':>10} "
          f"{'resid zlib':>11} {'true rate':>10}")
    for path in sorted(glob.glob(os.path.join(IMG_DIR, "*"))):
        if not path.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w, _ = rgb.shape
        tokens = np.asarray(proc(Image.fromarray(rgb), return_tensors="np")["input_ids"]).reshape(-1)

        low = palette[tokens].reshape(32, 32, 3)
        up = np.asarray(
            Image.fromarray((low * 255.0).round().astype(np.uint8)).resize((w, h), Image.BILINEAR)
        )
        # Residual must be coded per colour plane to match the 32x32x3 basis;
        # code the absolute reconstruction error, which is what a real codec sends.
        err = np.abs(rgb.astype(np.int16) - up.astype(np.int16)).astype(np.uint8)
        resid = len(zlib.compress(err.tobytes(), 9))

        n_tok = tokens.shape[0]
        h0 = order0(tokens.tolist()) * n_tok
        h1 = order1(tokens.tolist()) * max(1, n_tok - 1)
        print(f"{os.path.basename(path)[:21]:<22} {h*w:>10} {n_tok:>7} "
              f"{len(np.unique(tokens)):>5} {h0:>8.0f} {'--':>12} {'--':>10} "
              f"{resid:>11} {'--':>10}")
        print(f"    token stream entropy: H0={h0:.0f} bits, H1={h1:.0f} bits "
              f"({h0/n_tok:.3f} / {h1/n_tok:.3f} bits per token)")
        print(f"    residual after 32x32 reconstruction: {resid} bytes "
              f"({100*resid/(h*w*3):.2f}% of the {h*w*3} RGB bytes)")
        print(f"    token stream + residual = {resid} bytes -> "
              f"true lossless rate {100*resid/(h*w*3):.2f}% "
              f"(the token stream is ~330 bytes, i.e. noise)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
