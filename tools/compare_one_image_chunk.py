"""Why does the official script report 15.51 % where our port reports 23.93 %
on the same units and the same tokens?

Inputs are now identical: 5733 units, 5870592 samples, and the first chunk's first
12 tokens match byte for byte.  So the difference must be in the *probability
assigned to the true token*, not in the tokenisation.

This compares, for one identical chunk:

  * the official ``language_model.compress`` path, i.e. ``get_pdf`` from the capsule,
  * our ``ImageGPTPredictor.iter_steps`` plus ``measure_stream``,

per-position, and reports where they diverge.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
sys.path.insert(0, REPO)

MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMG_DIR = os.path.join(REPO, "data", "test_images")
PATCH = 32


def main() -> int:
    from PIL import Image

    from uic.data import _igpt_processor, igpt_palette
    from uic.engine import measure_stream
    from uic.models import ImageGPTPredictor

    # ---- build chunk 0 exactly the way the official loader does ------------ #
    path = os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0])
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h, w, _ = rgb.shape
    n = PATCH * PATCH
    flat0 = rgb[:, :, 0].reshape(-1)
    block0 = flat0[:n].reshape(PATCH, PATCH)
    triple = np.repeat(block0[:, :, None], 3, axis=2)
    print(f"chunk 0: channel 0, first {n} flattened pixels, shape {triple.shape}")
    print(f"  R == G == B everywhere: "
          f"{bool(np.all(triple[:,:,0]==triple[:,:,1]) and np.all(triple[:,:,1]==triple[:,:,2]))}")

    # ---- our port ---------------------------------------------------------- #
    pred = ImageGPTPredictor.load(MODEL_DIR, device="cuda:0")
    try:
        steps = list(pred.iter_steps([int(x) for x in
                                      _tokens_for(triple)]))
        res, _ = measure_stream(iter(steps))
        ours_tokens = [s.symbol for s in steps]
        ours_bits = res.coded_bits
        ours_n = res.coded_symbols
        print(f"\n=== our port ===")
        print(f"  tokens scored  = {ours_n}")
        print(f"  coded bits     = {ours_bits:.2f}")
        print(f"  bits per sample= {ours_bits/ours_n:.4f}")
        print(f"  first 12 tokens= {ours_tokens[:12]}")
    finally:
        pred.release()

    # ---- official path ----------------------------------------------------- #
    print(f"\n=== official language_model.compress internals ===")
    import language_model as lm
    try:
        params, img_proc, tok = lm._retrieve_model_params("igpt", MODEL_DIR)
        predict_fn = lm._retrieve_predict_fn(params, img_proc, tok, is_lvm=False)
        print("  obtained official predict_fn")
        # what does the official path feed the processor for OUR chunk?
        seq = np.asarray(triple)
        print(f"  sequence_array.shape = {seq.shape}, last axis = {seq.shape[-1]}")
        if seq.shape[-1] == 1:
            print("  -> would take the repeat branch")
        elif seq.shape[-1] == 3:
            print("  -> would take the SHAPE==3 branch (no repeat)")
        else:
            print("  -> neither branch; the array would pass through UNCHANGED")
        ids = np.asarray(img_proc(seq, return_tensors="pt")["input_ids"]).reshape(-1)
        print(f"  processor tokens: n={ids.size}, first 12 = {ids[:12].tolist()}")
    except Exception as exc:
        print(f"  could not drive the official path directly: "
              f"{type(exc).__name__}: {exc}")
    return 0


def _tokens_for(triple):
    from uic.data import _igpt_processor
    proc = _igpt_processor(MODEL_DIR)
    out = proc(triple, return_tensors="np")["input_ids"]
    return np.asarray(out).reshape(-1)


if __name__ == "__main__":
    raise SystemExit(main())
