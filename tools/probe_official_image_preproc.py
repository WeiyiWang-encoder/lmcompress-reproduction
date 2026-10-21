"""Confirm what preprocessing the official image path applies, and what it codes.

The previous probe recorded 0 processor calls because ``_retrieve_model_params``
re-fetches its own processor from ``from_pretrained``, replacing the instance we
wrapped.  This version patches at the *class* level, so every instance is observed,
and then verifies by hand:

  * the shape of the array the processor receives;
  * how many tokens come back and from how large a palette;
  * how many bytes the official coder produces for those tokens;
  * whether the official ratio denominator is the patch area or the source image.
"""

from __future__ import annotations

import os
import sys

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

IMAGE_DIR = os.path.join(REPO, "data", "test_images")
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")

CALLS: list = []


def main() -> int:
    from transformers import ImageGPTImageProcessor

    # ---- class-level spy -------------------------------------------------- #
    orig_call = ImageGPTImageProcessor.__call__

    def spy(self, images, *a, **kw):
        arr = np.asarray(images)
        rec = {"in": tuple(arr.shape), "dtype": str(arr.dtype),
               "range": (float(arr.min()), float(arr.max()))}
        out = orig_call(self, images, *a, **kw)
        ids = np.asarray(out["input_ids"])
        rec["out"] = tuple(ids.shape)
        rec["unique"] = int(len(np.unique(ids)))
        CALLS.append(rec)
        return out

    ImageGPTImageProcessor.__call__ = spy

    import language_model  # noqa: E402
    from data_loaders import get_imagenet_iterator  # noqa: E402

    gen = get_imagenet_iterator(num_chunks=2, is_channel_wised=True, is_seq=True,
                                data_path=IMAGE_DIR)
    chunk = next(iter(gen))[0]
    print("=== official loader chunk ===")
    print(f"  shape={chunk.shape}  dtype={chunk.dtype}  "
          f"range=[{chunk.min()}, {chunk.max()}]  pixels={chunk.shape[0]*chunk.shape[1]}")

    print("\n=== running the OFFICIAL language_model.compress() on it ===")
    before = len(CALLS)
    compressed, log_probs = language_model.compress(chunk, model_path=MODEL_DIR)
    print(f"  official coder produced {len(compressed)} bytes")
    print(f"  log_probs shape {tuple(np.asarray(log_probs).shape)} "
          f"(one row per coded symbol)")

    print(f"\n=== processor calls during that call: {len(CALLS) - before} ===")
    for i, c in enumerate(CALLS[before:]):
        print(f"  call {i}: input={c['in']} {c['dtype']} range={c['range']} "
              f"-> tokens={c['out']} unique_palette_ids={c['unique']}")

    n_tokens = CALLS[-1]["out"][-1] if CALLS else chunk.shape[0] * chunk.shape[1]
    n_sym = np.asarray(log_probs).shape[0]

    print("\n=== accounting ===")
    print(f"  loader chunk pixels (official denominator h*w) = {chunk.shape[0]*chunk.shape[1]}")
    print(f"  processor tokens                               = {n_tokens}")
    print(f"  symbols actually coded                         = {n_sym}")
    print(f"  official coder bytes                           = {len(compressed)}")
    print(f"  official ratio for this chunk                  = "
          f"{len(compressed)/(chunk.shape[0]*chunk.shape[1]):.6f} "
          f"({100*len(compressed)/(chunk.shape[0]*chunk.shape[1]):.4f} %)")

    # source image size for contrast
    from PIL import Image

    img = Image.open(os.path.join(IMAGE_DIR, os.listdir(IMAGE_DIR)[0])).convert("RGB")
    w, h = img.size
    print(f"\n  source image = {w}x{h} = {w*h} pixels = {w*h*3} RGB bytes")
    print(f"  official denominator per chunk = {chunk.shape[0]*chunk.shape[1]} pixels")
    print(f"  -> the denominator counts the 32x32 PATCH area, once per patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
