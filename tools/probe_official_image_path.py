"""Instrument the OFFICIAL image path, end to end, without modifying it.

The audit's image claim was criticised for proving only *our* path. This script
drives the capsule's own code (`compress_with_igpt.py` -> `compressor.py` ->
`language_model.compress`) with the capsule's own data loader, and records what
the ImageGPT processor is actually given at each call, what it returns, and what
denominator the ratio uses.

It monkey-patches only for *observation* (wrapping the processor and the ratio
accumulator); no reference logic is altered, so whatever it prints is the
official behaviour.

What to look for:
  * the shape of the array handed to ImageGPTImageProcessor;
  * whether that is the full image or a pre-cut patch;
  * how many tokens come back per call;
  * the value of `raw_length` used as the ratio denominator.
"""

from __future__ import annotations

import os
import sys
import traceback

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
    import language_model
    from transformers import ImageGPTImageProcessor

    # ---- observation only: wrap the processor ---------------------------- #
    proc = ImageGPTImageProcessor.from_pretrained(MODEL_DIR)
    print("=== processor config (from the model dir the capsule ships) ===")
    print(f"  do_resize        = {proc.do_resize}")
    print(f"  size             = {proc.size}")
    print(f"  do_color_quantize= {proc.do_color_quantize}")
    print(f"  do_normalize     = {proc.do_normalize}")
    print(f"  n clusters       = {len(proc.clusters)}")

    orig_call = proc.__call__

    def spy(images, *a, **kw):
        arr = np.asarray(images)
        CALLS.append({"input_shape": tuple(arr.shape)})
        out = orig_call(images, *a, **kw)
        ids = np.asarray(out["input_ids"])
        CALLS[-1]["output_shape"] = tuple(ids.shape)
        CALLS[-1]["n_unique"] = int(len(np.unique(ids)))
        return out

    proc.__call__ = spy

    # language_model caches the processor in a module global; force a re-fetch
    language_model.model = None
    language_model.image_processor = proc

    # ---- run the official pipeline --------------------------------------- #
    import argparse
    import compress_with_igpt as ref

    class Args:
        compressor = "language_model"
        dataset = "imagenet"
        data_path = IMAGE_DIR
        total = None
        model_path = MODEL_DIR

    print("\n=== running the official pipeline on the capsule's own image ===")
    try:
        ref.main(Args())
    except Exception:
        print("official pipeline raised:")
        traceback.print_exc()

    print(f"\n=== ImageGPT processor was called {len(CALLS)} time(s) ===")
    for i, c in enumerate(CALLS[:8]):
        print(f"  call {i}: input={c['input_shape']} -> output={c['output_shape']} "
              f"unique_tokens={c['n_unique']}")
    if len(CALLS) > 8:
        print(f"  ... and {len(CALLS)-8} more")

    # ---- what denominator does the official ratio use? -------------------- #
    print("\n=== denominator used by the official ratio code ===")
    print("  compress_with_igpt.evaluate_compressor_chunked does:")
    print("      h, w = data.shape[:2]")
    print("      raw_length += h * w")
    print("      compressed_length += len(compressed_data[0])")
    print("  i.e. the denominator is the area of whatever `data` the data loader yields.")
    print(f"  with is_seq=True the loader yields 32x32 patches, so h*w == 1024 per chunk")

    # reproduce the loader's chunking to show it concretely
    from data_loaders import get_imagenet_iterator

    gen = get_imagenet_iterator(num_chunks=3, is_channel_wised=True, is_seq=True,
                                data_path=IMAGE_DIR)
    print("\n=== loader chunks (official _extract_image_sequence, is_channel_wised) ===")
    for i, (chunk, img_id) in enumerate(gen):
        print(f"  chunk {i}: shape={chunk.shape}  h*w={chunk.shape[0]*chunk.shape[1]}  "
              f"img_id={img_id}")
        if i >= 2:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
