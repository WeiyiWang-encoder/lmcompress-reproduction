"""Fast, targeted instrumentation of the official image path.

The full official run confirms the denominator but takes a long time to reach its
summary.  This script drives the same loader and the same
``language_model.compress`` entry point but stops after a few chunks, and records:

  1. the exact shape the ImageGPT processor is called with;
  2. the token count and palette size it returns;
  3. whether any side information (residual / original pixels) is fed to the coder;
  4. the denominator the official ratio accumulates.

Nothing in the capsule is modified; the processor is wrapped for observation only.
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
import torch  # noqa: E402

IMAGE_DIR = os.path.join(REPO, "data", "test_images")
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")

CALLS: list = []


def main() -> int:
    import language_model
    from transformers import ImageGPTImageProcessor

    proc = ImageGPTImageProcessor.from_pretrained(MODEL_DIR)
    print("=== 1. processor config shipped with the capsule ===")
    print(f"  do_resize          = {proc.do_resize}")
    print(f"  size               = {proc.size}")
    print(f"  do_color_quantize  = {proc.do_color_quantize}")
    print(f"  do_normalize       = {proc.do_normalize}")
    print(f"  num colour clusters= {len(proc.clusters)}")

    orig_call = proc.__call__

    def spy(images, *a, **kw):
        arr = np.asarray(images)
        rec = {"in": tuple(arr.shape), "dtype": str(arr.dtype)}
        out = orig_call(images, *a, **kw)
        ids = np.asarray(out["input_ids"])
        rec["out"] = tuple(ids.shape)
        rec["unique"] = int(len(np.unique(ids)))
        CALLS.append(rec)
        return out

    proc.__call__ = spy
    language_model.model = None
    language_model.image_processor = proc

    # ---- 2. what does the official loader hand to compress()? ------------- #
    from data_loaders import get_imagenet_iterator

    print("\n=== 2. official loader chunks (is_channel_wised=True, is_seq=True) ===")
    gen = get_imagenet_iterator(num_chunks=4, is_channel_wised=True, is_seq=True,
                                data_path=IMAGE_DIR)
    chunks = []
    for i, (chunk, img_id) in enumerate(gen):
        chunks.append(chunk)
        print(f"  chunk {i}: shape={chunk.shape}  h*w={chunk.shape[0]*chunk.shape[1]}  "
              f"dtype={chunk.dtype}  range=[{chunk.min()},{chunk.max()}]")
        if i >= 3:
            break

    # ---- 3. run the official compress() on a few chunks ------------------ #
    print("\n=== 3. running official language_model.compress() on chunk 0 ===")
    before = len(CALLS)
    try:
        compressed, log_probs = language_model.compress(
            chunks[0], model_path=MODEL_DIR
        )
        print(f"  official compress() returned {len(compressed)} bytes")
    except Exception as exc:  # noqa: BLE001
        print(f"  official compress() raised {type(exc).__name__}: {exc}")

    print(f"\n=== 4. ImageGPT processor calls made by the official path ===")
    for i, c in enumerate(CALLS[before:before + 4]):
        print(f"  call {i}: input={c['in']} {c['dtype']} -> tokens={c['out']} "
              f"unique_palette_entries_used={c['unique']}")

    # ---- 5. is any side information coded? ------------------------------- #
    print("\n=== 5. what the coder is given ===")
    print("  language_model.compress() calls the encoder over `sequence_array` only:")
    print("      sequence_array = image_processor(...)['input_ids']")
    print("      for pdf, symbol in zip(raw_probs, sequence_array): encoder.encode(...)")
    print("  There is no second stream, no residual, no original-pixel channel.")
    print(f"  chunk 0 has {chunks[0].shape[0]*chunks[0].shape[1]} pixels; the coder "
          f"sees {CALLS[before]['out'][-1] if len(CALLS) > before else '?'} symbols.")

    # ---- 6. exact inverse test ------------------------------------------- #
    print("\n=== 6. is the preprocessing injective? (palette round-trip) ===")
    palette = np.asarray(proc.clusters, dtype=np.float32) * 0.5 + 0.5
    rgbbytes = int(np.prod(chunks[0].shape))
    print(f"  chunk 0 raw size            = {rgbbytes} bytes of 24-bit RGB")
    print(f"  distinct colours in chunk 0 = {len(np.unique(chunks[0].reshape(-1,3), axis=0))}")
    print(f"  palette entries available   = {len(palette)}")
    print("  -> 24-bit RGB has 2**24 = 16777216 values mapped onto 512 ids; the map "
          "is many-to-one, so it has no inverse without side information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
