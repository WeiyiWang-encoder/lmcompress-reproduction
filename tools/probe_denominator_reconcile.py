"""Reconcile the official denominator exactly.

Three candidate values for "how much data was compressed", and they differ:
  A = 5 836 800  what the official run actually logged as raw_length
  B = 5 870 592  what iterating the loader to exhaustion yields
                 (per channel floor(1957388/1024)*1024 = 1956864, x3)
  C = 5 872 164  the source RGB channel samples

This runs the official evaluation loop with a census wrapper so the chunk count and
the denominator are read off the same pass, removing the guesswork.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
os.chdir(REPO)
sys.path.insert(0, REPO)

IMAGE_DIR = os.path.join(REPO, "data", "test_images")
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")


def main() -> int:
    import language_model
    from data_loaders import get_imagenet_iterator

    # census: how many chunks and pixels does the generator really produce?
    raw_gen = get_imagenet_iterator

    def counting_gen(*a, **kw):
        n = 0
        px = 0
        for chunk, img in raw_gen(*a, **kw):
            n += 1
            px += chunk.shape[0] * chunk.shape[1]
            yield chunk, img
        print(f"  [census] generator exhausted after {n} chunks, sum(h*w) = {px}")

    import compress_with_igpt as ref

    # call the official evaluator, but with the census wrapper installed
    ref.data_loaders.GET_DATA_GENERATOR_FN_DICT["imagenet"] = counting_gen

    print("=== running the official evaluator with a census wrapper ===")
    class Args:
        compressor = "language_model"
        dataset = "imagenet"
        data_path = IMAGE_DIR
        total = None
        model_path = MODEL_DIR

    try:
        ref.main(Args())
    except Exception as exc:  # noqa: BLE001
        print(f"official evaluator raised {type(exc).__name__}: {exc}")

    print("\n=== candidates ===")
    h, w = 1142, 1714
    print(f"  A official logged raw_length        = 5836800")
    print(f"  B loader-to-exhaustion              = {(h*w//1024)*1024*3}")
    print(f"  C source RGB channel samples        = {h*w*3}")
    print(f"  A/B = {5836800/((h*w//1024)*1024*3):.6f}")
    print(f"  chunks in A = {5836800/1024:.0f}   chunks in B = {(h*w//1024)*3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
