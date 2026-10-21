"""Exact volume accounting for the official channel-wise image loader.

The official run logged ``raw_length = 5 836 800`` while the source image has
``1714 * 1142 * 3 = 5 872 164`` uint8 channel samples -- a deficit of 35 364
(0.602 %).  Whether that deficit is dropped edge data matters, so this counts the
loader's chunks exactly instead of inferring them.

Note the loader's own arithmetic: ``_extract_image_sequence`` computes
``total_chunks = total_pixels // sequence_length`` (floor division), so any tail
shorter than one 32x32 patch is never yielded.  It also ``reshape(-1, C)``, which
means the "sequence" is a flat raster walk, not a grid of aligned tiles.
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


def main() -> int:
    from PIL import Image

    import glob

    path = sorted(glob.glob(os.path.join(IMAGE_DIR, "*")))[0]
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h, w, c = rgb.shape
    print(f"source {os.path.basename(path)}: {w}x{h}x{c}")
    print(f"total channel samples = {h*w*c}")

    from data_loaders import get_imagenet_iterator

    for channel_wise, seq in ((True, True), (False, True), (True, False), (False, False)):
        gen = get_imagenet_iterator(num_chunks=10**9, is_channel_wised=channel_wise,
                                    is_seq=seq, data_path=IMAGE_DIR)
        n = 0
        pixels = 0
        shapes = set()
        for chunk, _img in gen:
            n += 1
            pixels += chunk.shape[0] * chunk.shape[1]
            shapes.add(chunk.shape)
            if n > 20000:
                break
        print(f"\nis_channel_wised={channel_wise} is_seq={seq}")
        print(f"  chunks = {n}   sum(h*w) = {pixels}   shapes = {sorted(shapes)}")
        print(f"  processed fraction of source samples = {100*pixels/(h*w*c):.4f} %")

    # what the official denominator should be for each configuration
    print("\n--- reconciliation ---")
    expected_channelwise = (h * w // 1024) * 1024 * c
    print(f"  floor(h*w/1024)*1024*c = {expected_channelwise}  (channel-wise, seq)")
    print(f"  h*w*c                  = {h*w*c}  (no chunking)")
    print(f"  observed official      = 5836800")
    print(f"  floor(h*w/1024) per ch = {h*w//1024}  -> x1024 = {(h*w//1024)*1024} per channel")
    print(f"  remainder per channel  = {h*w - (h*w//1024)*1024} samples never yielded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
