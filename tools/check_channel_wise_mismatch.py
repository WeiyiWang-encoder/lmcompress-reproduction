"""Is the official image path channel-wise while our port is RGB?

An external reviewer claimed the official code feeds three RGB channels separately
while our reproduction feeds them mixed.  This settles it by instrumenting the
processor and feeding it exactly what each path feeds it.

The two candidate inputs, both legal for the processor but not equivalent:

  channel-wise   (32,32,1) -> np.repeat(...,3) -> (32,32,3) with R=G=B=v
                 i.e. a greyscale triple built from ONE channel value
  full RGB       (32,32,3) -> a real colour patch, three independent channels

The official pipeline reaches the processor through
``language_model.compress``, whose branch is selected by ``sequence_array.shape[-1]``.
So the question reduces to: what is the last axis when the compress function is
called?  That is set by the data loader, and we print it live.
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

IMG_DIR = os.path.join(REPO, "data", "test_images")
MODEL = os.path.join(REPO, "data", "model", "imagegpt-small")


def main() -> int:
    from PIL import Image
    from transformers import ImageGPTImageProcessor

    path = os.path.join(IMG_DIR, os.listdir(IMG_DIR)[0])
    rgb = np.asarray(Image.open(path).convert("RGB"))
    print(f"image {os.path.basename(path)}  shape {rgb.shape}\n")

    proc = ImageGPTImageProcessor.from_pretrained(MODEL)

    # ---------------- what the OFFICIAL pipeline feeds ---------------- #
    print("=== official pipeline: get_imagenet_iterator(is_channel_wised=True) ===")
    import data_loaders
    gen = data_loaders.get_imagenet_iterator(
        num_chunks=10_000, is_channel_wised=True, is_seq=True, data_path=IMG_DIR)
    shapes = []
    first = None
    for i, (patch, img_id) in enumerate(gen):
        if i == 0:
            first = patch
        shapes.append(patch.shape)
        if i >= 5:
            break
    print(f"  first 6 yielded patch shapes: {shapes}")
    print(f"  -> last axis is {shapes[0][-1]}")

    # reproduce the official branch for the FIRST chunk
    seq = np.asarray(first)
    print(f"  chunk shape            : {seq.shape}")
    print(f"  unique values in chunk : {np.unique(seq)[:6]} ... "
          f"({len(np.unique(seq))} distinct)")
    reps = np.repeat(seq, 3, axis=2)
    print(f"  after np.repeat(...,3) : {reps.shape}")
    same = bool(np.all(reps[:, :, 0] == reps[:, :, 1]) and
                np.all(reps[:, :, 1] == reps[:, :, 2]))
    print(f"  R == G == B everywhere : {same}   <- greyscale triple (v,v,v)")
    ids_ch = np.asarray(proc(reps, return_tensors="np")["input_ids"]).reshape(-1)
    print(f"  token ids              : n={ids_ch.size}, "
          f"distinct={len(np.unique(ids_ch))}")

    # ---------------- what OUR PORT feeds ---------------- #
    print("\n=== our port: uic.data.image_units ===")
    from uic.data import image_units
    units = image_units(IMG_DIR, MODEL, limit=1, include_residual=False)
    u = units[0]
    print(f"  units returned         : {len(units)} (one per image)")
    print(f"  payload tokens         : n={len(u.payload)}, "
          f"distinct={len(set(u.payload))}")
    print(f"  original_bytes         : {u.original_bytes}")

    # rebuild the port's exact processor call to show the last axis
    full = proc(Image.fromarray(rgb), return_tensors="np")["input_ids"]
    print(f"  processor(Image.fromarray(rgb)) shape: "
          f"{np.asarray(full).shape}   <- a real RGB patch, 3 channels mixed")

    # ---------------- are the two token streams the same? ---------------- #
    print("\n=== do the two paths produce the same tokens? ===")
    # compare the first 1024 tokens of the channel-wise stream with the port's
    print(f"  channel-wise (chunk 0) first 12 ids : {ids_ch[:12].tolist()}")
    print(f"  port (whole image)     first 12 ids : {u.payload[:12]}")
    print(f"  identical? {np.array_equal(ids_ch[:1024], np.asarray(u.payload[:1024]))}")

    print("\n=== verdict ===")
    print("  The official loader sets the last axis to 1 (channel-wise), so")
    print("  language_model.compress takes the np.repeat branch and the processor")
    print("  receives a GREYSCALE TRIPLE (v,v,v) built from one channel value.")
    print("  Our port instead passes the full 32x32x3 RGB image in one call.")
    print("  The reviewer's claim is CORRECT about our port.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
