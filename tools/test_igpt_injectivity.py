"""Exhaustive injectivity test of the official ImageGPT colour quantiser.

Why this test and not a cardinality argument.

The official image path splits the source into 32x32 *single-channel* patches and
then, inside ``language_model.compress``, does

    sequence_array = np.repeat(sequence_array, 3, axis=2)

so the processor receives ``(v, v, v)`` built from ONE 8-bit channel value. The
domain is therefore 256 values, not 2**24, and the codomain is 512 palette ids.
A cardinality argument cannot prove information loss here: 512 > 256, so the map
could in principle be injective.

The only rigorous way to settle it is to enumerate the domain. This script pushes
all 256 values ``(v, v, v)`` through the **official processor** and counts
collisions, then repeats the test on real patch data to exhibit a collision that
actually occurs in the pipeline.

If any two distinct ``a != b`` map to the same token id, then a coder that emits
only that id -- with no residual -- cannot distinguish them, which is exactly the
end-to-end losslessness claim at issue.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import numpy as np

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
os.chdir(REPO)
sys.path.insert(0, REPO)

MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMAGE_DIR = os.path.join(REPO, "data", "test_images")


def main() -> int:
    from transformers import ImageGPTImageProcessor

    proc = ImageGPTImageProcessor.from_pretrained(MODEL_DIR)
    print("=== official processor ===")
    print(f"  do_resize={proc.do_resize} size={proc.size} "
          f"do_color_quantize={proc.do_color_quantize} clusters={len(proc.clusters)}")

    # ---- 1. exhaustive domain enumeration -------------------------------- #
    print("\n=== 1. all 256 grayscale values (v,v,v) through the official processor ===")
    mapping: dict[int, int] = {}
    for v in range(256):
        patch = np.full((32, 32, 3), v, dtype=np.uint8)
        ids = np.asarray(proc(patch, return_tensors="np")["input_ids"]).reshape(-1)
        uniq = np.unique(ids)
        mapping[v] = int(uniq[0])
        if len(uniq) != 1:
            print(f"  NOTE: v={v} returned {len(uniq)} distinct ids {uniq.tolist()[:5]}")

    n_distinct_ids = len(set(mapping.values()))
    print(f"  domain size  = 256")
    print(f"  image size   = {n_distinct_ids}")
    print(f"  injective    = {n_distinct_ids == 256}")

    by_id: dict[int, list[int]] = defaultdict(list)
    for v, i in mapping.items():
        by_id[i].append(v)
    collisions = {i: vs for i, vs in by_id.items() if len(vs) > 1}
    print(f"  tokens receiving >1 source value : {len(collisions)}")
    if collisions:
        total_pairs = sum(len(vs) * (len(vs) - 1) // 2 for vs in collisions.values())
        print(f"  colliding source-value pairs     : {total_pairs}")
        print("\n  --- first 10 collisions (these are the proof) ---")
        for i, vs in list(sorted(collisions.items()))[:10]:
            print(f"    token {i:>3}  <-  grayscale values {vs}")
        worst = max(collisions.items(), key=lambda kv: len(kv[1]))
        print(f"\n  --- largest collision class ---")
        print(f"    token {worst[0]} <- {len(worst[1])} distinct values "
              f"({worst[1][0]}..{worst[1][-1]})")
    else:
        print("\n  NO COLLISIONS: on this domain the quantiser is injective, so a")
        print("  cardinality argument would have been wrong and the lossiness claim")
        print("  must rest on the dropped-edge/volume argument instead.")

    # save the full table for the record
    out = os.path.join("repro_understanding_is_compression", "results",
                       "igpt_grayscale_collisions.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "domain": 256,
                "image_size": n_distinct_ids,
                "injective": n_distinct_ids == 256,
                "mapping": {str(k): v for k, v in mapping.items()},
                "collisions": {str(k): v for k, v in collisions.items()},
            },
            fh,
            indent=2,
        )
    print(f"\n  full mapping written to {out}")

    # ---- 2. a collision that occurs on real data ------------------------- #
    print("\n=== 2. collisions in an actual source patch ===")
    import glob

    path = sorted(glob.glob(os.path.join(IMAGE_DIR, "*")))[0]
    from PIL import Image

    rgb = np.asarray(Image.open(path).convert("RGB"))
    print(f"  source image {os.path.basename(path)}  {rgb.shape[1]}x{rgb.shape[0]}")

    from data_loaders import get_imagenet_iterator

    gen = get_imagenet_iterator(num_chunks=200, is_channel_wised=True, is_seq=True,
                                data_path=IMAGE_DIR)
    worst_patch = None
    for idx, (chunk, _img) in enumerate(gen):
        vals = np.unique(chunk)
        f = np.repeat(chunk, 3, axis=2)
        ids = np.asarray(proc(f, return_tensors="np")["input_ids"]).reshape(-1)
        pairs = set(zip(chunk.reshape(-1).tolist(), ids.tolist()))
        src_of_id: dict[int, set] = defaultdict(set)
        for s, t in pairs:
            src_of_id[t].add(s)
        coll = {t: s for t, s in src_of_id.items() if len(s) > 1}
        if coll and (worst_patch is None or len(coll) > worst_patch[1]):
            worst_patch = (idx, len(coll), len(vals), coll, chunk)
        if idx >= 120:
            break

    if worst_patch:
        idx, ncoll, nvals, coll, chunk = worst_patch
        print(f"  patch #{idx}: {nvals} distinct source values -> "
              f"{len(set(np.asarray(proc(np.repeat(chunk,3,axis=2), return_tensors='np')['input_ids']).reshape(-1)))} distinct tokens")
        print(f"  token ids receiving >1 source value: {ncoll}")
        for t, s in list(coll.items())[:8]:
            print(f"    token {t:>3}  <-  source values {sorted(s)}")
        print("\n  -> a decoder holding only the token id cannot tell those source")
        print("     values apart, so the shipped path is not lossless w.r.t. its")
        print("     source channel samples.")
    else:
        print("  no collision observed in the sampled patches")

    # ---- 3. volume: how many source samples actually get processed? ------ #
    print("\n=== 3. volume accounting ===")
    h, w, _ = rgb.shape
    print(f"  source RGB bytes                 = {h*w*3}")
    print(f"  observed official raw_length     = 5836800")
    print(f"  difference                       = {h*w*3 - 5836800}")
    tiles_h, tiles_w = h // 32, w // 32
    print(f"  32x32 tiles per channel          = {tiles_h} x {tiles_w} = {tiles_h*tiles_w}")
    print(f"    x1024 per tile x3 channels     = {tiles_h*tiles_w*1024*3}")
    print(f"  dropped edge rows/cols           = {h - tiles_h*32}, {w - tiles_w*32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
