"""Reproduce the official image evaluation loop and report its exact final numbers.

Reason: the official evaluator prints only every 100th chunk
(``if print_count % 100 == 0``), so the last ``raw_length`` in its log is stale —
it does not cover the final chunks.  Any rate quoted from that log is therefore the
wrong ratio, which is exactly the kind of thing this audit should not do.

This script reimplements the official accumulation faithfully

    raw_length        += h * w                 (per yielded chunk)
    compressed_length += len(compressed_data)  (per yielded chunk)

over the *same* generator the official code uses, and reports both quantities after
the generator is exhausted, alongside the census of chunks.  It also reports the
last printed checkpoint so the discrepancy can be quantified rather than guessed.
"""

from __future__ import annotations

import os
import sys
import time

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

    gen = get_imagenet_iterator(num_chunks=10**9, is_channel_wised=True, is_seq=True,
                                data_path=IMAGE_DIR)

    raw_length = 0
    compressed_length = 0
    n = 0
    last_checkpoint = None
    t0 = time.time()
    for chunk, _img in gen:
        comp, _lp = language_model.compress(chunk, model_path=MODEL_DIR)
        n += 1
        raw_length += chunk.shape[0] * chunk.shape[1]
        compressed_length += len(comp)
        if n % 100 == 0:
            last_checkpoint = (n, raw_length, compressed_length)
        if n % 500 == 0:
            print(f"  {n:>5} chunks  raw={raw_length}  comp={compressed_length}  "
                  f"{time.time()-t0:.0f}s")

    h, w = 1142, 1714
    print("\n=== exact official accounting after exhaustion ===")
    print(f"  chunks processed         = {n}")
    print(f"  raw_length   (final)     = {raw_length}")
    print(f"  compressed   (final)     = {compressed_length}")
    print(f"  rate = comp/raw          = {compressed_length/raw_length:.6f} "
          f"= {100*compressed_length/raw_length:.4f} %")
    print(f"  ratio = raw/comp         = {raw_length/compressed_length:.4f} x")
    print("\n  --- cross-checks ---")
    print(f"  source RGB channel samples = {h*w*3}")
    print(f"  raw_length / samples       = {raw_length/(h*w*3):.6f}")
    print(f"  per-chunk bytes            = {compressed_length/n:.2f}")
    if last_checkpoint:
        ck_n, ck_raw, ck_comp = last_checkpoint
        print(f"\n  last printed checkpoint (chunk {ck_n}): raw={ck_raw} comp={ck_comp}")
        print(f"    rate at that checkpoint  = {100*ck_comp/ck_raw:.4f} %")
        print(f"    final chunks beyond it   = {n - ck_n}")
        print(f"    -> a rate quoted from the official log understates compressed")
        print(f"       length by {compressed_length - ck_comp} bytes "
              f"({100*(compressed_length-ck_comp)/compressed_length:.2f} % of the code)")

    # save for the audit
    import json

    out = os.path.join("repro_understanding_is_compression", "results",
                       "official_image_accounting.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "chunks": n,
                "raw_length": raw_length,
                "compressed_length": compressed_length,
                "rate_pct": 100 * compressed_length / raw_length,
                "ratio": raw_length / compressed_length,
                "source_rgb_samples": h * w * 3,
                "per_chunk_bytes": compressed_length / n,
                "last_printed_checkpoint": last_checkpoint,
            },
            fh,
            indent=2,
        )
    print(f"\n  written to {out}")

    # separate: what does the exact ratio look like per colour channel?
    print("\n=== per-channel breakdown (for reference) ===")
    print(f"  raw_length / 3 = {raw_length/3:.0f} samples per channel")
    print(f"  source per channel = {h*w}")
    print(f"  dropped per channel = {h*w - raw_length//3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
