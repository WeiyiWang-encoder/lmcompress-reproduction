"""What exactly is the denominator of the official image/video compression rate?

Answer, from the authors' code rather than inference:

    compress_with_igpt.py:89-93
        if isinstance(data, tuple): h, w = data[0].shape[:2]
        else:                       h, w = data.shape[:2]
        raw_length += h * w

    compress_with_igpt.py:114
        return compressed_length / raw_length, running_time

so the rate is  (bytes of coded tokens) / (sum over chunks of h*w).

The decisive question is what `data` is at that point, because `compress_fn` is only
called on the NEXT line (84), i.e. `data` is the item yielded by the generator, before
any processor touches it.  And the generator is

    data_loaders.get_imagenet_iterator(is_channel_wised=True, is_seq=True)

which does `first_frame = process_img(path, True)` where `process_img` explicitly does
NOT resize (line 175 is commented out), then splits channel-wise and into 32x32 patches.

Therefore the denominator counts channel samples of the FULL-RESOLUTION SOURCE frame.
This script reconciles that statement against our instrumented measurement.
"""

from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W, H = 1714, 1142
CH = 3
PATCH = 32


def main() -> int:
    print("=== The three candidate denominators for the test image ===")
    source_px = W * H
    source_samples = source_px * CH
    print(f"  1. source PIXELS                       : {source_px:,}")
    print(f"  2. source RGB CHANNEL SAMPLES          : {source_samples:,}")
    print(f"  3. quantised grid (32x32, 512 colours)  : 1024 tokens/chunk")

    print("\n=== What the code sums ===")
    acc = os.path.join(HERE, "results", "official_image_accounting.json")
    data = json.load(open(acc, encoding="utf-8"))
    chunks = data["chunks"]
    raw = data["raw_length"]
    comp = data["compressed_length"]
    print(f"  measured chunks        = {chunks:,}")
    print(f"  measured raw_length    = {raw:,}   <- the denominator")
    print(f"  measured compressed    = {comp:,} bytes")
    print(f"  measured rate          = {data['rate_pct']:.4f} %")

    print("\n=== Reconciliation ===")
    print(f"  raw_length / chunks    = {raw/chunks:.1f} samples per chunk "
          f"(a 32x32 patch = {PATCH*PATCH})")
    print(f"  chunks x patch area    = {chunks*PATCH*PATCH:,}")
    print(f"  measured raw_length    = {raw:,}")
    print(f"  match: {chunks*PATCH*PATCH == raw}")
    print()
    print(f"  source channel samples = {source_samples:,}")
    print(f"  measured raw_length    = {raw:,}")
    print(f"  difference             = {source_samples - raw:,} "
          f"(loader floor division, 524 per channel)")
    print(f"  ratio raw/source       = {100*raw/source_samples:.4f} % "
          f"-> the denominator IS the source channel-sample count")
    print()
    print(f"  check the rate with each candidate denominator:")
    print(f"    / raw_length (as written)   = {100*comp/raw:.4f} %   <- matches the code")
    print(f"    / source samples            = {100*comp/source_samples:.4f} %")
    print(f"    / source pixels             = {100*comp/source_px:.4f} %  (mixes units)")

    print("\n=== Conclusion ===")
    print("  The denominator is the channel-sample count of the FULL-RESOLUTION")
    print("  source image (minus what the loader's floor division discards).")
    print("  It is NOT:")
    print("    * the downscaled/quantised grid -- that is only 1024 tokens/chunk and")
    print("      is what the NUMERATOR counts, which is the whole asymmetry;")
    print("    * the source pixel count -- dividing by that mixes units (bytes over")
    print("      pixels) and gives 46.53 %, which appears nowhere in the paper.")

    print("\n=== Video, same code path ===")
    video = os.path.join(HERE, "results", "video_igpt.jsonl")
    if os.path.exists(video):
        rows = [json.loads(l) for l in open(video, encoding="utf-8") if l.strip()]
        px = sum(r["pixels"] for r in rows)
        rgb = sum(r["rgb_bytes"] for r in rows)
        bits = sum(r["coded_bits"] for r in rows)
        print(f"  we reported rate = {100*(bits/8)/px:.6f} % on a PIXEL denominator")
        print(f"  paper's convention (channel samples) = "
              f"{100*(bits/8)/rgb:.6f} %")
        print(f"  full-resolution source frames, no resize (process_img line 176)")
        print(f"  -> the paper's video denominator is likewise the source frame's")
        print(f"     channel-sample count ({rgb:,}), not the quantised grid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
