"""Video modality via the ImageGPT path, with the paper's convention stated explicitly.

The paper's video method is: treat each frame as an image, code it independently with
iGPT, and never use interframe information (NMI, Fig. 4 discussion).  So the model
number for video is just the image model applied frame by frame.

Two conventions are reported, because the paper's is not self-evident and this
project has already been burned once by a denominator mismatch:

  * ``paper convention``: compressed bytes / raw pixel count, where each frame is
    first quantised by the ImageGPT processor to a 32x32, 512-colour token grid.
    This is the same convention §4.4.2 of the audit established for images.
  * ``raw RGB bytes``: the same numerator over the frame's true RGB byte count, so a
    reader cannot mistake it for a rate on the source video.

Per-frame results are appended to JSONL and each frame is checkpointed, so a power
cut or Ctrl-C costs at most one frame.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=os.path.join(
        HERE, "artifacts", "frames", "4k_aerial_first_second"))
    ap.add_argument("--model-dir", default=os.path.join(
        REPO, "data", "model", "imagegpt-small"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "video_igpt.jsonl"))
    ap.add_argument("--charge-residual", action="store_true", default=True)
    args = ap.parse_args()

    from uic.data import image_units
    from uic.models import ImageGPTPredictor

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.png")))
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"no frames in {args.frames_dir}; run tools/extract_video_frames.py first")
        return 1

    with open(os.path.join(os.path.dirname(args.frames_dir), "frames.json"),
              encoding="utf-8") as fh:
        fmeta = json.load(fh)["clips"][0]
    geom = (fmeta["width"], fmeta["height"])
    scale = fmeta.get("scale") or 0
    print(f"{len(frames)} frames from {fmeta['clip']} "
          f"({geom[0]}x{geom[1]}{f' scaled to {scale}' if scale else ''}), "
          f"raw RGB {fmeta['raw_rgb_bytes_per_frame']} B/frame")

    predictor = ImageGPTPredictor.load(args.model_dir, device="cuda:0")

    # Each frame is coded independently, exactly as the paper does.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["unit"])

    tot_tokens = tot_bits = tot_pixels = tot_rgb = tot_resid = 0
    t0 = time.time()
    try:
        for i, path in enumerate(frames, 1):
            name = os.path.basename(path)
            if name in done:
                print(f"[{i}/{len(frames)}] {name}: already measured, skipping")
                continue
            # one frame -> one unit; image_units reads the directory, so isolate it
            units = image_units(os.path.dirname(path) or ".", args.model_dir,
                                limit=None, extensions=(".png",),
                                include_residual=args.charge_residual)
            unit = next(u for u in units if os.path.basename(u.name) == name)

            import numpy as np

            from uic.engine import measure_stream
            from uic.runner import plan_windows

            windows = plan_windows(len(unit.payload), predictor.window_capacity, "igpt")
            bits = 0
            for w in windows:
                res, _ = measure_stream(predictor.iter_steps(unit.payload[w.start:w.stop]))
                bits += res.coded_bits

            h, w = unit.meta["height"], unit.meta["width"]
            pixels = h * w
            rgb_bytes = pixels * 3
            resid = len(unit.meta.get("residual") or b"")
            import zlib

            resid_z = len(zlib.compress(bytes(unit.meta.get("residual") or b""), 9))
            row = {
                "unit": name,
                "frame_index": i - 1,
                "tokens": len(unit.payload),
                "unique_tokens": len(set(unit.payload)),
                "pixels": pixels,
                "rgb_bytes": rgb_bytes,
                "coded_bits": round(bits, 6),
                "coded_bytes": round(bits / 8, 3),
                "rate_pct_paper_convention": round(100.0 * (bits / 8) / pixels, 6),
                "rate_pct_rgb_bytes": round(100.0 * (bits / 8) / rgb_bytes, 6),
                "ratio_paper_convention": round(pixels / (bits / 8), 6),
                "residual_bytes": resid,
                "residual_zlib9": resid_z,
                "exact_completion_bytes": round(bits / 8 + resid_z, 3),
                "exact_completion_rate_pct_rgb": round(
                    100.0 * (bits / 8 + resid_z) / rgb_bytes, 6),
                "wall_seconds": None,
            }
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            tot_tokens += row["tokens"]
            tot_bits += bits
            tot_pixels += pixels
            tot_rgb += rgb_bytes
            tot_resid += resid_z
            print(f"[{i}/{len(frames)}] {name}: {row['tokens']} tokens, "
                  f"{row['coded_bytes']:.0f} B, paper-convention "
                  f"{row['rate_pct_paper_convention']:.4f}%", flush=True)
    finally:
        predictor.release()

    print("\n=== video via iGPT (frame-independent, as the paper does) ===")
    print(f"  frames            = {len(frames)}")
    print(f"  total pixels      = {tot_pixels}")
    print(f"  total RGB bytes   = {tot_rgb}")
    print(f"  model code bytes  = {tot_bits/8:.1f}")
    print(f"  rate (paper conv) = {100*(tot_bits/8)/tot_pixels:.4f} %")
    print(f"  rate (RGB bytes)  = {100*(tot_bits/8)/tot_rgb:.4f} %")
    print(f"  + residual (zlib9) for exact reconstruction: {tot_resid} B")
    print(f"    -> exact-completion rate on RGB = "
          f"{100*((tot_bits/8)+tot_resid)/tot_rgb:.4f} %")
    print(f"  wall = {time.time()-t0:.0f}s")
    print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
