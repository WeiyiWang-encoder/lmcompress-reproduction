"""Extract video frames to PNG with the in-process Y4M reader.

Used instead of ffmpeg so the pipeline does not depend on an external binary, and so
frame extraction is bit-exact and reproducible from the repository alone.

Writes frames as ``frame_%06d.png`` plus a ``frames.json`` recording the source
clip's geometry, the raw byte count each frame represents, and the PNG sizes (which
are needed because the total PNG size is a plausible alternative denominator for a
video compression rate -- the paper's own video pipeline goes through PNG).
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--video", default=os.path.join(REPO, "data", "test_videos"))
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts", "frames"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--scale", type=int, default=0, help="downscale longest edge to N (0 = native)")
    args = ap.parse_args()

    from PIL import Image

    from uic import y4m

    videos = []
    if os.path.isfile(args.video):
        videos = [args.video]
    else:
        for name in sorted(os.listdir(args.video)):
            if name.lower().endswith((".y4m", ".yuv")):
                videos.append(os.path.join(args.video, name))
    if not videos:
        print(f"no Y4M files under {args.video}")
        return 1

    manifest = {"clips": []}
    for path in videos:
        info = y4m.probe(path)
        base = os.path.splitext(os.path.basename(path))[0]
        outdir = os.path.join(args.out, base)
        os.makedirs(outdir, exist_ok=True)

        limit = args.limit if args.limit > 0 else None
        w, h = info["width"], info["height"]
        scale = args.scale or 0

        n = 0
        png_bytes = 0
        t0 = time.time()
        for idx, rgb in enumerate(y4m.iter_frames(path, limit=limit)):
            img = Image.fromarray(rgb)
            if scale and max(img.size) > scale:
                ratio = scale / max(img.size)
                img = img.resize((max(1, round(w * ratio)), max(1, round(h * ratio))),
                                 Image.BILINEAR)
            dst = os.path.join(outdir, f"frame_{idx:06d}.png")
            img.save(dst, optimize=False, compress_level=6)
            png_bytes += os.path.getsize(dst)
            n += 1
            if n % 10 == 0:
                print(f"  {base}: {n} frames, {time.time()-t0:.0f}s", flush=True)

        entry = {
            "clip": base,
            "path": path,
            "width": w,
            "height": h,
            "fps": info["fps"],
            "chroma": info["chroma"],
            "frames": n,
            "frames_dir": outdir,
            "raw_yuv_bytes_per_frame": w * h * 3 // 2,
            "raw_rgb_bytes_per_frame": w * h * 3,
            "raw_yuv_bytes_total": w * h * 3 // 2 * n,
            "raw_rgb_bytes_total": w * h * 3 * n,
            "png_bytes_total": png_bytes,
            "png_bytes_per_frame": png_bytes / n if n else 0,
            "scale": scale,
            "seconds": round(time.time() - t0, 2),
        }
        manifest["clips"].append(entry)
        print(json.dumps(entry, indent=2))

    out = os.path.join(args.out, "frames.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
