"""Determine exactly how ImageGPT's processor quantises, so the residual is correct.

transformers 5.x removed ``decode_color_quantization``, so the inverse has to be
re-derived.  The open questions are what scale the 512 cluster centres live on and
what pre-processing is applied before nearest-centre lookup.  Rather than guess,
this compares candidate reconstructions against the real resized image and reports
which one actually inverts the processor -- and confirms it by checking that
nearest-centre lookup reproduces the token ids exactly.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMAGE_DIR = os.path.join(REPO, "data", "test_images")


def find_image() -> str:
    for name in sorted(os.listdir(IMAGE_DIR)):
        if name.lower().endswith((".png", ".bmp", ".jpg", ".jpeg")):
            return os.path.join(IMAGE_DIR, name)
    raise FileNotFoundError(IMAGE_DIR)


def main() -> int:
    from transformers import ImageGPTImageProcessor

    proc = ImageGPTImageProcessor.from_pretrained(MODEL_DIR)
    clusters = np.asarray(proc.clusters, dtype=np.float32)
    print(f"clusters: shape={clusters.shape} range=[{clusters.min():.4f}, {clusters.max():.4f}]")

    path = find_image()
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h, w, _ = rgb.shape
    print(f"image {os.path.basename(path)}  {w}x{h}")

    ids = np.asarray(proc(Image.fromarray(rgb), return_tensors="np")["input_ids"]).reshape(-1)
    print(f"tokens: n={ids.shape[0]}  unique={len(np.unique(ids))}  "
          f"range=[{ids.min()}, {ids.max()}]")

    resized = proc.resize(Image.fromarray(rgb), proc.size, proc.resample, "channels_last")
    resized = np.asarray(resized, dtype=np.float32) / 255.0
    print(f"resized before quantise: range=[{resized.min():.4f}, {resized.max():.4f}]")

    target = resized.reshape(-1, 3)
    candidates = {
        "centres*0.5+0.5": clusters * 0.5 + 0.5,
        "centres (as-is)": clusters,
        "(centres+1)/2": (clusters + 1.0) / 2.0,
    }
    print(f"\n{'candidate':22s} {'MAE vs resized':>15s} {'corr':>8s}")
    best = None
    for name, table in candidates.items():
        rec = table[ids]
        mae = float(np.abs(rec - target).mean())
        corr = float(np.corrcoef(rec.reshape(-1), target.reshape(-1))[0, 1])
        print(f"{name:22s} {mae:15.6f} {corr:8.4f}")
        if best is None or mae < best[1]:
            best = (name, mae, table)

    name, mae, table = best
    print(f"\nbest: {name}  MAE={mae:.6f}")

    # Nearest-centre lookup must reproduce the ids exactly.  If it does, the table
    # is the right one and the scale is just a linear map, so the inverse is exact
    # up to the resampling step.
    d = ((target[:, None, :] - table[None, :, :]) ** 2).sum(-1)
    agree = float((d.argmin(1) == ids).mean())
    print(f"nearest-centre recovers tokens: {agree*100:.2f}% of pixels")

    # How lossy is the whole pre-processing, in bytes?
    from PIL import Image as I

    rec = table[ids].reshape(resized.shape)
    up = np.asarray(I.fromarray((rec * 255).round().astype(np.uint8)).resize((w, h), I.BILINEAR))
    err = rgb.astype(np.int16) - up.astype(np.int16)
    print(f"\nresidual vs original {w}x{h} RGB: "
          f"MAE={np.abs(err).mean():.2f} levels, max|err|={np.abs(err).max()}")
    print(f"residual raw bytes = {err.astype(np.int8).nbytes} "
          f"(vs {rgb.nbytes} original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
