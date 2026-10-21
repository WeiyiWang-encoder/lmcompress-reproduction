"""iGPT rate on the same CLIC images as the baselines, official convention.

The baselines were re-measured on 12 CLIC professional images because the original
table used a single aerial photograph, which compresses very differently from CLIC
photography (PNG 84.4 % against a published 47.1 %).  This measures the model on
exactly the same 12 images, so the comparison finally holds the corpus fixed.

The accounting is the official one: `raw_length += h*w` per chunk, i.e. the denominator
is the source pixel count and the numerator is the code length of the 32x32,
512-colour token grid. `uic.data.image_units` implements that convention channel-wise
(units, denominator and token ids verified against the official loader).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
PATCH = 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(HERE, "artifacts", "clic", "raw"))
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "image_igpt_clic.json"))
    args = ap.parse_args()

    from PIL import Image

    from uic.data import _igpt_processor, image_units
    from uic.models import ImageGPTPredictor

    files = sorted(glob.glob(os.path.join(args.images, "**", "*.png"),
                             recursive=True))[: args.limit]
    if not files:
        print(f"no PNGs under {args.images}")
        return 1

    proc = _igpt_processor(MODEL_DIR)
    pred = ImageGPTPredictor.load(MODEL_DIR, device="cuda:0")

    per_image = []
    tot_bits = 0.0
    tot_samples = 0
    t0 = time.time()
    try:
        for k, path in enumerate(files, 1):
            name = os.path.basename(path)
            units = image_units(os.path.dirname(path), MODEL_DIR, limit=None,
                                extensions=(".png",), include_residual=False)
            units = [u for u in units if u.meta["source"] == name]
            if not units:
                print(f"  [{k}/{len(files)}] {name}: no units, skipped")
                continue

            bits = 0.0
            samples = 0
            buf_ids = []
            buf_sam = 0
            for u in units:
                buf_ids.append(u.payload)
                buf_sam += u.original_bytes
                if len(buf_ids) >= args.chunk:
                    arr = torch.tensor(np.asarray(buf_ids), dtype=torch.long,
                                       device="cuda:0")
                    with torch.no_grad():
                        lg = pred.model(arr).logits.float()
                    lp = torch.log_softmax(lg, dim=-1)[:, :-1, :]
                    tgt = arr[:, 1:]
                    bits += float((-lp.gather(-1, tgt.unsqueeze(-1))
                                   .squeeze(-1)).sum())
                    del lg, lp
                    buf_ids = []
            if buf_ids:
                arr = torch.tensor(np.asarray(buf_ids), dtype=torch.long,
                                   device="cuda:0")
                with torch.no_grad():
                    lg = pred.model(arr).logits.float()
                lp = torch.log_softmax(lg, dim=-1)[:, :-1, :]
                tgt = arr[:, 1:]
                bits += float((-lg.gather(-1, tgt.unsqueeze(-1))
                               .squeeze(-1)).sum())
                del lg, lp

            samples = buf_sam
            tot_bits += bits
            tot_samples += samples
            row = dict(image=name, units=len(units), samples=samples,
                       coded_bits=bits, coded_bytes=bits / 8,
                       rate_pct=100 * (bits / 8) / samples)
            per_image.append(row)
            print(f"[{k}/{len(files)}] {name}: {len(units)} units, "
                  f"{bits/8/1e6:.2f} MB, rate {row['rate_pct']:.4f} % "
                  f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        pred.release()

    out = dict(
        images=len(per_image),
        units=sum(r["units"] for r in per_image),
        samples=tot_samples,
        coded_bits=tot_bits,
        coded_bytes=tot_bits / 8,
        rate_pct=100 * (tot_bits / 8) / tot_samples,
        ratio=tot_samples / (tot_bits / 8) if tot_bits else None,
        published_rate_pct=14.62,
        convention=("official: compressed bytes / processed source samples, "
                    "denominator = sum(h*w) over channel-wise 32x32 chunks"),
        per_image=per_image,
        wall_seconds=round(time.time() - t0, 1),
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\n=== iGPT on CLIC professional, official convention ===")
    print(f"  images        = {out['images']}")
    print(f"  units         = {out['units']:,}")
    print(f"  samples       = {out['samples']:,}")
    print(f"  coded bytes   = {out['coded_bytes']:,.0f}")
    print(f"  RATE          = {out['rate_pct']:.4f} %")
    print(f"  published     = 14.62 %   delta = {out['rate_pct']-14.62:+.2f} pp")
    print(f"  wall          = {out['wall_seconds']}s")
    print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
