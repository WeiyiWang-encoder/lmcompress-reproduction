"""Port-side image accounting, channel-wise, over ALL units.

This is the end-to-end check that our own implementation now reproduces the official
figure.  It must be run over every unit: the units are consecutive raster bands of
one channel, so a prefix is not a random sample and its rate is not the file's rate.

Target: the official whole-file rate is 15.5144 % over 5 733 chunks / 5 870 592
samples (tools/probe_official_image_accounting.py).
"""

from __future__ import annotations

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
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
OUT = os.path.join(HERE, "results", "image_igpt_channelwise.json")


def main() -> int:
    from uic.data import image_units
    from uic.engine import measure_stream
    from uic.models import ImageGPTPredictor

    units = image_units(os.path.join(REPO, "data", "test_images"), MODEL_DIR,
                        limit=1, include_residual=False)
    print(f"units = {len(units)}")

    pred = ImageGPTPredictor.load(MODEL_DIR, device="cuda:0")
    bits = 0.0
    raw = 0
    per_channel = {0: 0.0, 1: 0.0, 2: 0.0}
    t0 = time.time()
    try:
        for i, unit in enumerate(units):
            res, _ = measure_stream(pred.iter_steps(unit.payload))
            bits += res.coded_bits
            raw += unit.original_bytes
            per_channel[unit.meta["channel"]] += res.coded_bits
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(units)}  running rate="
                      f"{100*(bits/8)/raw:.4f} %  ({time.time()-t0:.0f}s)",
                      flush=True)
    finally:
        pred.release()

    rate = 100 * (bits / 8) / raw
    out = {
        "units": len(units),
        "raw_samples": raw,
        "coded_bits": bits,
        "coded_bytes": bits / 8,
        "rate_pct": rate,
        "ratio": raw / (bits / 8),
        "bits_per_sample": bits / raw,
        "per_channel_rate_pct": {
            str(ch): 100 * (b / 8) / (raw / 3) for ch, b in per_channel.items()
        },
        "official_rate_pct": 15.514414900575614,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n=== port, channel-wise, whole image ===")
    print(f"  units            = {out['units']}        (official 5733)")
    print(f"  raw samples      = {out['raw_samples']}   (official 5870592)")
    print(f"  coded bytes      = {out['coded_bytes']:.1f}  (official 910788)")
    print(f"  rate             = {rate:.4f} %   (official 15.5144 %)")
    print(f"  ratio            = {out['ratio']:.4f}")
    print(f"  per-channel rate = "
          + ", ".join(f"ch{k}={v:.3f}%" for k, v in
                      out["per_channel_rate_pct"].items()))
    print(f"  wall             = {out['wall_seconds']}s")
    print(f"\n  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
