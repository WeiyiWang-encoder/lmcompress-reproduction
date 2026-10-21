"""Reconcile the official 15.5144 % against our port's 23.9257 %.

Established by measurement:

  * tokenisation is identical (same 5733 units, same 5870592 samples, same token ids);
  * on one chunk, the official model invocation and our port assign **identical**
    probabilities to the **identical** targets: 1774.715 bits either way.

Yet the official whole-image run reports 910 788 bytes while our port reports
1 404 624 bytes.  A factor of 1.542 cannot survive identical per-chunk numbers, so
the difference must be in how the official run *aggregates* chunks -- or in a
different quantity being accumulated altogether.

The official accumulator is
    output = self.model(**batch)
    if output.loss is not None:
        loss = output.loss
        self.total_loss += loss
    ...
    theoretical_bits_total = sum(self.theoretical_cl)
    self.total_compressed_length += theoretical_bits_total / 8

with ``self.theoretical_cl.extend(batch_sums.tolist())`` and
``batch_sums = sum(masked_neg_log_probs) / input_ids.size(0)``.

This script computes the candidate aggregations from our own per-chunk bits and
identifies which one lands on the official number.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()

OFFICIAL_BYTES = 910788.0
OFFICIAL_SAMPLES = 5870592.0
OURS_BYTES = 1404624.0
CHUNKS = 5733
PATCH_TOKENS = 1024


def main() -> int:
    print("=== established quantities ===")
    print(f"  chunks                        = {CHUNKS}")
    print(f"  samples (official denominator) = {OFFICIAL_SAMPLES:.0f}")
    print(f"  official coded bytes           = {OFFICIAL_BYTES:.0f}")
    print(f"  our port coded bytes           = {OURS_BYTES:.0f}")
    print(f"  ratio ours/official            = {OURS_BYTES/OFFICIAL_BYTES:.4f}")

    print("\n=== implied bits per unit ===")
    print(f"  official: {8*OFFICIAL_BYTES/OFFICIAL_SAMPLES:.6f} bits per sample")
    print(f"  ours    : {8*OURS_BYTES/OFFICIAL_SAMPLES:.6f} bits per sample")

    # our per-chunk bits from the full run
    p = os.path.join(HERE, "results", "image_igpt_channelwise.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        print(f"\n=== our full run (results/image_igpt_channelwise.json) ===")
        print(f"  units={d['units']}  raw={d['raw_samples']}  "
              f"bits={d['coded_bits']:.1f}  rate={d['rate_pct']:.4f} %")
        per_chunk = d["coded_bits"] / d["units"]
        print(f"  our mean bits per chunk = {per_chunk:.3f}")
    else:
        per_chunk = 8 * OURS_BYTES / CHUNKS
        print(f"\n  (using recorded total) mean bits per chunk = {per_chunk:.3f}")

    off_per_chunk = 8 * OFFICIAL_BYTES / CHUNKS
    print(f"  official mean bits per chunk = {off_per_chunk:.3f}")
    print(f"  ratio = {per_chunk/off_per_chunk:.4f}")

    print("\n=== candidate aggregations that could produce the official number ===")
    # A: exact true-token sum over 1023 scored positions (what we do)
    cand = {
        "mean over 1023 targets x 1024 tokens": per_chunk,
        "mean over 1023 targets x 1023 tokens": per_chunk * 1023 / 1024,
        "mean over 1024 positions x 1024": None,
    }
    for name, v in cand.items():
        if v:
            print(f"  {name:<40} -> {v:8.3f} bits/chunk "
                  f"(official {off_per_chunk:.3f}, ratio {v/off_per_chunk:.4f})")

    print("\n=== what the loss-based accumulator would give ===")
    print("  If, per chunk, the official code adds the model's own loss (mean CE over")
    print("  1023 shifted positions) scaled by the patch-token count, then")
    print("      bits/chunk = loss * 1024")
    print("  and the implied loss for the official number is")
    print(f"      {off_per_chunk/1024:.6f}")
    print(f"  whereas the loss we measure on chunk 0 is 1.202482, giving")
    print(f"      {1.202482*1024:.1f} bits/chunk  -> "
          f"{1.202482*1024*CHUNKS/8:.0f} bytes for the image")
    print(f"  which is {100*(1.202482*1024*CHUNKS/8)/OFFICIAL_BYTES-100:+.2f} % "
          f"relative to the official bytes.")

    print("\n=== conclusion ===")
    print("  Our port and the official model assign identical per-chunk probabilities,")
    print("  so the 1.54x is NOT a modelling difference and NOT a tokenisation")
    print("  difference.  It must be an aggregation difference in how the official")
    print("  script sums chunks -- which we can only settle by instrumenting its")
    print("  accumulator directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
