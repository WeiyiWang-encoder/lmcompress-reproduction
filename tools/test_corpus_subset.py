"""Is our 3-file subset unusually compressible? The cheap discriminator.

Four candidate mechanisms could make the paper's 9.06 % differ from our 6.33 %:

  A. convention   -- a different definition of what is counted (denominator, shift)
  B. formula      -- the same convention computed with a different expression
  C. code         -- an implementation defect in one of the two code paths
  D. corpus       -- a different subset of MeDAL, which genuinely compresses differently

A, B and C are hardware-free but need model forwards, which cost ~23 min per window on
CPU. D is testable **without any model**: if our three files sit in the easy tail of the
MeDAL distribution for classical compressors, then a corpus difference alone can explain
most of the gap, and the model comparison stops being apples-to-apples.

This measures brotli and lzma at the paper's 8192-byte window and unbounded, over all
108 CSVs in the sample, and reports where our three chosen files fall.
"""

from __future__ import annotations

import glob
import io
import lzma
import os
import sys
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDAL = os.path.join(_capsule_root(),
                     "data", "text", "MEDAL_sample")
CHOSEN = ["medal_102285.csv", "medal_103851.csv", "medal_108122.csv"]
WINDOWS = (8192, None)


def rate(blob: bytes, fn, w) -> float:
    if w is None:
        chunks = [blob]
    else:
        chunks = [blob[i:i + w] for i in range(0, len(blob), w)]
    comp = sum(len(fn(c)) for c in chunks)
    return 100 * comp / len(blob)


def main() -> int:
    import brotli

    lz = lambda b: lzma.compress(
        b, format=lzma.FORMAT_ALONE,
        filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}])
    br = lambda b: brotli.compress(b, quality=11)

    files = sorted(glob.glob(os.path.join(MEDAL, "*.csv")))
    print(f"MeDAL sample files: {len(files)}")

    for name, fn in (("brotli q11", br), ("lzma 9e", lz)):
        for w in WINDOWS:
            tag = "inf" if w is None else str(w)
            vals = []
            for p in files:
                blob = open(p, "rb").read()
                vals.append((os.path.basename(p), rate(blob, fn, w), len(blob)))
            vals.sort(key=lambda x: x[1])
            chosen = [(n, r) for n, r, _ in vals if n in CHOSEN]
            allr = [r for _, r, _ in vals]
            n = len(allr)
            mean = sum(allr) / n
            print(f"\n=== {name}, window {tag} ===")
            print(f"  all {n} files: min {allr[0]:.2f}  mean {mean:.2f}  "
                  f"median {allr[n//2]:.2f}  max {allr[-1]:.2f} %")
            print(f"  our three   : " +
                  ", ".join(f"{r:.2f}" for _, r in chosen))
            for cn, cr in chosen:
                rank = sum(1 for x in allr if x < cr)
                pct = 100 * rank / n
                print(f"    {cn:<22}{cr:>7.2f} %  percentile {pct:>5.1f} "
                      f"({'easier' if pct < 50 else 'harder'} than {100-pct:.0f}% of files)")
            ch_mean = sum(r for _, r in chosen) / len(chosen)
            print(f"  chosen mean {ch_mean:.2f} %  vs corpus mean {mean:.2f} %   "
                  f"difference {ch_mean-mean:+.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
