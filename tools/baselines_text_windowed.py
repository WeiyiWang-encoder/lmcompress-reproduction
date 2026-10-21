"""Reproduce the paper's fixed-window protocol for the text baselines.

The problem this fixes: our text baselines compressed each file in ONE stream, while the
paper's SI Table 11 runs every compressor at a fixed window of 8192 / 16384 / 32768 bytes
and finally unbounded. The same 7z spans 42.02 -> 23.89 % across those columns, so the
window alone moves the number by 18 pp. Comparing our whole-file figure with their 8192
column would flatter us by more than ten points.

This measures the same grid on our corpus, so the baseline table becomes like-for-like
with Table 11 and with our own Axis-A model numbers (which use windows of the same size
in characters, and for English text characters ~= bytes).

Protocol, per window W:
  * split the byte string into consecutive chunks of W bytes;
  * compress each chunk INDEPENDENTLY (a fresh compressor state, no dictionary carry-over);
  * rate = sum(compressed) / sum(original).
W = None means one stream over the whole file, which is the "unbounded" column.
"""

from __future__ import annotations

import io
import lzma
import os
import sys
import time
import zlib
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
FILES = ["medal_102285.csv", "medal_103851.csv", "medal_108122.csv"]
WINDOWS = [8192, 16384, 32768, None]

# paper SI Table 11, for the columns we can match
PAPER = {
    "lzma 9e (7z-equivalent)": {8192: 42.02, 16384: 38.21, 32768: 35.26, None: 23.89},
    "zstd -22": {8192: 39.54},
}


def codecs():
    out = {
        "brotli q11": None,
        "lzma 9e (7z-equivalent)": lambda b: lzma.compress(
            b, format=lzma.FORMAT_ALONE,
            filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}]),
        "zlib 9": lambda b: zlib.compress(b, 9),
        "zstd -22": None,
    }
    try:
        import brotli
        out["brotli q11"] = lambda b: brotli.compress(b, quality=11)
    except ImportError:
        out.pop("brotli q11")
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=22)
        out["zstd -22"] = lambda b: cctx.compress(b)
    except ImportError:
        out.pop("zstd -22")
    return out


def main() -> int:
    payload_total = 0
    blobs = {}
    for f in FILES:
        raw = open(os.path.join(MEDAL, f), "rb").read()
        blobs[f] = raw
        payload_total += len(raw)
    print(f"files: {len(FILES)}   total bytes: {payload_total:,}")

    impls = codecs()
    print(f"codecs available: {sorted(impls)}")
    missing = [k for k in ("brotli q11", "zstd -22") if k not in impls]
    if missing:
        print(f"  (missing, skipped: {missing})")

    print()
    header = f"{'codec':<26}" + "".join(f"{('inf' if w is None else str(w)):>10}"
                                        for w in WINDOWS)
    print(header)
    print("-" * len(header))

    results: dict[str, dict] = {}
    for name, fn in impls.items():
        row = {}
        for w in WINDOWS:
            comp = orig = 0
            for f, raw in blobs.items():
                if w is None:
                    chunks = [raw]
                else:
                    chunks = [raw[i:i + w] for i in range(0, len(raw), w)]
                for c in chunks:
                    comp += len(fn(c))
                    orig += len(c)
            row[w] = 100 * comp / orig
        results[name] = row
        print(f"{name:<26}" + "".join(f"{row[w]:>10.2f}" for w in WINDOWS))

    print()
    print("=== against paper SI Table 11 ===")
    for name, pubs in PAPER.items():
        if name not in results:
            continue
        for w, pv in pubs.items():
            ours = results[name][w]
            tag = "inf" if w is None else str(w)
            print(f"  {name:<26} window {tag:>6}:  ours {ours:6.2f} %   "
                  f"paper {pv:6.2f} %   delta {ours-pv:+6.2f} pp")

    print()
    print("=== does the window explain the earlier apparent gap? ===")
    seven = results.get("lzma 9e (7z-equivalent)")
    if seven:
        print(f"  our 7z-equivalent, unbounded  : {seven[None]:6.2f} %  "
              f"(paper 23.89)")
        print(f"  our 7z-equivalent, window 8192: {seven[8192]:6.2f} %  "
              f"(paper 42.02)")
        print(f"  swing from window alone       : {seven[8192]-seven[None]:6.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
