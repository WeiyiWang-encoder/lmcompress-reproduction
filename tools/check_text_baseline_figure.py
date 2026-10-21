"""Check the text baseline figures in this project's own documents.

Prompted by the question "why does the text baseline look so good compared with the
paper". Two things have to be separated:

1. **Is the number I quote correct?** The report cites brotli at 28.918 %, but the weighted
   payload figure in the result file is 28.672 %. If 28.918 is a single-file rate being
   quoted as an aggregate, that is an error in my document, not in anyone else's.
2. **Is the comparison with the paper like-for-like?** The paper's SI Table 11 reports the
   same compressors at a fixed *window* of 8192 / 16384 / 32768 bytes and finally
   unbounded, and 7z alone moves 42.02 -> 23.89 % across those columns. Our baselines
   compress the whole file in one stream, which corresponds to the unbounded column.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")


def main() -> int:
    path = os.path.join(R, "baselines_text_medal_python.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"=== {os.path.basename(path)}: {len(rows)} rows ===")

    # 1. where does 28.918 appear?
    print("\n--- rows whose rate is near 28.918 ---")
    for r in rows:
        v = r.get("compression_rate_pct") or r.get("rate_pct") or 0
        if abs(v - 28.918) < 0.02:
            print(f"  unit={r.get('unit'):<24} codec={r.get('codec'):<30} "
                  f"rate={v:.4f}  raw={r.get('original_bytes'):,}  "
                  f"comp={r.get('compressed_bytes'):,}")

    # 2. the correct aggregate, weighted by bytes
    print("\n--- weighted aggregate per codec (bytes-weighted, the usual convention) ---")
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r["codec"]][0] += r["compressed_bytes"]
        agg[r["codec"]][1] += r["original_bytes"]
    for codec, (c, raw) in sorted(agg.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {codec:<32}{100*c/raw:>8.4f} %")

    # 3. simple mean per codec, for contrast
    print("\n--- simple mean of per-unit rates per codec ---")
    by = defaultdict(list)
    for r in rows:
        by[r["codec"]].append(r.get("compression_rate_pct") or 0)
    for codec, vs in sorted(by.items()):
        print(f"  {codec:<32}{sum(vs)/len(vs):>8.4f} %   (n={len(vs)})")

    # 4. which unit did the report's 28.918 come from
    print("\n--- how many distinct units were compressed ---")
    units = sorted({r["unit"] for r in rows})
    print(f"  {len(units)} units: {units}")

    print("\n=== the protocol question ===")
    print("  ours: whole file in ONE stream (no window field in any row)")
    print("  paper SI Table 11, 7z:  8192 -> 42.02 | 16384 -> 38.21 |"
          " 32768 -> 35.26 | inf -> 23.89")
    print()
    ours7z = 100 * agg["lzma preset 9e (7z-equivalent)"][0] / \
        agg["lzma preset 9e (7z-equivalent)"][1]
    oursbr = 100 * agg["brotli q11"][0] / agg["brotli q11"][1]
    print(f"  our 7z-equivalent, whole file : {ours7z:.3f} %")
    print(f"  paper 7z, unbounded           : 23.89 %   -> we are {ours7z-23.89:+.2f} pp")
    print(f"  paper 7z, window 8192         : 42.02 %   -> we are {ours7z-42.02:+.2f} pp")
    print(f"  our brotli, whole file        : {oursbr:.3f} %")
    print()
    print("  So against the paper's *unbounded* column we are WORSE (31.5 vs 23.9),")
    print("  which is expected: they ran on ~180 MB and we ran on 313 KB, and a")
    print("  compressor with more data does better. Against the 8192 column we look")
    print("  13 pp better, but that column is a restricted-window protocol we did not")
    print("  reproduce. Neither comparison is like-for-like.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
