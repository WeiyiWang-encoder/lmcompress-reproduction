"""Re-measure the audio next-byte rate with the CURRENT code, and reconcile it
against every figure that appears in the documents.

A review caught an arithmetic inconsistency (13.12 bits/byte was quoted alongside
197.3 % / 0.51x, which correspond to 15.78 bits/byte).  The deeper issue is that
13.1233 came from an earlier revision of the alignment code, so the documents were
mixing two code generations.  This script establishes the single current number and
prints the cross-checks so the conversion cannot drift again.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()


def main() -> int:
    from uic.data import audio_units
    from uic.models import BGPTBytePredictor
    from uic.runner import measure_unit, summarize

    weights = os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth")
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    print(f"payload = {len(unit.payload)} bytes (8-bit PCM, 1 byte per sample)")
    print(f"sample width = {unit.meta['sampwidth']} byte -> original byte count is the PCM byte count\n")

    pred = BGPTBytePredictor.load(REPO, weights, device="cuda:0", extension=b"wav")
    try:
        res = measure_unit(pred, unit, os.path.join("runs", "cursors", "audio_recheck"))
    finally:
        pred.release()

    d = res.as_dict()
    bits = d["coded_bits"]
    raw = d["raw_bytes"]
    print("=== CURRENT code, next-byte scoring ===")
    print(f"  coded_bits              = {bits:.4f}")
    print(f"  raw_bytes (denominator) = {raw}")
    print(f"  bits_per_byte           = {bits/raw:.4f}")
    print(f"  compression rate        = {100*(bits/8)/raw:.4f} %")
    print(f"  compression ratio       = {raw/(bits/8):.4f} x")
    print(f"  windows                 = {d['windows']}")

    print("\n=== cross-checks (8-bit PCM: 1 byte = 8 bits) ===")
    bpb = bits / raw
    print(f"  rate from bpb           = 100*{bpb:.4f}/8 = {100*bpb/8:.4f} %")
    print(f"  ratio from bpb          = 8/{bpb:.4f} = {8/bpb:.4f} x")

    print("\n=== figures appearing in the documents ===")
    for label, v in (("13.1233 bpb (STALE, pre-alignment-fix)", 13.1233),
                     ("197.3 % (matches 15.78 bpb)", 15.7844)):
        if "bpb" in label:
            print(f"  {label:44s} -> rate {100*v/8:7.2f} %   ratio {8/v:.4f} x")
        else:
            print(f"  {label:44s} -> bpb  {v*8/100:7.4f}")
    print("\n  => 13.1233 and 197.3 % are NOT the same measurement.")
    print("     Current measurement is the one printed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
