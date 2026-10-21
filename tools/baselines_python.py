"""Python-native classical baselines.

Covers every baseline that can be run without external binaries, using the
paper's own settings where a Python equivalent exists:

===============  ==========================================================
paper tool       this module
===============  ==========================================================
zstd --ultra -22 ``python-zstandard`` level 22 (same library, same codec)
7z a -mx9        ``lzma`` preset 9e -- 7z's default method *is* LZMA2, so the
                 number is directly comparable even though the byte-exact
                 container differs
brotli q11       ``brotli`` quality 11 (the paper calls the Python module)
zlib / gzip      ``zlib`` level 9
===============  ==========================================================

Binaries that need Linux (optimfrog, static ffmpeg, WavPack CLI) are driven by
``tools/baselines.sh`` instead.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import time
import zlib
from typing import Callable, Dict, List
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

REPO = _capsule_root()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _zstd_compress() -> Callable[[bytes], bytes] | None:
    try:
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=22)
        return cctx.compress
    except ImportError:
        return None


def codecs() -> Dict[str, Callable[[bytes], bytes]]:
    out: Dict[str, Callable[[bytes], bytes]] = {
        "zlib 9": lambda b: zlib.compress(b, 9),
        "lzma preset 9e (7z-equivalent)": lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME),
    }
    try:
        import brotli

        out["brotli q11"] = lambda b: brotli.compress(b, quality=11)
    except ImportError:
        pass
    z = _zstd_compress()
    if z is not None:
        out["zstd --ultra -22"] = z
    return out


def measure_file(path: str) -> List[dict]:
    raw = open(path, "rb").read()
    orig = len(raw)
    rows = []
    for name, fn in codecs().items():
        t0 = time.perf_counter()
        comp = fn(raw)
        dt = time.perf_counter() - t0
        rows.append(
            {
                "codec": name,
                "unit": os.path.basename(path),
                "original_bytes": orig,
                "compressed_bytes": len(comp),
                "compression_rate_pct": round(100.0 * len(comp) / orig, 6),
                "compression_ratio": round(orig / len(comp), 6),
                "seconds": round(dt, 3),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="file with one WSL/Windows path per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = []
    with open(args.manifest, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # manifests are written in WSL form for the shell driver; convert back
            if line.startswith("/mnt/") and len(line) > 6:
                drive = line[5].upper()
                line = f"{drive}:\\" + line[6:].replace("/", "\\")
            paths.append(line)

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    total = 0
    with open(out, "w", encoding="utf-8") as fh:
        for path in paths:
            if os.path.getsize(path) > 64 * 1024 * 1024:
                print(f"skip (>{64} MB, use the shell driver): {path}")
                continue
            for row in measure_file(path):
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                total += 1
    print(f"wrote {total} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
