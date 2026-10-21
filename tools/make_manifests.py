"""Generate file manifests for the baseline measurements.

The manifests list payload paths in WSL form so ``tools/baselines.sh`` can feed
them to the paper's own Linux binaries (optimfrog, static ffmpeg, WavPack CLI).

The subset sizes are chosen to be comparable with the paper's per-dataset sums
while staying inside a sensible local runtime; ``--limit`` and the resulting
total bytes are printed so the report can state exactly what was measured.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

REPO = _capsule_root()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_wsl(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    if p[1:2] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def write(name: str, paths) -> None:
    out = os.path.join(HERE, "results", f"manifest_{name}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    total = 0
    # newline="\n" matters: the manifests are consumed by bash inside WSL, and a
    # CRLF line ending makes every path fail to resolve.
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for p in paths:
            fh.write(to_wsl(p) + "\n")
            total += os.path.getsize(p)
    print(
        f"{name:16s} {len(paths):>4} files  "
        f"{total / 1e6:>9.2f} MB  -> {os.path.basename(out)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-files", type=int, default=10)
    ap.add_argument("--image-files", type=int, default=1)
    ap.add_argument("--video-frames", type=int, default=1)
    args = ap.parse_args()

    data = os.path.join(REPO, "data")

    # Text: the paper's per-dataset total is ~180 MB of MeDAL.  Locally we use
    # the sampled CSVs, and we report both the container size and the raw
    # UTF-8 text size so the ratio is unambiguous.
    #
    # IMPORTANT: the baselines must be measured on exactly the same files as the
    # model, or the comparison is meaningless.  MEDAL_N / EURLEX_N below define
    # the shared subset.
    medal = sorted(glob.glob(os.path.join(data, "text", "MEDAL_sample", "*.csv")))[: args.text_files]
    eurlex = sorted(glob.glob(os.path.join(data, "text", "EURLEX_sample", "*.csv")))[: args.text_files]
    write("text_medal", medal)
    write("text_eurlex", eurlex)

    images = sorted(glob.glob(os.path.join(data, "test_images", "*")))[: args.image_files]
    write("image", images)

    videos = sorted(glob.glob(os.path.join(data, "test_videos", "*")))
    write("video", videos)

    audio = sorted(glob.glob(os.path.join(data, "test_audios", "*.wav")))
    write("audio", audio)

    # Raw text payloads (the true denominator for text) as standalone .txt files.
    raw_dir = os.path.join(HERE, "artifacts", "text_raw")
    os.makedirs(raw_dir, exist_ok=True)
    import pandas as pd

    raw_paths = []
    for src in medal:
        text = "".join(str(x) for x in pd.read_csv(src)["TEXT"].tolist())
        dst = os.path.join(raw_dir, os.path.basename(src).replace(".csv", ".txt"))
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        raw_paths.append(dst)
    write("text_medal_raw", raw_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
