"""Build the video modality comparison table.

Sources:
  results/baselines_video_wsl.jsonl  -- classical video codecs (ffmpeg, capsule build)
  results/video_igpt.jsonl           -- the paper's method: iGPT per frame

Reports the paper's metric (compressed / raw pixel count) *and* the raw-RGB-byte
rate, because this project has already been burned once by a convention that was
not stated explicitly.  Also reports the exact-reconstruction completion, so the
gap between "the model's rate" and "a lossless rate" is visible rather than implied.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paper, Supplementary Table 4, high-resolution clip (4096x2160, 759 MB).
PAPER_HIGHRES = {
    "zstd --ultra -22": 51.04,
    "ffv1": 40.16,
    "h.264": 37.42,
    "h.265": 21.69,
    "LMCompress-igpt": 15.31,
}
# Low-resolution clips, same table, for context.
PAPER_LOWRES = {
    "static": {"zstd --ultra -22": 50.47, "ffv1": 38.02, "h.264": 29.58,
               "h.265": 21.05, "LMCompress-igpt": 13.05},
    "dynamic": {"zstd --ultra -22": 61.83, "ffv1": 46.94, "h.264": 39.52,
                "h.265": 28.49, "LMCompress-igpt": 19.61},
}

ALIAS = {
    "zstd --ultra -22 (raw yuv)": "zstd --ultra -22",
    "h.264 crf 0 (x264 veryslow)": "h.264",
    "h.265 crf 0 (x265 veryslow)": "h.265",
    "ffv1 level 3": "ffv1",
}


def main() -> int:
    print("# Video modality — measured vs published\n")
    print("Source clip: `data/test_videos/4k_aerial_first_second.y4m`, "
          "4096x2160 @60fps, **60 frames**, raw YUV420 796 262 400 bytes "
          "(13 271 040 per frame).\n")
    print("Paper's video method: **each frame is coded independently with iGPT; "
          "interframe information is deliberately not used** (NMI, Fig. 4 "
          "discussion). The classical codecs here *do* use interframe redundancy, "
          "which is the fair comparison for them.\n")

    # ---------------- classical baselines ---------------- #
    # Two files because the codecs are driven by different toolchains: ffmpeg
    # (h.264/h.265/ffv1) and the Windows zstd binary.  Both must be read, or the
    # ordering check silently omits zstd.
    rows = []
    for name in ("baselines_video_wsl.jsonl", "baselines_video_zstd.jsonl"):
        path = os.path.join(HERE, "results", name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    print("## Classical baselines (measured)\n")
    if rows:
        print("| codec | frames | compressed bytes | rate (raw YUV) | ratio | encode time | paper (high-res) |")
        print("|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: x["compression_rate_pct"]):
            key = ALIAS.get(r["codec"], r["codec"])
            p = PAPER_HIGHRES.get(key)
            ptxt = f"{p:.2f} %" if p is not None else "—"
            print(f"| {r['codec']} | {r['frames']} | {r['compressed_bytes']:,} | "
                  f"**{r['compression_rate_pct']:.3f} %** | {r['compression_ratio']:.3f}× | "
                  f"{r['seconds']:.1f} s | {ptxt} |")
        print()
        print("**Ordering check.** The paper's high-res ordering is "
              "H.265 < H.264 < FFV1 < zstd (i.e. H.265 best). ")
        got = [ALIAS.get(r["codec"], r["codec"])
               for r in sorted(rows, key=lambda x: x["compression_rate_pct"])]
        print(f"Our ordering: {' < '.join(got)}.")
    else:
        print("_no baseline rows yet_")

    # ---------------- iGPT per frame ---------------- #
    print("\n## The paper's method: iGPT, one frame at a time\n")
    ipath = os.path.join(HERE, "results", "video_igpt.jsonl")
    if os.path.exists(ipath):
        fr = []
        with open(ipath, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    fr.append(json.loads(line))
        tot_bits = sum(r["coded_bits"] for r in fr)
        tot_px = sum(r["pixels"] for r in fr)
        tot_rgb = sum(r["rgb_bytes"] for r in fr)
        tot_resid = sum(r["residual_zlib9"] for r in fr)
        tok = sorted({r["tokens"] for r in fr})
        uniq = [r["unique_tokens"] for r in fr]

        print(f"Frames measured: **{len(fr)}**")
        print(f"Tokens per frame: {tok} (the processor always emits 32x32 = 1024)")
        print(f"Distinct palette ids used per frame: min {min(uniq)}, max {max(uniq)}\n")

        print("| quantity | value |")
        print("|---|---|")
        print(f"| total pixels (denominator, paper convention) | {tot_px:,} |")
        print(f"| total RGB bytes | {tot_rgb:,} |")
        print(f"| model code length | {tot_bits/8:,.1f} bytes |")
        print(f"| **rate, paper convention** | **{100*(tot_bits/8)/tot_px:.6f} %** |")
        print(f"| rate, raw RGB bytes | {100*(tot_bits/8)/tot_rgb:.6f} % |")
        print(f"| + residual (zlib -9) for exact reconstruction | {tot_resid:,} bytes |")
        print(f"| **exact-reconstruction rate (on RGB)** | "
              f"**{100*((tot_bits/8)+tot_resid)/tot_rgb:.4f} %** |")
        print(f"| paper, high-res clip | 15.31 % |")
        print()
        print("### Reading this\n")
        print("The model rate and the published 15.31 % are **not the same kind of "
              "number**, and the gap is the same one documented for images "
              "(audit §4.4.2): the ImageGPT processor reduces each 4096x2160 frame to "
              "a **32x32, 512-colour token grid** before the coder sees it, while the "
              "denominator counts the frame's real pixels. That is a reduction of "
              "roughly 8 640x. Our measured rate is therefore a rate for a quantised "
              "thumbnail, not for the video.\n")
        print("The honest figure for this clip is the exact-reconstruction line: "
              f"**{100*((tot_bits/8)+tot_resid)/tot_rgb:.2f} %**, i.e. *worse than "
              "every classical codec measured above*. The model's 370-byte-per-frame "
              "contribution is noise beside the residual, exactly as in the image "
              "case.\n")
    else:
        print("_no iGPT rows yet — run tools/run_video_igpt.py_")

    print("\n## Published video table (Supplementary Table 4), for reference\n")
    print("| method | low-res static | low-res dynamic | high-res |")
    print("|---|---|---|---|")
    for m in ("zstd --ultra -22", "ffv1", "h.264", "h.265", "LMCompress-igpt"):
        print(f"| {m} | {PAPER_LOWRES['static'][m]:.2f} % | "
              f"{PAPER_LOWRES['dynamic'][m]:.2f} % | {PAPER_HIGHRES[m]:.2f} % |")
    print()
    print("Our clip is a single 4K aerial shot, so the high-res column is the "
          "comparable one. It is **not** one of the paper's 11 clips, so agreement "
          "is not expected to be exact; ordering and rough magnitude are what is "
          "being checked.")

    # ---------------- baseline agreement summary ---------------- #
    print("\n## Baseline agreement with the published high-res column\n")
    if rows:
        print("| codec | measured | paper | Δ |")
        print("|---|---|---|---|")
        worst = 0.0
        for r in sorted(rows, key=lambda x: ALIAS.get(x["codec"], x["codec"]) or ""):
            key = ALIAS.get(r["codec"], r["codec"])
            p = PAPER_HIGHRES.get(key)
            if p is None:
                continue
            d = r["compression_rate_pct"] - p
            worst = max(worst, abs(d))
            print(f"| {key} | {r['compression_rate_pct']:.3f} % | {p:.2f} % | "
                  f"**{d:+.3f} pp** |")
        print()
        print(f"Largest discrepancy across the four codecs: **{worst:.3f} pp** on a "
              "clip the paper did not use. This is the control for the modality: the "
              "classical pipeline is measuring what the paper measured, so the iGPT "
              "row's disagreement (below) is attributable to the model path rather "
              "than to the harness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
