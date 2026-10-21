#!/usr/bin/env python3
"""What would it cost to make the iGPT image/video path actually lossless?

The iGPT route codes only the 512-way colour token, so a lossless codec built on
it needs a *side channel* that disambiguates the source value inside its token
class.  Because the authors feed the processor grey triples (v,v,v) built from a
single 8-bit channel value, and colour quantisation is per pixel, the map

    t : {0..255} -> {0..511}

depends only on v.  So the whole image reduces to that map plus the image's value
histogram, and the side channel's exact lower bound is the conditional entropy

    H(V | T) = - sum_v  p(v) * log2( p(v) / p(t(v)) )

which is achievable if encoder and decoder share the conditional distribution.

This script
  1. derives t(v) exhaustively from the processor,
  2. VALIDATES t(v) against a real (32,32,1) chunk taken from the authors' loader,
  3. computes H(V|T) for the shipped image and for real 4K video frames,
  4. adds the measured iGPT code length to get the lossless total, and compares
     it with the classical lossless codecs.

Writes results/_lossless_sideinfo.txt / .json
"""
from __future__ import annotations

import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
from PIL import Image

RESULTS = os.path.join(REPO, "repro_understanding_is_compression", "results")
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMG = os.path.join(REPO, "data", "test_images",
                   "0a8ea388b5df64e6337d65e65e7d5d55d74abbf5.png")
FRAMES = os.path.join(REPO, "repro_understanding_is_compression", "artifacts",
                      "frames", "4k_aerial_first_second")
IGPT_IMAGE_CODE_BYTES = 910788.0          # official_image_accounting.json (FULL run)
RAW_RGB_IMAGE = 5872164

OUT_TXT = os.path.join(RESULTS, "_lossless_sideinfo.txt")
OUT_JSON = os.path.join(RESULTS, "_lossless_sideinfo.json")
_lines: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    _lines.append(m)


def derive_map():
    """Exhaustively derive t(v): place value v at four known grid positions."""
    from transformers import ImageGPTImageProcessor
    proc = ImageGPTImageProcessor.from_pretrained(MODEL_DIR)
    vals = np.repeat(np.arange(256, dtype=np.uint8), 4).reshape(32, 32)
    rgb = np.repeat(vals[:, :, None], 3, axis=2)
    ids = np.asarray(proc(rgb, return_tensors="np")["input_ids"]).reshape(-1)
    # value k occupies flat positions 4k .. 4k+3
    grid = ids.reshape(256, 4)
    t = grid[:, 0].astype(np.int64)
    consistent = bool(np.all(grid == grid[:, 0:1]))
    return proc, t, consistent


def validate_map(proc, t, frame_uint8: np.ndarray) -> dict:
    """Run a real (32,32,1) chunk from the authors' loader through the processor."""
    ch = frame_uint8[:, :, 0]                       # channel 0, full resolution
    chunk = ch.reshape(-1)[:1024].reshape(32, 32, 1)
    rgb = np.repeat(chunk, 3, axis=2)
    got = np.asarray(proc(rgb, return_tensors="np")["input_ids"]).reshape(-1)
    want = t[chunk.reshape(-1)]
    return {"chunk_shape": list(chunk.shape),
            "n_positions": int(got.size),
            "matches_per_value_map": bool(np.array_equal(got, want)),
            "n_mismatch": int(np.sum(got != want))}


def conditional_entropy(arr_u8: np.ndarray, t: np.ndarray) -> dict:
    v = arr_u8.reshape(-1)
    n = v.size
    cnt_v = np.bincount(v, minlength=256).astype(np.float64)
    t_v = t[np.arange(256)]
    cnt_t = np.bincount(t_v, weights=cnt_v, minlength=512)
    pv = cnt_v / n
    pt = cnt_t / n
    nz = pv > 0
    h = -np.sum(pv[nz] * np.log2(pv[nz] / pt[t_v[nz]]))
    return {"samples": int(n), "H_V_given_T_bits_per_sample": float(h),
            "side_info_bytes": float(h * n / 8.0),
            "side_info_pct": float(100.0 * h * n / 8.0 / n)}


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    say("=" * 78)
    say("Cost of making the iGPT image/video path LOSSLESS (side-channel bound)")
    say("=" * 78)

    proc, t, consistent = derive_map()
    say(f"\ncolour map t(v): 256 source values -> {len(set(t.tolist()))} tokens")
    say(f"  per-value map is well defined : {consistent}")
    say(f"  class sizes: " + ", ".join(
        f"{k}:{int(c)}" for k, c in
        zip(*np.unique(np.bincount(t, minlength=512)[np.bincount(t, minlength=512) > 0],
                       return_counts=True))))

    res: dict = {"map": {"distinct_tokens": len(set(t.tolist())),
                         "per_value_map_well_defined": consistent}}

    # ---- validate on a real chunk from the authors' loader ---------------
    frame = np.asarray(Image.open(IMG).convert("RGB"))
    val = validate_map(proc, t, frame)
    res["validation"] = val
    say(f"\nvalidation on a real (32,32,1) chunk from the shipped image:")
    say(f"  positions {val['n_positions']}, mismatches vs t(v) = {val['n_mismatch']}"
        f"  -> {'MATCH' if val['matches_per_value_map'] else 'MISMATCH'}")

    # ---- image -----------------------------------------------------------
    say("\n" + "-" * 78)
    say("IMAGE (shipped 1714x1142 PNG, all samples)")
    say("-" * 78)
    h_img = conditional_entropy(frame, t)
    res["image_side_info"] = h_img
    say(f"  samples                    = {h_img['samples']}")
    say(f"  H(V|T)                     = {h_img['H_V_given_T_bits_per_sample']:.4f} bits/sample")
    say(f"  side-info lower bound      = {h_img['side_info_bytes']:.0f} bytes "
        f"= {100*h_img['side_info_bytes']/RAW_RGB_IMAGE:.4f} % of raw RGB")
    tot = IGPT_IMAGE_CODE_BYTES + h_img["side_info_bytes"]
    say(f"  iGPT code (full run)       = {IGPT_IMAGE_CODE_BYTES:.0f} bytes "
        f"= {100*IGPT_IMAGE_CODE_BYTES/RAW_RGB_IMAGE:.4f} %")
    say(f"  => LOSSLESS TOTAL (bound)  = {tot:.0f} bytes = "
        f"{100*tot/RAW_RGB_IMAGE:.4f} % of raw RGB  (ratio {RAW_RGB_IMAGE/tot:.4f}x)")
    res["image_lossless_total"] = {"bytes": tot,
                                   "rate_pct_raw_rgb": 100 * tot / RAW_RGB_IMAGE}

    # ---- video -----------------------------------------------------------
    say("\n" + "-" * 78)
    say("VIDEO (real 4096x2160 frames, per frame)")
    say("-" * 78)
    names = sorted(glob.glob(os.path.join(FRAMES, "*.png")))[:3]
    vid = []
    for p in names:
        fr = np.asarray(Image.open(p).convert("RGB"))
        h = conditional_entropy(fr, t)
        raw = fr.size
        vid.append({"frame": os.path.basename(p), **h, "raw_rgb": raw,
                    "side_info_pct": 100 * h["side_info_bytes"] / raw})
        say(f"  {os.path.basename(p)}: {h['samples']} samples, "
            f"H(V|T)={h['H_V_given_T_bits_per_sample']:.4f} bits/sample, "
            f"side-info={h['side_info_pct']:.4f} % of raw RGB")
    res["video_side_info"] = vid
    if vid:
        mean_pct = float(np.mean([v["side_info_pct"] for v in vid]))
        say(f"  mean side-info over {len(vid)} frames = {mean_pct:.4f} % of raw RGB")

    # ---- comparison ------------------------------------------------------
    say("\n" + "=" * 78)
    say("SO WHAT?  (image, % of raw RGB; lower is better)")
    say("=" * 78)
    rows = [
        ("LMCompress-igpt as shipped (LOSSY)", 100 * IGPT_IMAGE_CODE_BYTES / RAW_RGB_IMAGE),
        ("LMCompress-igpt + optimal side channel (LOSSLESS bound)",
         100 * tot / RAW_RGB_IMAGE),
        ("LMCompress-bgpt (authors, lossless)", 52.0896),
        ("WebP lossless (PIL, method 6)", 46.5672),
        ("JPEG-2000 lossless (PIL)", 50.2369),
        ("PNG (PIL optimize, level 9)", 52.0878),
        ("PNG (ffmpeg, earlier project run)", 84.3809),
    ]
    say(f"{'method':56s} {'rate':>9s}")
    for n, r in rows:
        say(f"{n:56s} {r:8.4f}%")
    res["comparison"] = [{"method": n, "rate_pct_raw_rgb": r} for n, r in rows]

    say("")
    say("Caveat: H(V|T) is a LOWER bound (it assumes the decoder knows the exact")
    say("conditional distribution). A real side channel costs more, so the")
    say("'lossless bound' row is optimistic for the iGPT route.")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nwritten -> {OUT_TXT}\nwritten -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
