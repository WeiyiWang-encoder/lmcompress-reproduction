#!/usr/bin/env python3
"""What rate does the AUTHORS' video path actually produce, and does the shipped
image path reproduce at the same per-chunk cost?

The paper (SI Table 3/4) reports LMCompress-igpt as
    image  CLIC2019 14.62 %      video (high-res) 15.31 %
under the convention

    rate = arithmetic-coded bytes / (chunks * 1024)

because ``compress_with_igpt.evaluate_compressor_chunked`` does
``raw_length += h*w`` with an already-quantised (32,32) chunk.

A prior investigation in this project measured "video via iGPT" with its **own**
construction: one whole frame resized to a single 32x32 grid = 1024 tokens/frame.
The authors' own loader does something different -- ``_extract_image_sequence``
takes 1024-pixel *raster segments* of each channel of the full-resolution frame,
so a 4096x2160 frame yields 3 * 8640 = 25920 chunks of 1024 tokens.

This script measures the authors' real path with a bounded number of chunks and
reports bytes/chunk, then extrapolates.  It first validates the estimator on the
image path against the known full-run value (158.8676 bytes/chunk).

Runs on CPU by default so it never contends with the GPU jobs.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np

MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMG_DIR = os.path.join(REPO, "data", "test_images")
FRAMES_SRC = os.path.join(REPO, "repro_understanding_is_compression",
                          "artifacts", "frames", "4k_aerial_first_second")
RESULTS = os.path.join(REPO, "repro_understanding_is_compression", "results")
OUT = os.path.join(RESULTS, "_verify_video_rate.txt")

KNOWN_IMAGE_BYTES_PER_CHUNK = 158.8676085818943   # results/official_image_accounting.json
CHUNK_TOKENS = 1024
FRAME_W, FRAME_H = 4096, 2160
N_FRAMES = 60

_lines: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    _lines.append(m)


class StopAfter(Exception):
    pass


def run_bounded(dataset: str, data_path: str, n_chunks: int) -> dict:
    """Drive the authors' compress_with_igpt.main() for exactly n_chunks."""
    import compressor
    real = compressor.COMPRESS_FN_DICT["language_model"]
    acc = {"bytes": 0, "chunks": 0, "shapes": set()}

    def counting(data, *a, **kw):
        if acc["chunks"] >= n_chunks:
            raise StopAfter()
        out = real(data, *a, **kw)
        # language_model.compress returns (compressed_bytes, log_probs)
        acc["bytes"] += len(out[0])
        acc["chunks"] += 1
        try:
            acc["shapes"].add(tuple(np.asarray(data).shape))
        except Exception:
            pass
        return out

    compressor.COMPRESS_FN_DICT["language_model"] = counting

    import compress_with_igpt as C
    args = argparse.Namespace(compressor="language_model", dataset=dataset,
                              data_path=data_path, total=None, model_path=MODEL_DIR)
    buf = io.StringIO()
    status = "ok"
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            C.main(args)
    except StopAfter:
        status = "stopped-after-limit"
    except Exception as e:
        status = f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    compressor.COMPRESS_FN_DICT["language_model"] = real
    return {"status": status, "bytes": acc["bytes"], "chunks": acc["chunks"],
            "shapes": sorted(str(s) for s in acc["shapes"]), "secs": dt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200)
    ap.add_argument("--cpu", action="store_true", default=True)
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    say("=" * 78)
    say("What rate does the authors' OWN video path produce?")
    say("=" * 78)

    # ---- force CPU so this cannot contend with the GPU jobs ---------------
    import torch
    import language_model
    if a.cpu:
        language_model.device = torch.device("cpu")
        say("device forced to CPU (language_model.device patched)")
    say(f"chunks per run = {a.chunks}")

    # ---- 1. image path: validate the estimator --------------------------
    say("\n" + "-" * 78)
    say("1) IMAGE path (--dataset imagenet) on the SHIPPED full-resolution PNG")
    say("-" * 78)
    r_img = run_bounded("imagenet", IMG_DIR, a.chunks)
    say(f"   status = {r_img['status']}   shapes seen = {r_img['shapes']}")
    say(f"   chunks = {r_img['chunks']}   coded bytes = {r_img['bytes']}   [{r_img['secs']:.0f}s]")
    bpc_img = r_img["bytes"] / max(r_img["chunks"], 1)
    say(f"   bytes/chunk = {bpc_img:.4f}")
    say(f"   known FULL-run value (all 5733 chunks) = {KNOWN_IMAGE_BYTES_PER_CHUNK:.4f}")
    say(f"   estimator error = {100*(bpc_img/KNOWN_IMAGE_BYTES_PER_CHUNK - 1):+.2f} %")
    say(f"   implied rate on the shipped image = {100*bpc_img/CHUNK_TOKENS:.4f} %")
    say(f"   [paper SI Table 3, CLIC2019, LMCompress-igpt = 14.62 %]")

    # ---- 2. video path ---------------------------------------------------
    say("\n" + "-" * 78)
    say("2) VIDEO path (--dataset raw_video_frames) on real 4K PNG frames")
    say("-" * 78)
    d = os.path.join(tempfile.gettempdir(), "verify_video_rate_frames")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    names = sorted(glob.glob(os.path.join(FRAMES_SRC, "*.png")))[:4]
    for f in names:
        shutil.copy2(f, d)
    say(f"   frames supplied = {len(names)}  ({FRAME_W}x{FRAME_H}, real clip)")
    t0 = time.time()
    r_vid = run_bounded("raw_video_frames", d, a.chunks)
    say(f"   status = {r_vid['status']}   shapes seen = {r_vid['shapes']}")
    say(f"   chunks = {r_vid['chunks']}   coded bytes = {r_vid['bytes']}   [{r_vid['secs']:.0f}s]")
    bpc_vid = r_vid["bytes"] / max(r_vid["chunks"], 1)
    say(f"   bytes/chunk = {bpc_vid:.4f}")
    rate_vid = 100 * bpc_vid / CHUNK_TOKENS
    say(f"   rate (paper convention, bytes/(chunks*1024)) = {rate_vid:.4f} %")

    chunks_per_frame = 3 * (FRAME_W * FRAME_H) // CHUNK_TOKENS
    say(f"\n   chunks per frame  = 3 channels x {FRAME_W*FRAME_H//CHUNK_TOKENS} "
        f"= {chunks_per_frame}")
    say(f"   per-frame samples = {chunks_per_frame*CHUNK_TOKENS}  "
        f"(= 3 x {FRAME_W*FRAME_H} pixels = {3*FRAME_W*FRAME_H})")
    tot_chunks = chunks_per_frame * N_FRAMES
    say(f"   whole clip ({N_FRAMES} frames) = {tot_chunks} chunks, "
        f"denominator {tot_chunks*CHUNK_TOKENS} samples")
    say(f"   estimated code size for the clip = "
        f"{bpc_vid*tot_chunks/1e6:.1f} MB")
    say(f"   [paper SI Table 4, high-resolution video, LMCompress-igpt = 15.31 %]")

    # ---- 3. the prior project's convention, for contrast -----------------
    say("\n" + "-" * 78)
    say("3) contrast with the whole-frame convention used earlier in this project")
    say("-" * 78)
    say("   earlier run (tools/run_video_igpt.py): ONE 32x32 grid per frame")
    say(f"     -> 1024 tokens/frame, denominator {FRAME_W*FRAME_H} pixels/frame,")
    say("        reported rate 0.004200 %")
    say(f"   authors' loader: {chunks_per_frame} chunks/frame "
        f"({chunks_per_frame}x more tokens)")
    say("   -> the two are different objects; the 0.0042 % figure is not the")
    say("      authors' video path.")

    say("\n" + "=" * 78)
    say("SUMMARY")
    say("=" * 78)
    say(f"  image  bytes/chunk {bpc_img:9.4f}  -> rate {100*bpc_img/CHUNK_TOKENS:8.4f} %"
        f"   (paper 14.62 %)")
    say(f"  video  bytes/chunk {bpc_vid:9.4f}  -> rate {rate_vid:8.4f} %"
        f"   (paper 15.31 %)")
    say(f"  both paths reach ImageGPTImageProcessor with do_color_quantize=True")
    say(f"  video chunks/frame = {chunks_per_frame} (channel-wise raster segments)")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwritten -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
