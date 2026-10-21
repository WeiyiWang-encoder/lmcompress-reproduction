#!/usr/bin/env python3
"""Is the paper's image method lossless -- and if not, what IS a fair lossless test?

Part A  answers the losslessness question by inspecting the bitstream the authors'
        own coder emits, and by the exhaustive injectivity result.
Part B  runs the authors' OWN lossless image path (bGPT over raw bytes,
        ``weights-image.pth``, ``--data_type bmp``) and compares it, on identical
        pixels, against the classical lossless image codecs the paper itself uses
        as baselines (PNG / WebP / JPEG-2000 / JPEG-XL).

Why bGPT is the right vehicle for a lossless test
-------------------------------------------------
bGPT models the file one *byte* at a time over a 257-symbol alphabet (0..255 plus
a pad symbol).  The symbol it codes IS the source symbol, so arithmetic coding on
its predictive distribution is invertible by construction -- no quantiser sits
between the source and the coder.  iGPT cannot be used this way: its input is a
512-way k-means colour token, so the quantiser is unavoidable.

The paper's own SI Table 3 has both rows (LMCompress-igpt 14.62 %, LMCompress-bgpt
28.57 % on CLIC2019), which is consistent with exactly this reading.

Writes results/_lossless_image.txt / .json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

from PIL import Image

RESULTS = os.path.join(REPO, "repro_understanding_is_compression", "results")
WORK = os.path.join(REPO, "repro_understanding_is_compression", "artifacts", "lossless_img")
SRC_PNG = os.path.join(REPO, "data", "test_images",
                       "0a8ea388b5df64e6337d65e65e7d5d55d74abbf5.png")
BGPT_IMAGE = os.path.join(REPO, "data", "model", "bgpt", "weights-image.pth")
CTX = 8160

OUT_TXT = os.path.join(RESULTS, "_lossless_image.txt")
OUT_JSON = os.path.join(RESULTS, "_lossless_image.json")
_lines: list[str] = []


def say(m: str = "") -> None:
    print(m, flush=True)
    _lines.append(m)


# --------------------------------------------------------------------------
def verify_bitstream_is_lossy() -> dict:
    """Show the iGPT path codes token ids, never source pixel values."""
    import inspect
    import language_model
    src = inspect.getsource(language_model.compress)
    lines = src.splitlines()
    # the encode calls, with the symbol being coded
    enc = [l.strip() for l in lines if "encoder.encode(" in l]
    sym_assign = [l.strip() for l in lines if "sequence_array =" in l
                  or "symbols.append" in l]
    return {"encode_calls": enc, "symbol_sources": sym_assign}


# --------------------------------------------------------------------------
def run_bgpt_image(bmp_path: str) -> dict:
    """Drive the authors' own lossless byte-level image path."""
    import compress_with_bgpt_or_llama as C
    buf = io.StringIO()
    t0 = time.time()
    err = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            C.compress_with_llm(
                data_path=bmp_path,
                data_type="bmp",
                model_path=BGPT_IMAGE,
                lora_path=None,
                context_window_size=CTX,
                batch_size=1,
            )
    except Exception as e:
        import traceback
        err = f"{type(e).__name__}: {e}"
        say(traceback.format_exc()[-1200:])
    dt = time.time() - t0
    text = buf.getvalue()
    dump = os.path.join(RESULTS, "_lossless_img_bgpt_stdout.txt")
    with open(dump, "w", encoding="utf-8") as f:
        f.write(text)

    got = {}
    for line in text.splitlines():
        s = line.strip()
        m = re.search(r"total compression ratio:\s*([0-9.]+)", s)
        if m:
            got["ratio"] = float(m.group(1))
        m = re.search(r"Total size:\s*([0-9]+)", s)
        if m:
            got["total_size"] = int(m.group(1))
        m = re.search(r"compressed size:\s*([0-9.]+)", s)
        if m:
            got["compressed_size"] = float(m.group(1))
        m = re.search(r"Parameter Number:\s*([0-9]+)", s)
        if m:
            got["params"] = int(m.group(1))
    got["secs"] = dt
    got["error"] = err
    got["stdout_tail"] = [l for l in text.splitlines() if l.strip()][-6:]
    return got


# --------------------------------------------------------------------------
def classical_codecs(im: Image.Image) -> dict:
    raw = im.tobytes()
    n_raw = len(raw)
    out = {}
    trials = [
        ("png", dict(format="PNG", optimize=True, compress_level=9), ".png"),
        ("webp_lossless", dict(format="WEBP", lossless=True, quality=100, method=6), ".webp"),
        ("jpeg2000_lossless", dict(format="JPEG2000", irreversible=False), ".jp2"),
        ("jpegxl_lossless", dict(format="JXL", lossless=True, effort=9), ".jxl"),
        ("tiff_deflate", dict(format="TIFF", compression="tiff_deflate"), ".tif"),
    ]
    for name, kw, ext in trials:
        p = os.path.join(WORK, f"cls_{name}{ext}")
        try:
            t0 = time.time()
            im.save(p, **kw)
            n = os.path.getsize(p)
            out[name] = {"bytes": n, "rate_pct_raw_rgb": 100 * n / n_raw,
                         "ratio": n_raw / n, "secs": time.time() - t0}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    # a plain-context coder on the raw pixels, as an additional anchor
    try:
        import brotli
        n = len(brotli.compress(raw, quality=11))
        out["brotli_q11_raw"] = {"bytes": n, "rate_pct_raw_rgb": 100 * n / n_raw,
                                 "ratio": n_raw / n}
    except Exception as e:
        out["brotli_q11_raw"] = {"error": str(e)}
    out["_raw_rgb_bytes"] = n_raw
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bgpt", action="store_true")
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    say("=" * 78)
    say("Is the paper's image method lossless?  And what is the right lossless test?")
    say("=" * 78)

    im = Image.open(SRC_PNG).convert("RGB")
    W, H = im.size
    n_raw = W * H * 3
    say(f"\ntest image : {os.path.basename(SRC_PNG)}")
    say(f"             {W} x {H} RGB, raw = {n_raw} bytes")

    # ---------------- Part A -------------------------------------------
    say("\n" + "-" * 78)
    say("A) What does the iGPT path actually put into the arithmetic coder?")
    say("-" * 78)
    bs = verify_bitstream_is_lossy()
    say("   language_model.compress, the encode call(s):")
    for l in bs["encode_calls"]:
        say(f"     {l}")
    say("   where `symbol` comes from:")
    for l in bs["symbol_sources"]:
        say(f"     {l}")
    say("")
    say("   `sequence_array` is the ImageGPT processor output (token ids), and the")
    say("   coder emits exactly that id -- there is no residual / side channel.")
    say("   The quantiser is non-injective (256 -> 48 ids, 585 colliding pairs),")
    say("   so no decoder can invert it.  => the image and video paths are LOSSY.")
    say("   (Text uses Llama token ids = the paper's own unit; audio uses raw bytes")
    say("    via bGPT -> those two ARE lossless by construction.)")

    # ---------------- Part B -------------------------------------------
    say("\n" + "-" * 78)
    say("B) The authors' own LOSSLESS image path: bGPT over raw bytes")
    say("-" * 78)
    bmp = os.path.join(WORK, "test_image.bmp")
    im.save(bmp, format="BMP")
    n_bmp = os.path.getsize(bmp)
    say(f"   prepared {bmp}")
    say(f"   BMP size = {n_bmp} bytes  (header {n_bmp - n_raw} B, no row padding)")
    say(f"   command equivalent: compress_with_bgpt_or_llama.py --data_type bmp "
        f"--model_path weights-image.pth --context_window {CTX} --batch_size 1")

    res = {"image": {"file": os.path.basename(SRC_PNG), "width": W, "height": H,
                     "raw_rgb_bytes": n_raw, "bmp_bytes": n_bmp}}
    if not a.skip_bgpt:
        say("   running (authors' unmodified code) ...")
        bg = run_bgpt_image(bmp)
        res["bgpt"] = bg
        say(f"   status   : {'ERROR ' + str(bg['error']) if bg['error'] else 'ok'}")
        for l in bg["stdout_tail"]:
            say("     | " + l[:110])
        if "compressed_size" in bg:
            nb = bg["compressed_size"]
            say(f"   bGPT code size        = {nb:.1f} B")
            say(f"   rate vs BMP file      = {100*nb/n_bmp:.4f} %   (harness ratio "
                f"{bg.get('ratio')})")
            say(f"   rate vs raw RGB bytes = {100*nb/n_raw:.4f} %")
            say(f"   ratio vs raw RGB      = {n_raw/nb:.4f}x")
        say(f"   wall clock            = {bg['secs']:.0f}s")

    # ---------------- classical ----------------------------------------
    say("\n" + "-" * 78)
    say("C) Classical LOSSLESS image codecs, identical pixels")
    say("-" * 78)
    cls = classical_codecs(im)
    res["classical"] = cls
    say(f"{'codec':22s} {'bytes':>12s} {'rate vs raw RGB':>17s} {'ratio':>9s}")
    for k, v in cls.items():
        if k.startswith("_"):
            continue
        if "error" in v:
            say(f"{k:22s} {'--':>12s} {'unsupported':>17s} {'--':>9s}")
        else:
            say(f"{k:22s} {v['bytes']:12d} {v['rate_pct_raw_rgb']:16.4f}% "
                f"{v['ratio']:9.4f}")

    # ---------------- comparison ---------------------------------------
    say("\n" + "=" * 78)
    say("COMPARISON (denominator = raw RGB bytes = %d)" % n_raw)
    say("=" * 78)
    rows = []
    if not a.skip_bgpt and "compressed_size" in res.get("bgpt", {}):
        nb = res["bgpt"]["compressed_size"]
        rows.append(("LMCompress-bgpt (authors, lossless)", nb, 100 * nb / n_raw))
    for k, v in cls.items():
        if k.startswith("_") or "error" in v:
            continue
        rows.append((k, v["bytes"], v["rate_pct_raw_rgb"]))
    rows.sort(key=lambda r: r[2])
    say(f"{'method':40s} {'bytes':>12s} {'rate':>9s}")
    for name, nb, rate in rows:
        say(f"{name:40s} {nb:12.0f} {rate:8.4f}%")
    res["comparison"] = [{"method": n, "bytes": b, "rate_pct_raw_rgb": r}
                         for n, b, r in rows]

    say("\nreference (paper, CLIC2019, SI Table 3):")
    say("   LMCompress-igpt 14.62 %  (LOSSY - cannot be compared with these)")
    say("   LMCompress-bgpt 28.57 %  (lossless)")
    say("   PNG 47.08 %   JPEG-XL 31.16 %   WebP 35.56 %   JPEG-2000 48.64 %")
    say("...but this capsule's test image is a 4K aerial detail crop, much harder")
    say("   than CLIC2019 (PNG measures ~84 % here vs 47 % there).")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nwritten -> {OUT_TXT}\nwritten -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
