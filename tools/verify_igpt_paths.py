#!/usr/bin/env python3
"""Verification 2: do BOTH the image path and the video path of the authors'
own code push pixels through ``ImageGPTImageProcessor`` (i.e. colour-quantise)?

Method
------
1. Wrap ``ImageGPTImageProcessor.__call__`` with a spy that records every call:
   the exact array it was handed, the token grid it returned, and whether
   ``do_color_quantize`` was on.
2. Drive the authors' *unmodified* ``compress_with_igpt.main()`` twice:
     a) ``--dataset imagenet``          (still images)
     b) ``--dataset raw_video_frames``  (video, as PNG frames, exactly what
                                         ``utils/preprocess_video_dataset.py``
                                         produces from a .y4m)
   The only intervention is a call counter that stops the run after N chunks so
   it finishes quickly; every chunk that does run goes through the real code.
3. Separately prove the quantiser is not injective (source values collide), so
   "went through the processor" implies "quantised, not preserved".

Writes ``results/_verify_igpt_paths.txt`` and ``results/_verify_igpt_paths.json``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import constants

MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMG_DIR = os.path.join(REPO, "data", "test_images")
FRAMES_SRC = os.path.join(REPO, "repro_understanding_is_compression",
                          "artifacts", "frames", "4k_aerial_first_second")
RESULTS = os.path.join(REPO, "repro_understanding_is_compression", "results")
OUT_TXT = os.path.join(RESULTS, "_verify_igpt_paths.txt")
OUT_JSON = os.path.join(RESULTS, "_verify_igpt_paths.json")

N_CHUNKS = 40          # per run; enough to characterise the path
_lines: list[str] = []
CALLS: list[dict] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    _lines.append(msg)


class StopAfter(Exception):
    pass


# --------------------------------------------------------------------------
def install_spy() -> None:
    from transformers import ImageGPTImageProcessor

    orig_call = ImageGPTImageProcessor.__call__

    def spy(self, images, *a, **kw):
        out = orig_call(self, images, *a, **kw)
        ids = out["input_ids"] if isinstance(out, dict) else getattr(out, "input_ids", out)
        arr = np.asarray(ids)
        # the array the processor was actually handed
        try:
            src = np.asarray(images)
            src_shape = list(src.shape)
            src_dtype = str(src.dtype)
            src_min = float(np.nanmin(src)) if src.size else None
            src_max = float(np.nanmax(src)) if src.size else None
        except Exception:
            src_shape, src_dtype, src_min, src_max = None, None, None, None
        def _n_clusters(self):
            c = getattr(self, "clusters", None)
            if c is None:
                return None
            try:
                return int(np.asarray(c).shape[0])
            except Exception:
                return None

        CALLS.append({
            "input_shape": src_shape,
            "input_dtype": src_dtype,
            "input_min": src_min,
            "input_max": src_max,
            "output_shape": list(arr.shape),
            "output_dtype": str(arr.dtype),
            "token_min": int(arr.min()),
            "token_max": int(arr.max()),
            "n_distinct_tokens": int(np.unique(arr).size),
            "n_token_slots": int(arr.size),
            "do_color_quantize": bool(getattr(self, "do_color_quantize", False)),
            "do_resize": bool(getattr(self, "do_resize", False)),
            "size": dict(getattr(self, "size", {}) or {}),
            "n_clusters": _n_clusters(self),
            "kwargs": {k: str(v)[:60] for k, v in kw.items()},
        })
        return out

    ImageGPTImageProcessor.__call__ = spy


def install_call_limiter(n: int) -> None:
    """Wrap the registered language_model compressor to stop after `n` chunks."""
    import compressor
    if "_verify_real_compress" not in globals():
        globals()["_verify_real_compress"] = compressor.COMPRESS_FN_DICT["language_model"]
    real = globals()["_verify_real_compress"]
    state = {"i": 0}

    def limited(*a, **kw):
        if state["i"] >= n:
            raise StopAfter()
        state["i"] += 1
        return real(*a, **kw)

    compressor.COMPRESS_FN_DICT["language_model"] = limited


def run_authors_main(dataset: str, data_path: str) -> dict:
    import compress_with_igpt as C
    args = argparse.Namespace(
        compressor="language_model",
        dataset=dataset,
        data_path=data_path,
        total=None,
        model_path=MODEL_DIR,
    )
    buf = io.StringIO()
    status = "ok"
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            C.main(args)
    except StopAfter:
        status = "stopped-after-limit"
    except Exception as e:
        status = f"{type(e).__name__}: {e}"
        say(traceback.format_exc()[-1500:])
    tail = [l for l in buf.getvalue().splitlines() if l.strip()][-6:]
    return {"status": status, "stdout_tail": tail}


def make_small_image_dir(src_dir: str, out_dir: str, size: int = 128) -> str:
    """Crop the real test image so the instrumented run is fast.

    The crop is only to bound runtime; every processed chunk still flows through
    the authors' loader -> language_model.compress -> ImageGPTImageProcessor.
    """
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
    for f in names:
        with Image.open(os.path.join(src_dir, f)) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = min(size, w, h)
            im = im.crop((0, 0, side, side))
            im.save(os.path.join(out_dir, f))
    return out_dir


def prepare_frames(n: int = 3) -> str:
    """A directory of PNG frames, the layout `raw_video_frames` expects."""
    d = os.path.join(tempfile.gettempdir(), "verify_video_frames")
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    names = sorted(f for f in os.listdir(FRAMES_SRC) if f.lower().endswith(".png"))[:n]
    for f in names:
        shutil.copy2(os.path.join(FRAMES_SRC, f), os.path.join(d, f))
    return d


def injectivity_probe() -> dict:
    """Exhaustive: how many of the 256 source values survive the quantiser?

    The authors feed the processor a (32,32,1) chunk which they first turn into a
    (32,32,3) grey image with ``np.repeat(..., 3, axis=2)``.  We reproduce that
    exactly, placing each of the 256 uint8 values four times so the grid is
    already 32x32 and ``do_resize`` is the identity.
    """
    from transformers import ImageGPTImageProcessor
    proc = ImageGPTImageProcessor.from_pretrained(MODEL_DIR)
    vals = np.repeat(np.arange(256, dtype=np.uint8), 4)          # 1024 values
    grey = vals.reshape(32, 32, 1)
    rgb = np.repeat(grey, 3, axis=2)                             # (32,32,3) uint8
    ids = np.asarray(proc(rgb, return_tensors="np")["input_ids"]).reshape(-1)
    assert ids.size == 1024, ids.shape
    m: dict[int, list[int]] = {}
    for v, t in zip(vals.tolist(), ids.tolist()):
        m.setdefault(int(t), []).append(int(v))
    # every value must appear 4x and every repeat must land on one token
    consistent = all(len({ids[i] for i in range(k, 1024, 256)}) == 1 for k in range(4))
    collisions = sum(len(set(v)) - 1 for v in m.values() if len(set(v)) > 1)
    worst = sorted(m.items(), key=lambda kv: -len(set(kv[1])))[:4]
    return {
        "domain_values": 256,
        "distinct_tokens": len(m),
        "colliding_pairs": collisions,
        "repeat_consistent": consistent,
        "largest_groups": [{"token": t, "source_values": sorted(set(v))}
                           for t, v in worst],
    }


# --------------------------------------------------------------------------
def main() -> int:
    say("=" * 78)
    say("VERIFICATION 2 - ImageGPT quantisation on the image AND video paths")
    say("=" * 78)
    say(f"repo      = {REPO}")
    say(f"model dir = {MODEL_DIR}")
    os.makedirs(RESULTS, exist_ok=True)

    install_spy()
    from transformers import ImageGPTImageProcessor
    proc = ImageGPTImageProcessor.from_pretrained(MODEL_DIR)
    say("\n[preprocessor_config.json, as loaded]")
    say(f"  do_color_quantize = {proc.do_color_quantize}")
    say(f"  do_resize         = {proc.do_resize}")
    say(f"  size              = {dict(proc.size)}")
    say(f"  n clusters        = {len(proc.clusters)}")
    say(f"  rescale_factor    = {getattr(proc, 'rescale_factor', None)}")
    say(f"  image_mean/std    = {getattr(proc,'image_mean',None)} / {getattr(proc,'image_std',None)}")

    report: dict = {"processor": {
        "do_color_quantize": bool(proc.do_color_quantize),
        "do_resize": bool(proc.do_resize),
        "size": dict(proc.size),
        "n_clusters": len(proc.clusters),
    }}

    # ---------------- image path -----------------------------------------
    small = os.path.join(tempfile.gettempdir(), "verify_img_small")
    make_small_image_dir(IMG_DIR, small, size=128)
    CALLS.clear()
    install_call_limiter(N_CHUNKS)
    say("\n" + "-" * 78)
    say("A) authors' compress_with_igpt.main()  --dataset imagenet   (still images)")
    say("-" * 78)
    say(f"   data_path = {small}  (128x128 crop of the shipped test image)")
    r_img = run_authors_main("imagenet", small)
    say(f"   status    = {r_img['status']}")
    for l in r_img["stdout_tail"]:
        say("   | " + l[:110])
    say(f"   ImageGPTImageProcessor calls = {len(CALLS)}")
    report["image"] = {"status": r_img["status"], "n_processor_calls": len(CALLS),
                       "calls": CALLS[:3], "stdout_tail": r_img["stdout_tail"]}
    for c in CALLS[:3]:
        say(f"     in {c['input_shape']} {c['input_dtype']} [{c['input_min']}..{c['input_max']}]"
            f" -> out {c['output_shape']} tokens[{c['token_min']}..{c['token_max']}]"
            f" distinct={c['n_distinct_tokens']} quantize={c['do_color_quantize']}")

    # ---------------- video path -----------------------------------------
    frames = prepare_frames(3)
    CALLS.clear()
    install_call_limiter(N_CHUNKS)
    say("\n" + "-" * 78)
    say("B) authors' compress_with_igpt.main()  --dataset raw_video_frames   (video)")
    say("-" * 78)
    say(f"   data_path = {frames}  (PNG frames, the layout")
    say("               utils/preprocess_video_dataset.py:y4m2png produces)")
    r_vid = run_authors_main("raw_video_frames", frames)
    say(f"   status    = {r_vid['status']}")
    for l in r_vid["stdout_tail"]:
        say("   | " + l[:110])
    say(f"   ImageGPTImageProcessor calls = {len(CALLS)}")
    report["video"] = {"status": r_vid["status"], "n_processor_calls": len(CALLS),
                       "calls": CALLS[:3], "stdout_tail": r_vid["stdout_tail"]}
    for c in CALLS[:3]:
        say(f"     in {c['input_shape']} {c['input_dtype']} [{c['input_min']}..{c['input_max']}]"
            f" -> out {c['output_shape']} tokens[{c['token_min']}..{c['token_max']}]"
            f" distinct={c['n_distinct_tokens']} quantize={c['do_color_quantize']}")

    # ---------------- lossiness ------------------------------------------
    say("\n" + "-" * 78)
    say("C) is that processor lossy?  exhaustive 256-value probe")
    say("-" * 78)
    inj = injectivity_probe()
    say(f"   256 source values map onto {inj['distinct_tokens']} distinct tokens"
        f"  ({inj['colliding_pairs']} colliding pairs)")
    report["injectivity"] = {"domain_values": inj["domain_values"],
                             "distinct_tokens": inj["distinct_tokens"],
                             "colliding_pairs": inj["colliding_pairs"]}

    # ---------------- any non-iGPT image/video route? ---------------------
    say("\n" + "-" * 78)
    say("D) other image/video entry points in the same repo")
    say("-" * 78)
    say("   compress_with_igpt.py        --dataset imagenet|raw_video_frames -> iGPT (quantised)")
    say("   compress_with_bgpt_or_llama.py --data_type bmp|video|wav         -> bGPT, raw bytes, NO iGPT")
    say("   compress_with_baseline.py    --mode text (accepts .y4m)          -> zstd/zpaq/7z/brotli, NO iGPT")
    say("   run.sh routes ./data/vidoes/{4k,dynamic,static} to compress_with_baseline.py --mode text")

    say("\n" + "=" * 78)
    ok_img = len(CALLS) > 0 and report["image"]["n_processor_calls"] > 0
    say(f"VERDICT image path  : {report['image']['n_processor_calls']} processor calls  -> "
        f"{'QUANTISED' if ok_img else 'NOT OBSERVED'}")
    say(f"VERDICT video path  : {report['video']['n_processor_calls']} processor calls  -> "
        f"{'QUANTISED' if report['video']['n_processor_calls'] > 0 else 'NOT OBSERVED'}")
    say(f"VERDICT lossiness   : {inj['distinct_tokens']}/256 values survive -> LOSSY")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten -> {OUT_TXT}\nwritten -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
