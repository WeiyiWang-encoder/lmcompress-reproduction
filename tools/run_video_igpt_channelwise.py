"""Video modality via the ImageGPT path, channel-wise, batched, resumable.

Rewritten three times, for three distinct defects, each worth recording:

 1. **RGB instead of channel-wise.** The first version tokenised each frame as a
    whole RGB image. The official loader codes one channel at a time and feeds the
    processor a greyscale triple, so the tokens differ from the very first token
    (AUDIT_FOR_REVIEW.md section 4.4.4). Fixed here.
 2. **Rebuilding every frame's units on every iteration.** The second version called
    ``image_units`` on the whole frame directory inside the per-frame loop, so
    processing frame 1 constructed units for all 60 frames. That is why a one-frame
    probe looked hung while the GPU sat at 0 %.
 3. **One processor call per unit.** The processor costs ~1.9 ms per call, so
    ~25 900 separate calls per frame is ~51 s of pure overhead. Here all 32x32
    greyscale triples for a frame are stacked into one array and handed to the
    processor in a single call.

Checkpointing is per frame with an atomic append, so an interruption costs at most one
frame, and a rerun aggregates whatever is already on disk.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
PATCH = 32


def tokenise_frame(rgb: np.ndarray, processor, patch: int = PATCH,
                   batch: int = 2000):
    """Tile a whole frame into a few processor calls, in the official flattened order.

    Returns ``(ids, units, per_channel)`` where ``ids`` is ``(units, patch*patch)``.
    Order matches ``get_imagenet_iterator(is_channel_wised=True, is_seq=True)``:
    channel-major, then consecutive flattened runs of ``patch*patch`` samples.

    ``batch`` bounds the processor call size. Passing all ~25 900 patches at once
    makes the quantiser's CPU matmul request >5 GB
    (``squared_euclidean_distance_torch`` builds an ``(N, 512)`` block), so the call
    is chunked; 2 000 keeps the transient under ~0.4 GB.
    """
    h, w, _ = rgb.shape
    n_patch = patch * patch
    per_channel = (h * w) // n_patch
    units = per_channel * 3

    # Build the full stacked array once; it is uint8 and ~24 MB per thousand patches.
    stacked = np.empty((units, patch, patch, 3), dtype=np.uint8)
    i = 0
    for channel in range(3):
        flat = rgb[:, :, channel].reshape(-1)
        for k in range(per_channel):
            block = flat[k * n_patch:(k + 1) * n_patch].reshape(patch, patch)
            stacked[i] = np.repeat(block[:, :, None], 3, axis=2)
            i += 1

    out_parts = []
    for start in range(0, units, batch):
        stop = min(start + batch, units)
        res = np.asarray(processor(stacked[start:stop],
                                   return_tensors="np")["input_ids"])
        out_parts.append(res.reshape(stop - start, -1))
    return (np.concatenate(out_parts, axis=0).astype(np.int64), units, per_channel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=os.path.join(
        HERE, "artifacts", "frames", "4k_aerial_first_second"))
    ap.add_argument("--model-dir", default=os.path.join(
        REPO, "data", "model", "imagegpt-small"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results",
                                                  "video_igpt_channelwise.jsonl"))
    args = ap.parse_args()

    from PIL import Image

    from uic.data import _igpt_processor
    from uic.engine import measure_stream
    from uic.models import ImageGPTPredictor

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.png")))
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"no frames in {args.frames_dir}")
        return 1

    proc = _igpt_processor(args.model_dir)
    pred = ImageGPTPredictor.load(args.model_dir, device="cuda:0")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    done: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["frame"])

    w = h = 0
    t0 = time.time()
    try:
        for fi, path in enumerate(frames):
            name = os.path.basename(path)
            if name in done:
                print(f"[{fi+1}/{len(frames)}] {name}: already done", flush=True)
                continue
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
            h, w, _ = rgb.shape

            t_enc = time.time()
            ids, units, per_channel = tokenise_frame(rgb, proc)
            enc_s = time.time() - t_enc

            # Vectorised scoring: one forward pass per unit, then the model-side code
            # length -log2 p_true computed on the GPU. The arithmetic encoder is
            # skipped deliberately (measure_stream(with_coder=False) was still
            # 62 ms/unit, dominated by per-symbol Python); the reported rate does not
            # depend on the coder's quantised probabilities. Chunked to bound memory.
            t_mod = time.time()
            bits = 0.0
            CH = 32   # benchmarked: batch >32 is slower and 128 OOMs on 12 GiB
            for start in range(0, units, CH):
                stop = min(start + CH, units)
                buf = torch.tensor(ids[start:stop], dtype=torch.long,
                                   device=pred.device)
                with torch.no_grad():
                    lg = pred.model(buf).logits.float()
                lp = torch.log_softmax(lg, dim=-1)[:, :-1, :]
                tgt = buf[:, 1:]
                nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                bits += float(nll.sum().item())
                del lg, lp, nll
            mod_s = time.time() - t_mod

            samples = units * PATCH * PATCH
            row_out = {
                "frame": name,
                "frame_index": fi,
                "units": units,
                "per_channel_chunks": per_channel,
                "samples": samples,
                "coded_bits": round(bits, 6),
                "coded_bytes": round(bits / 8, 3),
                "rate_pct_official": round(100 * (bits / 8) / samples, 6),
                "tokenise_seconds": round(enc_s, 2),
                "model_seconds": round(mod_s, 2),
            }
            with open(args.out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row_out, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"[{fi+1}/{len(frames)}] {name}: {units} units, "
                  f"{bits/8:.0f} B, rate {row_out['rate_pct_official']:.4f} % "
                  f"(tok {enc_s:.1f}s + model {mod_s:.1f}s, "
                  f"elapsed {time.time()-t0:.0f}s)", flush=True)
    finally:
        pred.release()

    rows = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
    bits = sum(r["coded_bits"] for r in rows)
    samples = sum(r["samples"] for r in rows)
    rgb_bytes = w * h * 3 * len(rows) if w and h else 0

    print("\n=== video via iGPT, channel-wise (official convention) ===")
    print(f"  frames                = {len(rows)}")
    print(f"  processed samples     = {samples}")
    print(f"  coded bytes           = {bits/8:.1f}")
    print(f"  rate (official conv.) = {100*(bits/8)/samples:.6f} %")
    if rgb_bytes:
        print(f"  rate (on RGB bytes)   = {100*(bits/8)/rgb_bytes:.6f} %")
    print(f"  wall = {time.time()-t0:.0f}s")
    print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
