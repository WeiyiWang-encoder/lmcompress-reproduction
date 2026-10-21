"""Compare the official and port probability models on one identical token sequence.

Tokenisation is settled: the port and the official processor produce identical token
ids for the same chunk (verified).  The remaining 1.54x discrepancy in bits per
sample must therefore come from the probability each implementation assigns to the
true token.

This feeds BOTH the same 1024-token sequence and compares, position by position:

  * the official ``get_pdf`` sequence from ``language_model._retrieve_predict_fn``
  * our ``ImageGPTPredictor.iter_steps``

If the per-position probabilities agree, the difference is in the aggregation.  If
they disagree, the model invocation differs -- context length, padding, or dtype.
"""

from __future__ import annotations

import os
import sys

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
sys.path.insert(0, REPO)

MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMG_DIR = os.path.join(REPO, "data", "test_images")
PATCH = 32


def main() -> int:
    from PIL import Image
    from uic.data import _igpt_processor
    from uic.models import ImageGPTPredictor

    path = os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0])
    rgb = np.asarray(Image.open(path).convert("RGB"))
    n = PATCH * PATCH
    block = rgb[:, :, 0].reshape(-1)[:n].reshape(PATCH, PATCH)
    triple = np.repeat(block[:, :, None], 3, axis=2)
    proc = _igpt_processor(MODEL_DIR)
    ids = np.asarray(proc(triple, return_tensors="np")["input_ids"]).reshape(-1)
    print(f"sequence: {ids.size} tokens, first 12 = {ids[:12].tolist()}")

    # ---- our port ---------------------------------------------------------- #
    pred = ImageGPTPredictor.load(MODEL_DIR, device="cuda:0")
    try:
        steps = list(pred.iter_steps([int(x) for x in ids]))
        ours_p = np.array([s.p_true for s in steps], dtype=np.float64)
        ours_sym = np.array([s.symbol for s in steps], dtype=np.int64)
    finally:
        pred.release()
    ours_bits = float(-np.log2(np.clip(ours_p, 1e-300, None)).sum())
    print(f"\nour port   : n={len(ours_p)}  bits={ours_bits:.2f}  "
          f"bpb={ours_bits/len(ours_p):.4f}")
    print(f"  symbols[0:6]={ours_sym[:6].tolist()}  p[0:6]="
          f"{np.round(ours_p[:6],4).tolist()}")

    # ---- official ---------------------------------------------------------- #
    import language_model as lm
    params, img_proc, tok = lm._retrieve_model_params("igpt", MODEL_DIR)
    predict_fn = lm._retrieve_predict_fn(params, img_proc, tok, is_lvm=False)

    seq = np.asarray(ids).reshape(1, -1)
    print(f"\nofficial predict_fn on the same ids, shape {seq.shape}")
    try:
        probs = predict_fn(seq)
        probs = np.asarray(probs)
        print(f"  returned shape {probs.shape}, dtype {probs.dtype}")
    except Exception as exc:
        print(f"  direct call failed: {type(exc).__name__}: {exc}")
        return 1

    # interpret: get_pdf typically returns the probability of the true token
    flat = probs.reshape(-1)
    if flat.size == seq.size:
        print(f"  -> per-token probabilities, n={flat.size}")
        off_bits = float(-np.log2(np.clip(flat, 1e-300, None)).sum())
        print(f"  official  : n={flat.size}  bits={off_bits:.2f}  "
              f"bpb={off_bits/flat.size:.4f}")
        print(f"  p[0:6]={np.round(flat[:6],4).tolist()}")
        print(f"\n  ratio official/port = {off_bits/ours_bits:.4f}")
    elif flat.size == seq.size - 1:
        print(f"  -> n-1 probabilities (last token unscored), n={flat.size}")
        off_bits = float(-np.log2(np.clip(flat, 1e-300, None)).sum())
        print(f"  official  : bits={off_bits:.2f}  bpb={off_bits/flat.size:.4f}")
    else:
        print(f"  -> unexpected size {flat.size}; raw head: {flat[:8]}")

    print("\n=== per-position comparison (first 10) ===")
    k = min(10, len(ours_p), flat.size)
    for i in range(k):
        print(f"  pos {i:3d}: port p={ours_p[i]:.6f}   official p={flat[i]:.6f}"
              f"   ratio={flat[i]/ours_p[i]:.4f}" if ours_p[i] else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
