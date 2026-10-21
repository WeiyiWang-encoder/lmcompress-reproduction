"""Same token ids, two model invocations: why do they disagree?

Both paths now receive the identical 1024-token sequence from the processor.  The
official-style call (``ImageGPTForCausalImageModeling`` with ``labels=ids``) gives
mean CE 1.2025, i.e. 1231 bits for the window.  Our port's ``iter_steps`` reported
1774.71 bits on the same ids -- 1.44x more.

That gap cannot come from aggregation.  It must come from the model invocation, so
this script runs both on the same tensor and compares the logits directly.
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
    from transformers import ImageGPTForCausalImageModeling

    from uic.data import _igpt_processor

    path = os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0])
    rgb = np.asarray(Image.open(path).convert("RGB"))
    n = PATCH * PATCH
    block = rgb[:, :, 0].reshape(-1)[:n].reshape(PATCH, PATCH)
    triple = np.repeat(block[:, :, None], 3, axis=2)
    ids = np.asarray(_igpt_processor(MODEL_DIR)(triple, return_tensors="np")
                     ["input_ids"]).reshape(-1)
    print(f"ids: n={ids.size}")

    # ---- invocation A: exactly what the official code does ---------------- #
    modelA = ImageGPTForCausalImageModeling.from_pretrained(MODEL_DIR).to("cuda:0").eval()
    inp = torch.tensor(ids, dtype=torch.long, device="cuda:0").unsqueeze(0)
    with torch.no_grad():
        outA = modelA(inp, labels=inp)
    lpA = torch.log_softmax(outA.logits.float(), dim=-1)[0]
    tgt = torch.tensor(ids[1:], dtype=torch.long, device="cuda:0")
    bitsA = float(-lpA[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())
    print(f"\nA official-style  : loss={float(outA.loss):.6f}  bits={bitsA:.2f}  "
          f"bpb={bitsA/1023:.4f}")
    print(f"  logits shape {tuple(outA.logits.shape)}  dtype {outA.logits.dtype}")

    # ---- invocation B: our port's predictor ------------------------------- #
    from uic.models import ImageGPTPredictor
    pred = ImageGPTPredictor.load(MODEL_DIR, device="cuda:0")
    try:
        steps = list(pred.iter_steps([int(x) for x in ids]))
        pB = np.array([s.p_true for s in steps], dtype=np.float64)
        sB = np.array([s.symbol for s in steps], dtype=np.int64)
    finally:
        pred.release()
    bitsB = float(-np.log2(np.clip(pB, 1e-300, None)).sum())
    print(f"\nB our port        : bits={bitsB:.2f}  bpb={bitsB/len(pB):.4f}  "
          f"n={len(pB)}")
    print(f"  symbols scored  : first 6 = {sB[:6].tolist()}")
    print(f"  true tokens     : first 6 = {ids[1:7].tolist()}")
    print(f"  symbols == ids[1:]? {np.array_equal(sB, ids[1:])}")
    print(f"  ratio B/A       = {bitsB/bitsA:.4f}")

    print(f"\n=== is B scoring the same targets as A? ===")
    print(f"  B p[:6] = {np.round(pB[:6],6).tolist()}")
    keep = lpA[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp().cpu().numpy()
    print(f"  A p[:6] = {np.round(keep[:6],6).tolist()}")

    # per-position divergence
    d = np.abs(np.log2(np.clip(pB, 1e-300, None)) -
               np.log2(np.clip(keep, 1e-300, None)))
    print(f"\n  max |delta bits| = {d.max():.4f} at position {int(d.argmax())}")
    print(f"  mean |delta bits| = {d.mean():.6f}")
    worst = np.argsort(d)[::-1][:5]
    print("  worst positions:")
    for i in worst:
        print(f"    pos {i:4d}: B={pB[i]:.6f}  A={keep[i]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
