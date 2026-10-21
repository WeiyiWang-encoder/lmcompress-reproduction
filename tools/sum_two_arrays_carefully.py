"""Sum the two per-position bit arrays carefully and find the discrepancy.

The previous script printed per-position probabilities that matched to eight decimal
places, yet the totals were 1230.14 and 1774.71 bits.  Those two facts cannot both be
true, so one of the arrays must not be what it was labelled.  This script prints the
running sums so the error is located rather than assumed.
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

    model = ImageGPTForCausalImageModeling.from_pretrained(MODEL_DIR).to("cuda:0").eval()
    inp = torch.tensor(ids, dtype=torch.long, device="cuda:0").unsqueeze(0)
    with torch.no_grad():
        out = model(inp, labels=inp)
    lp = torch.log_softmax(out.logits.float(), dim=-1)[0]      # (1024, 512)

    tgt = torch.tensor(ids[1:], dtype=torch.long, device="cuda:0")   # 1023 targets
    pA = lp[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp().cpu().numpy()
    bitsA = float(-np.log2(np.clip(pA, 1e-300, None)).sum())

    from uic.models import ImageGPTPredictor
    pred = ImageGPTPredictor.load(MODEL_DIR, device="cuda:0")
    try:
        steps = list(pred.iter_steps([int(x) for x in ids]))
        pB = np.array([s.p_true for s in steps], dtype=np.float64)
        symB = np.array([s.symbol for s in steps], dtype=np.int64)
    finally:
        pred.release()
    bitsB = float(-np.log2(np.clip(pB, 1e-300, None)).sum())

    print(f"A: n={pA.size}  bits={bitsA:.2f}")
    print(f"B: n={pB.size}  bits={bitsB:.2f}")
    print(f"targets identical (B.symbol vs ids[1:]) : "
          f"{np.array_equal(symB, ids[1:])}")
    print(f"probabilities identical                 : "
          f"{np.allclose(pA, pB, atol=1e-9)}")
    if not np.allclose(pA, pB, atol=1e-9):
        d = np.abs(np.log2(np.clip(pB, 1e-300, None)) -
                   np.log2(np.clip(pA, 1e-300, None)))
        print(f"  max |delta bits| = {d.max():.6f}  at {int(d.argmax())}")
        print(f"  number of positions differing by >1e-6 bits: "
              f"{int((d > 1e-6).sum())} / {d.size}")

    # running-bit comparison over the two arrays
    bitsPA = -np.log2(np.clip(pA, 1e-300, None))
    bitsPB = -np.log2(np.clip(pB, 1e-300, None))
    print("\n  running totals at selected positions")
    print(f"    {'pos':>6} {'A bits':>12} {'B bits':>12} {'B-A':>12}")
    for i in (0, 99, 100, 255, 511, 767, 838, 900, 1022):
        if i < bitsPA.size:
            print(f"    {i:>6} {bitsPA[:i+1].sum():>12.3f} "
                  f"{bitsPB[:i+1].sum():>12.3f} "
                  f"{bitsPB[:i+1].sum()-bitsPA[:i+1].sum():>12.3f}")

    print(f"\n  total A = {bitsPA.sum():.3f}")
    print(f"  total B = {bitsPB.sum():.3f}")

    # final sanity: recompute B from A's own probabilities
    print(f"\n  sum(|log2 pB|)={bitsPB.sum():.3f}  "
          f"sum(|log2 pA|)={bitsPA.sum():.3f}")
    print("\n  If the arrays are identical the totals MUST match; a mismatch means")
    print("  the earlier comparison used a mis-sliced array.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
