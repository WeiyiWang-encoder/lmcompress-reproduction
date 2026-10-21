"""Where inside a long bGPT window does prediction quality collapse?

If the first patches are predicted well and later ones are not, the fault is in
how the hierarchical forward pass handles a long patch sequence (position or mask
related), not in the per-byte alignment.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

from uic.data import audio_units  # noqa: E402
from uic.models import BGPT_PATCH_SIZE, BGPTBytePredictor  # noqa: E402

REPO = _capsule_root()
N = 4096


def stats(flat, grid, offset, n, label):
    lp = torch.log_softmax(flat[:, offset : offset + n, :], dim=-1)
    tgt = grid[offset : offset + n].unsqueeze(0)
    p_true = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
    bpb = -torch.log2(p_true.clamp_min(1e-30))
    print(f"\n{label}")
    print("  patch_idx | mean bpb | mean_p")
    for p in range(0, n // BGPT_PATCH_SIZE, max(1, (n // BGPT_PATCH_SIZE) // 12)):
        seg = bpb[0, p * BGPT_PATCH_SIZE : (p + 1) * BGPT_PATCH_SIZE]
        print(f"  {p:>9} | {seg.mean().item():>8.4f} | {p_true[0, p*BGPT_PATCH_SIZE:(p+1)*BGPT_PATCH_SIZE].mean().item():.4f}")
    return bpb


def main() -> int:
    pred = BGPTBytePredictor.load(
        REPO,
        os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"),
        device="cuda:0",
    )
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    data = unit.payload[:N]
    patches, mask_t, n_patches = pred.build_batch(data)
    grid = patches.reshape(1, 512, BGPT_PATCH_SIZE)[0].reshape(-1)
    offset = 15  # N is a multiple of 16

    with torch.no_grad():
        logits = pred.model(patches=patches, masks=mask_t).logits
    flat = logits.reshape(1, -1, logits.shape[-1]).float()
    stats(flat, grid, offset, N, "WITH patch attention mask (as coded)")
    del logits, flat
    torch.cuda.empty_cache()

    with torch.no_grad():
        logits2 = pred.model(patches=patches, masks=None).logits
    flat2 = logits2.reshape(1, -1, logits2.shape[-1]).float()
    print("no-mask logits shape:", tuple(logits2.shape))
    stats(flat2, grid, offset, N, "WITHOUT mask (all patches attended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
