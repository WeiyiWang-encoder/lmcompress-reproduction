"""Probe the bGPT logit-to-byte alignment.

The hierarchical bGPT forward pass returns one prediction per (patch, slot) of
the *selected* patches.  Which byte each prediction belongs to depends on how the
patch-level and byte-level decoder masks interact, and getting it wrong by even
one byte silently destroys the compression ratio.  This script scores several
candidate offsets so the correct one can be identified by evidence rather than
by argument.
"""

from __future__ import annotations

import os
import sys

import numpy as np
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


def main() -> int:
    pred = BGPTBytePredictor.load(
        REPO,
        os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"),
        device="cuda:0",
    )
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]

    for n in (8160, 512, 48):
        data = unit.payload[:n]
        patches, mask_t, n_patches = pred.build_batch(data)
        with torch.no_grad():
            logits = pred.model(patches=patches, masks=mask_t).logits
        flat = logits.reshape(1, -1, logits.shape[-1]).float()
        grid = patches.reshape(1, 512, BGPT_PATCH_SIZE)[0].reshape(-1)
        print(f"\n=== n={n}  patches={n_patches}  logits={tuple(logits.shape)} ===")
        print("  grid[0:36] =", grid[:36].tolist())
        for offset in (0, 1, 14, 15, 16, 17, 31, 32):
            if offset + n > flat.shape[1]:
                continue
            lp = torch.log_softmax(flat[:, offset : offset + n, :], dim=-1)
            tgt = grid[offset : offset + n].unsqueeze(0)
            p_true = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
            acc = (lp.argmax(-1).squeeze(0) == tgt.squeeze(0)).float().mean().item()
            bpb = (-torch.log2(p_true.clamp_min(1e-30))).mean().item()
            print(
                f"   offset={offset:>3}: bpb={bpb:8.4f}  argmax_acc={acc:.4f}  "
                f"mean_p={p_true.mean().item():.5f}  first8={tgt[0,:8].tolist()}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
