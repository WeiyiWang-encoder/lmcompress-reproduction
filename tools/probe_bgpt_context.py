"""Map bGPT compression quality against context length.

Motivation: with a correct patch/byte alignment the model should get *better*
with more context, not catastrophically worse.  This script scans window sizes
and reports bits-per-byte for each, which localises the regime where the
hierarchical forward pass stops behaving.
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


def offset_for(n_data: int) -> int:
    return 15 if (n_data and n_data % BGPT_PATCH_SIZE == 0) else 16


def main() -> int:
    pred = BGPTBytePredictor.load(
        REPO,
        os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"),
        device="cuda:0",
    )
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    payload = unit.payload
    print(f"payload = {len(payload)} bytes")

    print(f"{'n':>7} {'patches':>8} {'bpb':>9} {'argmax_acc':>11} {'mean_p':>8}")
    for n in (256, 512, 1024, 2048, 4096, 6144, 8160):
        for start in ((0,) if n > 2048 else (0, 8192)):
            data = payload[start : start + n]
            if len(data) < n:
                continue
            patches, mask_t, n_patches = pred.build_batch(data)
            with torch.no_grad():
                logits = pred.model(patches=patches, masks=mask_t).logits
            flat = logits.reshape(1, -1, logits.shape[-1]).float()
            grid = patches.reshape(1, 512, BGPT_PATCH_SIZE)[0].reshape(-1)
            off = offset_for(n)
            lp = torch.log_softmax(flat[:, off : off + n, :], dim=-1)
            tgt = grid[off : off + n].unsqueeze(0)
            p_true = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
            acc = (lp.argmax(-1).squeeze(0) == tgt.squeeze(0)).float().mean().item()
            bpb = (-torch.log2(p_true.clamp_min(1e-30))).mean().item()
            print(
                f"{n:>7} {n_patches:>8} {bpb:>9.4f} {acc:>11.4f} {p_true.mean().item():>8.4f}"
                f"   (start={start}, offset={off})"
            )
            del logits, flat, lp
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
