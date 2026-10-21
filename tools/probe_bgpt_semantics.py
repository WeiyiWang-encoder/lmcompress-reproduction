"""Does the released bGPT audio checkpoint predict the NEXT byte or the CURRENT one?

If the model's distribution at a slot matches the byte stored *at that slot*
better than the byte stored *after* it, then the checkpoint is an autoencoder-like
"reproduce the input" model, and any code that scores it against the following
byte -- or worse, that pairs slot distributions with the wrong targets -- is
measuring something other than next-byte prediction.

This matters because the reference implementation's index arithmetic produces
exactly such a pairing, and reports 4.1x on data whose order-1 entropy is only
3.17 bits/byte (i.e. a maximum honest ratio of ~2.5x).
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
from uic.models import BGPTBytePredictor  # noqa: E402

REPO = _capsule_root()


def main() -> int:
    pred = BGPTBytePredictor.load(
        REPO, os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"), device="cuda:0"
    )
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]

    for n in (512, 8160):
        data = unit.payload[:n]
        patches, mask_t, n_patches = pred.build_batch(data)
        with torch.no_grad():
            logits = pred.model(patches=patches, masks=mask_t).logits
        flat = logits.reshape(-1, logits.shape[-1]).float()
        grid = patches.reshape(-1)
        lp = torch.log_softmax(flat, dim=-1)
        arg = lp.argmax(-1)

        print(f"\n=== n={n}: {flat.shape[0]} prediction rows ===")
        print(f"{'row->grid shift':>16} {'argmax_match':>13} {'meandef  bpb':>14}")
        for shift in (-2, -1, 0, 1, 2, 15, 16, 17):
            lo = max(0, -shift)
            hi = min(flat.shape[0], flat.shape[0] - shift)
            tgt = grid[lo + shift : hi + shift]
            if tgt.numel() == 0:
                continue
            rows = lp[lo:hi]
            acc = (arg[lo:hi] == tgt).float().mean().item()
            p = rows.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
            bpb = (-torch.log2(p.clamp_min(1e-30))).mean().item()
            tag = ""
            if shift == 0:
                tag = "  <- distribution at slot vs byte AT that slot"
            if shift == 1:
                tag = "  <- vs the NEXT byte"
            print(f"{shift:>16} {acc:>13.4f} {bpb:>14.4f}{tag}")

        # Where is the model's peak probability concentrated?
        top = arg[:64]
        print("  argmax[:32]      =", top[:32].tolist())
        print("  grid[0:32]       =", grid[:32].tolist())
        print("  grid[1:33]       =", grid[1:33].tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
