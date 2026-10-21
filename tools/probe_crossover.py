"""Crossover finder: where does the honest bGPT score stop agreeing with the reference?

Scores the same window at increasing prefix lengths with (a) the honest alignment
(contiguous payload coverage) and (b) the reference's row selection, so the point
where they diverge is visible directly.
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
    payload = unit.payload

    print(f"{'n':>6} {'ours bpb':>10} {'ref-ish bpb':>12} {'argmax@+1':>10} {'argmax@0':>10}")
    for n in (512, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8160):
        data = payload[:n]
        patches, mask_t, n_patches = pred.build_batch(data)
        with torch.no_grad():
            logits = pred.model(patches=patches, masks=mask_t).logits
        flat = logits.reshape(-1, logits.shape[-1]).float()
        grid = patches.reshape(-1)
        lp = torch.log_softmax(flat, dim=-1)

        off = 16
        ours = lp[off : off + n]
        t_ours = grid[off : off + n]
        p = ours.gather(-1, t_ours.unsqueeze(-1)).squeeze(-1).exp()
        bpb_ours = (-torch.log2(p.clamp_min(1e-30))).mean().item()
        acc_next = (ours.argmax(-1) == t_ours).float().mean().item()

        # reference-style: drop last slot of each patch row, then 16 rows
        refrows = logits[:-1, :-1, :].reshape(1, -1, logits.shape[-1]).float()
        lpr = torch.log_softmax(refrows, dim=-1)
        m = min(refrows.shape[1], grid.numel() - off)
        t_ref = grid[off : off + m]
        pr = lpr[0, :m].gather(-1, t_ref.unsqueeze(-1)).squeeze(-1).exp()
        bpb_ref = (-torch.log2(pr.clamp_min(1e-30))).mean().item()
        acc_cur = (lpr[0, :m].argmax(-1) == t_ref).float().mean().item()

        print(f"{n:>6} {bpb_ours:>10.4f} {bpb_ref:>12.4f} {acc_next:>10.4f} {acc_cur:>10.4f}")
        del logits, flat, lp, lpr
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
