"""Instrument the live bGPT code path to find where p(true) collapses."""

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
from uic.models import BGPTBytePredictor, bgpt_prediction_indices  # noqa: E402

REPO = _capsule_root()


def main() -> int:
    pred = BGPTBytePredictor.load(
        REPO, os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"), device="cuda:0"
    )
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    data = unit.payload[:8160]

    row_indices = bgpt_prediction_indices(len(data))
    patches, mask_t, n_patches = pred.build_batch(data)
    with torch.no_grad():
        logits = pred.model(patches=patches, masks=mask_t).logits
    print("logits", tuple(logits.shape), "n_patches", n_patches,
          "first/last selected row", row_indices[0], row_indices[-1])

    flat = logits.reshape(1, -1, logits.shape[-1])
    print("flat", tuple(flat.shape))
    rows = flat.index_select(1, torch.tensor(row_indices, device=flat.device))
    grid = patches.reshape(1, 512, 16).reshape(1, -1)
    tgt = grid[:, 16 : 16 + len(data)]
    print("rows", tuple(rows.shape), "tgt", tuple(tgt.shape))
    print("data[0:8]      ", data[:8])
    print("tgt[0,0:8]     ", tgt[0, :8].tolist())

    probs = torch.softmax(rows.float(), dim=-1)
    g = probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    print("p_true[0,:8]   ", [round(x, 6) for x in g[0, :8].tolist()])
    print("argmax[0,:8]   ", probs.argmax(-1)[0, :8].tolist())
    print("mean p_true    ", g.mean().item())
    print("mean bpb       ", (-torch.log2(g.clamp_min(1e-30))).mean().item())

    # cross-check with an independent log_softmax computation
    lp = torch.log_softmax(rows.float(), dim=-1)
    g2 = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
    print("independent mean p_true", g2.mean().item(), "bpb", (-torch.log2(g2.clamp_min(1e-30))).mean().item())

    # and through the real generator
    n_seen = 0
    total = 0.0
    import math

    for step in pred.iter_steps(data):
        total += -math.log2(max(step.p_true, 1e-30))
        n_seen += 1
        if n_seen <= 8:
            print(f"   step {n_seen-1}: symbol={step.symbol} p_true={step.p_true:.6f} probs_argmax={int(step.probs.argmax())}")
    print(f"generator: {n_seen} steps, bpb={total/n_seen:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
