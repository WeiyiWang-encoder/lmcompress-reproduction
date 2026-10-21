"""Measure our port's bpb on exactly ONE 8 160-byte context window.

This exists so our implementation can be compared with the authors' unmodified script on
identical input.  The reference, run as shipped, can only complete a full block (a
short tail block makes it raise), so a single window is the largest directly comparable
input.

Our port's own scoring uses the row mapping established in ``uic.models``
(``bgpt_prediction_indices``): the reference drops slot 16 of every 17-slot patch row,
so payload position ``k = 16p + j`` uses flat row ``17p + j`` and target grid position
``16 + k``.  An earlier probe here used row ``16 + k``, which is a different mapping and
produced nonsense; that error is documented in docs/REVIEW_GUIDE_FOR_AGENT.md §5.2.
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
WINDOW = 8160


def main() -> int:
    from uic.data import audio_units
    from uic.models import BGPTBytePredictor, bgpt_prediction_indices

    pred = BGPTBytePredictor.load(
        REPO, os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"),
        device="cuda:0")
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    data = unit.payload[:WINDOW]
    print(f"payload window = {len(data)} bytes (LJ001-0004, first context window)")

    rows = bgpt_prediction_indices(len(data))
    patches, masks, n_patches = pred.build_batch(data)
    with torch.no_grad():
        logits = pred.model(patches=patches, masks=masks).logits

    # The model pads the window by PATCH_SIZE at each end, so the scored targets live on
    # the flattened grid at [PATCH_SIZE : PATCH_SIZE + n].
    grid = patches.reshape(-1)
    lp = torch.log_softmax(logits.reshape(-1, logits.shape[-1]).float(), dim=-1)
    n = len(data)
    row_idx = torch.tensor(rows[:n])
    tgt = grid[16 : 16 + n]
    assert int(tgt[0]) == data[0] and int(tgt[-1]) == data[-1], "target misaligned"
    p = lp[row_idx].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
    bits = float(-torch.log2(p.clamp_min(1e-30)).sum())
    bpb = bits / n

    print(f"  n_patches      = {n_patches}")
    print(f"  scored rows    = {len(rows)} (first {rows[0]}, last {rows[-1]})")
    print(f"  coded bits     = {bits:.2f}")
    print(f"  bits per byte  = {bpb:.4f}")
    print(f"  compression ratio = {8.0/bpb:.6f}")
    print()
    print("  reference script on the same window: ratio 3.666938 "
          "= 2.1941 bits/byte")
    print(f"  our port                            : ratio {8.0/bpb:.6f} "
          f"= {bpb:.4f} bits/byte")
    print(f"  difference                          : "
          f"{100*(bpb-2.1941)/2.1941:+.2f} %")
    pred.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
