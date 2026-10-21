"""Audit the reference bGPT audio path.

Question: the reference reports ~4.13x (1.94 bits/byte) on the bundled audio
file.  The clean-room port, scoring every payload byte against the model's own
distribution, gets ~13 bits/byte.  Which one is measuring the model?

Plan:
  1. Establish ground truth: the empirical order-0/order-1 entropy of the payload.
     No model can beat the true entropy of the source, so a cross-entropy far
     below it is proof of a measurement artefact rather than compression.
  2. Find the (prediction row, target byte) pairing the reference actually uses.
  3. Check whether the reference's scored target set is even a subset of the
     payload.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

import numpy as np
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
N = 8160


def order1_entropy(data) -> float:
    pairs = Counter(zip(data[:-1], data[1:]))
    singles = Counter(data[:-1])
    total = sum(pairs.values())
    h = 0.0
    for (a, b), c in pairs.items():
        h -= (c / total) * math.log2(c / singles[a])
    return h


def order0_entropy(data) -> float:
    counts = Counter(data)
    total = len(data)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def main() -> int:
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    payload = unit.payload
    print(f"file = {unit.name}")
    print(f"meta = { {k: v for k, v in unit.meta.items() if k != 'residual'} }")
    print(f"payload = {len(payload)} bytes")
    print(f"order-0 entropy = {order0_entropy(payload):.4f} bits/byte")
    print(f"order-1 entropy = {order1_entropy(payload):.4f} bits/byte")
    print(f"reference claims  = {8 / (82264 / 19924.81):.4f} bits/byte")
    print(
        "\n  -> a cross-entropy BELOW the order-1 entropy is impossible for a "
        "correctly scored code;\n     it means the reference is not scoring the "
        "bytes it thinks it is.\n"
    )

    pred = BGPTBytePredictor.load(
        REPO, os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"), device="cuda:0"
    )
    data = payload[:N]
    patches, mask_t, n_patches = pred.build_batch(data)
    with torch.no_grad():
        logits = pred.model(patches=patches, masks=mask_t).logits

    grid = patches.reshape(1, -1)[0]
    lp = torch.log_softmax(logits.reshape(1, -1, 257).float(), dim=-1)[0]

    # (a) our alignment: contiguous payload coverage, offset 16
    ours = lp[16 : 16 + N]
    t = grid[16 : 16 + N]
    p_ours = ours.gather(-1, t.unsqueeze(-1)).squeeze(-1).exp()
    bpb_ours = (-torch.log2(p_ours.clamp_min(1e-30))).mean().item()
    print(f"[ours]  contiguous rows 16..{16+N-1}, targets 16..{16+N-1}: bpb = {bpb_ours:.4f}")

    # (b) the reference's row selection, scored against the byte actually stored
    #     at the same grid position (this is what the reference's index arithmetic
    #     amounts to: rows and targets share an index, but the rows are shifted)
    ref_rows = logits[:-1, :-1, :].reshape(1, -1, 257).float()
    lp_ref = torch.log_softmax(ref_rows, dim=-1)
    for off in (0, 1, 15, 16):
        tgt = grid[off : off + lp_ref.shape[1]]
        if tgt.shape[0] < lp_ref.shape[1]:
            continue
        p = lp_ref.gather(-1, tgt.unsqueeze(0).unsqueeze(-1)).squeeze(-1).squeeze(0).exp()
        print(
            f"[ref ]  rows [:-1,:-1], targets offset {off:>2}: "
            f"bpb = {(-torch.log2(p.clamp_min(1e-30))).mean().item():.4f}"
        )

    # (c) what byte values does the reference end up scoring?
    ref_targets = grid[16 : 16 + lp_ref.shape[1]]
    if ref_targets.shape[0] == lp_ref.shape[1]:
        vals = ref_targets[:8160]
        print(
            f"\n[ref ]  scored target values: first12={vals[:12].tolist()} "
            f"mean={float(vals.float().mean()):.2f} "
            f"unique={int(torch.unique(vals).numel())}"
        )
        print(
            f"[ours]  scored target values: first12={t[:12].tolist()} "
            f"mean={float(t.float().mean()):.2f} "
            f"unique={int(torch.unique(t).numel())}"
        )
        print(f"        payload mean = {float(np.mean(data)):.2f}, unique = {len(set(data))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
