"""Locate the (prediction row, target byte) pairing the reference code uses.

The reference deletes the last slot of every patch row and then drops 16 rows
from the front, so its scored positions are *not* a contiguous byte range.  This
script brute-forces the question: for the actual logits and the actual payload,
which target offset makes the reference's own scoring produce its reported
2.17 bits/byte?  That tells us what it is really measuring.
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
N = 8160


def main() -> int:
    pred = BGPTBytePredictor.load(
        REPO, os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"), device="cuda:0"
    )
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    data = unit.payload[:N]

    patches, mask_t, n_patches = pred.build_batch(data)
    with torch.no_grad():
        logits = pred.model(patches=patches, masks=mask_t).logits

    # the reference's exact row selection
    ref_rows = logits[:-1, :-1, :].reshape(1, -1, 257).float()
    print("reference-selected rows:", ref_rows.shape[1])
    lp = torch.log_softmax(ref_rows, dim=-1)

    grid = patches.reshape(1, -1)
    best = None
    print(f"{'target_offset':>14} {'bpb':>9}")
    for target_offset in (15, 16, 17, 31, 32, 33):
        tgt = grid[:, target_offset : target_offset + ref_rows.shape[1]]
        if tgt.shape[1] < ref_rows.shape[1]:
            continue
        p = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
        bpb = (-torch.log2(p.clamp_min(1e-30))).mean().item()
        print(f"{target_offset:>14} {bpb:>9.4f}")
        if best is None or bpb < best[1]:
            best = (target_offset, bpb)

    # Which payload indices does the reference's selection actually cover?
    sel = []
    for p in range(logits.shape[0] - 1):
        for j in range(logits.shape[1] - 1):
            sel.append(p * 16 + j + 1)
    sel = torch.tensor(sel)
    print("\nreference covers grid positions:", sel[:20].tolist(), "...", sel[-4:].tolist())
    print("covers a contiguous grid range?", bool((sel[1:] - sel[:-1] == 1).all()))
    skipped = sorted(set(range(16, 16 + N)) - set(sel.tolist()))
    print(f"grid positions in the payload that the reference never scores: {len(skipped)}")
    print("  first 20:", skipped[:20])

    # our covering: contiguous 16 .. 16 + N - 1
    ours = torch.arange(16, 16 + N)
    print(f"\nours covers grid positions 16..{16+N-1} (contiguous, all {N} bytes)")

    # Do the two agree on the bytes they both score?
    common = sorted(set(sel.tolist()) & set(ours.tolist()))
    idx_ref = {g: i for i, g in enumerate(sel.tolist())}
    idx_our = {g: i for i, g in enumerate(ours.tolist())}
    diffs = []
    for g in common[:2000]:
        a = lp[0, idx_ref[g]]
        b = lp[0, idx_our[g]]
        diffs.append(float((a - b).abs().max()))
    print(f"logit distributions identical on common positions? max|diff| = {max(diffs):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
