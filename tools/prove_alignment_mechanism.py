"""Minimal controlled proof of the reference's alignment off-by-one-patch.

The search established that the reference's 8160 prediction rows are
"patch p, slots 0..15" (patch-major), and that its targets are exactly the payload
``grid[16:8176]``.  This script states the consequence as a single controlled
comparison: take ONE row and evaluate it against two candidate targets that differ
only by which patch they come from.

If the row is markedly more confident about the *next* patch's byte than about its
own, that is the mechanism, and it explains both why the reference's number is
reproducible and why it is not a code length for the payload.
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np
import torch

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "repro_understanding_is_compression", "src"))

from data.model.bgpt.config import *  # noqa: E402,F403
from data.model.bgpt.utils import bGPTLMHeadModel  # noqa: E402
from transformers import GPT2Config  # noqa: E402
import compress_with_bgpt_or_llama as ref  # noqa: E402

WEIGHTS = os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth")
N = 8160


def build_model():
    pc = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, max_length=PATCH_LENGTH,
                    max_position_embeddings=PATCH_LENGTH, hidden_size=HIDDEN_SIZE,
                    n_head=HIDDEN_SIZE // 64, vocab_size=1)
    bc = GPT2Config(num_hidden_layers=BYTE_NUM_LAYERS, max_length=PATCH_SIZE + 1,
                    max_position_embeddings=PATCH_SIZE + 1, hidden_size=HIDDEN_SIZE,
                    n_head=HIDDEN_SIZE // 64, vocab_size=256 + 1)
    m = bGPTLMHeadModel(pc, bc)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu",
                                 weights_only=False)["model"], strict=False)
    return m.to("cuda:0").eval()


def main() -> int:
    from uic.data import audio_units

    model = build_model()
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    data = [int(x) for x in unit.payload[:N]]
    ext = [ord(c) for c in "wav"]

    loader = torch.utils.data.DataLoader(
        ref.split(data, N), batch_size=1, shuffle=False,
        collate_fn=functools.partial(ref.byte_collate, appendix={"ext": ext},
                                     device="cuda:0"),
    )
    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(**batch).logits
    grid = batch["patches"].reshape(1, -1)[0]
    flat = logits.reshape(-1, logits.shape[-1]).float()

    print("Setup: row (p, j) = logits[p, j].  Its natural target is grid[16p + j + 1],")
    print("which for j < 15 is the byte in the SAME patch (p).  The reference pairs")
    print("row k with payload[k] = grid[16 + k]; for k = 16p + j that is grid[16p+j+16],")
    print("i.e. slot j+1 of the byte 16 positions later -- a different patch.\n")

    print(f"{'row (p,j)':>12} {'target used by ref':>20} {'p(ref target)':>14} "
          f"{'p(own patch next byte)':>24}")

    # Compare, over many rows, the probability of
    #   (i)  the byte the reference actually pairs the row with
    #   (ii) the byte the row would naturally predict (its own patch, next slot)
    ref_logp, own_logp = [], []
    for p in range(0, 400, 7):
        for j in range(0, 8, 3):
            k = p * 16 + j
            row_idx = p * 17 + j
            if row_idx >= flat.shape[0]:
                continue
            row = torch.log_softmax(flat[row_idx], dim=-1)
            ref_t = int(grid[16 + k]) if (16 + k) < grid.numel() else None
            own_t = int(grid[p * 16 + j + 1])
            if ref_t is None:
                continue
            pr = float(row[ref_t].exp())
            po = float(row[own_t].exp())
            ref_logp.append(np.log2(max(pr, 1e-30)))
            own_logp.append(np.log2(max(po, 1e-30)))
            if len(ref_logp) <= 10:
                print(f"  p={p:<4} j={j:<3} {ref_t:>12} {pr:>20.4f} {po:>24.4f}")

    print(f"\nmean bits for the REFERENCE's pairing      : {-np.mean(ref_logp):.4f} bpb")
    print(f"mean bits for the row's OWN natural target : {-np.mean(own_logp):.4f} bpb")
    print(f"advantage of the reference pairing         : "
          f"{-np.mean(ref_logp) / -np.mean(own_logp):.2f}x fewer bits")

    # Whole-stream versions, to tie back to the headline numbers.
    lp = torch.log_softmax(flat, dim=-1)

    def stream_bpb(row_idx, tgt_idx):
        n = min(len(row_idx), len(tgt_idx))
        rows = lp[torch.tensor(row_idx[:n])]
        tgt = grid[torch.tensor(tgt_idx[:n])]
        pp = rows.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
        return (-torch.log2(pp.clamp_min(1e-30))).mean().item()

    ref_rows = list(range(0, 510 * 17, 17))          # placeholder, replaced below
    # reference row set: patch p, slots 0..15
    r_idx, t_idx = [], []
    for p in range(0, 510):
        for j in range(16):
            k = len(r_idx)
            r_idx.append(p * 17 + j)
            t_idx.append(16 + k)
    print(f"\nwhole-window (n={len(r_idx)}):")
    print(f"  reference rows vs reference targets : {stream_bpb(r_idx, t_idx):.4f} bpb")
    # honest: prediction for payload[k] lives at row (patch of payload[k] minus 1)
    h_idx, h_t = [], []
    for k in range(16, 16 + N):
        patch = (k - 1) // 16
        j = (k - 1) % 16
        h_idx.append(patch * 17 + j)
        h_t.append(k)
    print(f"  honest rows     vs the same targets : {stream_bpb(h_idx, h_t):.4f} bpb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
