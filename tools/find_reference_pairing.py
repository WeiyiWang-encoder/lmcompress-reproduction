"""Brute-force: which (prediction row, target byte) pairing yields 2.2561 bits/byte?

Two hypotheses have already failed:
  * honest next-byte scoring          -> 13.12 bpb
  * "distribution i vs payload[i]"    -> 13.08 bpb
yet the reference code as shipped yields 2.2561 bpb on the same logits.

So the reference's slice+mask selects something else.  This enumerates candidate
pairings -- every (row shift, target shift) within a patch -- and reports which one
reproduces 2.2561, printing enough structure to explain *why*.
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
        logits = model(**batch).logits          # (P, 17, 257)
    grid = batch["patches"].reshape(1, -1)[0]   # (8192,)
    print(f"logits {tuple(logits.shape)}   grid {tuple(grid.shape)}")

    # ---- reproduce the reference's own high-level slices ------------------- #
    ref_rows = logits[:-1, :-1, :].reshape(1, -1, 257).float()
    ref_ids = batch["patches"][:, PATCH_SIZE:-PATCH_SIZE]
    ref_mask = batch["masks"].repeat_interleave(PATCH_SIZE, dim=1)[:, PATCH_SIZE:-PATCH_SIZE]
    print(f"reference rows {tuple(ref_rows.shape)}  ids {tuple(ref_ids.shape)} "
          f" mask {tuple(ref_mask.shape)}  mask.sum {int(ref_mask.sum())}")

    lp_ref = torch.log_softmax(ref_rows, dim=-1)
    p = lp_ref.gather(-1, ref_ids.unsqueeze(-1)).squeeze(-1).exp()
    nll = -torch.log2(p.clamp_min(1e-30)) * ref_mask
    print(f"reference bpb = {nll.sum().item()/ref_mask.sum().item():.4f}")

    # ---- what is ref_ids actually equal to? -------------------------------- #
    ids0 = ref_ids[0]
    print(f"\nref_ids[:12]        = {ids0[:12].tolist()}")
    print(f"grid[16:28]         = {grid[16:28].tolist()}")
    print(f"grid[1:13]          = {grid[1:13].tolist()}")
    print(f"payload[:12]        = {data[:12]}")
    for off in (0, 1, 15, 16, 17):
        seg = grid[off : off + ids0.numel()]
        same = int((seg == ids0).sum())
        print(f"  ref_ids vs grid[{off}:{off+ids0.numel()}]  identical={same}/{ids0.numel()}")

    # ---- brute force (row shift, target shift) ----------------------------- #
    flat = logits.reshape(-1, logits.shape[-1]).float()
    lp = torch.log_softmax(flat, dim=-1)
    nrows = flat.shape[0]
    print(f"\nflat rows = {nrows}")

    print(f"\n{'row0':>6} {'tgt0':>6} {'n':>6} {'bpb':>9}  note")
    best = None
    for row0 in range(0, 33):
        for tgt0 in range(0, 33):
            n = min(nrows - row0, grid.numel() - tgt0)
            if n < 1000:
                continue
            tgt = grid[tgt0 : tgt0 + n]
            pp = lp[row0 : row0 + n].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
            bpb = (-torch.log2(pp.clamp_min(1e-30))).mean().item()
            if best is None or bpb < best[2]:
                best = (row0, tgt0, bpb, n)
    print(f"  best over small shifts: row0={best[0]} tgt0={best[1]} "
          f"bpb={best[2]:.4f} n={best[3]}")

    for row0, tgt0 in ((0, 15), (0, 16), (16, 16), (1, 15), (1, 16), (0, 1), (15, 16)):
        n = min(nrows - row0, grid.numel() - tgt0)
        tgt = grid[tgt0 : tgt0 + n]
        pp = lp[row0 : row0 + n].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
        bpb = (-torch.log2(pp.clamp_min(1e-30))).mean().item()
        print(f"{row0:>6} {tgt0:>6} {n:>6} {bpb:>9.4f}")

    # ---- is the reference's metric dominated by the mask? ------------------ #
    print("\n=== is it the mask or the pairing? ===")
    p_all = lp_ref.gather(-1, ref_ids.unsqueeze(-1)).squeeze(-1).exp()
    bpb_masked = (-torch.log2(p_all.clamp_min(1e-30)) * ref_mask).sum().item() / ref_mask.sum().item()
    bpb_unmasked = (-torch.log2(p_all.clamp_min(1e-30))).mean().item()
    print(f"  masked   bpb = {bpb_masked:.4f}   (this is what the reference reports)")
    print(f"  unmasked bpb = {bpb_unmasked:.4f}")
    print(f"  mask selects {int(ref_mask.sum())} of {ref_mask.numel()} positions "
          f"({100*ref_mask.sum().item()/ref_mask.numel():.2f} %)")
    print("  -> if these differ a lot, the reference's *denominator* is a subset,")
    print("     so its bpb is the mean over a curated subset of positions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
