"""Identify the reference's 8160 prediction rows by search, not by derivation.

Established facts:
  * the reference's targets ARE the payload: ref_ids == grid[16:8176] exactly;
  * the reference's mask covers 100 % of positions, so its denominator is not a
    curated subset;
  * pairing payload[0:8160] against the *first* 8160 flat logit rows gives 13.15 bpb,
    but the reference reports 2.2561.

Therefore its row selection must be something other than the first 8160 rows.  This
searches for the row set that reproduces 2.2561, over several structured candidate
families, and prints the winner so the mechanism is explicit.
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
TARGET_BPB = 2.2561


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
    target = grid[16 : 16 + N]          # confirmed == payload

    # The reference's own row tensor, to confirm the number.
    ref_rows = logits[:-1, :-1, :].reshape(1, -1, 257).float()
    lp_ref = torch.log_softmax(ref_rows, dim=-1)
    p_ref = lp_ref.gather(-1, target.unsqueeze(0).unsqueeze(-1)).squeeze(-1).exp()
    print(f"reference bpb reproduced = {(-torch.log2(p_ref.clamp_min(1e-30))).mean().item():.4f}")
    print(f"reference rows shape = {tuple(ref_rows.shape)}")

    flat = logits.reshape(-1, logits.shape[-1]).float()
    lp = torch.log_softmax(flat, dim=-1)

    def bpb_for_rows(rows):
        n = min(rows.numel(), N)
        pp = rows[:n].gather(-1, target[:n].unsqueeze(-1)).squeeze(-1).exp()
        return (-torch.log2(pp.clamp_min(1e-30))).mean().item()

    print("\n=== structured candidate row sets ===")
    cands = {}

    # (a) same as ref_rows construction, but on the flat view, to check equality
    cands["flat[0:8160]"] = lp[0:N]

    # (b) the reference's row order reconstructs patch-by-patch as (17 slots, drop last)
    idx = []
    P = logits.shape[0]
    for p in range(P - 1):
        for j in range(16):
            idx.append(p * 17 + j)
    cands["patch-major, 16 of 17 slots"] = lp[torch.tensor(idx[:N])]

    # (c) drop the FIRST row of each patch instead of the last
    idx2 = []
    for p in range(P - 1):
        for j in range(1, 17):
            idx2.append(p * 17 + j)
    cands["patch-major, slots 1..16"] = lp[torch.tensor(idx2[:N])]

    # (d) row index = same as target index but from the flat (patch*16) grid
    cands["flat rows at 16+k"] = lp[16 : 16 + N]

    for name, rows in cands.items():
        print(f"  {name:<34} bpb = {bpb_for_rows(rows):.4f}")

    # The reference's own tensor, expressed as the row indices it actually used:
    # ref_rows = logits[:-1,:-1,:].reshape(...) -> row p covers patch p, slots 0..15
    print("\n=== what the reference's reshape does ===")
    print("  logits[:-1, :-1, :] drops the LAST patch row and the LAST slot of each")
    print("  row, so its 8160 rows are: patch p (p=0..509), slot j (j=0..15)")
    print("  i.e. rows are patch-major, and the reference then aligns row k with")
    print("  payload[k].  Row (p, j) is the model's 17-slot prediction vector for")
    print("  patch p at slot j -- whose natural target is patch p slot j+1.")
    print("  Payload[k] for k = 16p + s lives at patch p+1, slot s.")
    print("  => row (p,j) is paired with a byte from patch p+1, i.e. ONE PATCH LATER")
    print("     than the patch whose features produced the prediction.")

    # directly compare: does row (p,j) predict better the byte in patch p+1 or p?
    print("\n=== per-row confidence: predict own patch vs next patch ===")
    for p in (1, 2, 5):
        for j in (0, 7):
            row = lp[p * 17 + j]
            own = int(grid[p * 16 + j + 1])          # byte right after the row's own slot
            nxt = int(grid[(p + 1) * 16 + j + 1])    # same slot, next patch
            po = float(row[own].exp())
            pn = float(row[nxt].exp())
            print(f"  patch {p} slot {j}: p(own next byte)={po:.4f}  "
                  f"p(next patch same slot)={pn:.4f}")

    print(f"\n  target reference bpb = {TARGET_BPB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
