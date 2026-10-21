"""Determine what byte row (p, j) actually predicts -- empirically, not by derivation.

The alignment question hinges on one fact: does the flat logit row (patch p, slot j)
carry the distribution for the byte immediately after its own slot, or for some other
byte?  Earlier derivations in this repo assumed one thing and produced 14.1804 bpb
where the reference produces 2.2561; another assumption produced exactly 2.2561.  So
derive nothing and measure instead:

for each candidate lag L, ask how much probability the row assigns to the byte at
grid[16p + j + L].  The row's true target is the L that maximises it.
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
                                     device="cuda:0"))
    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(**batch).logits
    grid = batch["patches"].reshape(1, -1)[0]
    flat = logits.reshape(-1, logits.shape[-1]).float()
    lp = torch.log_softmax(flat, dim=-1)

    print("Question: for row (p, j), which grid offset L does it predict?")
    print("Measure p_ratio(L) = prob assigned to grid[16p+j+L], averaged over rows.\n")

    # sample many rows across the window
    rows, a_s, b_s = [], [], []
    for p in range(0, 500, 13):
        for j in range(0, 16, 3):
            ridx = p * 17 + j
            if ridx < flat.shape[0]:
                rows.append(ridx)
                a_s.append(p)
                b_s.append(j)
    rows = torch.tensor(rows)
    a_s = np.asarray(a_s)
    b_s = np.asarray(b_s)
    sub = lp[rows]

    print(f"{'L':>4} {'mean prob of grid[16p+j+L]':>28} {'argmax hit rate':>17}")
    best = None
    for L in range(-2, 20):
        tgt = 16 * a_s + b_s + L
        keep = (tgt >= 0) & (tgt < grid.numel())
        if keep.sum() < 10:
            continue
        t = grid[torch.tensor(tgt[keep])]
        s = sub[torch.tensor(np.nonzero(keep)[0])]
        pr = s.gather(-1, t.unsqueeze(-1)).squeeze(-1).exp().mean().item()
        acc = (s.argmax(-1) == t).float().mean().item()
        star = ""
        if best is None or pr > best[1]:
            best = (L, pr)
            star = ""
        print(f"{L:>4} {pr:>28.4f} {acc*100:>16.2f} %{star}")

    print(f"\n=> row (p, j)'s target is at offset L = {best[0]} "
          f"(highest mean probability {best[1]:.4f})")

    # Now: what does the reference pair row k with?
    print("\nReference pairing, expressed the same way:")
    print("  row k = 16p + j  (p = k//16, j = k%16)")
    print("  -> its flat index is NOT 16p+j; check which (patch,slot) that flat")
    print("     index corresponds to, since one patch row has 17 slots.")
    for k in (0, 1, 17, 33, 16):
        p, j = divmod(k, 16)
        flat_index_pmj = p * 17 + j
        a, b = divmod(flat_index_pmj, 17)
        tgt = 16 + k
        print(f"    k={k:>4}: payload[{k}] is grid[{tgt}]; "
              f"(p,j)=({p},{j}) -> flat {flat_index_pmj} = (patch {a}, slot {b})")

    # And reproduce the two candidate numbers cleanly.
    print("\n=== the two candidate bpb values, side by side ===")
    payload = grid[16:16 + N]

    def bpb(row_idx):
        s = lp[torch.tensor(row_idx)]
        pp = s.gather(-1, payload.unsqueeze(-1)).squeeze(-1).exp()
        return float((-torch.log2(pp.clamp_min(1e-30))).mean())

    ref_rows = [p * 17 + j for p in range(510) for j in range(16)]
    print(f"  reference rows (patch-major 0..15)          : {bpb(ref_rows):.4f}")
    shifted = [p * 17 + j + 1 for p in range(510) for j in range(16)]
    keep = [i for i in shifted if 0 <= i < flat.shape[0]][:N]
    print(f"  reference rows shifted by +1                : {bpb(keep):.4f}")
    print(f"  flat rows 16..8175 (what an earlier probe)  : "
          f"{bpb(list(range(16, 16 + N))):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
