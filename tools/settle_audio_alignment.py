"""Settle the audio alignment question by measuring every candidate pairing.

A reviewer-facing harness caught that the row/target offset is **15**, not 16, which
contradicts an earlier derivation in this repo.  Both cannot be right, and the
difference decides whether the finding stands at all.  So rather than argue, this
enumerates every plausible row-selection rule against the reference's own targets
and reports the bits/byte each produces.

Facts to be consistent with:
  * the reference reports 2.2561 bpb on this window;
  * its targets are exactly payload[0:8160] (verified element-wise);
  * the model's row at (patch p, slot j) is logits[p, j], with 17 slots per patch.
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

    # Confirm the reference's targets really are the payload.
    ref_ids = batch["patches"][:, PATCH_SIZE:-PATCH_SIZE][0]
    payload = grid[16:16 + N]
    print(f"ref targets == payload[0:{N}]: {bool((ref_ids == payload).all())}")

    flat = logits.reshape(-1, logits.shape[-1]).float()
    lp = torch.log_softmax(flat, dim=-1)

    def bpb(row_idx):
        rows = lp[torch.tensor(row_idx)]
        pp = rows.gather(-1, payload.unsqueeze(-1)).squeeze(-1).exp()
        return float((-torch.log2(pp.clamp_min(1e-30))).mean())

    print(f"\n{'row-selection rule':<52} {'bpb':>9}")
    print("-" * 63)

    # 1. what the reference does: logits[:-1,:-1,:] flattened -> patch-major, 16/17
    idx = [p * 17 + j for p in range(510) for j in range(16)]
    print(f"{'reference: patch-major, slots 0..15':<52} {bpb(idx):>9.4f}")

    # 2. rows shifted so that row k predicts payload[k] under the stated rule
    #    If row (p,j) predicts grid[16p+j+1] == payload[16p+j], then row k=16p+j does it.
    #    Such a row sits at flat index 17p+j, so the flat index is NOT 16+k in general.
    idx2 = []
    for k in range(N):
        p, j = divmod(k, 16)
        idx2.append(p * 17 + j)
    print(f"{'row (p,j) -> payload[16p+j]   [p*17+j]':<52} {bpb(idx2):>9.4f}")

    # 3. the flat offset rule used by an earlier probe in this repo
    print(f"{'flat rows 16+k':<52} {bpb(list(range(16, 16 + N))):>9.4f}")
    print(f"{'flat rows k':<52} {bpb(list(range(N))):>9.4f}")

    # 4. slot 16 instead of slot 0..15
    idx4 = [p * 17 + 16 for p in range(510) for _ in range(1)][:N]
    print(f"{'patch-major, slot 16 only (n grows to 510)':<52} "
          f"{'--':>9}")

    # 5. patch-major, slots 1..16
    idx5 = [p * 17 + j for p in range(509) for j in range(1, 17)][:N]
    print(f"{'patch-major, slots 1..16':<52} {bpb(idx5):>9.4f}")

    print("\n=== the structural statement ===")
    p, j = 3, 7
    k = 16 * p + j
    print(f"  reference pairs row k={k} (patch {p}, slot {j}) with payload[{k}]")
    print(f"  payload[{k}] lives in patch {(16 + k) // 16} at slot {(16 + k) % 16}")
    print(f"  -> the prediction comes from patch {p}, the scored byte from patch "
          f"{(16 + k) // 16}: one entire patch later, same slot.")
    rule = bpb(idx2)
    print(f"\n  best semantically-correct rule gives {rule:.4f} bpb")
    print(f"  reference gives 2.2561 bpb")
    print(f"  => ratio {rule/2.2561:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
