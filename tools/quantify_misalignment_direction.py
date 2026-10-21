"""Quantify the direction of the reference misalignment, now that the
data-independence hypothesis has been falsified.

The sensitivity test showed the reference metric *does* track the data (2.17 bpb on
real audio vs 8.35 on random bytes), so the number is not a constant artifact.
What remains is to explain why the reference's pairing yields 2.17 bpb while honest
next-byte scoring of the same logits yields 13.12 bpb.

Hypothesis: the model's next-symbol distribution is far more confident about the
byte occupying the *current* slot than about the following byte.  If true, then
pairing a distribution with the current byte instead of the next one inflates
confidence enormously.

This measures both pairings directly, and also the "oracle" for each, so the size
of the effect is separated from the model's overall quality.
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

    # Reference pairing: drop last slot per row, drop last row, target = patches[16:-16]
    ref_rows = logits[:-1, :-1, :].reshape(1, -1, 257).float()
    lp_ref = torch.log_softmax(ref_rows, dim=-1)
    grid = batch["patches"].reshape(1, -1)

    def bpb_for(lp, tgt):
        p = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
        return (-torch.log2(p.clamp_min(1e-30))).mean().item()

    n = len(data)
    # honest: logits[i] predicts payload[i]  (position i holds payload[i-1])
    honest_lp = torch.log_softmax(
        logits.reshape(1, -1, 257).float()[:, 16 : 16 + n, :], dim=-1)
    honest_tgt = grid[:, 16 : 16 + n]

    # "current byte" pairing: logits[i] against payload[i+1] (what the reference
    # selects, since its row offset is one patch behind its target offset)
    cur_lp = torch.log_softmax(
        logits.reshape(1, -1, 257).float()[:, 15 : 15 + n, :], dim=-1)
    cur_tgt = grid[:, 16 : 16 + n]

    print(f"payload = {n} bytes\n")
    print(f"{'pairing':<46} {'bits/byte':>10}")
    print(f"{'honest: distribution i -> payload[i+1]  (next byte)':<46} "
          f"{bpb_for(honest_lp, honest_tgt):>10.4f}")
    print(f"{'reference-like: distribution i -> payload[i]  (cur byte)':<46} "
          f"{bpb_for(cur_lp, cur_tgt):>10.4f}")
    print(f"{'reference code as shipped (its own slice+mask)':<46} "
          f"{bpb_for(lp_ref, grid[:, 16:16+ref_rows.shape[1]]):>10.4f}")

    # oracle: how well does the model's argmax do at each task?
    print("\n=== argmax accuracy (the mechanism) ===")
    for label, lp, tgt in (
        ("next byte (honest)", honest_lp, honest_tgt),
        ("current byte (reference-like)", cur_lp, cur_tgt),
    ):
        acc = (lp.argmax(-1) == tgt).float().mean().item()
        print(f"  {label:<32} {acc*100:6.2f} %")

    print("\n=== mean probability assigned to the true symbol ===")
    for label, lp, tgt in (
        ("next byte (honest)", honest_lp, honest_tgt),
        ("current byte (reference-like)", cur_lp, cur_tgt),
    ):
        p = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp().mean().item()
        print(f"  {label:<32} {p:.4f}")

    # order-1 self-prediction baseline: predict current byte = previous byte
    arr = np.asarray(data, dtype=np.int64)
    self_acc = float((arr[1:] == arr[:-1]).mean())
    print(f"\n  trivial 'byte[i] == byte[i-1]' accuracy: {self_acc*100:.2f} %")
    print("  -> this is the floor the 'current byte' task inherits for free from")
    print("     highly autocorrelated audio, and it is what makes 2.17 bpb look good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
