"""Does the released bGPT-audio checkpoint predict causally (past -> next byte)?

My misalignment claim is dead: the reference pairs row (p,j) with the byte at
grid[16p+j+16], and measurement shows that IS the byte the row is most confident about
(mean prob 0.3052 at L=16, versus 0.0804 at L=1).  So the reference is internally
coherent.

What remains is a different and narrower question, and it is the one that matters for
whether a *lossless* code is possible: at encode time, when the decoder reaches byte
k, the only information available is bytes < k.  If the model's confidence at a row
depends on bytes at or after its target, the scheme is not a causal code.

This tests that directly: recompute the model's probability for one target byte under
two inputs that differ ONLY in bytes strictly after that target.  A causal model must
give an identical result.
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


def logits_for(model, data, ext):
    loader = torch.utils.data.DataLoader(
        ref.split(data, N), batch_size=1, shuffle=False,
        collate_fn=functools.partial(ref.byte_collate, appendix={"ext": ext},
                                     device="cuda:0"))
    batch = next(iter(loader))
    with torch.no_grad():
        lg = model(**batch).logits
    return lg, batch["patches"].reshape(1, -1)[0]


def main() -> int:
    from uic.data import audio_units

    model = build_model()
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    base = [int(x) for x in unit.payload[:N]]
    ext = [ord(c) for c in "wav"]

    lg0, grid0 = logits_for(model, base, ext)
    flat0 = lg0.reshape(-1, lg0.shape[-1]).float()

    # A target well inside the window, and a perturbation strictly AFTER it.
    K = 4000                      # payload index of the target
    p, j = divmod(K, 16)
    row = p * 17 + j              # the row the reference uses for payload[K]
    target = int(grid0[16 + K])
    print(f"target = payload[{K}] = {target}, scored via row {row} (patch {p}, slot {j})")
    print(f"perturbation will touch payload indices > {K + 200} only\n")

    perturbed = list(base)
    rng = np.random.default_rng(0)
    for i in range(K + 200, min(N, K + 800)):
        perturbed[i] = int(rng.integers(0, 256))

    lg1, _ = logits_for(model, perturbed, ext)
    flat1 = lg1.reshape(-1, lg1.shape[-1]).float()

    lp0 = torch.log_softmax(flat0[row], dim=-1)
    lp1 = torch.log_softmax(flat1[row], dim=-1)
    d = (lp0 - lp1).abs().max().item()
    p0 = float(lp0[target].exp())
    p1 = float(lp1[target].exp())

    print(f"  p(target) before perturbation = {p0:.6f}")
    print(f"  p(target) after  perturbation = {p1:.6f}")
    print(f"  max |delta log-prob| over vocab = {d:.6e}")
    causal = d < 1e-6
    print(f"\n  CAUSAL (unchanged): {causal}")
    if causal:
        print("  -> the model's score for byte K does not depend on later bytes,")
        print("     so the reference's pairing is a legitimate causal code and the")
        print("     reported 1.94 bpb is a genuine code length for THIS scheme.")
    else:
        print("  -> the model's score for byte K DOES depend on later bytes; the")
        print("     scheme is not a causal code.")

    # For contrast: perturb BEFORE the target -- must change the result.
    perturbed2 = list(base)
    for i in range(max(0, K - 800), K - 200):
        perturbed2[i] = int(rng.integers(0, 256))
    lg2, _ = logits_for(model, perturbed2, ext)
    lp2 = torch.log_softmax(lg2.reshape(-1, lg2.shape[-1]).float()[row], dim=-1)
    d2 = (lp0 - lp2).abs().max().item()
    print(f"\n  control: perturbing bytes BEFORE the target changes the row by "
          f"{d2:.6e} (must be large, or the probe is broken)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
