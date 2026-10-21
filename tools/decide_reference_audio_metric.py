"""Decisive test: is the reference's audio number tracking the model, or the artifact?

The user's experience is that running the authors' code reproduces the paper's audio
result, while this port finds the same pipeline gives 15.78 bits/byte under honest
next-byte scoring.  Both observations could be true if the reference's number is
*insensitive to the data* -- i.e. if it is produced by the scoring misalignment
rather than by the model's predictive quality.

That is directly testable.  If the reported bits/byte is driven by the model
predicting each byte from its own context (self-reconstruction), then it should
barely change when the model's *predictive* content is destroyed.  So:

  arms:
    A  real audio
    B  real audio, byte-shuffled      (destroys all sequential structure)
    C  uniform random bytes           (destroys everything)
    D  zeros                          (maximally predictable)

  for each arm, compute
    * the reference's own metric, exactly as ``compress_with_bgpt_or_llama`` scores,
    * honest next-byte scoring with the same model.

If the reference's number stays low across B and C while honest scoring blows up,
the reference is measuring the artifact.  That would reconcile the user's
observation with ours: the number is reproducible because it does not depend on
the data's actual predictability.
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
    patch_config = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS, max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH, hidden_size=HIDDEN_SIZE,
        n_head=HIDDEN_SIZE // 64, vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=BYTE_NUM_LAYERS, max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1, hidden_size=HIDDEN_SIZE,
        n_head=HIDDEN_SIZE // 64, vocab_size=256 + 1,
    )
    m = bGPTLMHeadModel(patch_config, byte_config)
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"], strict=False)
    return m.to("cuda:0").eval()


def score_reference(model, data, ext):
    """Exactly the reference's post-processing, including its index arithmetic."""
    loader = torch.utils.data.DataLoader(
        ref.split(data, N), batch_size=1, shuffle=False,
        collate_fn=functools.partial(ref.byte_collate, appendix={"ext": ext},
                                     device="cuda:0"),
    )
    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(**batch).logits
    ref_logits = logits[:-1, :-1, :].reshape(1, -1, 257)
    ids = batch["patches"][:, PATCH_SIZE:-PATCH_SIZE]
    mask = batch["masks"].repeat_interleave(PATCH_SIZE, dim=1)[:, PATCH_SIZE:-PATCH_SIZE]
    probs = ref_logits.softmax(dim=-1)
    pd = torch.gather(probs, -1, ids.unsqueeze(-1)).squeeze(-1).add_(6e-8)
    nll = -torch.log2(pd) * mask
    bits = nll.sum().item() / mask.sum().item() * len(data)
    scored_grid = []
    for p in range(logits.shape[0] - 1):
        for j in range(logits.shape[1] - 1):
            scored_grid.append(p * 16 + j + 1)
    return bits, min(scored_grid), max(scored_grid)


def score_honest(model, data, ext):
    """Our alignment: every payload byte scored against its own predictor."""
    loader = torch.utils.data.DataLoader(
        ref.split(data, N), batch_size=1, shuffle=False,
        collate_fn=functools.partial(ref.byte_collate, appendix={"ext": ext},
                                     device="cuda:0"),
    )
    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(**batch).logits
    flat = logits.reshape(1, -1, logits.shape[-1]).float()
    grid = batch["patches"].reshape(1, -1)
    n = len(data)
    off = 16
    lp = torch.log_softmax(flat[:, off : off + n, :], dim=-1)
    tgt = grid[:, off : off + n]
    # alignment assertion: the scored targets must be the payload
    assert tgt[0, 0].item() == data[0] and tgt[0, -1].item() == data[-1], "misaligned"
    p = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).exp()
    bits = -torch.log2(p.clamp_min(1e-30)).sum().item()
    return bits


def main() -> int:
    model = build_model()
    ext = [ord(c) for c in "wav"]

    from uic.data import audio_units

    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    real = unit.payload[:N]
    rng = np.random.default_rng(0)

    arms = {
        "A real audio": list(real),
        "B real audio, byte-shuffled": list(rng.permutation(np.asarray(real, dtype=np.uint8))),
        "C uniform random bytes": list(rng.integers(0, 256, size=N, dtype=np.uint8)),
        "D all zeros": [0] * N,
    }

    print(f"{'arm':<30} {'reference bpb':>14} {'honest next-byte bpb':>21} {'ratio':>8}")
    results = {}
    for name, data in arms.items():
        data = [int(x) for x in data]
        rbits, lo, hi = score_reference(model, data, ext)
        hbits = score_honest(model, data, ext)
        rb = rbits / len(data)
        hb = hbits / len(data)
        results[name] = (rb, hb)
        print(f"{name:<30} {rb:>14.4f} {hb:>21.4f} {hb/rb:>8.2f}x")

    print("\n=== interpretation ===")
    rb_real = results["A real audio"][0]
    for name in ("B real audio, byte-shuffled", "C uniform random bytes"):
        rb = results[name][0]
        hb = results[name][1]
        print(f"  {name}:")
        print(f"    reference metric {rb:.3f} bpb vs real audio {rb_real:.3f} bpb "
              f"-> changes by {100*(rb-rb_real)/rb_real:+.1f}%")
        print(f"    honest metric    {hb:.3f} bpb vs real audio {results['A real audio'][1]:.3f} bpb "
              f"-> changes by {100*(hb-results['A real audio'][1])/results['A real audio'][1]:+.1f}%")
    print("\n  If the reference metric barely moves while the honest one explodes,")
    print("  the reference number is not measuring the data's predictability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
