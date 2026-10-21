"""The decisive question: does the officially used logit row depend on the token it codes?

``language_model.compress`` codes token ``t`` with ``raw_probs[t]``, where
``raw_probs = predict_fn(sequence_array)`` and ``predict_fn`` runs the model on the
whole (already complete) token sequence.  So the question is whether model logits at
position ``t`` depend on tokens at positions after ``t``.

A causal transformer must not: changing tokens from ``K+1`` onward must leave rows
``0..K`` untouched.  If that holds, then coding token ``t`` with row ``t`` uses a
distribution computed from a prefix that already *contains* token ``t`` -- it is not
a predictive distribution for token ``t``, and the resulting bit count is not a
lossless code length for the sequence.

This measures it directly rather than arguing from architecture.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
sys.path.insert(0, REPO)

MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMG_DIR = os.path.join(REPO, "data", "test_images")


def main() -> int:
    from PIL import Image
    from transformers import ImageGPTForCausalImageModeling

    from uic.data import _igpt_processor

    path = os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0])
    rgb = np.asarray(Image.open(path).convert("RGB"))
    blk = rgb[:, :, 0].reshape(-1)[:1024].reshape(32, 32)
    ids = np.asarray(_igpt_processor(MODEL_DIR)(
        np.repeat(blk[:, :, None], 3, axis=2), return_tensors="np")
        ["input_ids"]).reshape(-1)

    model = ImageGPTForCausalImageModeling.from_pretrained(MODEL_DIR)
    model = model.to("cuda:0").eval()

    def logits(seq):
        inp = torch.tensor(seq, dtype=torch.long, device="cuda:0").unsqueeze(0)
        with torch.no_grad():
            return model(inp).logits.float()[0]

    lp = logits(ids)
    K = 500
    mod = ids.copy()
    mod[K + 1:] = (mod[K + 1:] + 137) % 512
    lp2 = logits(mod)

    d_before = float((lp[:K + 1] - lp2[:K + 1]).abs().max())
    d_after = float((lp[K + 1:] - lp2[K + 1:]).abs().max())

    print("Causality probe on the official model invocation:")
    print(f"  perturbed tokens [{K+1}:] (a suffix)")
    print(f"  rows 0..{K}  max|delta logit| = {d_before:.3e}   (0 => causal)")
    print(f"  rows {K+1}..  max|delta logit| = {d_after:.3e}")

    causal = d_before < 1e-5
    print(f"\n  causal: {causal}")
    if causal:
        print("\n  CONSEQUENCE")
        print("  Row t is computed from tokens[0..t], so coding token t with row t")
        print("  means the distribution already conditioned on token t itself.")
        print("  The official image 'code length' therefore uses, at every position,")
        print("  a distribution that has seen the symbol it is coding.")
        print("  That is not a lossless code length for the sequence, and it cannot")
        print("  be reproduced by any decoder that does not already know the token.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
