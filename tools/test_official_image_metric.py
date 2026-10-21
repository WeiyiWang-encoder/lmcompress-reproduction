"""The official image metric uses the model's built-in loss, not the true-token likelihood.

``language_model.get_pdf`` returns a full ``(window, 512)`` probability matrix, not the
probability of the true token.  In ``compress_with_bgpt_or_llama.py`` the loss comes from
the model call itself:

    output = self.model(**batch)
    logits = output.logits
    if output.loss is not None:
        loss = output.loss
        self.total_loss += loss

and ``ImageGPTForCausalImageModeling`` computes ``loss`` as the mean cross-entropy
between ``logits[:, :-1]`` and ``labels[:, 1:]``.  Two consequences follow, and this
script tests both:

  H1  the official code scores the model's *own shifted* next-token prediction, whereas
      our port scores the true token at each position.  If the model's argmax differs
      from the true token, our port is penalised and reports MORE bits.
  H2  the official aggregation is ``mean_loss * (n_patches * PATCH_SIZE)`` bits per
      chunk, i.e. **the whole window including the first token**, whereas our port
      codes ``n-1`` tokens (the first has no causal predecessor).

Either way the fix is on our side: we must reproduce what the official code measures.
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
PATCH = 32


def main() -> int:
    from PIL import Image
    from transformers import ImageGPTForCausalImageModeling

    from uic.data import _igpt_processor

    path = os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0])
    rgb = np.asarray(Image.open(path).convert("RGB"))
    n = PATCH * PATCH
    block = rgb[:, :, 0].reshape(-1)[:n].reshape(PATCH, PATCH)
    triple = np.repeat(block[:, :, None], 3, axis=2)
    proc = _igpt_processor(MODEL_DIR)
    ids = np.asarray(proc(triple, return_tensors="np")["input_ids"]).reshape(-1)

    model = ImageGPTForCausalImageModeling.from_pretrained(MODEL_DIR).to("cuda:0").eval()
    inp = torch.tensor(ids, dtype=torch.long, device="cuda:0").unsqueeze(0)

    with torch.no_grad():
        out = model(inp, labels=inp)
    print(f"official-style call: logits {tuple(out.logits.shape)}, "
          f"loss = {float(out.loss):.6f}")

    lp = torch.log_softmax(out.logits.float(), dim=-1)[0]     # (1024, 512)
    n_tok = lp.shape[0]

    # ---- H1: the model's own shifted prediction vs the true token --------- #
    # labels[:, 1:] against logits[:, :-1]
    shift_lp = lp[:-1]                       # rows 0..1022
    shift_tgt = torch.tensor(ids[1:], dtype=torch.long, device="cuda:0")
    shift_nll = -shift_lp.gather(-1, shift_tgt.unsqueeze(-1)).squeeze(-1)
    print(f"\nshifted scoring (official convention: logits[t] vs token[t+1])")
    print(f"  positions scored = {shift_nll.numel()}")
    print(f"  bits = {float(shift_nll.sum()):.2f}   "
          f"bpb = {float(shift_nll.mean()):.4f}")

    # the model's cross-entropy loss should equal this mean
    print(f"  model loss (mean CE) = {float(out.loss):.6f}   "
          f"match={abs(float(out.loss) - float(shift_nll.mean())) < 1e-4}")
    print(f"  mean_loss * n_tok    = {float(out.loss) * n_tok:.2f} bits "
          f"<- what the official accumulator adds")

    # ---- what our port scored --------------------------------------------- #
    # our iter_steps scores token i+1 using logits[i], and skips the first token
    ours_bits = float(shift_nll.sum())
    print(f"\n  our port reported (from earlier run) = 1774.71 bits")
    print(f"  shifted scoring here                = {ours_bits:.2f} bits")
    print(f"  difference                          = {ours_bits - 1774.71:+.2f}")

    # ---- H2: include the first token? ------------------------------------- #
    all_nll = -lp.gather(
        -1, torch.tensor(ids, dtype=torch.long, device="cuda:0").unsqueeze(-1)
    ).squeeze(-1)
    print(f"\nfull-window true-token scoring (all {n_tok} rows vs their own token)")
    print(f"  bits = {float(all_nll.sum()):.2f}  (includes a meaningless row 0)")

    # ---- argmax agreement: why the two differ ----------------------------- #
    pred = lp[:-1].argmax(-1)
    agree = int((pred == shift_tgt).sum())
    print(f"\nargmax agreement on the shifted task: {agree}/{pred.numel()} "
          f"= {100*agree/pred.numel():.2f} %")
    print("  -> where the model's own prediction disagrees with the true token,")
    print("     TRUE-token scoring costs more bits than the model's reported loss")

    # what the official number implies
    print("\n=== reconciliation ===")
    print(f"  official reported rate for the whole image = 15.5144 %")
    print(f"  our port (true-token, shifted)             = 23.9257 %")
    print(f"  ratio                                      = "
          f"{23.9257/15.5144:.4f}")
    print(f"  NOTE: if the official accumulator uses mean_loss * n_tokens with the")
    print(f"  model's own shifted predictions, it measures a DIFFERENT quantity")
    print(f"  from a true-token code length, and the two are not interchangeable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
