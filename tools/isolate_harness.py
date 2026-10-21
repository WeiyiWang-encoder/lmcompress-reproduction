"""Isolate harness from weights: score one window two ways on the same checkpoint.

The paired A100 run (6.33 %) and our local port (9.04 %) disagree by 46 % on the same
corpus. The weights are ruled out: the capsule checkpoint is byte-identical to the
public one (`tools/compare_llama_weights.py`, two tensors including a 1 GB embedding).

So the difference is in the harness. This computes, on ONE 4096-token window of the same
file, using the SAME capsule checkpoint on CPU in bf16:

  A. a plain HuggingFace forward pass, scoring token j+1 from position j  -- the
     convention the A100 notebook used, and the one its `output.loss` cross-check
     validated;
  B. the port's `LlamaTokenPredictor.iter_steps` on the same window.

If A and B agree, the harnesses are equivalent and the A100 figure is the puzzle.
If they disagree, our port is the defective one and 9.04 % is wrong.

A third number is also computed for reference: the reference implementation's convention
(row j scores token j), which the earlier iGPT work showed to be optimistic.
"""

from __future__ import annotations

import os
import sys
import time

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
MODEL = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B")
CSV = os.path.join(REPO, "data", "text", "MEDAL_sample", "medal_102285.csv")
WINDOW = 4096


def main() -> int:
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    df = pd.read_csv(CSV)
    text = ""
    for _, row in df.iterrows():
        text += row["TEXT"]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    w = ids[:WINDOW]
    print(f"file tokens {len(ids):,}   window {len(w):,}   raw bytes {len(text.encode()):,}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print(f"loaded on CPU in {time.time()-t0:.0f}s  dtype={next(model.parameters()).dtype}")

    # ---------- A: plain HF forward, causal shift ---------- #
    t0 = time.time()
    t = torch.tensor([w], dtype=torch.long)
    with torch.no_grad():
        logits = model(t).logits            # (1, L, V) bf16
    print(f"forward in {time.time()-t0:.0f}s  logits {tuple(logits.shape)} "
          f"{logits.dtype} {logits.numel()*2/2**30:.2f} GiB")

    bits_shift = 0.0
    bits_same = 0.0
    SLICE = 512
    L = t.shape[1]
    for a in range(0, L - 1, SLICE):
        b = min(a + SLICE, L - 1)
        seg = logits[:, a:b, :].float()
        lp = torch.log_softmax(seg, dim=-1)
        bits_shift += float(-lp.gather(-1, t[:, a+1:b+1].unsqueeze(-1)).squeeze(-1).sum())
        bits_same += float(-lp.gather(-1, t[:, a:b].unsqueeze(-1)).squeeze(-1).sum())
        del seg, lp

    # cross-check A against the model's own loss (the notebook's assertion)
    with torch.no_grad():
        out = model(t, labels=t)
    ref = float(out.loss) * (L - 1)
    print()
    print("A. plain HF, causal shift (position j scores token j+1)")
    print(f"   bits = {bits_shift:12.3f}   bits/token = {bits_shift/(L-1):.4f}")
    print(f"   output.loss*(L-1) = {ref:12.3f}   rel diff = "
          f"{abs(bits_shift-ref)/bits_shift:.2e}")
    print()
    print("   for contrast, position j scores token j (the reference's convention)")
    print(f"   bits = {bits_same:12.3f}   bits/token = {bits_same/(L-1):.4f}")

    del logits, out

    # ---------- B: our port on the same window ---------- #
    from uic.models import LlamaTokenPredictor

    print()
    print("B. our port, LlamaTokenPredictor.iter_steps on the same window")
    t0 = time.time()
    pred = LlamaTokenPredictor(
        model=model, tokenizer=tok, device="cpu", context_window=WINDOW,
        prepend_bos=True)
    port_bits = 0.0
    n = 0
    for step in pred.iter_steps(w, prior=0, stop_at=len(w)):
        port_bits += -np.log2(max(step.p_true, 1e-300))
        n += 1
    print(f"   bits = {port_bits:12.3f}   scored {n} tokens   "
          f"bits/token = {port_bits/max(n,1):.4f}")
    print(f"   elapsed {time.time()-t0:.0f}s")

    print()
    print("=== comparison ===")
    print(f"   A (plain HF)   {bits_shift:12.3f} bits")
    print(f"   B (our port)   {port_bits:12.3f} bits")
    if port_bits:
        print(f"   ratio B/A      {port_bits/bits_shift:.4f}")
        if abs(port_bits - bits_shift) / bits_shift < 0.01:
            print("   -> the two harnesses AGREE on identical weights.")
            print("      The A100 figure of 6.33 % is then the puzzle, not the port.")
        else:
            print("   -> the two harnesses DISAGREE on identical weights.")
            print("      Our port is the defective one and 9.04 % must be withdrawn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
