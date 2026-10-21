"""Measure how many scored tokens a Llama forward pass yields.

Rather than deriving the budget from the causal-shift contract (which turned out
to disagree with what the library actually returns), measure it: for a given
context length, sequence length and prefix length, how many predictions come
back?  This is the constant the window planner needs.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

REPO = _capsule_root()
MODEL = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B")
LORA = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora")


def main() -> int:
    from uic.models import LlamaTokenPredictor

    p = LlamaTokenPredictor.load(
        MODEL, lora_path=LORA, device="cuda:0", load_in_4bit=True, context_window=4096
    )
    bos = p.bos_id

    print(f"{'prefix':>7} {'tokens':>7} {'seq':>6} {'logits':>7} {'available':>10}")
    for prefix in (0, 1, 2, 100):
        for n in (100, 4094, 4095, 4096):
            lead = [bos] * (1 if prefix else 0) + list(range(500, 500 + max(0, prefix - 1)))
            window = list(range(1000, 1000 + n))
            seq = lead + window
            if len(seq) > 8192:
                continue
            ids = torch.tensor([seq], dtype=torch.long, device="cuda:0")
            with torch.no_grad():
                logits = p.model(ids).logits
            off = len(lead)
            # candidate slices
            a = logits[:, off:-1, :].shape[1]
            b = logits[:, off : off + n, :].shape[1]
            print(
                f"{prefix:>7} {n:>7} {len(seq):>6} {logits.shape[1]:>7} "
                f"slice[:-1]={a:>6} slice[off:off+n]={b:>6}"
            )
            del logits, ids
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
