"""Print the exact tensor shapes on the Llama coding path."""

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
CTX = 4096


def main() -> int:
    from uic.models import LlamaTokenPredictor

    p = LlamaTokenPredictor.load(
        MODEL, lora_path=LORA, device="cuda:0", load_in_4bit=True, context_window=CTX
    )
    print("bos_id =", p.bos_id, " context_window =", CTX, " n_symbols =", p.n_symbols)

    n_tokens = 4090
    tokens = list(range(1000, 1000 + n_tokens))
    bos = [p.bos_id] if p.bos_id is not None else []
    seq = bos + tokens
    print("seq len =", len(seq))
    ids = torch.tensor([seq], dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        out = p.model(ids)
    print("logits shape =", tuple(out.logits.shape))
    offset = len(bos)
    scope = out.logits[:, offset:-1, :]
    print(f"offset={offset}  scope={tuple(scope.shape)}  tokens={len(tokens)}")
    print("=> predictions available:", scope.shape[1])

    # how many tokens can this context actually score on window 0?
    for n in (4094, 4093, 4092, 4090, 2048):
        seq = bos + list(range(n))
        ids = torch.tensor([seq], dtype=torch.long, device="cuda:0")
        with torch.no_grad():
            o = p.model(ids)
        got = o.logits[:, len(bos) : -1, :].shape[1]
        print(f"   tokens={n:>5} seq={len(seq):>5} logits={o.logits.shape[1]:>5} scorable={got:>5}")
        del o, ids
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
