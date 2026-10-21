"""Reproduce the Llama window shape problem in isolation."""

from __future__ import annotations

import os
import sys
import traceback

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
    from uic.models import LlamaTokenPredictor, _gather_true_probs

    p = LlamaTokenPredictor.load(
        MODEL, lora_path=LORA, device="cuda:0", load_in_4bit=True, context_window=4096
    )

    for n in (100, 4094):
        for prior in (0, 4094):
            doc = list(range(1000, 1000 + prior + n))
            p.set_document(doc)
            window = doc[prior : prior + n]
            bos = [p.bos_id] if (p.bos_id is not None and prior == 0) else []
            history = max(0, p.context_window - len(window))
            start = max(0, prior - history)
            prefix = p._all_tokens[start:prior] if prior > 0 else []
            seq = bos + prefix + window
            ids = torch.tensor([seq], dtype=torch.long, device="cuda:0")
            with torch.no_grad():
                logits = p.model(ids).logits
            offset = len(bos) + len(prefix)
            hard = min(len(window), max(0, logits.shape[1] - offset))
            scope = logits[:, offset : offset + hard, :]
            print(
                f"n={n} prior={prior}: seq={len(seq)} logits={logits.shape[1]} "
                f"offset={offset} hard={hard} scope={tuple(scope.shape)} "
                f"tgt_slice={ids[:, offset+1:offset+1+hard].shape[1]}"
            )
            try:
                pt = _gather_true_probs(scope.to(torch.float32), ids[:, offset + 1 : offset + 1 + hard])
                print("   gathered:", tuple(pt.shape))
            except Exception:
                traceback.print_exc()
            # now the real generator
            try:
                got = sum(1 for _ in p.iter_steps(window, prior=prior))
                print(f"   iter_steps yielded {got} (window {len(window)})")
            except Exception as exc:
                print(f"   iter_steps FAILED: {type(exc).__name__}: {exc}")
            del logits, scope, ids
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
