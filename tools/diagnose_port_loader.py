"""What does OUR port actually load, when asked for 4-bit on this machine?

`bitsandbytes` here has no CUDA binary, and requesting 4-bit raises
`ImportError: ... requires accelerate`. So the loader in `src/uic/models.py` either
has a fallback we have not read, or every "4-bit NF4" claim in this project's
documentation is wrong.

This calls the port's own loader and reports what came back.
"""

from __future__ import annotations

import os
import sys

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
LORA = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora")


def main() -> int:
    from uic.models import LlamaTokenPredictor

    print(f"GPU total : {torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")

    for label, lora, merge in (("base", None, True),
                               ("+LoRA (merged)", LORA, True),
                               ("+LoRA (unmerged)", LORA, False)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n=== load: {label} ===", flush=True)
        try:
            pred = LlamaTokenPredictor.load(
                MODEL, device="cuda:0", lora_path=lora, merge_lora=merge)
        except Exception as exc:
            print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
            continue
        m = pred.model
        inner = getattr(m, "model", m)
        emb = inner.get_input_embeddings().weight
        print(f"  embedding dtype        : {emb.dtype}")
        print(f"  quantization_method    : {getattr(m, 'quantization_method', None)}")
        try:
            import bitsandbytes as bnb
            n4 = sum(1 for x in m.modules() if isinstance(x, bnb.nn.Linear4bit))
            print(f"  Linear4bit layers      : {n4}")
        except Exception:
            pass
        print(f"  peak CUDA allocated    : "
              f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        print(f"  peak CUDA reserved     : "
              f"{torch.cuda.max_memory_reserved()/2**30:.2f} GiB")
        # is any parameter on CPU?
        devices = {}
        for p in m.parameters():
            devices[str(p.device)] = devices.get(str(p.device), 0) + 1
        print(f"  parameter devices      : {devices}")
        pred.release()
        del pred
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
