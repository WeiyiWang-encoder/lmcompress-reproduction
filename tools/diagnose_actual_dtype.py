"""What did our text port ACTUALLY load?

Context: `bitsandbytes` in this environment reports "compiled without GPU support" and
cannot find `libbitsandbytes_cuda128.dll`, so 4-bit NF4 quantisation should be
impossible. Yet our text results exist and are documented as "4-bit".

Possibilities:
  * transformers silently fell back to an unquantised dtype,
  * the model is fp16/bf16 (≈16 GB for 8B -- would not fit a 12 GB card),
  * or something else is going on.

This prints the ground truth: dtype of the embeddings, whether any quantisation state
exists, and the process's GPU memory footprint.
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


def describe(model, label: str) -> None:
    print(f"\n=== {label} ===")
    try:
        emb = model.get_input_embeddings().weight
        print(f"  embedding dtype      : {emb.dtype}")
        print(f"  embedding shape      : {tuple(emb.shape)}")
    except Exception as exc:
        print(f"  embedding probe failed: {exc}")
    quant = getattr(model, "quantization_method", None)
    print(f"  quantization_method  : {quant}")
    hfq = getattr(model, "hf_quantizer", None)
    print(f"  hf_quantizer         : {type(hfq).__name__ if hfq else None}")
    if hfq is not None:
        print(f"  quantizer.is_quantized: {getattr(hfq, 'is_quantized', None)}")
        print(f"  quantizer.quantization_config: "
              f"{getattr(hfq, 'quantization_config', None)}")
    # look for bitsandbytes Linear4bit layers
    try:
        import bitsandbytes as bnb
        n4 = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
        n8 = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear8bitLt))
        print(f"  Linear4bit layers    : {n4}")
        print(f"  Linear8bitLt layers  : {n8}")
    except Exception as exc:
        print(f"  bitsandbytes probe failed: {exc}")
    if torch.cuda.is_available():
        print(f"  torch.cuda.memory_allocated: "
              f"{torch.cuda.memory_allocated()/2**30:.2f} GiB")
        print(f"  torch.cuda.max_memory_allocated: "
              f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        print(f"  device               : {next(model.parameters()).device}")


def main() -> int:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print("=== does `import bitsandbytes` even work? ===")
    try:
        import bitsandbytes as bnb
        print(f"  version {bnb.__version__}")
    except Exception as exc:
        print(f"  import failed: {type(exc).__name__}: {exc}")

    print("\n=== attempt 4-bit, exactly as our port requests ===")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True))
        model = model.to("cuda:0")
        describe(model, "4-bit request (what our port asked for)")
        ok4 = True
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        ok4 = False

    print("\n=== attempt bf16, which is what our port falls back to ===")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        model2 = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, dtype=torch.bfloat16)
        model2 = model2.to("cuda:0")
        describe(model2, "bf16 loading")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    print("\n=== verdict ===")
    if not ok4:
        print("  4-bit quantisation does NOT work in this environment.")
        print("  -> every '4-bit NF4' claim in this project's documentation is WRONG")
        print("     and must be re-checked: the real dtype is whatever loaded instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
