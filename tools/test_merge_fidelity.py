"""Does `merge_and_unload()` distort the adapter, or faithfully fold it in?

This is the question that decides whether the +0.312 pp LoRA penalty is trustworthy.

It matters because our port merges:

    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

`merge_and_unload` computes W + scale * B@A and writes it into the base tensors. On a
plain floating-point base that is exact (up to rounding). On a **quantised** base it is
not, and bitsandbytes is broken in this environment anyway -- so the first thing to
settle is what the base dtype actually was, and whether merging changed anything.

Method: load the base **without any quantisation** and offload to CPU, so the 11.94 GiB
GPU is irrelevant and nothing can silently fall back. Then score one fixed sequence
three ways and compare:

    1. base, no adapter
    2. + adapter, NOT merged   (adapter applied live -- the faithful evaluation)
    3. + adapter, MERGED       (what our port did)

Verdict:
  * (3) == (1)                -> merging discarded the adapter; our LoRA arm was base
  * (3) != (2), by > ~1e-3    -> merging is lossy; the merged arm is not a faithful
                                 evaluation of the adapter
  * (3) ~= (2)                -> merging is faithful, and the penalty is real
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

TEXT = ("The patient was admitted with acute exacerbation of chronic obstructive "
        "pulmonary disease and was treated with bronchodilators and systemic "
        "corticosteroids. Follow-up spirometry showed improvement in forced "
        "expiratory volume in one second, and the patient was discharged.")


def load_base(dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # low_cpu_mem_usage keeps peak RAM sane; no device_map, so no accelerate needed
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    return model, tok


def last_logits(model, tok):
    ids = tok(TEXT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids).logits.float()
    return out[0, -1].clone()


def main() -> int:
    dtype = torch.bfloat16
    print(f"base dtype requested: {dtype}")
    print("loading base on CPU (no quantisation, no device_map) ...", flush=True)

    model, tok = load_base(dtype)
    emb = model.get_input_embeddings().weight
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  embedding dtype : {emb.dtype}")
    print(f"  parameters      : {n_params/1e9:.3f} B")
    print(f"  quantisation    : {getattr(model, 'quantization_method', None)}")

    base_logits = last_logits(model, tok)
    print(f"  argmax(last)    : {int(base_logits.argmax())}")

    # ---- arm 2: adapter applied live (no merge) ---------------------------- #
    from peft import PeftModel
    print("\nattaching adapter (NOT merged) ...", flush=True)
    live = PeftModel.from_pretrained(model, LORA, is_trainable=False)
    live.eval()
    live_logits = last_logits(live, tok)
    print(f"  argmax(last)    : {int(live_logits.argmax())}")

    # ---- arm 3: merged ---------------------------------------------------- #
    print("\nmerging adapter into the base ...", flush=True)
    merged = live.merge_and_unload()
    merged.eval()
    merged_logits = last_logits(merged, tok)
    print(f"  argmax(last)    : {int(merged_logits.argmax())}")

    # ---- comparison ------------------------------------------------------- #
    d_base_live = (base_logits - live_logits).abs().max().item()
    d_base_merged = (base_logits - merged_logits).abs().max().item()
    d_live_merged = (live_logits - merged_logits).abs().max().item()

    print("\n=== last-position logit differences ===")
    print(f"  base   vs live   : max|Δ| = {d_base_live:.6e}")
    print(f"  base   vs merged : max|Δ| = {d_base_merged:.6e}")
    print(f"  live   vs merged : max|Δ| = {d_live_merged:.6e}")

    print("\n=== verdict ===")
    if d_base_merged == 0.0:
        print("  merged == base  -> merge_and_unload() DISCARDED the adapter.")
        print("  Our 'LoRA' arm would have been the base model; the penalty is void.")
    elif d_base_live == 0.0:
        print("  live == base    -> the adapter had NO effect at all; arm is void.")
    elif d_live_merged > 1e-3:
        print(f"  live != merged  -> merging is LOSSY (max|Δ| = {d_live_merged:.3e}).")
        print("  The merged arm is not a faithful evaluation of the adapter;")
        print("  the +0.312 pp penalty must be re-measured unmerged.")
    else:
        print(f"  live ~= merged  -> merging is faithful (max|Δ| = {d_live_merged:.3e}).")
        print("  The adapter genuinely changes the distribution, so the measured")
        print("  penalty is an effect of the adapter, not of the merge step.")
    print("\n  Note: which of these holds is independent of the 4-bit problem,")
    print("  because this run uses no quantisation at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
