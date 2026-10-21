"""Is our LoRA actually taking effect?

Hypothesis for why our port shows the adapter as *worse* than base while the paper
shows it slightly better: our loader does

    model = AutoModelForCausalLM.from_pretrained(..., load_in_4bit=True, ...)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()      # <-- suspect

`merge_and_unload()` folds the adapter delta W = B@A * scale into the base weights.
On a **quantised** (bitsandbytes NF4) base those weights are not ordinary float
tensors, so merging is at best lossy and at worst a silent no-op.  If the delta is
not applied, the "LoRA" arm is really just the base model again -- which would make
any measured difference noise, and would explain a result that disagrees with the
reference harness.

Test: score one fixed token sequence three ways and compare the logits.
  1. base, 4-bit, no adapter
  2. 4-bit + adapter, merged      (our current path)
  3. 4-bit + adapter, NOT merged  (adapter active at runtime)

If (2) is bit-identical to (1), the merge silently discards the adapter.
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
        "pulmonary disease and treated with bronchodilators and systemic "
        "corticosteroids. Follow-up showed improvement in forced expiratory volume.")


def load(use_lora: bool, merge: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Load exactly the way our port does (no `accelerate` requirement), so the test
    # exercises the same code path that produced the +0.312 pp figure.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model = model.to("cuda:0")
    if use_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA, is_trainable=False)
        if merge:
            model = model.merge_and_unload()
    model.eval()
    return model, tok


def probe(model, tok):
    ids = tok(TEXT, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids).logits.float()
    # last-position distribution and a summary of the whole tensor
    last = out[0, -1]
    return last, float(out.sum()), int(out[0, -1].argmax())


def main() -> int:
    arms = {}
    for label, use_lora, merge in (("base (no adapter)", False, False),
                                   ("adapter, MERGED (our path)", True, True),
                                   ("adapter, NOT merged", True, False)):
        print(f"loading: {label} ...", flush=True)
        model, tok = load(use_lora, merge)
        last, total, arg = probe(model, tok)
        arms[label] = (last.clone(), total, arg, model)
        print(f"  argmax(last) = {arg}   sum(logits) = {total:.4f}")
        del model
        torch.cuda.empty_cache()

    print("\n=== pairwise comparison of the last-position distribution ===")
    labels = list(arms)
    base = arms[labels[0]][0]
    for label in labels[1:]:
        other = arms[label][0]
        d = (base - other).abs().max().item()
        print(f"  base vs {label:<28}: max|Δ| = {d:.6e}")

    merged = arms[labels[1]][0]
    unmerged = arms[labels[2]][0]
    d_mu = (merged - unmerged).abs().max().item()
    print(f"  MERGED vs NOT-merged          : max|Δ| = {d_mu:.6e}")

    print("\n=== verdict ===")
    d_base_merged = (base - merged).abs().max().item()
    if d_base_merged == 0.0:
        print("  The merged adapter is BIT-IDENTICAL to base.")
        print("  -> merge_and_unload() silently discarded the adapter on the 4-bit base.")
        print("  -> our 'LoRA' arm was really the base model; the reported penalty is")
        print("     not a property of the adapter at all.")
    elif d_mu > 1e-3:
        print("  Merging changes the model, but differs from running the adapter live.")
        print("  -> merging on a quantised base is lossy; the merged path is not a")
        print("     faithful evaluation of the adapter.")
    else:
        print("  Merging preserves the adapter's effect; this path is fine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
