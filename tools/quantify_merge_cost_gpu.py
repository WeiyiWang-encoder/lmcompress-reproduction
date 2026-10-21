"""Merge fidelity, measured the way the port actually scores.

The first attempt used 1 200 characters of a generic clinical sentence and got a
suspiciously clean `merged - base = +0.000000`. That is a symptom worth understanding
rather than trusting: merging adds a small delta to bf16 base weights, and bf16 has ~8
mantissa bits, so a small delta can be **rounded away entirely**.

This run instead scores a real window of the actual MeDAL file -- 4 096 tokens, the
port's context length -- so the adapter's effect is measured on in-domain text at the
scale that produced the reported numbers. GPU, bf16, no quantisation.
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
MEDAL = os.path.join(REPO, "data", "text", "MEDAL_sample", "medal_102285.csv")
WINDOW = 4096
PENALTY_BPT = 0.113002          # the reported LoRA penalty, bits/token


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import pandas as pd

    text = "".join(str(x) for x in pd.read_csv(MEDAL)["TEXT"])
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    ids = ids_all[:WINDOW].unsqueeze(0).to("cuda:0")
    print(f"MeDAL payload {len(text)} chars -> scoring the first {ids.shape[1]} tokens")

    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16).to("cuda:0")
    base.eval()

    def nll_bits(model):
        with torch.no_grad():
            logits = model(ids).logits.float()
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = ids[0, 1:]
        return (-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).sum().item(), lp.shape[0]

    b_base, n = nll_bits(base)
    bpt_base = b_base / n
    print(f"  base, no adapter   : {bpt_base:.6f} bits/token")

    live = PeftModel.from_pretrained(base, LORA, is_trainable=False).to("cuda:0")
    live.eval()
    b_live, _ = nll_bits(live)
    bpt_live = b_live / n
    print(f"  + adapter, live    : {bpt_live:.6f} bits/token")

    merged = live.merge_and_unload()
    merged.eval()
    b_merged, _ = nll_bits(merged)
    bpt_merged = b_merged / n
    print(f"  + adapter, merged  : {bpt_merged:.6f} bits/token")

    print(f"\n=== on {n} in-domain tokens ===")
    print(f"  adapter effect (live   - base) : {bpt_live - bpt_base:+.6f} bits/token")
    print(f"  adapter effect (merged - base) : {bpt_merged - bpt_base:+.6f} bits/token")
    print(f"  MERGE DISTORTION               : {bpt_merged - bpt_live:+.6f} bits/token")
    print(f"\n  reported LoRA penalty          : {PENALTY_BPT:+.6f} bits/token")
    ratio = (bpt_merged - bpt_live) / PENALTY_BPT
    print(f"  distortion as share of penalty : {100*ratio:+.2f} %")

    print("\n=== did the merge preserve the adapter at all? ===")
    if bpt_merged == bpt_base:
        print("  merged == base (to the last bit): the adapter's delta was ROUNDED AWAY")
        print("  by the bf16 fold. On a bf16 base, merge_and_unload() is effectively a")
        print("  no-op, so the merged arm silently becomes the base model.")
    elif abs(bpt_merged - bpt_live) < 0.1 * abs(bpt_live - bpt_base or 1):
        print("  merged tracks live closely: the fold preserved the adapter.")
    else:
        print("  merged sits between base and live: the fold is partial.")

    print("\n=== what this means for the reported penalty ===")
    print(f"  The penalty we reported is {PENALTY_BPT:.6f} bits/token, measured with the")
    print("  MERGED path. Compare it with the distortion above:")
    if abs(bpt_merged - bpt_live) >= 0.25 * PENALTY_BPT:
        print("  -> distortion is >= 25 % of the penalty, so the penalty CANNOT be")
        print("     attributed to the adapter without re-measuring unmerged.")
    else:
        print("  -> distortion is < 25 % of the penalty, so it cannot by itself")
        print("     explain the adapter appearing worse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
