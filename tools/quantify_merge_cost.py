"""How many bits does the merge distortion actually cost, in our metric?

`tools/test_merge_fidelity.py` established that `merge_and_unload()` is **not** an exact
fold: on an unquantised bf16 base, the merged model's last-position logits differ from
the live (unmerged) adapter by up to 1.875e-1, while the adapter's total effect on those
logits is 6.81. So merging preserves ~97 % of the adapter's effect and distorts ~2.7 %.

That answers "is merging lossy?" but not "does it matter for the number we report". This
script answers the second question in the actual unit of the thesis:

    Δ bits/token = mean over tokens of [ -log2 q_merged(t) + log2 q_live(t) ]

over a real MeDAL excerpt, which is directly comparable to the +0.113 bits/token we
measured as the LoRA penalty.
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


def read_payload(path: str, limit: int = 0) -> str:
    """Read the MEDAL TEXT column exactly as the port and the reference both do."""
    import pandas as pd
    df = pd.read_csv(path)
    text = "".join(str(x) for x in df["TEXT"])
    return text if limit <= 0 else text[:limit]


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    text = read_payload(MEDAL)
    print(f"MeDAL payload: {len(text)} characters (full file, as scored in the "
          f"reported runs)")

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base.eval()
    ids = tok(text, return_tensors="pt").input_ids

    def nll_bits(model):
        with torch.no_grad():
            logits = model(ids).logits.float()
        # token i is predicted by position i-1
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = ids[0, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return nll.sum().item(), nll.numel()

    print("\nscoring with the adapter applied live (unmerged) ...", flush=True)
    live = PeftModel.from_pretrained(base, LORA, is_trainable=False)
    live.eval()
    live_bits, n = nll_bits(live)
    live_bpt = live_bits / n

    print("scoring after merge_and_unload() ...", flush=True)
    merged = live.merge_and_unload()
    merged.eval()
    merged_bits, _ = nll_bits(merged)
    merged_bpt = merged_bits / n

    print("scoring the base with no adapter ...", flush=True)
    base_bits, _ = nll_bits(base)
    base_bpt = base_bits / n

    print(f"\n=== bits/token on this excerpt ({n} scored tokens) ===")
    print(f"  base, no adapter        : {base_bpt:.6f}")
    print(f"  + adapter, live         : {live_bpt:.6f}")
    print(f"  + adapter, merged       : {merged_bpt:.6f}")
    print()
    print(f"  adapter effect (live - base)   : {live_bpt - base_bpt:+.6f} bits/token")
    print(f"  adapter effect (merged - base) : {merged_bpt - base_bpt:+.6f} bits/token")
    print(f"  MERGE DISTORTION               : {merged_bpt - live_bpt:+.6f} bits/token")
    print()
    print("  For scale, the LoRA penalty we reported on medal_102285.csv was")
    print("  +0.113002 bits/token.")
    d = (merged_bpt - live_bpt) / 0.113002
    print(f"  -> merge distortion is {100*d:+.2f} % of the reported penalty")
    print()
    if abs(d) < 0.25:
        print("  VERDICT: the merge distortion is small relative to the effect being")
        print("  measured, so it cannot by itself explain the adapter appearing worse.")
    else:
        print("  VERDICT: the merge distortion is comparable to the effect, so the")
        print("  reported penalty is NOT trustworthy until re-measured unmerged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
