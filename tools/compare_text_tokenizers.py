"""Do the base and checkpoint tokenizers produce the same ids on this corpus?

The two tokenizer.json files hash differently, which matters: the notebook asserts the
corpus tokenises to exactly 57 547 ids, so if the HF base tokenizer disagrees with the
one the model was trained with, that assertion fails and the run stops.

Two questions:
  1. Do the two local tokenizers produce identical ids on the corpus?
  2. Does the HF Llama-3-8B tokenizer (what the notebook downloads) agree with them?
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
BASE = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B")
CKPT = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora",
                    "checkpoint-5120")
MEDAL = os.path.join(REPO, "data", "text", "MEDAL_sample")
FILES = ["medal_102285.csv", "medal_103851.csv", "medal_108122.csv"]


def corpus() -> list[str]:
    out = []
    for f in FILES:
        df = pd.read_csv(os.path.join(MEDAL, f))
        s = ""
        for _, row in df.iterrows():
            s += row["TEXT"]
        out.append(s)
    return out


def main() -> int:
    from transformers import AutoTokenizer

    tb = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tc = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)

    print("=== special tokens ===")
    for name, t in (("base", tb), ("ckpt-5120", tc)):
        print(f"  {name:<10} vocab={t.vocab_size:<7} "
              f"len={len(t):<7} bos={t.bos_token!r} eos={t.eos_token!r} "
              f"pad={t.pad_token!r}")

    print("\n=== identical ids on the corpus? ===")
    texts = corpus()
    total_b = total_c = 0
    agree = True
    for f, s in zip(FILES, texts):
        ib = tb(s, add_special_tokens=False)["input_ids"]
        ic = tc(s, add_special_tokens=False)["input_ids"]
        total_b += len(ib)
        total_c += len(ic)
        same = ib == ic
        agree &= same
        print(f"  {f:<20} base {len(ib):>6}  ckpt {len(ic):>6}  "
              f"identical={same}")
        if not same:
            for i, (a, b) in enumerate(zip(ib, ic)):
                if a != b:
                    print(f"    first difference at index {i}: {a} vs {b}")
                    print(f"    base decodes to: {tb.decode(ib[max(0,i-5):i+5])!r}")
                    print(f"    ckpt decodes to: {tc.decode(ic[max(0,i-5):i+5])!r}")
                    break
    print(f"  totals: base {total_b:,}  ckpt {total_c:,}  all identical={agree}")

    print("\n=== does add_special_tokens change the counts? ===")
    s = texts[0]
    for flag in (False, True):
        print(f"  add_special_tokens={flag!s:<5} base {len(tb(s, add_special_tokens=flag)['input_ids']):>6}")

    print("\n=== the notebook's assertion ===")
    print(f"  it requires 57 547 tokens with add_special_tokens=False")
    print(f"  base tokenizer gives {total_b:,} -> "
          f"{'PASS' if total_b == 57547 else 'FAIL'}")
    print(f"  ckpt tokenizer gives {total_c:,} -> "
          f"{'PASS' if total_c == 57547 else 'FAIL'}")

    print("\n=== char window counts under each tokenizer ===")
    for name, t in (("base", tb), ("ckpt", tc)):
        counts = []
        for ctx in (4096, 6144, 8192):
            k = 0
            for s in texts:
                for i in range(0, len(s), ctx):
                    if len(t(s[i:i+ctx], add_special_tokens=False)["input_ids"]) >= 2:
                        k += 1
            counts.append(k)
        print(f"  {name:<6} char windows {counts}  (notebook asserts [78, 51, 39])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
