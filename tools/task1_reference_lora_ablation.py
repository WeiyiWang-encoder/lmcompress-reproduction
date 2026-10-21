#!/usr/bin/env python3
"""Task 1: does the AUTHORS' own script show LoRA helping?

Our port measures the published LoRA as *worse* than base llama-3-8B on MeDAL
(9.353 % vs 9.041 %, +0.312 pp).  The paper reports it as slightly *better*
(9.06 vs 9.08).  Two explanations compete:

  (a) our harness is wrong somewhere, or
  (b) the reference harness has a property ours lacks -- e.g. it applies the LoRA
      differently, or its context handling differs.

The only way to separate them is to run the reference harness itself, on the same
file, with and without the adapter, and read the numbers it prints.

This script drives ``compress_with_bgpt_or_llama`` directly so both arms use
*identical* code, batch construction and aggregation; only ``lora_path`` changes.
One file only, to keep it quick.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import time

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
os.chdir(REPO)
sys.path.insert(0, REPO)

MEDAL = os.path.join(REPO, "data", "text", "MEDAL_sample")
MODEL = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B")
LORA = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora")
ONE = os.path.join(REPO, "repro_understanding_is_compression", "artifacts", "task1_one")


def run_arm(label: str, lora_path):
    import compress_with_bgpt_or_llama as C

    print(f"\n{'='*70}\nARM: {label}\n{'='*70}", flush=True)
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        C.compress_with_llm(
            data_path=ONE,
            data_type="csv",
            model_path=MODEL,
            lora_path=lora_path,
            context_window_size=8192,
            batch_size=1,
        )
    dt = time.time() - t0
    text = buf.getvalue()
    # the script prints the ratio; grab every numeric line that looks like a result
    interesting = [l for l in text.splitlines()
                   if re.search(r"ratio|size|loss|time cost|Parameter", l, re.I)]
    for line in interesting[-8:]:
        print("   " + line.strip()[:110])
    print(f"   [{dt:.0f}s]")
    ratio = None
    for line in interesting:
        m = re.search(r"total compression ratio:\s*([0-9.]+)", line)
        if m:
            ratio = float(m.group(1))
    return ratio


def main() -> int:
    # one file, so the comparison is cheap and the file is identical between arms
    src = os.path.join(MEDAL, "medal_102285.csv")
    os.makedirs(ONE, exist_ok=True)
    raw = open(src, "rb").read()
    open(os.path.join(ONE, "medal_102285.csv"), "wb").write(raw)
    print(f"test file: medal_102285.csv, {len(raw)} container bytes "
          f"(MEDAL TEXT payload is smaller)")

    base = run_arm("base, no LoRA", None)
    lora = run_arm("with the authors' published LoRA", LORA)

    print(f"\n{'='*70}\nRESULT (reference harness, ctx 8192)\n{'='*70}")
    print(f"  base ratio          = {base}")
    print(f"  LoRA ratio          = {lora}")
    if base and lora:
        print(f"  delta ratio         = {lora - base:+.4f}")
        print(f"  LoRA better?        = {lora > base}")
    print()
    print("  For comparison, OUR port at ctx 4096 on this file:")
    print("    base 8.8631 %, LoRA 9.1269 %  (rate; lower is better -> LoRA worse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
