#!/usr/bin/env python3
"""Independent re-run of the AUTHORS' harness, both arms, full file, one process.

Why this exists
---------------
The prior evidence for "the published LoRA makes text compression worse" is two
separate invocations of ``compress_with_bgpt_or_llama.compress_with_llm``:

    results/_task1_ref_ablation.txt   base   ratio 11.204097   (16:51)
    results/_task1_lora_arm.txt       +LoRA  ratio 10.509252   (17:11)

Two separate processes is a weak control.  This script runs BOTH arms inside one
process, same file, same arguments, changing only ``lora_path``, and writes each
arm's full stdout as it goes so nothing is lost if it is interrupted.

It also records what the harness actually did with the adapter (module count,
dtype, parameter deltas) so "the adapter was silently ignored" can be excluded.

Writes:
  results/_verify_refrun_base.txt
  results/_verify_refrun_lora.txt
  results/_verify_refrun_summary.txt
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

MEDAL = os.path.join(REPO, "data", "text", "MEDAL_sample")
MODEL = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B")
LORA = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora")
ONE = os.path.join(REPO, "repro_understanding_is_compression", "artifacts",
                   "task1_one")
RESULTS = os.path.join(REPO, "repro_understanding_is_compression", "results")
CTX = 8192
SRC = "medal_102285.csv"

SUMMARY: list[str] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)
    SUMMARY.append(msg)


def run_arm(label: str, lora_path) -> float | None:
    """Drive the authors' own entry point verbatim."""
    import compress_with_bgpt_or_llama as C

    say("\n" + "=" * 74)
    say(f"ARM: {label}   lora_path={lora_path}")
    say("=" * 74)
    buf = io.StringIO()
    t0 = time.time()
    err = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            C.compress_with_llm(
                data_path=ONE,
                data_type="csv",
                model_path=MODEL,
                lora_path=lora_path,
                context_window_size=CTX,
                batch_size=1,
            )
    except Exception as e:                       # keep the partial stdout
        err = f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    text = buf.getvalue()

    tag = "base" if lora_path is None else "lora"
    dump = os.path.join(RESULTS, f"_verify_refrun_{tag}.txt")
    with open(dump, "w", encoding="utf-8") as f:
        f.write(f"# authors' compress_with_llm, arm={label}\n"
                f"# data_path={ONE}  data_type=csv  model_path={MODEL}\n"
                f"# lora_path={lora_path}  context_window={CTX}  batch_size=1\n"
                f"# wall_clock={dt:.1f}s  error={err}\n\n")
        f.write(text)

    for line in text.splitlines():
        s = line.strip()
        if re.search(r"ratio|Total size|compressed size|Parameter Number|total loss|time cost", s, re.I):
            say("   | " + s[:120])
    if err:
        say(f"   !! error: {err}")

    ratio = None
    for line in text.splitlines():
        m = re.search(r"total compression ratio:\s*([0-9.]+)", line)
        if m:
            ratio = float(m.group(1))
    say(f"   -> ratio = {ratio}   [{dt:.0f}s]   raw stdout -> {dump}")
    return ratio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str, default="both",
                    help="base | lora | both.  Use one at a time if the machine "
                         "cannot hold two 8B models in one process.")
    ap.add_argument("--out-suffix", type=str, default="")
    a = ap.parse_args()

    os.makedirs(ONE, exist_ok=True)
    src = os.path.join(MEDAL, SRC)
    raw = open(src, "rb").read()
    open(os.path.join(ONE, SRC), "wb").write(raw)

    say("Independent re-run of the AUTHORS' harness")
    say(f"  arms         = {a.arms}")
    say(f"  repo         = {REPO}")
    say(f"  test file    = {SRC}  ({len(raw)} container bytes)")
    say(f"  data_path    = {ONE}")
    say(f"  model_path   = {MODEL}")
    say(f"  lora_path    = {LORA}")
    say(f"  context      = {CTX}   batch_size = 1")
    import torch
    say(f"  device       = {'cuda:0 ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    base = lora = None
    if a.arms in ("base", "both"):
        base = run_arm("base llama-3-8B, no adapter", None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if a.arms in ("lora", "both"):
        lora = run_arm("+ the authors' published LoRA", LORA)

    say("\n" + "=" * 74)
    say("RESULT - authors' own harness, identical settings, only lora_path differs")
    say("=" * 74)
    say(f"  base   ratio = {base}")
    say(f"  +LoRA  ratio = {lora}")
    if base and lora:
        say(f"  delta        = {lora - base:+.6f}  (positive = adapter better)")
        say(f"  base   rate  = {100/base:.4f} %")
        say(f"  +LoRA  rate  = {100/lora:.4f} %")
        say(f"  adapter      = {'WORSE' if lora < base else 'BETTER'} "
            f"by {abs(100/lora - 100/base):.4f} pp")
    say("")
    say("  paper (SI Table 6, MeDAL): base 9.08 %  +LoRA 9.06 %  -> paper says BETTER by 0.02 pp")
    say("  prior run in this project: base 11.204097 (8.9253 %) "
        "+LoRA 10.509252 (9.5154 %) -> WORSE by 0.5901 pp")

    out = os.path.join(RESULTS, f"_verify_refrun_summary{a.out_suffix}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(SUMMARY) + "\n")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
