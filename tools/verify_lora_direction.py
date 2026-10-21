#!/usr/bin/env python3
"""Independent verification: does the AUTHORS' published LoRA make text compression worse?

This does NOT use our port.  It rebuilds the authors' own csv path from
``compress_with_bgpt_or_llama.LLM_Compressor`` (same model load, same
``tokenize_collate``, same shift/mask/NLL arithmetic) and measures the bits the
arithmetic coder would need, for several variants of the adapter.

Arms
----
base        : Llama-3-8B, no adapter                (authors' from_pretrained)
lora_live   : + PeftModel.from_pretrained           (authors' exact call, unmerged)
lora_lam=x  : same adapter with its delta scaled by x  (scaling sweep)

The scaling sweep tests the hypothesis raised by the adapter's own
``trainer_state.json``: the shipped weights are the step-5120 / 10-epoch
checkpoint, whose eval loss (2.6728) is far above the best (2.4708 @ step 1020).
If the delta is simply too large, shrinking it should recover the base model and
then beat it.

Writes ``results/_verify_lora_direction.txt``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B")
LORA = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora")
CSV = os.path.join(REPO, "data", "text", "MEDAL_sample", "medal_102285.csv")
CTX = 8192
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

OUT = os.path.join(REPO, "repro_understanding_is_compression", "results",
                   "_verify_lora_direction.txt")
_lines: list[str] = []
MAX_WINDOWS = 2


def say(msg: str = "") -> None:
    print(msg, flush=True)
    _lines.append(msg)


def load_payload() -> str:
    """Byte-identical to the authors' csv branch: concat every row['TEXT']."""
    df = pd.read_csv(CSV)
    return "".join(str(t) for t in df["TEXT"].tolist())


def tokenize_window(text: str, tok, ws: int):
    """Byte-identical to the authors' split()+tokenize_collate for batch_size=1."""
    out = []
    for i in range(0, len(text), ws):
        chunk = text[i:i + ws]
        enc = tok([chunk], padding="longest", truncation=False, return_tensors="pt")
        out.append({k: v.to(DEVICE) for k, v in enc.items()})
    return out[:MAX_WINDOWS]


def nll_of_batch(model, batch):
    """Byte-identical to LLM_Compressor.forward_next for the csv branch."""
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]
    shifted_attention_mask = attention_mask[:, 1:]
    probs = shifted_logits.softmax(dim=-1)
    pd = torch.gather(probs, -1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    pd = pd.add_(6e-8)
    neg_log_probs = -torch.log2(pd)
    masked = neg_log_probs * shifted_attention_mask
    bits = torch.sum(masked, dim=1) / input_ids.size(0)
    return bits.tolist(), int(shifted_attention_mask.sum().item())


_NATIVE_SCALE: list[tuple[object, dict[str, float]]] = []


def capture_native_scale(model) -> dict:
    """Record peft's own LoRA scaling (lora_alpha / r) per module.

    The delta a LoRA layer contributes is ``B @ A * scaling``, and peft sets
    ``scaling = lora_alpha / r`` = 32/8 = 4.0 for the released adapter.  Any
    scaling experiment must be expressed *relative to that*, otherwise "scale
    1.0" silently means one quarter strength.
    """
    _NATIVE_SCALE.clear()
    for mod in model.modules():
        sc = getattr(mod, "scaling", None)
        if isinstance(sc, dict):
            _NATIVE_SCALE.append((mod, {k: float(v) for k, v in sc.items()}))
    vals = {v for _, d in _NATIVE_SCALE for v in d.values()}
    return {"n_modules": len(_NATIVE_SCALE), "native_scaling_values": sorted(vals)}


def set_lora_scale(model, lam: float) -> int:
    """Set every LoRA delta to `lam` x its native (alpha/r) magnitude."""
    n = 0
    for mod, nat in _NATIVE_SCALE:
        sc = getattr(mod, "scaling", None)
        if isinstance(sc, dict):
            for k, v in nat.items():
                sc[k] = v * lam
            n += 1
    return n


def main() -> int:
    global MAX_WINDOWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=2,
                    help="how many ctx-8192 windows of the payload to measure")
    ap.add_argument("--arms", type=str, default="base,lora",
                    help="base | lora | scale  (scale = delta-scaling sweep)")
    ap.add_argument("--scale-lams", type=str, default="0.75,0.5,0.25",
                    help="delta scales to try when --arms includes 'scale'")
    ap.add_argument("--scale-windows", type=str, default="",
                    help="comma-separated window indices for the sweep "
                         "(default: all measured windows)")
    ap.add_argument("--only-windows", type=str, default="",
                    help="restrict the base / lora arms to these window indices")
    ap.add_argument("--out", type=str, default=OUT)
    a = ap.parse_args()
    MAX_WINDOWS = a.windows
    want = set(a.arms.split(","))
    lams = [float(x) for x in a.scale_lams.split(",") if x.strip()]
    swin = [int(x) for x in a.scale_windows.split(",") if x.strip()] or None
    owin = [int(x) for x in a.only_windows.split(",") if x.strip()] or None
    out_path = a.out

    t_start = time.time()
    say(f"repo       = {REPO}")
    say(f"device     = {DEVICE}  ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        say(f"vram       = {p.total_memory/1024**3:.2f} GiB")
    say(f"windows    = {MAX_WINDOWS}   arms = {sorted(want)}")

    payload = load_payload()
    say(f"payload    = {len(payload)} chars  (authors' 'Total size' denominator)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    batches = tokenize_window(payload, tok, CTX)
    ntokens = [int(b["attention_mask"].sum().item()) for b in batches]
    measured_chars = sum(len(payload[i * CTX:(i + 1) * CTX]) for i in range(len(batches)))
    say(f"measuring  = first {len(batches)} window(s) = {measured_chars} chars  "
        f"tokens/window={ntokens}")

    t0 = time.time()
    say("\nloading base Llama-3-8B exactly as the authors do ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    dtype = next(model.parameters()).dtype
    say(f"  params   = {params/1e9:.3f} B   dtype = {dtype}")
    if torch.cuda.is_available():
        say(f"  vram allocated = {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
    say(f"  [{time.time()-t0:.0f}s]")

    # ---- arm: base -------------------------------------------------------
    arms: dict[str, dict] = {}

    def run_arm(name: str, only: list[int] | None = None) -> None:
        idxs = list(range(len(batches)) if not only else only)
        pw: dict[int, float] = {}
        pw_tok: dict[int, int] = {}
        total_bits = 0.0
        total_tok = 0
        chars = 0
        t = time.time()
        for k in idxs:
            b = batches[k]
            bits, ntok = nll_of_batch(model, b)
            pw[k] = bits[0]
            pw_tok[k] = ntok
            total_bits += float(np.sum(bits))
            total_tok += ntok
            chars += len(payload[k * CTX:(k + 1) * CTX])
            say(f"    window {k:2d}: {bits[0]:9.3f} bits over {ntok:5d} tokens "
                f"= {bits[0]/ntok:.6f} bits/token")
        ratio = chars / (total_bits / 8.0)
        arms[name] = dict(bits=total_bits, tokens=total_tok, ratio=ratio,
                          bpt=total_bits / total_tok, secs=time.time() - t,
                          windows=idxs, chars=chars,
                          per_window=pw, per_window_tokens=pw_tok)
        say(f"  {name:12s} bits/token={total_bits/total_tok:.6f}  "
            f"total_bits={total_bits:.1f}  ratio={ratio:.6f}  "
            f"rate={100/ratio:.4f}%  [{time.time()-t:.0f}s]")

    if "base" in want:
        say("\n--- ARM: base (no adapter) ---")
        run_arm("base", only=owin)

    # ---- attach the authors' adapter, unmerged ---------------------------
    say("\nattaching the published adapter exactly as the authors do:")
    say("   PeftModel.from_pretrained(model, lora_path, device_map=self.device)")
    from peft import PeftModel
    t0 = time.time()
    model = PeftModel.from_pretrained(model, LORA, device_map=torch.device(DEVICE))
    model.eval()
    info = capture_native_scale(model)
    say(f"  LoRA modules        = {info['n_modules']}")
    say(f"  native scaling      = {info['native_scaling_values']}  "
        f"(= lora_alpha / r = 32 / 8)")
    say(f"  -> 'lam' below is RELATIVE to that native magnitude")
    say(f"  [{time.time()-t0:.0f}s]")

    if "lora" in want:
        say("\n--- ARM: lora_live (lam = 1.0 x native, the authors' setting) ---")
        set_lora_scale(model, 1.0)
        run_arm("lora_lam=1.0", only=owin)

    # ---- scaling sweep ---------------------------------------------------
    if "scale" in want:
        for lam in lams:
            say(f"\n--- ARM: delta = {lam} x native ({lam*4.0:.2f} x raw B@A) "
                f"(windows={swin if swin else 'all'}) ---")
            set_lora_scale(model, lam)
            run_arm(f"lora_lam={lam}", only=swin)

    # ---- summary ---------------------------------------------------------
    say("\n" + "=" * 78)
    say(f"SUMMARY (authors' csv path, ctx {CTX}, first {len(batches)} window(s) "
        f"of medal_102285.csv)")
    say("=" * 78)
    say(f"{'arm':16s} {'windows':>10s} {'bits/token':>12s} {'total bits':>12s} "
        f"{'ratio':>10s} {'rate %':>9s} {'vs base (same set)':>19s}")
    ref = arms.get("base") or arms.get("lora_lam=1.0")
    ref_pw = ref["per_window"] if ref else {}
    for name, a_ in arms.items():
        if ref is not None and a_["windows"] and all(w in ref_pw for w in a_["windows"]):
            b_shared = sum(ref_pw[w] for w in a_["windows"])
            a_shared = sum(a_["per_window"][w] for w in a_["windows"])
            tag = f"{100 * (a_shared - b_shared) / b_shared:+19.4f}"
        else:
            tag = f"{'n/a (diff set)':>19s}"
        wtag = (f"{a_['windows'][0]}-{a_['windows'][-1]}"
                if len(a_["windows"]) > 1 else str(a_["windows"][0]))
        say(f"{name:16s} {wtag:>10s} {a_['bpt']:12.6f} {a_['bits']:12.1f} "
            f"{a_['ratio']:10.6f} {100/a_['ratio']:9.4f} {tag}")

    # ---- per-window detail where both base and adapter were measured ------
    if "base" in arms and "lora_lam=1.0" in arms:
        say("\nper-window detail (bits; positive delta = adapter costs MORE bits)")
        say(f"{'window':>6s} {'tokens':>7s} {'base':>11s} {'+LoRA':>11s} "
            f"{'delta bits':>11s} {'delta bits/tok':>15s} {'adapter':>9s}")
        bp, lp = arms["base"], arms["lora_lam=1.0"]
        for w in sorted(set(bp["per_window"]) & set(lp["per_window"])):
            t_ = bp["per_window_tokens"][w]
            d = lp["per_window"][w] - bp["per_window"][w]
            say(f"{w:6d} {t_:7d} {bp['per_window'][w]:11.1f} "
                f"{lp['per_window'][w]:11.1f} {d:+11.1f} {d/t_:+15.6f} "
                f"{'WORSE' if d > 0 else 'better':>9s}")
        nworse = sum(1 for w in sorted(set(bp["per_window"]) & set(lp["per_window"]))
                     if lp["per_window"][w] - bp["per_window"][w] > 0)
        ntot = len(set(bp["per_window"]) & set(lp["per_window"]))
        say(f"  -> adapter worse on {nworse} of {ntot} windows")
    say("")
    say("paper (MeDAL table)  LMCompress* base 9.08 %   LMCompress +LoRA 9.06 %")
    say("      -> the paper says the adapter HELPS by 0.02 pp")
    if "base" in arms and "lora_lam=1.0" in arms:
        b, lv = arms["base"], arms["lora_lam=1.0"]
        say(f"      this run: base {100/b['ratio']:.4f} %  +LoRA {100/lv['ratio']:.4f} %"
            f"   -> adapter {'HURTS' if lv['ratio'] < b['ratio'] else 'HELPS'} "
            f"by {abs(100/lv['ratio'] - 100/b['ratio']):.4f} pp")
    say(f"\ntotal wall clock {time.time()-t_start:.0f}s")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    print(f"\nwritten -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
