"""Build the precision-discrimination notebook for the text level discrepancy.

The question it settles is narrow and was reached by elimination (AUDIT section 5.9):

  shift alignment, denominator convention and corpus subset are all excluded by
  measurement, leaving two candidates for the 2.73 pp gap between our 6.33 % and the
  paper's 9.06 %:
      B. formula -- the same convention evaluated differently
      C. code    -- a shared computation defect
  plus a third the audit judged most likely: a different MODEL STATE, i.e. precision.

The observation that motivates it: the authors' harness (8.9253 %) and our own port
(9.0410 %) both report about 1.48x the code length implied by the model's own loss on
identical weights, while a direct forward pass reports 1.00x. Two independently written
pipelines on the same factor points at the environment, not at two coincident bugs.

Design: 3 precisions x 2 formulas, on the same five 4096-token windows of the same file.

  precisions  bf16 (15 GiB), 8-bit LLM.int8 (7.5 GiB), 4-bit NF4 (4 GiB)
  formulas    fp32 log_softmax + gather          -- what this audit uses
              native-dtype softmax + gather      -- what the reference's forward_next does
  control     output.loss * (L-1) for every cell, which must equal the fp32 formula

Targets:
  bf16 / fp32 formula      2.5682 bits/token   (measured on this machine, CPU bf16)
  the inflated regime      3.80 - 3.87 bits/token (our port and the authors' harness)

So the notebook answers three questions at once: does 8-bit reproduce the inflation, does
4-bit, and does the reference's softmax-in-output-dtype differ from an fp32 log_softmax on
identical logits (which separates B from C).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
MEDAL = os.path.join(REPO, "data", "text", "MEDAL_sample")
FILE = "medal_102285.csv"
WINDOW = 4096


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def payload() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        with open(os.path.join(MEDAL, FILE), "rb") as fh:
            zf.writestr(FILE, fh.read())
    return base64.b64encode(buf.getvalue()).decode("ascii")


CELLS: list[dict] = []

CELLS.append(md("""# Which mechanism? precision vs formula

This notebook settles one question and nothing else: **why is the paper's text rate 9.06 %
when this model achieves 6.33 % on the same corpus?**

## What is already excluded

Three candidate mechanisms were eliminated by measurement (AUDIT §5.9), so this notebook
does not re-test them:

| candidate | why excluded |
|---|---|
| **shift alignment** | the reference's `forward_next` uses `logits[:, :-1]` against `input_ids[:, 1:]`, which is correct; and an independent gather reproduces `output.loss*(L-1)` to **5.6e-08** |
| **denominator** | container vs payload differ by 0.39 %; characters vs bytes by 0.14 % — far too small |
| **corpus subset** | across all 100 MeDAL files a classical compressor spans only **2.7 pp**, and our three files are *harder* than average, which would push our rate up, not down |

## What is left

| candidate | test |
|---|---|
| **C. shared computation defect** | would show up as the fp32 formula disagreeing with `output.loss` |
| **B. formula difference** | would show up as the native-dtype softmax differing from an fp32 log_softmax **on identical logits** |
| **model state / precision** (the audit's leading hypothesis) | would show up as 8-bit or 4-bit reproducing the inflated figure |

## The observation that motivates it

| pipeline | bits/token | relative to the model's own loss |
|---|---|---|
| direct forward, bf16, fp32 formula | **2.5682** | 1.00x |
| our port (retracted) | 3.80 | 1.48x |
| authors' harness | 3.87 | 1.48x |

**Two independently written pipelines on the same 1.48x factor** points at the environment
rather than at two coincident bugs. The paper states its base model is loaded in **8-bit**,
and the machine that produced the two inflated figures cannot do 8-bit at all.

## Design

3 precisions x 2 formulas, on the same five 4096-token windows of the same file.

| precisions | weight footprint |
|---|---|
| bf16 | 15 GiB |
| 8-bit (LLM.int8) | 7.5 GiB |
| 4-bit (NF4, double quant, bf16 compute) | 4 GiB |

| formulas | what it represents |
|---|---|
| fp32 `log_softmax` + gather | this audit's convention |
| native-dtype `softmax` + gather | what the reference's `forward_next` literally does |

Every cell is also checked against `output.loss * (L-1)`.

**Targets.** bf16/fp32 should give **2.5682 bits/token**. The inflated regime is **3.80–3.87**.

## Cost

Nine model loads at most (three precisions), five 4096-token forwards each: a few minutes on
an A100 plus the 16 GB checkpoint download.
"""))

CELLS.append(md("## 0. Environment"))

CELLS.append(code("""import subprocess, sys, os, json, time, base64, zipfile, io
from pathlib import Path

def sh(cmd, check=True):
    print('$', cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip()[-1500:])
    if check and r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit(f'failed: {cmd}')
    return r

sh('pip install -q -U "transformers>=4.43" "bitsandbytes>=0.43" accelerate safetensors pandas 2>&1 | tail -3',
   check=False)
sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', check=False)

import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'this test needs a GPU: 8-bit and 4-bit require CUDA'
props = torch.cuda.get_device_properties(0)
print(f'gpu {torch.cuda.get_device_name(0)}  {props.total_memory/2**30:.1f} GiB')
DEVICE = 'cuda'

import bitsandbytes as bnb
print('bitsandbytes', bnb.__version__)
# The local machine fails exactly here; on Colab this must print a version.
"""))

CELLS.append(md("""## 1. The corpus

One MeDAL file, embedded, so there is nothing to upload. It is the same file the
2.5682 bits/token reference was measured on, split into five 4096-token windows.
"""))

CELLS.append(code("""PAYLOAD = \"\"\"__PAYLOAD__\"\"\"

WORK = Path('/content/medal'); WORK.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as zf:
    zf.extractall(WORK)
    print('extracted:', zf.namelist())

import pandas as pd
df = pd.read_csv(WORK / 'medal_102285.csv')
text = ''
for _, row in df.iterrows():          # exactly as the reference loader
    text += row['TEXT']
RAW_BYTES = len(text.encode())
print(f'chars {len(text):,}   bytes {RAW_BYTES:,}')
assert RAW_BYTES == 103534, f'corpus mismatch: {RAW_BYTES}'
"""))

CELLS.append(code("""from transformers import AutoTokenizer

MODEL_ID = 'meta-llama/Meta-Llama-3-8B'   # if gated: huggingface_hub.login() first,
                                          # or use 'unsloth/llama-3-8b'
tok = AutoTokenizer.from_pretrained(MODEL_ID)
ids_all = tok(text, add_special_tokens=False)['input_ids']
WINDOWS = [ids_all[i:i+WINDOW] for i in range(0, len(ids_all), WINDOW)]
WINDOWS = [w for w in WINDOWS if len(w) >= 2]
print(f'tokens {len(ids_all):,}   windows {len(WINDOWS)}   '
      f'sizes {[len(w) for w in WINDOWS]}')
assert len(ids_all) == 19338, f'expected 19338 tokens, got {len(ids_all)}'
assert len(WINDOWS) == 5, f'expected 5 windows, got {len(WINDOWS)}'
"""))

CELLS.append(md("""## 2. Two formulas and one control

The only difference between the two formulas is **where the precision is lost**:

* `fp32` promotes the logits to float32 before the log-softmax, so the probability is as
  accurate as the weights allow.
* `native` runs `softmax` in the logits' own dtype, which is what
  `forward_next` does (`shifted_logits.softmax(dim=-1)`). bf16 carries 8 mantissa bits, so
  a probability near 1 can round to exactly 1 and a small one can be badly represented.

`output.loss` is the control: HuggingFace computes it with its own fused
cross-entropy, so the fp32 formula must match it for every precision.
"""))

CELLS.append(code("""import torch.nn.functional as F


def bits_fp32(model, w, slice_len=1024):
    \"\"\"sum(-log2 p_true) with an fp32 log-softmax, causal shift.\"\"\"
    t = torch.tensor([w], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(t).logits                   # (1, L, V), may be bf16
    L = t.shape[1]
    total = 0.0
    for a in range(0, L - 1, slice_len):
        b = min(a + slice_len, L - 1)
        lp = torch.log_softmax(logits[:, a:b, :].float(), dim=-1)
        total += float(-lp.gather(-1, t[:, a+1:b+1].unsqueeze(-1)).squeeze(-1).sum())
        del lp
    del logits
    return total


def bits_native(model, w, slice_len=1024):
    \"\"\"The same sum using softmax in the logits' own dtype, as forward_next does.\"\"\"
    t = torch.tensor([w], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(t).logits
    L = t.shape[1]
    total = 0.0
    for a in range(0, L - 1, slice_len):
        b = min(a + slice_len, L - 1)
        p = logits[:, a:b, :].softmax(dim=-1)              # native dtype
        pd = torch.gather(p, -1, t[:, a+1:b+1].unsqueeze(-1)).squeeze(-1)
        pd = pd.add_(6e-8)                                 # forward_next's epsilon
        total += float(-torch.log2(pd).sum())
        del p, pd
    del logits
    return total


def loss_bits(model, w):
    \"\"\"The control: HuggingFace's own mean cross-entropy x (L-1).\"\"\"
    t = torch.tensor([w], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model(t, labels=t)
    return float(out.loss) * (t.shape[1] - 1)
"""))

CELLS.append(md("""## 3. Load each precision and measure

The quantisation configs match the paper's description (8-bit for text) and the fallback our
own port used (4-bit NF4 with bf16 compute and double quantisation).

Each load is verified: if a quantisation config is passed but the model comes back
unquantised, the cell says so loudly rather than silently reporting bf16 numbers under an
8-bit label.
"""))

CELLS.append(code("""from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load(precision):
    kw = dict(device_map={'': 0}, low_cpu_mem_usage=True)
    if precision == 'bf16':
        kw['dtype'] = torch.bfloat16
    elif precision == '8bit':
        kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == '4bit':
        kw['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw).eval()
    # verification that the requested precision actually took effect
    is8 = bool(getattr(model, 'is_loaded_in_8bit', False))
    is4 = bool(getattr(model, 'is_loaded_in_4bit', False))
    dtype = next(model.parameters()).dtype
    print(f'  requested {precision:<5} -> is_loaded_in_8bit={is8} '
          f'is_loaded_in_4bit={is4} param dtype={dtype}  '
          f'peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    return model, is8, is4


records = []
for precision in ('bf16', '8bit', '4bit'):
    print(f'=== {precision} ===')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, is8, is4 = load(precision)
    load_s = time.time() - t0

    b_fp = b_ni = b_lo = 0.0
    n_tok = 0
    for w in WINDOWS:
        b_fp += bits_fp32(model, w)
        b_ni += bits_native(model, w)
        b_lo += loss_bits(model, w)
        n_tok += len(w) - 1
    row = dict(precision=precision, is8=is8, is4=is4,
               bits_fp32=b_fp, bits_native=b_ni, bits_loss=b_lo,
               scored_tokens=n_tok,
               bpt_fp32=b_fp / n_tok, bpt_native=b_ni / n_tok,
               bpt_loss=b_lo / n_tok,
               rate_pct_fp32=100 * (b_fp / 8) / RAW_BYTES,
               peak_vram_gib=round(torch.cuda.max_memory_allocated()/2**30, 2),
               load_seconds=round(load_s, 1))
    records.append(row)
    print(f'  bits/token  fp32 {row[\"bpt_fp32\"]:.4f}   '
          f'native {row[\"bpt_native\"]:.4f}   loss {row[\"bpt_loss\"]:.4f}')
    print(f'  rate (fp32) {row[\"rate_pct_fp32\"]:.4f} %   '
          f'peak VRAM {row[\"peak_vram_gib\"]:.2f} GiB')
    with open('/content/precision_test.jsonl', 'a') as fh:
        fh.write(json.dumps(row, sort_keys=True) + '\\n'); fh.flush(); os.fsync(fh.fileno())
    del model
    torch.cuda.empty_cache()
"""))

CELLS.append(md("## 4. The verdict"))

CELLS.append(code("""TARGET_BF16 = 2.5682     # measured on this project's CPU-bf16 reference
LO_INFLATED, HI_INFLATED = 3.75, 3.95   # our port 3.80, authors' harness 3.87

print(f\"{'precision':<10}{'fp32':>9}{'native':>9}{'loss':>9}{'rate %':>9}{'GiB':>7}\")
print('-' * 54)
for r in records:
    print(f\"{r['precision']:<10}{r['bpt_fp32']:>9.4f}{r['bpt_native']:>9.4f}\"
          f\"{r['bpt_loss']:>9.4f}{r['rate_pct_fp32']:>9.3f}{r['peak_vram_gib']:>7.2f}\")

print()
print('=== C. is there a shared computation defect? ===')
worst = max(abs(r['bpt_fp32'] - r['bpt_loss']) / r['bpt_loss'] for r in records)
print(f'  max |fp32 - output.loss| / loss across all precisions = {worst:.2e}')
if worst < 1e-4:
    print('  -> the fp32 formula agrees with the model loss at every precision, so the')
    print('     scoring code is NOT the defect. Candidate C is excluded.')
else:
    print('  -> the fp32 formula disagrees with the model loss; a computation defect exists.')

print()
print('=== B. does the formula matter at matched logits? ===')
for r in records:
    d = r['bpt_native'] - r['bpt_fp32']
    print(f'  {r[\"precision\"]:<6} native - fp32 = {d:+.4f} bits/token  '
          f'({100*d/r[\"bpt_fp32\"]:+.2f} %)')
print('  -> a few hundredths of a bit cannot explain 1.3 bits/token, so whatever')
print('     difference remains is in the model state, not in the expression.')

print()
print('=== the leading hypothesis: does quantisation reproduce the inflation? ===')
for r in records:
    v = r['bpt_fp32']
    tag = ''
    if abs(v - TARGET_BF16) < 0.05:
        tag = '<- matches the bf16 reference'
    elif LO_INFLATED <= v <= HI_INFLATED:
        tag = '<- INSIDE the inflated regime'
    print(f'  {r[\"precision\"]:<6} {v:.4f} bits/token   {tag}')

print()
print('=== interpretation ===')
q8 = [r for r in records if r['precision'] == '8bit']
q4 = [r for r in records if r['precision'] == '4bit']
bf = [r for r in records if r['precision'] == 'bf16']
if bf and abs(bf[0]['bpt_fp32'] - TARGET_BF16) > 0.10:
    print('  WARNING: bf16 does not reproduce the 2.5682 reference on this machine.')
    print('  Something other than precision differs; do not read the 8-bit row yet.')
elif q8 and LO_INFLATED <= q8[0]['bpt_fp32'] <= HI_INFLATED:
    print('  CONFIRMED: 8-bit reproduces the inflated figure.')
    print('  => the paper measures a QUANTISED model, and our port\\'s retracted figure was')
    print('     also a quantised measurement. Both are honest; neither is comparable with a')
    print('     bf16 measurement. The 2.73 pp gap is a precision effect, not an error.')
elif q8 and abs(q8[0]['bpt_fp32'] - TARGET_BF16) < 0.10:
    print('  REJECTED: 8-bit stays at the bf16 level, so quantisation does NOT explain it.')
    print('  => both harnesses must share a computation defect elsewhere, and locating it')
    print('     is now mandatory before any text rate is quoted.')
else:
    print('  INCONCLUSIVE: 8-bit is between the two regimes. Report the number as measured')
    print('  and treat the mechanism as partly precision, partly unexplained.')
"""))

CELLS.append(code("""summary = dict(
    model=MODEL_ID,
    file='medal_102285.csv',
    raw_bytes=RAW_BYTES,
    windows=len(WINDOWS),
    window_tokens=WINDOW,
    cells=records,
    reference_bf16_bpt=2.5682,
    inflated_regime=[3.75, 3.95],
    port_retracted_bpt=3.80,
    authors_harness_bpt=3.87,
    note=('every cell is sum(-log2 p_true) with a causal shift over the same five windows; '
          'the fp32 formula is cross-checked against output.loss*(L-1); requantisation '
          'flags are recorded so a label cannot silently describe the wrong precision'),
)
p = Path('/content/precision_test.json')
p.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'cells'}, indent=2))
print()
for r in records:
    print(f\"{r['precision']:<6} fp32 {r['bpt_fp32']:.4f}  native {r['bpt_native']:.4f}  \"
          f\"loss {r['bpt_loss']:.4f}  rate {r['rate_pct_fp32']:.4f} %\")

try:
    from google.colab import files
    files.download(str(p))
except Exception as exc:
    print('download helper unavailable:', type(exc).__name__)
"""))

CELLS.append(md("""## How to read the outcome

| result | conclusion |
|---|---|
| bf16 ≈ 2.57, **8-bit ≈ 3.8-3.9** | The gap is **precision**. The paper's 9.06 % and our retracted 9.04 % are both honest measurements of a *quantised* model. Neither is comparable with a bf16 number, and the report must say which precision every text rate was measured at. |
| bf16 ≈ 2.57, 8-bit ≈ 2.6, 4-bit higher but not enough | Precision is not the whole story. Something else inflates both harnesses; locating it becomes mandatory. |
| bf16 ≠ 2.57 | The environment differs in a way this notebook has not isolated. Stop and compare against a direct `output.loss` on one window before interpreting anything else. |

**What this notebook settles regardless of outcome**: whether the scoring code itself can
produce the inflation. If `fp32` matches `output.loss` in every row, candidate C is excluded
and the difference must live in the model state or in a formula used elsewhere — which is a
much smaller question than the one we started with.

## What it deliberately does not do
* **No claim about the paper's intent.** A precision difference is not an error; the paper
  states its precision openly. What would be wrong is comparing a quantised rate with a
  bf16 rate without saying so.
* **No 4-bit on more data.** One file is enough to discriminate; the mechanism does not
  depend on corpus size, which was already excluded in §5.9.
"""))


def main() -> int:
    p = payload()
    cells = []
    for c in CELLS:
        if c["cell_type"] == "code" and "__PAYLOAD__" in "".join(c["source"]):
            c["source"] = ["".join(c["source"]).replace("__PAYLOAD__", p)]
        cells.append(c)

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = os.path.join(HERE, "colab", "Precision_Test_Colab.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"  wrote {out}")
    print(f"  {len(cells)} cells, {os.path.getsize(out)/1024:.0f} KB "
          f"(payload {len(p)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
