"""Build the Colab notebook for the paired text experiment at context 8192.

Why this notebook exists
------------------------
The report measures the text fine-tuning result at context 4096 because our port's
default is 4096, then has to carry a caveat: the paper's own Supplementary Table 9 puts
the context effect at ~0.3 pp, which is the same size as the adapter penalty being
reported (+0.312 pp). So the two effects cannot be separated from the existing data.

That is fixable, and this notebook fixes it with a 2x2 design:

    arm  x  ctx  =  {base, +LoRA} x {4096, 8192}

Four numbers turn two confounded effects into four separable ones. The comparison the
report needs is simply whether the LoRA penalty persists at the published context.

Design decisions, each bought with a failure earlier in this project
--------------------------------------------------------------------
* **bf16, no quantisation.** The paper's 15.49 GB peak GPU figure for text is an 8-bit
  number. Our local machine cannot do 8-bit (bitsandbytes has no CUDA binary), which is
  a real constraint but one that applies equally at 4096 and 8192. An A100 40 GB holds
  the model at bf16 outright, so quantisation is removed as a variable rather than
  argued about. The notebook records the actual dtype and peak VRAM per cell.
* **One model, `disable_adapter()`.** Loading base and LoRA separately invites the
  objection that the two loads differed. Toggling the adapter on one PeftModel makes the
  base weights provably identical between arms.
* **The rate is `sum(-log2 p_true)`, never the coder's quantised distribution.** Same
  convention as every other number in this project. The arithmetic coder is not run; it
  cannot change the reported rate.
* **Cross-checked against the model's own loss.** For a window of L tokens the
  causal-shift code length must equal `output.loss * (L-1)`. The notebook asserts this,
  because a mis-shifted scorer is exactly the defect that invalidated an earlier audio
  finding in this project.
* **Data embedded, adapter uploaded.** The three MeDAL test files gzip to 110 KB
  (143 KB base64), so they ride inside the notebook. Only the 40 MB adapter needs
  uploading. No 16 GB model upload, no dataset download.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
MEDAL_DIR = os.path.join(REPO, "data", "text", "MEDAL_sample")
FILES = ["medal_102285.csv", "medal_103851.csv", "medal_108122.csv"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_payload() -> str:
    """gzip the three CSVs into one archive and return base64 text."""
    import io as _io
    import zipfile

    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in FILES:
            with open(os.path.join(MEDAL_DIR, name), "rb") as fh:
                zf.writestr(name, fh.read())
    return base64.b64encode(buf.getvalue()).decode("ascii")


CELLS: list[dict] = []

CELLS.append(md("""# LMCompress - text, paired at context 4096 and 8192

Measures the text modality of Li et al., *Lossless data compression by large models*
(NMI 7:794-799, 2025), in a **2x2 design** that separates the LoRA adapter effect from
the context-length effect.

## Why the 2x2

The reproduction measured the fine-tuned text result at **context 4096**, then had to
carry a caveat: the paper's Supplementary Table 9 reports

| context | rate |
|---|---|
| 8192 | 9.06 % |
| 6144 | 9.16 % |
| 4096 | **9.36 %** |

so the context effect is about **0.30 pp** -- essentially the same size as the adapter
penalty being reported (**+0.312 pp**). From a 4096-only measurement the two cannot be
told apart.

Four measurements settle it:

|  | ctx 4096 | ctx 8192 |
|---|---|---|
| **base** | known: 9.0410 % | **new** |
| **+LoRA** | known: 9.3529 % | **new** |

* If the LoRA penalty **persists** at 8192, the adapter really is harmful here and
  context is not the explanation.
* If the penalty **shrinks** at 8192, the earlier finding was a context artefact and
  must be withdrawn.

## What you need to supply

| item | size | how |
|---|---|---|
| base model | ~16 GB | downloaded automatically (Llama-3-8B, ungated mirror with fallbacks) |
| LoRA adapter | **40 MB** | **you upload it** (`adapter_model.safetensors` + `adapter_config.json`) |
| MeDAL test files | 313 KB | **embedded in this notebook**, nothing to do |

## Cost

An A100 40 GB holds the model at bf16 with room to spare, so no quantisation is needed.
The whole corpus is only **57 547 tokens**, so all four cells together are a few minutes
of GPU time plus the ~16 GB model download.
"""))

CELLS.append(md("## 0. Environment"))

CELLS.append(code("""import subprocess, sys, os, json, time, math, base64, zipfile, io
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
print('python', sys.version.split()[0])

def sh(cmd, check=True):
    print('$', cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip()[-1500:])
    if check and r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit(f'command failed: {cmd}')

sh('pip install -q -U "transformers>=4.43" "peft>=0.11" accelerate safetensors pandas 2>&1 | tail -3',
   check=False)
sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', check=False)

import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu  ', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print(f'vram {props.total_memory/2**30:.1f} GiB, sm_{props.major}{props.minor}')
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
    print('WARNING: no GPU. bf16 Llama-3-8B needs ~15 GiB of VRAM; on CPU this will be '
          'extremely slow. Switch the runtime to A100 before continuing.')
"""))

CELLS.append(md("""## 1. The base model

Llama-3-8B. The canonical repo is gated, so several mirrors are tried in order and the
first that loads is used -- the point is to obtain *these weights*, and the notebook
verifies that rather than trusting the repo name.

`max_position_embeddings` is **8192**, so context 8192 is native and needs no RoPE
scaling.
"""))

CELLS.append(code("""from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CANDIDATES = [
    'meta-llama/Meta-Llama-3-8B',        # canonical, gated (works if you have a token)
    'unsloth/llama-3-8b',                # ungated mirror
    'NousResearch/Meta-Llama-3-8B',      # ungated mirror
    'gradientai/Llama-3-8B-Instruct-262k-80B',  # last resort, wrong tune -- see check below
]

BASE_ID, tok = None, None
for cand in CANDIDATES:
    try:
        cfg = AutoConfig.from_pretrained(cand)
        # Structural identity of Llama-3-8B. Rejects 3B/70B and mistuned variants.
        ok = (cfg.hidden_size == 4096 and cfg.num_hidden_layers == 32
              and cfg.intermediate_size == 14336 and cfg.num_attention_heads == 32
              and cfg.vocab_size in (128000, 128256))
        if not ok:
            print(f'  skip {cand}: not Llama-3-8B shape '
                  f'(h={cfg.hidden_size}, L={cfg.num_hidden_layers}, V={cfg.vocab_size})')
            continue
        tok = AutoTokenizer.from_pretrained(cand)
        BASE_ID = cand
        print(f'  using {cand}')
        break
    except Exception as exc:
        print(f'  skip {cand}: {type(exc).__name__}')

assert BASE_ID is not None, (
    'Could not resolve a Llama-3-8B checkpoint. If you have a HF token, run '
    '`from huggingface_hub import login; login()` first, or put the model in Drive '
    'and set BASE_ID to that path.')

cfg = AutoConfig.from_pretrained(BASE_ID)
print()
print('hidden   :', cfg.hidden_size)
print('layers   :', cfg.num_hidden_layers)
print('heads    :', cfg.num_attention_heads)
print('vocab    :', cfg.vocab_size)
print('max_pos  :', cfg.max_position_embeddings, '<- context 8192 is native')
assert cfg.max_position_embeddings >= 8192, 'model cannot do context 8192 natively'

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    print('pad_token set to eos:', tok.pad_token)
"""))

CELLS.append(code("""t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(
    BASE_ID, torch_dtype=torch.bfloat16, device_map={'': 0},
    low_cpu_mem_usage=True)
base.eval()
print(f'loaded in {time.time()-t0:.0f}s')
print('dtype :', next(base.parameters()).dtype)
n = sum(p.numel() for p in base.parameters())
print(f'params: {n/1e9:.2f} B   bf16 weights {n*2/2**30:.2f} GiB')
print(f'peak VRAM after load: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
"""))

CELLS.append(md("""## 2. The LoRA adapter

Upload `adapter_model.safetensors` and `adapter_config.json` from the capsule's
`data/model/finetuned_llama/Llama-3-8B_lora/`. Roughly 40 MB.

The published config is r=8, alpha=32, dropout 0.05, on all seven projection modules
across all 32 layers -- 20,971,520 parameters. The cell prints the config it actually
loaded so a wrong upload is visible immediately.

The adapter's `base_model_name_or_path` points at a path on the authors' machine, which
does not exist here; it is rewritten to the resolved checkpoint id so the loader does not
try to follow it.
"""))

CELLS.append(code("""import shutil
from pathlib import Path

LORA_DIR = Path('/content/lora')
LORA_DIR.mkdir(parents=True, exist_ok=True)

have = (LORA_DIR / 'adapter_model.safetensors').exists() and \\
       (LORA_DIR / 'adapter_config.json').exists()

if not have:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as exc:
        print('Drive not mounted:', type(exc).__name__)

    # Look for an already-uploaded copy before asking again.
    for cand in Path('/content/drive/MyDrive').rglob('adapter_model.safetensors'):
        if cand.stat().st_size > 10e6:
            print('found on Drive:', cand)
            shutil.copy2(cand, LORA_DIR / 'adapter_model.safetensors')
            cfgp = cand.parent / 'adapter_config.json'
            if cfgp.exists():
                shutil.copy2(cfgp, LORA_DIR / 'adapter_config.json')
            break

if not (LORA_DIR / 'adapter_model.safetensors').exists():
    from google.colab import files
    print('Upload adapter_model.safetensors and adapter_config.json (both at once):')
    up = files.upload()
    for name, blob in up.items():
        tgt = LORA_DIR / Path(name).name
        tgt.write_bytes(blob)
        print('  saved', tgt, f'{len(blob)/1e6:.1f} MB')

assert (LORA_DIR / 'adapter_model.safetensors').exists(), 'adapter weights missing'
assert (LORA_DIR / 'adapter_config.json').exists(), 'adapter_config.json missing'

# Rewrite the authors' absolute path so the loader does not chase it.
acfg = json.loads((LORA_DIR / 'adapter_config.json').read_text())
print('adapter_config as shipped:')
for k in ('r', 'lora_alpha', 'lora_dropout', 'target_modules', 'task_type'):
    print(f'  {k:16}: {acfg.get(k)}')
acfg['base_model_name_or_path'] = BASE_ID
(LORA_DIR / 'adapter_config.json').write_text(json.dumps(acfg, indent=2))
print('rewrote base_model_name_or_path ->', BASE_ID)
"""))

CELLS.append(code("""from peft import PeftModel

t0 = time.time()
model = PeftModel.from_pretrained(base, str(LORA_DIR), device_map={'': 0})
model.eval()
print(f'peft model ready in {time.time()-t0:.0f}s')

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
adapter_n = sum(v.numel() for k, v in model.named_parameters() if 'lora_' in k)
print(f'adapter parameters: {adapter_n:,}   '
      f'(published: 20,971,520 -> match: {adapter_n == 20971520})')
assert adapter_n > 0, 'no LoRA parameters found -- the adapter did not load'
print(f'peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
"""))

CELLS.append(md("""## 3. The corpus

The paper's MeDAL test material, as three of the capsule's sample CSVs, concatenated
exactly the way the reference loader does it -- every `TEXT` cell appended with no
separator -- so the token stream is the one the earlier 9.0410 % figure was measured on.

Embedded here as a base64 zip, so there is nothing to download or upload.
"""))

CELLS.append(code("""import pandas as pd

PAYLOAD = \"\"\"__PAYLOAD__\"\"\"

WORK = Path('/content/medal'); WORK.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as zf:
    zf.extractall(WORK)
    names = sorted(zf.namelist())
print('extracted:', names)

FILES = ['medal_102285.csv', 'medal_103851.csv', 'medal_108122.csv']
texts = {}
for f in FILES:
    df = pd.read_csv(WORK / f)
    s = ''
    for _, row in df.iterrows():          # exactly as the reference loader
        s += row['TEXT']
    texts[f] = s
    print(f'  {f:<20} {len(s):>8,} chars   {len(s.encode()):>8,} bytes')

TOTAL_RAW = sum(len(t.encode()) for t in texts.values())
print(f'  total raw bytes: {TOTAL_RAW:,}')
assert TOTAL_RAW == 312053, (
    f'corpus differs from the one the 9.0410 %% baseline was measured on '
    f'(expected 312053 bytes, got {TOTAL_RAW})')
"""))

CELLS.append(code("""tokenised = {}
print(f\"{'file':<22}{'tokens':>9}{'win@4096':>11}{'win@8192':>11}\")
for f, s in texts.items():
    ids = tok(s, add_special_tokens=False)['input_ids']
    tokenised[f] = ids
    print(f'{f:<22}{len(ids):>9,}{math.ceil(len(ids)/4096):>11}{math.ceil(len(ids)/8192):>11}')

N_TOK = sum(len(v) for v in tokenised.values())
W4 = sum(math.ceil(len(v)/4096) for v in tokenised.values())
W8 = sum(math.ceil(len(v)/8192) for v in tokenised.values())
print()
print(f'total tokens      : {N_TOK:,}')
print(f'windows @4096     : {W4}')
print(f'windows @8192     : {W8}')
print(f'scored @4096      : {N_TOK-W4:,}  (one unscored first token per window)')
print(f'scored @8192      : {N_TOK-W8:,}')
print(f'boundary overhead : {100*(W4-W8)/N_TOK:.5f} % of tokens')

# Fallback: if the downloaded tokenizer disagrees, the tokenizer the model was actually
# trained with may be available in the adapter upload (checkpoint-5120/tokenizer.json).
# The two local tokenizers differ only in pad_token and give identical ids, but this
# removes a hard stop if an HF mirror ships a different vocabulary.
if N_TOK != 57547:
    alt = LORA_DIR / 'tokenizer.json'
    if alt.exists():
        print(f'token count {N_TOK:,} != 57,547 -- retrying with {alt}')
        from transformers import AutoTokenizer as _AT
        tok = _AT.from_file(str(alt))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tokenised = {f: tok(s, add_special_tokens=False)['input_ids']
                     for f, s in texts.items()}
        N_TOK = sum(len(v) for v in tokenised.values())
        W4 = sum(math.ceil(len(v)/4096) for v in tokenised.values())
        W8 = sum(math.ceil(len(v)/8192) for v in tokenised.values())
        print(f'  retry gives {N_TOK:,} tokens, windows {W4}/{W8}')

assert N_TOK == 57547, (
    f'expected 57547 tokens, got {N_TOK}. The base tokenizer does not match the one '
    f'this corpus was measured with. Upload checkpoint-5120/tokenizer.json alongside '
    f'the adapter and re-run, or set BASE_ID to a checkpoint whose tokenizer matches.')
assert W4 == 15 and W8 == 9, f'expected 15/9 windows, got {W4}/{W8}'
print()
print('The boundary-token difference between the two contexts is 0.005 % of the corpus,')
print('so the ~0.30 pp context effect in the paper is a real model effect and not an')
print('artefact of merging windows.')
"""))

CELLS.append(md("""## 4. Scoring convention

For a window of `L` tokens, position `i` predicts token `i+1`, so `L-1` tokens are
scored and the first token of each window is coded at no cost -- the same convention as
the rest of this reproduction.

The reported rate is `sum(-log2 p_true)` on the model's own float32 probabilities divided
by the window's raw byte count. The arithmetic coder is **not** run: it cannot change the
reported rate (its quantisation reserve only affects `ac_bits`), and running it for
230 000 tokens would dominate the runtime -- in the video work it was 78 % of wall time.

**Cross-check.** For the same window, `output.loss * (L-1)` must equal the computed code
length, because HuggingFace's causal-LM loss shifts by one internally. The cell below
asserts this. A mis-shifted scorer is precisely the defect that invalidated an earlier
audio finding in this project, so it is checked rather than assumed.
"""))

CELLS.append(code("""def window_bits(w, slice_len=2048):
    \"\"\"Causal-shift code length in bits for one window of token ids.\"\"\"
    t = torch.tensor([w], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(t).logits                 # (1, L, V) bf16
    L = t.shape[1]
    total = 0.0
    for a in range(0, L - 1, slice_len):
        b = min(a + slice_len, L - 1)
        seg = logits[:, a:b, :].float()
        lp = torch.log_softmax(seg, dim=-1)
        tgt = t[:, a+1:b+1]
        total += float(-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())
        del seg, lp
    del logits
    return total


# --- self-check on the first window of the first file --------------------- #
_w = tokenised[FILES[0]][:4096]
_b = window_bits(_w)
_t = torch.tensor([_w], dtype=torch.long, device=DEVICE)
with torch.no_grad():
    _out = model(_t, labels=_t)
_loss_bits = float(_out.loss) * (len(_w) - 1)
print(f'gather bits = {_b:.4f}')
print(f'loss*(L-1)  = {_loss_bits:.4f}')
print(f'rel. diff   = {abs(_b-_loss_bits)/_b:.2e}')
assert abs(_b - _loss_bits) / _b < 1e-4, 'scorer disagrees with the model loss'
print('scorer validated against output.loss')
"""))

CELLS.append(md("""## 5. Two axes, because the two "context lengths" are not the same quantity

This is the correction that makes this notebook worth running.

The reference builds the text corpus as a **string** and windows it with

```python
def split(s, max_length):
    return [s[i : i + max_length] for i in range(0, len(s), max_length)]
...
split(self.data[0], context_window_size)      # self.data[0] is text_str
```

so `--context_window 8192` means **8192 characters**, not tokens. On MeDAL one character
is about 0.18 Llama-3 tokens, so the paper's three settings are:

| paper's context | unit | mean tokens/window |
|---|---|---|
| 4096 | characters | 739 |
| 6144 | characters | 1129 |
| 8192 | characters | **1476** |

Our port windows by **tokens**, so its "context 4096" is **3836 tokens per window** --
**2.6x more context than the paper's largest setting**. That is why our base rate came out
at 9.0410 %, near the paper's 8192 row (9.06 %) rather than its 4096 row (9.36 %): it was
never the same experiment.

So both axes are measured. **Axis A** (character windows, 4096/6144/8192) is directly
comparable with Supplementary Table 9, and the authors' own harness at 8192 characters
reported base **8.9253 %** / +LoRA **9.5154 %** on this corpus, which gives Axis A an
independent check. **Axis B** (token windows, 4096/8192) connects to our port's existing
9.0410 % / 9.3529 %.

Ten cells: 2 arms x (3 character + 2 token contexts). The corpus is 57 547 tokens, so
this stays a few minutes of GPU time.
"""))

CELLS.append(code("""def windows_chars(text, ctx_chars):
    \"\"\"Split a string into ctx_chars chunks, then tokenise each -- the reference way.

    Each chunk is tokenised separately, so cross-boundary merges are lost and the token
    total runs a few tens above tokenising the whole file at once.
    \"\"\"
    out = []
    for i in range(0, len(text), ctx_chars):
        ids = tok(text[i:i+ctx_chars], add_special_tokens=False)['input_ids']
        if len(ids) >= 2:
            out.append(ids)
    return out


def windows_tokens(ids, ctx_tokens):
    \"\"\"Split an id list into ctx_tokens chunks -- our port's convention.\"\"\"
    return [ids[i:i+ctx_tokens] for i in range(0, len(ids), ctx_tokens)
            if len(ids[i:i+ctx_tokens]) >= 2]


AXIS_A = {}
for ctx in (4096, 6144, 8192):
    wins = []
    for f in FILES:
        wins.extend(windows_chars(texts[f], ctx))
    AXIS_A[ctx] = wins
    print(f'A chars {ctx:<5}: {len(wins):>3} windows, '
          f'mean {sum(len(w) for w in wins)/len(wins):7.1f} tokens')

AXIS_B = {}
for ctx in (4096, 8192):
    wins = []
    for f in FILES:
        wins.extend(windows_tokens(tokenised[f], ctx))
    AXIS_B[ctx] = wins
    print(f'B tokens {ctx:<4}: {len(wins):>3} windows, '
          f'mean {sum(len(w) for w in wins)/len(wins):7.1f} tokens')

assert [len(AXIS_A[c]) for c in (4096, 6144, 8192)] == [78, 51, 39], \\
    f'char window counts unexpected: {[len(AXIS_A[c]) for c in (4096,6144,8192)]}'
assert [len(AXIS_B[c]) for c in (4096, 8192)] == [15, 9], \\
    f'token window counts unexpected: {[len(AXIS_B[c]) for c in (4096,8192)]}'
print()
print('window counts match the reference arithmetic (78/51/39 chars, 15/9 tokens)')
"""))

CELLS.append(md("""## 6. Measure all ten cells

One model, adapter toggled with `disable_adapter()`, so the base arm runs the identical
weights and the pairing is structural rather than hoped for. Every cell records the rate,
window count, mean context actually used, dtype and peak VRAM, and is appended to a
JSONL as it finishes.
"""))

CELLS.append(code("""OUT = Path('/content/text_paired.jsonl')
if OUT.exists():
    OUT.unlink()


def measure(windows, use_lora):
    bits = 0.0
    for w in windows:
        if use_lora:
            bits += window_bits(w)
        else:
            with model.disable_adapter():
                bits += window_bits(w)
    return bits


records = []
for use_lora in (False, True):
    arm = '+LoRA' if use_lora else 'base'
    for axis, table in (('A_chars', AXIS_A), ('B_tokens', AXIS_B)):
        for ctx in sorted(table):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            bits = measure(table[ctx], use_lora)
            rate = 100 * (bits / 8) / TOTAL_RAW
            wins = table[ctx]
            row = dict(arm=arm, axis=axis, ctx=ctx, bits=bits, windows=len(wins),
                       mean_window_tokens=round(sum(len(w) for w in wins)/len(wins), 1),
                       rate_pct=rate,
                       dtype=str(next(base.parameters()).dtype),
                       peak_vram_gib=round(torch.cuda.max_memory_allocated()/2**30, 2),
                       seconds=round(time.time()-t0, 1))
            records.append(row)
            with open(OUT, 'a') as fh:
                fh.write(json.dumps(row, sort_keys=True) + '\\n')
                fh.flush(); os.fsync(fh.fileno())
            print(f'{arm:<6} {axis} ctx {ctx:<5} rate {rate:7.4f} %  '
                  f'windows {len(wins):>3}  mean {row["mean_window_tokens"]:6.1f} tok  '
                  f'{row["peak_vram_gib"]:.2f} GiB  {row["seconds"]:.0f}s', flush=True)
"""))

CELLS.append(md("## 7. The result"))

CELLS.append(code("""def g(axis, ctx, arm):
    for r in records:
        if r['axis'] == axis and r['ctx'] == ctx and r['arm'] == arm:
            return r
    raise KeyError((axis, ctx, arm))


print('=== Axis A: character windows (comparable with SI Table 9) ===')
print(f\"{'':<10}{'4096':>11}{'6144':>11}{'8192':>11}    paper 9.36 / 9.16 / 9.06\")
for arm in ('base', '+LoRA'):
    print(f'{arm:<10}' + ''.join(
        f'{g("A_chars", c, arm)["rate_pct"]:>11.4f}' for c in (4096, 6144, 8192)))
print()
print('=== Axis B: token windows (comparable with our port) ===')
print(f\"{'':<10}{'4096':>11}{'8192':>11}    our port 9.0410 / 9.3529\")
for arm in ('base', '+LoRA'):
    print(f'{arm:<10}' + ''.join(
        f'{g("B_tokens", c, arm)["rate_pct"]:>11.4f}' for c in (4096, 8192)))

print()
print('=== adapter penalty on every context (pp) ===')
pen = {}
for axis, ctxs in (('A_chars', (4096, 6144, 8192)), ('B_tokens', (4096, 8192))):
    for c in ctxs:
        b = g(axis, c, 'base')['rate_pct']
        l = g(axis, c, '+LoRA')['rate_pct']
        pen[(axis, c)] = l - b
        print(f'  {axis} ctx {c:<5} base {b:7.4f}  +LoRA {l:7.4f}  penalty {l-b:+.4f}')

print()
print('=== validation against independent measurements ===')
a8192 = g('A_chars', 8192, 'base')['rate_pct']
b4096 = g('B_tokens', 4096, 'base')['rate_pct']
print(f'  A/@8192 base {a8192:.4f} %  vs authors harness 8.9253 %  '
      f'(delta {a8192-8.9253:+.4f} pp)')
print(f'  B/@4096 base {b4096:.4f} %  vs our port 9.0410 %          '
      f'(delta {b4096-9.0410:+.4f} pp)')
print(f'  -> Axis A agreement: {\"OK\" if abs(a8192-8.9253)<0.15 else \"MISMATCH\"}'
      f'   Axis B agreement: {\"OK\" if abs(b4096-9.0410)<0.15 else \"MISMATCH\"}')
if abs(a8192 - 8.9253) >= 0.15 or abs(b4096 - 9.0410) >= 0.15:
    print('  WARNING: an axis does not reproduce its independent reference. Check the')
    print('  corpus bytes, the window unit, and the causal shift before reading further.')

print()
print('=== the question this settles ===')
vals = list(pen.values())
print(f'  penalties span {min(vals):+.4f} .. {max(vals):+.4f} pp over five contexts')
if all(v > 0 for v in vals):
    print('  -> the adapter is HARMFUL at every context, on both axes.')
    print('     Context length is not the explanation. The earlier +0.312 pp finding')
    print('     stands, and the spread bounds what context could ever have contributed:')
    print(f'     at most {max(vals)-min(vals):.4f} pp.')
elif all(v < 0 for v in vals):
    print('  -> the adapter HELPS at every context. Our earlier finding was wrong and')
    print('     must be withdrawn.')
else:
    print('  -> the sign DEPENDS on context. The earlier single-context finding cannot')
    print('     be generalised; the report must say so with these numbers attached.')

print()
print('=== why our 4096 was never the paper\\'s 4096 ===')
print(f'  paper ctx 4096 =  739 tokens/window -> {g("A_chars",4096,"base")["rate_pct"]:.4f} %')
print(f'  our   ctx 4096 = 3836 tokens/window -> {g("B_tokens",4096,"base")["rate_pct"]:.4f} %')
"""))

CELLS.append(md("## 8. Save"))

CELLS.append(code("""summary = dict(
    base_model=BASE_ID,
    adapter_params=adapter_n,
    corpus_files=FILES,
    corpus_bytes=TOTAL_RAW,
    cells=records,
    penalties={f'{a}@{c}': round(g(a, c, '+LoRA')['rate_pct']
                               - g(a, c, 'base')['rate_pct'], 6)
               for a, cs in (('A_chars', (4096, 6144, 8192)),
                             ('B_tokens', (4096, 8192))) for c in cs},
    published_table9=dict(ctx8192=9.06, ctx6144=9.16, ctx4096=9.36),
    authors_harness=dict(base_8192chars=8.9253, lora_8192chars=9.5154),
    our_port=dict(base_4096tokens=9.0410, lora_4096tokens=9.3529),
    window_units=("Axis A windows the text by CHARACTERS, matching the reference "
                  "implementation's split() on a string; Axis B windows by tokens, "
                  'matching our port'),
    note=('reported rate is sum(-log2 p_true) on the model float32 probabilities; the '
          'arithmetic coder is not run because it cannot change the reported rate; the '
          'base arm uses disable_adapter() on the same PeftModel so the weights are '
          'identical between arms'),
)
SUM = Path('/content/text_paired.json')
SUM.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

try:
    from google.colab import files
    files.download(str(SUM))
except Exception as exc:
    print('download helper unavailable:', type(exc).__name__)
"""))

CELLS.append(md("""## If something goes wrong

| symptom | cause |
|---|---|
| `Could not resolve a Llama-3-8B checkpoint` | all mirrors gated. Run `from huggingface_hub import login; login()` with a token that has Llama-3 access, then re-run cell 4. |
| CUDA out of memory on load | you are not on a 40 GB GPU. The model is 15 GiB in bf16; a T4 (16 GB) will not hold it with room for activations at context 8192. |
| `base @4096` far from 9.0410 % | wrong tokenizer or corpus. The corpus byte count is asserted at 312 053, so the census is fixed; if the bytes match and the rate does not, the scorer is shifted. |
| adapter parameters != 20,971,520 | the wrong adapter was uploaded, or only part of the directory. |
| very slow | the runtime is on CPU. Check the first cell printed a GPU name. |

## What this notebook deliberately does not do

* **No retraining.** The "just train more" hypothesis would need MeDAL's ~180 MB training
  split, which the capsule does not ship. Training on these three test files would be
  circular -- more epochs would compress the very data being measured -- so it is not
  attempted rather than attempted badly.
* **No quantisation.** bf16 throughout, because the A100 does not need 8-bit and mixing
  a quantisation change into a context experiment would add a third variable.
"""))


def main() -> int:
    payload = build_payload()
    cells = []
    for c in CELLS:
        cells.append(c)

    # substitute the payload placeholder
    for c in cells:
        if c["cell_type"] == "code" and "__PAYLOAD__" in "".join(c["source"]):
            c["source"] = ["".join(c["source"]).replace("__PAYLOAD__", payload)]

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
    out = os.path.join(HERE, "colab", "Text_Paired_Context_Colab.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    size = os.path.getsize(out)
    print(f"  wrote {out}")
    print(f"  {len(cells)} cells, {size/1024:.0f} KB "
          f"(payload {len(payload)/1024:.0f} KB base64)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
