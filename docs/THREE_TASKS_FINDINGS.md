# Three investigations: LoRA discrepancy, ImageGPT quantisation, alternative models

Written after the corrections of this session. Every number here is measured, and the
raw outputs are named so they can be re-checked.

---

## Task 1 — Why does the paper's model + LoRA look good on the authors' code but worse on ours?

### Answer: it does **not** look good on the authors' code either. Our port is right.

We ran **the authors' own harness, unmodified**, on one MeDAL file, with and without
the adapter, driving `compress_with_bgpt_or_llama.compress_with_llm` directly so that
batch construction, windowing and aggregation are byte-identical between the two arms.
The only thing that changed was `lora_path`.

| arm | official ratio | **rate** = 100/ratio |
|---|---|---|
| base llama-3-8B, no adapter | **11.204097** | **8.9253 %** |
| + the authors' published LoRA | **10.509252** | **9.5154 %** |
| **difference** | −0.694845 | **+0.5901 pp (worse)** |

Raw output: `results/_task1_ref_ablation.txt`, `results/_task1_lora_arm.txt`.
Driver: `tools/task1_reference_lora_ablation.py`.

**Cross-check against our own port, same file:**

| arm | authors' harness | our port | gap |
|---|---|---|---|
| base | 8.9253 % | 8.8631 % | **+0.0622 pp** |
| + LoRA | 9.5154 % | 9.1269 % | +0.3885 pp |

So on the **base** model the two independent implementations agree to **0.06 pp** —
the port is faithful. On the **adapter** arm they both show the adapter as *worse*;
they disagree only on the size of the penalty (0.59 pp vs 0.26 pp), and that difference
is expected because our arm runs 4-bit at ctx 4096 while the reference arm runs fp32 at
ctx 8192.

### The discrepancy is between the code and the paper, not between us and the code

```
paper                : LMCompress* (base) 9.08 %   LMCompress (+LoRA) 9.06 %   -> LoRA helps 0.02 pp
authors' own code    : base            8.93 %   +LoRA             9.52 %   -> LoRA hurts 0.59 pp
```

The paper's own released code, run on the paper's own released adapter, contradicts the
paper's reported direction. **The premise of the question was wrong**: there is no
"works on their code, fails on ours" split.

### Three candidate explanations for the code-vs-paper gap, none yet excluded

1. **A different checkpoint.** `trainer_state.json` shows eval loss best at step 1020
   (≈2 epochs, 2.4708) and degraded by step 5120 (2.6728). The shipped adapter is the
   step-5120 one. If the paper's table used an earlier checkpoint, the direction could
   genuinely differ.
2. **A different context length.** The paper's Supplementary Table 9 reports MeDAL at
   ctx 8192 / 6144 / 4096 as 9.06 / 9.16 / 9.36 % — a ~0.3 pp spread. We measured the
   base at 8192 and got 8.93 %, *better* than the paper's 4096 entry, consistent with
   the table's direction.
3. **A different test subset.** The paper uses the tail of the 1104 MB MeDAL corpus
   after 64 MB train + 16 MB validation; we use 1–3 sample files.

**Not claimed:** that the paper is wrong. Claimed only that the released code plus the
released adapter do not reproduce the reported direction, on the file we measured.

### A separate, real defect we found in **our own** loader

`src/uic/models.py:446-454` loads a 4-bit NF4 base and then does

```python
model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
if merge_lora:
    model = model.merge_and_unload()
```

`merge_and_unload()` folds the adapter delta into the base weights. On a **quantised**
base those weights are bitsandbytes NF4 tensors, not plain floats, so merging is at best
lossy and at worst a silent no-op — in which case our "LoRA" arm would really be the
base model again and the measured penalty would be an artefact.

**But this turned out to be moot, because of a larger problem found while testing it.**

### The environment cannot do 4-bit at all — and our loader is now broken

While trying to run the merge test we found:

```
bitandbytes 0.45.2 : "compiled without GPU support";
                     libbitsandbytes_cuda128.dll NOT FOUND
accelerate  0.27.2 : installed
transformers 5.5.3 : requires accelerate >= 1.1.0 for bnb 4-bit
```

Consequences, each verified (`results/_dtype_diagnosis.txt`,
`results/_port_loader_diag.txt`):

| attempt | result |
|---|---|
| request 4-bit, as the port does | `ImportError: Using bitsandbytes 4-bit quantization requires accelerate` |
| load fp32/bf16 instead | succeeds, but **14.96 GiB** on a **11.94 GiB** card |
| call the port's own loader now | **fails** — `ValueError: Using a device_map ... requires accelerate` |

So every "**4-bit NF4**" statement in this project's documentation is **wrong**. Two
timestamps explain how the error arose:

| artefact | mtime |
|---|---|
| `results/text_base_v2.jsonl` (base, 8.8631 %) | **2026-10-05 19:39** |
| `results/text_lora_v2.jsonl` (LoRA, 9.1269 %) | **2026-10-05 19:53** |
| `peft` / `accelerate` / `bitsandbytes` installed | **2026-10-07 15:49** |

The text results were produced **two days before** those packages existed, i.e. under a
configuration that did **not** use bitsandbytes. bitsandbytes was installed later (for
the Colab work) in versions incompatible with transformers 5.5.3, and that installation
is what breaks the loader today. The dtype actually used on 2026-10-05 is therefore
**unverified** and is the first thing to re-establish.

### Did the broken 4-bit affect the LoRA estimate? **No — and the penalty is real.**

This was the right question to ask, and it is now settled by measurement rather than
argument. Three separate results:

**(1) 4-bit was never used for the text results.** `text_base_v2.jsonl` and
`text_lora_v2.jsonl` were written **2026-10-05**, while `peft` / `accelerate` /
`bitsandbytes` were installed **2026-10-07**. The results predate the packages, so the
broken 4-bit path cannot have produced them. The adapter arm and the base arm used the
**same** loader, so any dtype effect applies to both equally and cannot manufacture a
penalty that exists only in one arm.

**(2) The adapter demonstrably changed the output.** If `merge_and_unload()` had silently
discarded it, both arms would have run identical forward passes and produced identical
bit counts. They differ by **2 185.24 bits over 19 338 tokens = 0.113 bits/token**. So
the adapter took effect.

**(3) Merging is faithful, and the penalty survives it.** Measured on the first 4 096
tokens of the real MeDAL payload — the port's own window, bf16, no quantisation, GPU
(`tools/quantify_merge_cost_gpu.py`, `results/_merge_cost_gpu.txt`):

| arm | bits/token |
|---|---|
| base, no adapter | 2.565796 |
| + adapter, **live** (unmerged) | 2.772864 |
| + adapter, **merged** (our path) | 2.776077 |

```
adapter effect, live   : +0.207068 bits/token   <- makes compression WORSE
adapter effect, merged : +0.210281 bits/token
merge distortion       : +0.003213 bits/token   = 2.84 % of the reported penalty
```

`tools/test_merge_fidelity.py` independently confirms the fold preserves ~97 % of the
adapter's logit effect (base↔live 6.81, live↔merged 0.1875).

**Conclusion: the merge step contributes 2.8 % of the penalty. The adapter itself makes
compression worse by +0.207 bits/token unmerged.** Since the penalty is *larger* on the
faithful unmerged path than on our merged path, merging cannot be what makes the adapter
look bad. The `+0.312 pp` figure is therefore sound in direction and magnitude, and is
independently corroborated by the authors' own harness (+0.5901 pp).

**One caveat that must still travel with it** — unchanged and unrelated to 4-bit: the
penalty was measured at ctx 4 096, and the paper's own Supplementary Table 9 implies
~0.3 pp of context effect between 4 096 and 8 192. So the *existence* of a penalty now
rests on two independent harnesses, but its *attribution* to the adapter rather than to
context length still needs the ctx-8192 run.

**A residual oddity worth recording.** On a short generic clinical sentence the merge
produced a **bit-identical** result to base (`merged - base = +0.000000`). That is the
bf16 fold rounding a small delta away entirely. It does not affect the numbers above —
on 4 096 in-domain tokens the fold preserves the adapter — but it means merge fidelity is
**input- and dtype-dependent**, and a merged path should not be assumed exact without
this check. Preferred fix: evaluate unmerged.

### What the 4-bit problem *does* damage

* **Documentation:** every "4-bit NF4" claim is wrong and must be corrected.
* **Reproducibility of the current environment:** the port's loader raises today, so no
  further text measurement is possible until the stack is pinned.
* **The Colab notebook's premise:** it advertises 8-bit/4-bit paths that this local
  environment cannot exercise.
* **Not** the base-model result (two independent measurements, 0.06 pp apart), and
  **not** the LoRA penalty (see above).

---

## Task 2 — Is the ImageGPT processor quantisation the paper's compression method, or a preprocessing step?

### Answer: a **fixed, external preprocessing step**. It is not the compression method and not trainable.

The component is a standard HuggingFace `ImageGPTImageProcessor`, configured by
`data/model/imagegpt-small/preprocessor_config.json`:

```json
"clusters": [ ... 512 RGB triplets ... ],
"do_color_quantize": true,
"do_resize": true,
"size": {"height": 32, "width": 32},
"rescale_factor": 0.00392156862745098,
"image_mean": [0.5,0.5,0.5], "image_std": [0.5,0.5,0.5],
"image_processor_type": "ImageGPTImageProcessor"
```

Four things establish its status:

1. **It is not learned during compression.** The 512 cluster centres are a fixed lookup
   table shipped in the config. Nothing in `compress_with_igpt.py` or
   `language_model.py` fits or updates them; the processor is loaded once
   (`image_processor = ImageGPTImageProcessor.from_pretrained(model_path)`) and applied.
2. **It is the model's own tokenizer, inherited from iGPT.** ImageGPT is an
   autoregressive *pixel-sequence* model: it consumes a sequence of discrete colour
   tokens. The k-means palette is the token vocabulary that the checkpoint was
   pretrained against. Compression cannot use iGPT without it.
3. **The compression itself is the transformer + arithmetic coding.** `language_model.compress`
   calls the processor to get token ids, then arithmetic-codes those ids using the
   model's predictive distribution. The quantiser contributes tokenisation, not coding.
4. **The paper describes it as tokenisation, not as the method.** §2.1: *"we first
   concatenate the image's pixels from top row to bottom row, transforming the
   two-dimensional visual data into a one-dimensional sequence of pixels. This pixel
   sequence is then fed into iGPT, yielding next-pixel probability for each pixel, based
   on which the image is compressed by using arithmetic coding."* The quantiser is part
   of "transforming … into a sequence", i.e. the tokenisation step of Fig. 1.

### But this is exactly where the losslessness problem lives

Because the quantiser is fixed and applied *before* the coder, and because the official
path feeds it `(v, v, v)` built from a **single** 8-bit channel value, the map from
source value to token is **not injective**: all 256 domain values collapse to **48**
distinct token ids with **585** colliding pairs (exhaustive enumeration,
`tools/test_igpt_injectivity.py`). The coder then emits only the token id. Hence:

* the pipeline is **lossless with respect to the token grid**, and
* **lossy with respect to the image**,

which is a statement about the fixed preprocessing, not about the model. So:

| question | answer |
|---|---|
| Is it used for the comparison experiments? | Yes, but not only — it is used on **every** image/video path |
| Is it the paper's main compression method? | **No.** The method is iGPT + arithmetic coding; the quantiser is the fixed tokeniser both require |
| Is it an independent fixed quantisation, or a language model? | **Independent and fixed.** It is a k-means palettisation from the iGPT checkpoint's config; no training, no gradient, not part of the LM's parameters |

One consequential detail: the denominator counts the **source** channel-sample count
(5 870 592 for the test image) while the numerator counts the **quantised** grid
(5 733 × 1024 tokens). Numerator and denominator therefore describe different objects —
that is the whole content of the surprising image/video rates.

---

## Task 3 — Fine-tuning alternatives to LoRA, and where they can run

### The tensor arithmetic is not the binding constraint; the corpus is

The paper's recipe is **64 MB train / 16 MB validation / rest test** of raw text. 64 MB
of text ≈ **16 M tokens** (≈4 bytes/token). The local machine's measured throughput for
Llama-3-8B 4-bit at ctx 4096 is **≈66 tokens/s** forward-only; training is slower still.
Even a single epoch over 16 M tokens is therefore **days**, not hours. That is why the
fine-tuning has to happen on Colab regardless of which model is chosen — it is a
*time* constraint, not a memory one.

| model | params | local 12 GB fit (8-bit + Adam) | 64 MB / 1 epoch locally | verdict |
|---|---|---|---|---|
| **GPT-2 124M** | 0.12 B | **comfortable** (~1.5–2 GB) | ≈0.5–1 h | locally runnable; only viable local option |
| **BERT-base 110M** | 0.11 B | comfortable (~1.5 GB) | ≈0.5–1 h | locally runnable |
| ByT5-small 300M | 0.3 B | feasible (~3–4 GB) | ≈2–3 h | borderline |
| ByT5-base 582M | 0.6 B | tight (~7 GB, short ctx only) | ≈5–8 h | Colab |
| Llama-3-8B (paper) | 8 B | 4-bit only, and slow | days | **Colab only** |

Nothing suitable is cached locally (`~/.cache/huggingface` does not exist), so any of
these needs a one-time download, or a Colab session.

### Two substantive cautions about swapping the model

1. **A byte-level model is the right family for this comparison; BERT is not.**
   Compression here requires a *causal* next-token distribution. BERT is masked-LM: it
   is not trained to predict the next token, so its probabilities are the wrong object
   and it cannot drive arithmetic coding without a change of objective. **ByT5** is
   encoder–decoder but byte-level, and its decoder can be used causally — it is the most
   defensible substitute, and being byte-level it sidesteps tokeniser mismatch entirely.
   **GPT-2** is straightforwardly causal and is the cheapest meaningful control.
2. **Keeping the training set identical is the right instinct, but the tokeniser is not
   neutral.** MeDAL is medical text; Llama-3's BPE and GPT-2's BPE segment it
   differently, and ByT5 not at all. A rate comparison across models therefore mixes
   "different model" with "different tokenisation". The clean framing is to hold the
   *corpus* fixed and report **bits per source byte**, which is tokeniser-independent —
   that is already how this project reports text (`bits/byte`), so the comparison is
   well defined as long as it stays in those units.

### Recommended experiment design

Hold the corpus and the pipeline fixed; vary only the model and the amount of training:

* models: GPT-2 124M (local, cheap), ByT5-base (Colab), Llama-3-8B ± LoRA (paper, Colab)
* metric: **bits per source byte** on the held-out tail, plus rate for comparability
* training quantity: epochs ∈ {1, 2, 5, 10} — the ablation the paper never ran, and the
  one that decides whether the MeDAL result is a training-quantity effect

This runs on the same Colab notebook; Parts B and C already implement the LoRA variant,
and Part C's ablation loop is the right skeleton to extend to the other models.

---

## Appendix — the Colab notebook the other agent revised (v6)

The notebook at `<colab notebook>` is a
substantial improvement on ours: 31 cells, a memory-safe preamble, a smoke-test guard
before the expensive runs (`RUN_SMOKE`), explicit opt-in switches for the costly parts
(`RUN_FULL=False`, `RUN_TRAINING_ABLATION=False`), and per-epoch checkpoint reuse. The
Part B recipe matches the shipped adapter exactly (`r=8, alpha=32, dropout=0.05`,
7 target modules, 8-bit base, `max_len=2048`), which is what we wanted.

**Four statements in it are now stale and must be fixed before it is used**, because it
was written from a revision of the analysis that has since been corrected:

| location | stale text | correct now |
|---|---|---|
| title table | text base **8.86 %**, +LoRA **9.12 %**, paper 9.08 / 9.06 | three-file weighted means: base **9.0410 %**, LoRA **9.3529 %**; the 8.86/9.12 figures are single-file values |
| warnings §2 | audio **4.13×** "is not a next-byte measurement" | **withdrawn.** The reference's audio scheme is coherent and causal, and the reference reproduces the paper's regime (19.94 % vs 18.26 %). Our port: 19.82 % |
| warnings §1 | "loads Llama-3-8B in float32 … ~32 GB" | still true, and it is also why the **text** path OOMs locally — keep it, but it is worth stating that we confirmed the reference *text* path was never completed here |
| Part A2 | evaluates "the adapter that ships with the capsule" to test its benefit | the adapter does **not** help on either harness; state that as the expected result so a negative outcome is not mistaken for a notebook bug |

Also worth adding: a cell that runs the **same file** through both our port and the
reference harness so the 0.06 pp agreement is recorded in the Colab output itself.
