# Audit package — LMCompress reproduction

**Purpose of this document.** It is written to be handed to an independent
reviewer (human or model) with **no access to the conversation that produced it**.
Every substantive claim is stated together with the evidence that supports it, the
derivation behind any non-obvious step, and an explicit marker of whether it is
*measured*, *derived*, or *inferred*. The reviewer is invited to attack the
inferences and to re-run the measurements from the raw data listed in §7.

**Subject.** Li, Huang, Wang, Hu, Wyeth, Bu, Yu, Gao, Liu, Li,
*Lossless data compression by large models*, Nature Machine Intelligence **7**,
794–799 (2025), DOI 10.1038/s42256-025-01033-7 (preprint: *Understanding is
Compression*, arXiv:2407.07723). Method name **LMCompress**. Reference
implementation: Code Ocean capsule 4957991.

**What was done.** A from-scratch reimplementation of the paper's pipeline
(tokenise → autoregressive model → per-symbol probability → arithmetic coding),
applied to the paper's three model families and four modalities, then compared
against the paper's published numbers and against classical baselines measured
with the paper's own toolchain.

---

## 0. How to read the claim markers

| Marker | Meaning |
|---|---|
| **[M]** | Directly measured by us; raw data in §7. |
| **[D]** | Derived by argument; derivation given inline. |
| **[I]** | Inferred from [M] + [D]; these are the claims most worth attacking. |
| **[P]** | Quoted from the paper or its supplementary information. |
| **[U]** | Unknown / not established. |

---

## 1. Summary of claims

| # | Claim | Marker | Confidence |
|---|---|---|---|
| C1 | The text-modality pipeline is reproduced: base model 9.041 % vs paper 9.08 % | [M] | high |
| C2 | The LoRA adapter **published by the authors with their capsule** (their own fine-tuning, not ours) makes MeDAL compression **worse** (9.353 % vs 9.041 %), consistently on every file | [M] | high |
| C3 | That adapter is the most overfitted checkpoint of its own run | [M] (from its `trainer_state.json`) | high |
| C4 | The fine-tuning gain is governed by domain match, not training quantity | [I] | medium |
| ~~C5~~ | ~~The reference implementation's audio figure is not a code length for those bytes~~ **RETRACTED** — the argument depended on the withdrawn contiguous-row probe | — | withdrawn |
| ~~C6~~ | ~~The released bGPT audio checkpoint reconstructs rather than predicts the next byte~~ **RETRACTED** — the checkpoint's offset-16 pairing is the reference pairing | — | withdrawn |
| C7 | The **shipped reference image path** is lossless w.r.t. the quantised token grid and lossy w.r.t. the source image (official code instrumented end to end) | [M] | high |
| C8 | Our port's coder had a latent interval-collapse bug (demonstrated, fixed). The reference implementation's quantiser is **structurally exposed to the same failure mode** — a risk note, *not* a demonstrated defect, since no failing input was constructed for their code | [M] + [D] | medium |
| ~~C9~~ | ~~The bGPT reference code misaligns its audio scoring~~ **RETRACTED** — the reference's pairing is the one the checkpoint is most confident about (0.3052 at offset 16 vs 0.0804 at offset 1), the scheme is causal (Δ = 0 under a future-byte perturbation), and its number is realized coder output. Our earlier probes used row `16+k` instead of `17(k//16)+(k%16)` | — | **withdrawn** |

---

## 2. Experimental design

### 2.1 What is measured, and why the metric is subtle

The paper's metric is **compression rate = compressed size / original size, in
percent, lower is better** [P, NMI Methods]. The reciprocal (ratio) is used in the
preprint. This document uses the NMI convention throughout.

The pipeline is: `data → tokens → model posteriors → arithmetic coding`. The code
length for a token sequence `ω` under model distribution `q` is
`⌈-log₂ q(ω)⌉ + 1` bits [P, Suppl. §S4.1]. We therefore define:

* **reported code length** = `Σᵢ -log₂ q(ωᵢ | ω_{<i})` — the model's exact
  cross-entropy. This is what all comparisons use. It needs no coder at all.
* **arithmetic-coder output length** = the actual bits emitted, always ≥ the
  above minus coder rounding. We track this as a consistency check, not as the
  headline number.

**Why this distinction is the crux of two of our findings.** A compression rate is
only interpretable if the object that was coded is the object in the denominator.

* For **audio**, the ratio's denominator is raw PCM bytes. The reference's
  17-slot row mapping is coherent for the released checkpoint; the remaining
  question is reproducibility because the supplied invocation fails on the
  bundled file's final non-patch-aligned chunk without a tail-padding policy.
* For **images**, the denominator is internally consistent (it counts the samples
  the pipeline actually processed, as §4.4.2 shows by measurement), and the
  scoring is fine. The defect is upstream: the **preprocessing is non-injective**,
  so the object that gets coded is a 512-colour quantised surrogate from which the
  source samples cannot be recovered. This is a **tokenisation** defect.

These are *two different failure modes* and are deliberately **not** unified into a
single slogan. An earlier revision called both "numerator/denominator mismatch",
which is wrong for images: there the denominator is internally consistent and the
token stream genuinely is the coded object — the defect is that this object is not
a recoverable representation of the source. See §9.

### 2.2 Local subset, and why the sample sizes differ from the paper

| Modality | Local sample | Paper's sample | Marker |
|---|---|---|---|
| text | 3 MeDAL CSVs, 312 053 bytes | ~180 MB [P, NMI] | [M] |
| audio | 1 WAV, 82 220 PCM bytes (16 kHz mono 8-bit) | ~100 MB per dataset [P, Suppl.] | [M] |
| image | 1 PNG, 1714×1142, 5 872 164 RGB bytes | 197 images, ~128 MB [P, Suppl.] | [M] |
| video | 1 Y4M, 4096×2160, 60 frames, 796 MB | 11 clips, ~1.16 GB [P, Suppl.] | [M] |

**Consequence, stated up front:** absolute agreement with the paper is *not*
expected and is not what is being tested. Classical codecs improve monotonically
with input size, so we expect our classical baselines to be systematically *worse*
(higher rate) than the paper's, by roughly 1–4 pp at these sizes. The tests that
carry weight are: (a) the ordering of methods, (b) the model-vs-baseline gap, and
(c) model-vs-model differences measured on **identical bytes with identical
code**.

### 2.3 Controls that were run deliberately

A measurement claim is only as good as its negative controls. Four were used:

1. **Classical baselines on the same files, with the paper's own tool
   parameters.** If our harness can reproduce FLAC/WavPack/OptimFROG and
   zstd/brotli/lzma to within a few points, then the harness is sound, and a
   grossly discordant model result is attributable to the model path rather than
   to the measurement setup. This is the control that makes claim C5 credible.
2. **Base model vs fine-tuned model on identical files and identical code path.**
   The only difference is whether the LoRA adapter is attached. This is what makes
   C2 a like-for-like comparison rather than a cross-run comparison.
3. **Entropy bounds.** For any source we can compute its empirical entropy. A
   correctly scored lossless code cannot beat the *true* entropy of the source. The
   subtlety that a reviewer correctly forced us to state plainly: an **order-1
   empirical** entropy is *not* a lower bound for a model with longer context, so
   this control is suggestive, not conclusive, and C5 does not rest on it. The
   load-bearing control for audio is the direct scoring test in §4.3.3.
4. **Domain enumeration for the image quantiser.** The image claim is settled by
   enumerating the entire input domain of the quantiser (256 grayscale values)
   rather than by a cardinality argument, which a reviewer correctly showed was
   invalid for the repeated-grayscale inputs this path actually feeds (§4.4.2).

### 2.4 Reproducibility protocol

```
# environment
.venv/Scripts/python.exe          # Python 3.11.15, torch 2.11.0+cu128, RTX 5070 12 GB
                                  # transformers 5.5.3, peft 0.21.2, bitsandbytes 0.50.2

# coder self-tests (must be 0 failures before any measurement is trusted)
python tests/test_coder.py

# text: base model, 3 files, identical code path to the LoRA run
python -m uic.cli text --dataset medal --limit 3 --context-window 4096 \
       --no-lora --out text_base_3_final.jsonl

# text: same files, the AUTHORS' published LoRA attached
python -m uic.cli text --dataset medal --limit 3 --context-window 4096 \
       --out text_lora_3.jsonl

# audio baselines, using the capsule's own Linux binaries (WSL)
bash tools/baselines_linux.sh audio results/manifest_audio.txt

# image, incl. the honest lossless accounting
# image: the C7 evidence is the four probes listed in section 7.2, in this order
python tools/test_igpt_injectivity.py
python tools/probe_official_image_accounting.py
python tools/probe_official_image_path.py
python tools/probe_loader_volume.py

# the audit that establishes the entropy bounds
python tools/audit_bgpt_audio.py
# (tools/audit_igpt_image.py is SUPERSEDED -- whole-image path, NOT C7 evidence)
python tools/probe_bgpt_semantics.py
```

All run commands are idempotent: progress cursors are keyed by configuration, so
re-running resumes rather than restarting.

---

## 3. The pipeline, derived

This section contains the derivations a reviewer needs in order to check that the
implementation is faithful — and to check the two alignment claims (C5, C9).

### 3.1 Arithmetic coding with a model that can be wrong

Standard range coding on a sequence of symbols with per-step distributions. The
essential detail for *losslessness* is the quantisation of a real-valued
probability vector into integer cumulative frequencies.

Write `FULL = 2^32` for the coder's range. For a symbol `i` with probability
`pᵢ`, we assign an integer interval `[cdf[i], cdf[i+1])` with

```
cdf[i+1] - cdf[i] ≥ w_min ,   Σ (cdf[i+1] - cdf[i]) = FULL
```

and the widths otherwise proportional to `pᵢ`.

**Why `w_min ≥ 1` is not optional — this is the bug in C8.** The encoder narrows
the interval as

```
new_low  = low  + (width * cdf_lo) // FULL
new_high = low  + (width * cdf_hi) // FULL - 1
```

with `width = high - low + 1`. If `cdf_hi - cdf_lo = 1` and `width < FULL`, then
`width * 1 // FULL = 0`, so `new_high = low - 1 < new_low`. The interval inverts;
the subsequent renormalisation loop then subtracts `HALF` or `QUARTER` from a
negative `low` and the coder emits a stream that decodes to a **different symbol**
— silently, with no exception.

We hit this concretely: a 70-symbol distribution where the true symbol had
`p = 9.06e-11` received an interval of exactly 1 unit while the current width was
`2 159 099 904 < FULL`, and the decoder returned symbol 56 instead of 57.

Two facts a reviewer should note:

* the fix is `w_min = FULL / 2^20`, plus an assertion in the encoder so the failure
  mode can never be silent again;

  **A reviewer correctly rejected the justification originally given here.** An
  earlier revision said the floor "cannot distort a reported ratio because 20 bits
  exceeds the 8 bits/byte ceiling of a byte alphabet". That is wrong: having 256
  symbols does not bound the surprisal of the true byte. A badly calibrated model
  can assign the true symbol a probability below `2^-20`, in which case
  `-log₂ p > 20` bits and the floor does change the distribution the coder uses.

  The floor does not affect the headline numbers, but for a different and simpler
  reason: **the reported rate is never computed from the coder's quantised
  probabilities.** It is `Σ -log₂ q(true symbol)` evaluated on the model's own
  float32 probabilities, computed before and independently of any quantisation
  (see §2.1). The quantised cdf affects only the arithmetic-coded bitstream
  length, which is tracked separately as a consistency check. So the correct
  statement is: *the floor bounds the coder's worst case, and the reported rate is
  unaffected because it does not consume the coder's probabilities at all* — not
  that the floor is somehow beneath the alphabet's granularity.

  This also means the two quantities can legitimately differ: in the image run the
  reported code length is 910 788 bytes while the arithmetic coder's own output is
  slightly larger, and the difference is coder rounding plus the floor, not a
  measurement inconsistency.
* **the reference implementation has the same structural hole.** Its
  `normalize_pdf_for_arithmetic_coding` adds a machine-epsilon term to every
  probability and then renormalises by dividing by the sum. A symbol whose
  probability is far below the epsilon still ends up with an arbitrarily small
  quantised mass, and its coder performs the same integer division. [D] We have
  not demonstrated a concrete failing input for *their* code, so this is stated as
  a structural risk, not a demonstrated defect.

### 3.2 bGPT: the patch/byte alignment (source of C9)

bGPT is hierarchical: a 12-layer patch-level decoder over 512 patches of 16 bytes,
then a 3-layer byte-level decoder that emits the bytes within each patch.

Let `PATCH_SIZE = 16`, `PATCH_LENGTH = 512`. The input grid for a payload of `n`
bytes is

```
patch 0        : file-extension bytes, padded with token 256
patches 1..k   : the payload, zero-padded up to a patch boundary
patch k+1      : token 256 repeated (end-of-sequence patch)
```

`bGPTLMHeadModel.forward` does two things that matter:

```python
left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
masks[:, 0] = 0
encoded_patches = encoded_patches[left_shift_masks == 1]
patches         = patches[masks == 1]
```

so patch 0 is dropped, and the returned logits cover patches `1..P` where `P` is
the number of payload-carrying patches. The byte-level decoder yields
`PATCH_SIZE + 1 = 17` positions per patch, i.e. 17 next-byte predictions.

**Derivation of the row mapping.** Slot `j` of returned patch `p` predicts grid
position `16p + j + 1`. The reference removes slot 16 from every 17-slot patch
row before flattening. Therefore payload byte `k` (at grid position `16 + k`) is
paired with original logit row:

```
payload byte k  ←  original flattened logit row 17 * (k // 16) + (k % 16)
```

**Verified by measurement [M]**, and **corrected**: an earlier revision of this section
presented a sweep of candidate flat offsets (13.1784 / 13.1281 / **13.1233** / 13.1271 /
13.1669 bits/byte) as evidence for the alignment. **That sweep is void** — it indexed
rows as `16 + k` when a patch row carries 17 slots, so it was scoring every row against
a byte roughly `k // 16` positions away from the one it predicts.

The correct measurement asks which byte a row is actually most confident about, over
offsets rather than flat indices:

| offset L from the row's own slot | mean probability of `grid[16p+j+L]` | argmax hit rate |
|---|---|---|
| 1 (the immediately following byte) | 0.0804 | 7.26 % |
| 8 | 0.1448 | 15.81 % |
| 15 | 0.2199 | 26.07 % |
| **16** | **0.3052** | **37.18 %** |
| 17 | 0.1923 | 20.94 % |

(`tools/measure_row_target_offset.py`.) The row is most confident about offset **16** —
and that is precisely the byte the reference pairs it with. So the reference's pairing
is coherent for this checkpoint. The port now selects those non-contiguous rows
explicitly and checks that its entire target slice equals the payload.

**The reference implementation's offset.** `compress_with_bgpt_or_llama.py`
computes

```python
logits   = logits[:-1, :-1, :].reshape(1, -1, 257)
input_ids = input_ids[:, PATCH_SIZE:-PATCH_SIZE]
```

The first line deletes the 17th slot of every patch row and the trailing patch;
the second selects grid positions `16 .. 16 + n - 1`. Thus the first retained row
of patch `p` is paired with the corresponding byte in patch `p + 1`. This is the
same 17-slot mapping above, not a contiguous slice of the untrimmed logit tensor.

### 3.3 Sequential (causal) models: window bookkeeping

For Llama, position `p` predicts token `p + 1`, so a sequence of `L` positions
yields `L − 1` scorable tokens; BOS costs one further position on the first
window. Windows must therefore hand the model **one token of lookahead** beyond
the tokens they score.

This was established empirically rather than assumed, because a first-principles
derivation disagreed with the library's actual output by one. The implementation
now **calibrates** `window_budget()` by running one throwaway forward pass at
maximum context and counting the predictions that come back; the runner then
asserts that every payload symbol is coded exactly once (see §4.1, the
`uncoded_symbols` column).

---

## 4. Measurements

### 4.1 Text — the reproduction that succeeds

> **Note (post-review revision).** These numbers were re-measured after a
> target-alignment defect was found and fixed in this port; see
> `docs/REVIEW_RESPONSE.md` §5. The previous values were 9.0333 % (base) and
> 9.3459 % (LoRA). The fix moved both slightly *up* and, importantly, made the
> base figure **closer** to the paper. All 57 547 tokens are now coded exactly
> once (`uncoded_symbols = 0`).

> **Adapter provenance — do not misread this section.** The two arms compared here
> are (a) raw Llama-3-8B and (b) **the LoRA adapter that the paper's authors
> published inside their Code Ocean capsule**
> (`data/model/finetuned_llama/Llama-3-8B_lora/`). It is *their* fine-tuning, not
> ours. Evidence: its `adapter_config.json` records `base_model_name_or_path` as a
> path on their machine (`<a path on the authors' machine>`), and the
> same directory still carries their DeepSpeed 8-way pipeline optimiser shards
> (`bf16_zero_pp_rank_0..7_*_optim_states.pt`). This section therefore answers
> *"is the authors' published adapter better than the base model?"* — and **not**
> *"how good is our own fine-tuning?"*, which is the job of §5.4 and Colab
> Parts B-C. No adapter was trained during this reproduction.

Identical code path for both rows; the only difference is whether the LoRA adapter
is attached. Code length is `Σ -log₂ q(true token)`.

| File | raw bytes | base coded bits | base rate | +LoRA coded bits | +LoRA rate | Δ |
|---|---|---|---|---|---|---|
| medal_102285.csv | 103 534 | 73 410.306 | 8.8631 % | 75 595.548 | 9.1269 % | +0.264 pp |
| medal_103851.csv | 104 332 | 77 092.631 | 9.2365 % | 79 715.076 | 9.5506 % | +0.314 pp |
| medal_108122.csv | 104 187 | 75 199.549 | 9.0222 % | 78 176.908 | 9.3794 % | +0.357 pp |
| **total** | **312 053** | **225 702.486** | **9.0410 %** | **233 487.532** | **9.3529 %** | **+0.312 pp** |

Ratios: base **11.0607×**, LoRA **10.6919×**.

Settings: Llama-3-8B, 4-bit NF4, context window 4 096, 5 windows per file,
57 547 of 57 547 tokens coded.

Comparison with the paper [P, Suppl. Table 6]:

| | ours | paper | Δ |
|---|---|---|---|
| LMCompress\* (no fine-tuning) | **9.041 %** | 9.08 % | **−0.04 pp** |
| LMCompress (fine-tuned) | **9.353 %** | 9.06 % | **+0.29 pp** |

**C1 [M].** The base-model figure lands 0.04 pp from the published value. Given
4-bit vs the paper's 8-bit and 4 096 vs the paper's 8 192 context, this is a
like-for-like match and is the primary evidence that the port is faithful.

**C2 [M].** The adapter is *worse*, by 0.312 pp, on **all three** files. The
direction is opposite to the paper, and the magnitude is larger than the effect
the paper claims (0.02 pp).

#### 4.1.1 The uncoded-token disclosure is obsolete

An earlier revision of this port left exactly one token per document uncoded and
disclosed it here. That was caused by a target-alignment defect, now fixed; the
corrected runs code every token (`coded_symbols == symbols == 57 547`). The
disclosure is retained only to record that the defect existed and how it was
found — see `docs/REVIEW_RESPONSE.md` §5.

### 4.2 Text — baselines (the harness control)

Same 3 files. **Two denominators are in play here and the reviewer should notice
it:** the classical codecs compress the whole CSV container (313 272 bytes), while
the model codes the *extracted* `TEXT` payload (312 053 bytes) because
`pandas.read_csv` strips the quoting and the `TEXT` header (~420 bytes per file,
0.13 %). Both are reported so the comparison can be made on either basis.

| Method | compressed bytes | rate ÷ container (313 272) | rate ÷ payload (312 053) | paper [P] |
|---|---|---|---|---|
| brotli quality 11 | 90 238 | 28.8050 % | 28.9175 % | 27.93 % |
| lzma preset 9e (≡ 7z `-mx9`) | 99 388 | 31.7258 % | 31.8497 % | 30.25 % |
| zstd `--ultra -22` | 100 228 | 31.9939 % | 32.1189 % | 30.45 % |
| zlib 9 | 109 702 | 35.0181 % | 35.1549 % | — |

**Effect of the denominator choice: ≤ 0.13 pp on every row** — i.e. the classical
baselines are insensitive to it. The model rows in §4.1 use the payload
denominator. To compare strictly like-for-like, use the payload column: brotli
**28.92 %** against the model's **9.041 %**.

**Interpretation [D].** Ordering reproduces exactly (brotli best; zstd ≈ 7z; zlib
worst). The uniform +1.0 to +1.7 pp offset against the paper is the expected
direction for 0.31 MB versus ~180 MB of input. The model is **3.20×** better than
the best classical baseline on the payload basis (9.041 % vs 28.92 %), consistent
with the paper's "nearly one-third of zpaq" claim [P].

#### 4.2.1 Two corrections to our own text numbers [M]

Prompted by a reader asking why the text baselines looked so good relative to the paper.
Two errors on our side, both leaning in our own favour:

**The brotli figure was 28.918 % and is wrong.** The bytes-weighted payload value in
`results/baselines_text_medal_python.jsonl` is **28.6187 %** over 10 units and **28.6717 %**
over the 3 raw files. 28.918 appears in no result row. Corrected throughout. The other
text baselines, bytes-weighted over 10 units, are lzma 9e **31.4479 %**, zstd -22
**31.6755 %**, zlib 9 **34.7582 %**.

**The denominator sentence conflated a size with a rate.** The CSV container and the
extracted payload differ in size by **0.389 %** (313 272 vs 312 053 bytes), whereas the
*measured rate* moves by **0.133 pp** between the two denominators (brotli 28.8050 ->
28.6717). The earlier wording said the denominators differ by 0.13 %, which is the rate
shift, not the size difference.

**And the comparison with the paper's text baselines is not like-for-like.** SI Table 11
runs each compressor at a fixed window of 8192 / 16384 / 32768 bytes and then unbounded;
7z alone spans **42.02 -> 23.89 %** across those columns. Our baselines compress the whole
file in one stream, which corresponds to their *unbounded* column -- against which our
7z-equivalent is **31.45 %**, i.e. **worse** than the paper's 23.89 %, exactly as expected
from 313 KB of input against their ~180 MB. Comparing our whole-file 28.6 % with their
8192-byte window figures would flatter us by more than ten points and is not done here.

#### 4.2.2 Text baselines at the paper's fixed-window protocol [M]

The earlier version of this section reported whole-file compression and then declined to
compare it with the paper, because SI Table 11 uses a fixed window and the same compressor
spans 42.02 -> 23.89 % across its columns. Declining to compare was the wrong call: the
protocol is cheap to reproduce, so it was reproduced
(`tools/baselines_text_windowed.py`). Each file is split into consecutive chunks of 8192 /
16384 / 32768 bytes, every chunk compressed with a **fresh** compressor state, and the
whole file compressed once for the unbounded column.

| codec | 8192 | 16384 | 32768 | unbounded |
|---|---|---|---|---|
| brotli q11 | 31.84 | 30.91 | 30.17 | 28.81 |
| lzma 9e (7z-equivalent) | **41.44** | **38.40** | **35.78** | **31.68** |
| zlib 9 | 41.78 | 39.67 | 37.70 | 35.02 |
| zstd -22 | **40.38** | 37.93 | 35.69 | 31.99 |

Against SI Table 11:

| codec | window | ours | paper | delta |
|---|---|---|---|---|
| 7z | 8192 | 41.44 | 42.02 | **-0.58** |
| 7z | 16384 | 38.40 | 38.21 | **+0.19** |
| 7z | 32768 | 35.78 | 35.26 | **+0.52** |
| 7z | unbounded | 31.68 | 23.89 | +7.79 |
| zstd | 8192 | 40.38 | 39.54 | **+0.84** |

**Three of the four 7z columns agree to better than 0.6 pp.** That is a reproduction of the
paper's baseline grid, and it is a stronger statement than the hedge it replaces.

**The unbounded column is the exception and the reason is known.** The paper compressed on
the order of 180 MB; we compress 313 KB. With no window to limit it, a compressor benefits
directly from more input, and our own table shows the same gradient (7z improves
41.44 -> 31.68 % as the window grows). So +7.79 pp is a corpus-size effect, not a protocol
error, and it is exactly the column where the two corpora are least comparable.

**Consequence for the model comparison.** The like-for-like comparison is now available and
is the one that should be quoted: our model at an 8192-character window (**6.3321 %**)
against our baselines at an 8192-byte window (best is brotli at **31.84 %**), a factor of
**5.0x**. The paper's own pair is 9.06 % against 39.54 % (zstd), a factor of 4.4x. The
model's margin over classical compression therefore reproduces at matched windows.

### 4.3 The audio result, after three corrections [M]

This section has been wrong three times. The final, measured position follows.

**Canonical numbers.** Sample `LJ001-0004.wav`, 16 kHz mono 8-bit PCM,
**82 220 payload bytes**. All of them are coded in every figure below.

| implementation | coded output | bits/byte | **rate** (compressed/original) | ratio |
|---|---|---|---|---|
| **our port** (`results/audio_bgpt_rows17_tailpad_v1.jsonl`) | 130 383 bits = 16 298 B | 1.5858 | **19.82 %** | 5.046× |
| **reference, unmodified** (`artifacts/refrun`, one full run) | 16 399.95 B | 1.5957 | **19.94 %** | 5.016× |
| paper headline (LJSpeech) | — | 1.4608 | **18.26 %** | 5.477× |

Our arithmetic coder's **real emitted output** is 1.5858 bits/byte against the
reference's reported 1.5957 — a **0.6 %** difference. The reported figure computed as
`Σ −log₂ q` is 1.6641 bits/byte (20.80 %), 4.29 % above the reference; the gap between
our two internal figures is the usual coder-quantisation overhead, and both are tracked
separately (`coded_bits` vs `ac_bits`).

**So audio does reproduce.** The reference lands **9.2 % relative** from the published
18.26 %, which is the same order as the classical baseline gaps elsewhere in this
audit (audio baselines: OptimFROG +1.89 pp, WavPack +2.15 pp, FLAC +4.07 pp), and the
ordering is unchanged: the model beats every classical codec measured, and by a wide
margin (19.8 % vs OptimFROG's 27.3 %).

**Three errors, recorded so they are not repeated.**

1. **"The scoring is misaligned by 16 positions" — RETRACTED.** Our probe indexed flat
   logits as `16 + k`, but a patch row carries **17** slot predictions, so the row for
   payload position `k` is `17*(k//16) + (k%16)`. Measured under the correct mapping,
   the reference's pairing is the one the checkpoint is most confident about
   (mean probability **0.3052** at offset 16 vs **0.0804** at offset 1), the scheme is
   causal (`max |Δ log-prob| = 0` when bytes after the target are perturbed, versus
   **9.44** when bytes before it are), and the coder really emits the bytes. Every
   figure derived from that probe (13.12, 13.1233, 14.1804, **15.7844** bits/byte and
   the 197.31 % rate) is **void**.
2. **"The tail block makes the script crash" — RETRACTED.** `split(..., 8160)` does
   yield 10 full windows plus a 620-byte tail, and `byte_collate` does pad that tail to
   652 bytes, which is not a multiple of `PATCH_SIZE`. But the unmodified script
   **completes all 11 blocks** and reports `Total size: 82264`. The predicted crash does
   not occur.
3. **"`samplings` blocks the run" — correct, and now fixed.** The capsule genuinely
   omits `samplings.py` while `data/model/bgpt/utils.py:9` imports it, so every entry
   point fails at import time. `tools/verify_reference_blockers.py` confirms the file is
   absent. A reconstruction lives in `replacement_modules/samplings.py`; the three
   functions are used only by the custom `generate` (lines 178-180), **never** by
   `forward`, so they cannot affect a compression measurement.
   `tools/run_reference_audio_onewindow.sh` copies the capsule to a temp tree, installs
   that module there, and runs the authors' script **unmodified** — the published
   capsule is never altered.

**Withdrawn: the "missing `samplings` dependency" claim.**

An earlier revision of this audit asserted that `data/model/bgpt/utils.py:9` imports a
`samplings` module the capsule fails to ship, so that the reference code "fails at
import time", and that this blocked reproducing the paper. **That is false.**

`samplings` is a real PyPI package (by Sander Wood) and it is **already installed** in
the capsule's virtualenv:

```
<capsule>/.venv/Lib/site-packages/samplings/__init__.py   (21 316 bytes)
```

It is correctly *not* vendored inside the capsule source tree — it is a dependency, and
a virtualenv is where dependencies live; it is also absent from `requirements.txt`,
which is a separate (minor) packaging gap. Our search walked the capsule directory while
excluding `.venv`, found nothing, and drew the wrong conclusion.

**The reference code runs, unmodified, on the original capsule.** Verified twice:

```
import compress_with_bgpt_or_llama   ->  IMPORT OK (exit 0)

python compress_with_bgpt_or_llama.py \
  --data_path data/test_audios --data_type wav \
  --model_path data/model/bgpt/weights-audio.pth \
  --context_window 8160 --batch_size 1

Theoretical compression ratio: 5.016113
Total size: 82264
compressed size: 16399.949935913086
```

**The tail-block claim is also withdrawn.** The arithmetic was right — `split(..., 8160)`
yields 10 full windows plus a 620-byte tail, and `byte_collate` pads that to 652 bytes,
which is not a multiple of `PATCH_SIZE` — but the reference run **completes all 11
blocks** and reports `Total size: 82264`. The predicted crash does not happen.

**Consequence for this audit.** Nothing about the audio *result* changes: the reference
reproduces the paper's audio regime, and so does our port (§4.3). What changes is that
the reproduction is **easier** than we claimed — no repair, no patched copy, no
reconstruction was needed. Anyone can reproduce it with the two commands above.

**Process note.** This is the third claim in this section to be withdrawn, and the
failure mode is the same each time: we searched for an artifact, searched in the wrong
place (excluding `.venv`, or indexing a 17-slot row grid as 16), and reported the
absence of evidence as evidence. We had also run the import check, seen it succeed, and
explained it away rather than following it. The reviewer should treat any remaining
"the reference cannot do X" statement in this document as suspect until re-run.

**Reproduction procedure** (`tools/confirm_tail_and_our_window.sh`):
the reference was run as shipped on the full file and on a single 8 160-byte window;
our port was measured on the same inputs. Raw outputs are in
`results/_tail_and_window.txt` and `results/_refrun_onewindow.txt`.

### 4.4.3 Withdrawn: the 330-byte / 67.43 % completion

An earlier revision of this audit contained a section deriving a "true lossless
rate" of 67.43 % from a **330-byte** token-stream cost plus a zlib-coded residual.
**That section has been withdrawn and is not evidence for C7.** It is recorded here
only so a reviewer who saw the earlier revision knows why it is gone.

Two independent reasons:

1. **Wrong pipeline.** The 330 bytes came from our own whole-image experiment, in
   which one image was reduced to a single 32×32 token grid. The official path is
   patch-wise and channel-wise: the final run produced **910 788 bytes** of arithmetic code
   over 5 733 chunks (its log's last *printed* value was 906 459). Mixing the 330-byte numerator with official-path reasoning
   was a version-merge error, and it is also what corrupted the rate table in the
   previous revision (§4.4.2).
2. **Wrong claim even for the right pipeline.** 67.43 % is one particular
   reconstruction completion, not a bound. This port's own WebP result (48.28 %
   lossless on the same image) proves it cannot be a bound, so presenting it as a
   "true lossless rate" was unsupportable.

**The number is therefore absent from the conclusions.** C7 no longer rests on any
residual-completion argument at all; it rests on the exhaustive injectivity test in
§4.4.2, which is a statement about the quantiser alone and needs no residual model.

A properly-scoped replacement — de-quantise each channel patch, reassemble, code a
residual, and charge for the dropped edge samples — would be a legitimate
*illustrative* completion for the official path. It has not been run, and it is not
needed for C7.

### 4.4.4 A channel-wise mismatch in **our own** image and video port [M]

**Found by external review, then verified here. The reviewer is correct.**

The official pipeline is *channel-wise*, and it is worth stating exactly how, because
our port does something different.

`get_imagenet_iterator(is_channel_wised=True)` splits the image into single-channel
patches, so every chunk it yields has last axis 1. `language_model.compress` then takes
its `np.repeat(..., 3, axis=2)` branch and hands the processor a **greyscale triple**
`(v, v, v)` built from one channel value. Verified live
(`tools/check_channel_wise_mismatch.py`, `results/_channel_wise_check.txt`):

| path | processor input | tokens | distinct ids |
|---|---|---|---|
| official, chunk 0 | `(32,32,1)` → repeat → `(32,32,3)`, R = G = B everywhere | 1024 | **9** |
| our port, whole image | `(32,32,3)` real RGB patch | 1024 | **101** |

The two token streams differ from the first token: official
`[71, 71, 71, 71, 71, 71, 96, ...]` against our port
`[51, 51, 51, 51, 51, 51, 51, ...]`.

**What this does and does not affect.**

| Item | Status |
|---|---|
| Image accounting 15.5144 % / chunks 5733 / raw_length 5870592 | **unaffected** — produced by `tools/probe_official_image_accounting.py`, which drives the *official* channel-wise iterator |
| Image non-injectivity (256 → 48 ids, 585 pairs) | **unaffected** — a property of the official path, established through the actual processor on `(v,v,v)` inputs |
| Video iGPT 0.0042 % and exact-completion 73.05 % | **WITHDRAWN for now** — produced by `tools/run_video_igpt.py` via `image_units`, i.e. the RGB path, not the official one |
| A correction previously applied here — "use the channel-sample denominator, i.e. divide the video rate by 3" | **WRONG and withdrawn** — our port codes one channel's worth of tokens while the official path codes three, so scaling the denominator by 3 without also scaling the numerator is not a valid conversion |

**Required fix.** `uic.data.image_units` must mirror the official loader: split the
image into channels, replicate each channel to a greyscale triple, and emit one unit per
(patch, channel). Until that is done, no image or video number produced by our own port
should be quoted; the image figures in this audit stand because they come from the
official code, not from our port.

**Why this is recorded rather than quietly patched.** It is the fourth instance in this
project of the same failure mode — a plausible reconstruction of someone else's pipeline
substituted for the pipeline itself. The difference here is that it was caught by an
external reviewer rather than by our own tooling, which had no check that would have
noticed.

### 4.4.5 The official image metric codes each token against a distribution that has seen it [M]

This is the reason our channel-corrected port still reports 23.93 % where the official
script reports 15.51 %, and it is a stronger finding than the denominator question.

**What the official loop does.** `language_model.compress` builds

```python
log_probs = predict_fn(sequence_array)          # (1024, 512)
raw_probs = log_probs.reshape(-1, log_probs.shape[-1])
...
for pdf, symbol in zip(raw_probs, sequence_array):
    encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)
```

`predict_fn` runs the model on the **complete** token sequence and returns, at row `t`,
the softmax of the logits at position `t`. The zip then codes token `t` with row `t`.

**Why that matters.** We verified the model is causal, so row `t` is computed from
tokens `0..t` — which *includes token `t` itself*. Perturbing the token suffix from
position 501 onward leaves rows 0–500 bit-identical (`max |Δ logit| = 0`) while moving
rows 501+ by up to 33: the probe is live and the rows are causal
(`tools/test_official_row_causality.py`).

So at every position the distribution used to code a symbol has already conditioned on
that symbol. That is not a predictive distribution for the token being coded, and the
resulting bit count is **not a lossless code length for the sequence**: no decoder that
does not already know token `t` can reproduce it. Our port codes token `t` with the
distribution from position `t-1`, which is the conventional and decodable convention.

**Measured.** On one identical chunk (same ids, same model):

| quantity | value |
|---|---|
| official rows (row `t` for token `t`) | **1.2839** bits/sample |
| proper causal (row `t-1` for token `t`) | 1.7363 bits/sample |
| our port | **1.7348** bits/sample |
| official arithmetic coder, actually emitted | 165 bytes = 1.2891 bits/sample |

Our port agrees with the conventional causal figure to **0.09 %**. The official figure
is **1.35×** smaller, and its own coder really emits the smaller stream — so this is
realised output, not a modelling artefact.

**Whole-image consequence.** Official 15.5144 % against our channel-corrected 23.9257 %
— a factor of **1.542**.

**Reconciliation of the two effects.** Before this section the image discrepancy had two
candidate causes; both turn out to be real and they are independent:

| cause | direction | status |
|---|---|---|
| channel-wise vs whole-RGB tokenisation | our port was wrong; fixed in `uic.data.image_units` | **resolved** — the port now produces 5733 units, 5870592 samples and identical ids |
| row `t` used to code token `t` | official figure is optimistic relative to any decodable code | **open** — and not something our port can or should reproduce |

Consequently the published image rates should be read as the size of a stream produced
under a convention whose decodability we cannot reconstruct. Our port's figure is the
conventional one. **Neither number is a like-for-like comparison with the classical
baselines**, which remain valid.

**Note on the same pattern elsewhere.** The audio path was checked for exactly this and
did **not** show it: the reference audio routine scores each byte against the
distribution its own context produced, and a future-byte perturbation leaves the score
unchanged (`max |Δ log p| = 0`). The two modalities therefore differ, and the audio
result is unaffected by this section.

### 4.4.6 Image baselines and the model, on the actual CLIC corpus [M]

The earlier version of this section used a single 1714x1142 aerial photograph and was
the weakest table in the audit: PNG compressed to **84.4 %** here against a published
**47.1 %**, so the distribution gap dwarfed anything the table could show. A reviewer
asked why the measurement was not simply taken on CLIC. That was the right question and
this section replaces the old table with the answer.

Corpus: **CLIC professional validation**, downloaded from the official archive
(`downloads.compression.cc`, 128.6 MiB, **41 images**), which is almost exactly the
paper's reported corpus size (197 images, 128 MB). No upload was required; the archive is
public.

**Classical baselines, 12 images**, same harness as every other baseline in this audit:

| codec | ours (%) | paper CLIC2019 (%) | delta |
|---|---|---|---|
| WebP --lossless -q100 | 39.362 | 35.56 | **+3.80** |
| JPEG-2000 lossless | 50.215 | 48.64 | **+1.57** |
| PNG (ffmpeg, cl 100) | 76.848 | 47.08 | +29.77 |

Moving from the aerial photograph to CLIC collapses the WebP gap from **+12.7 pp** to
**+3.8 pp** and the JPEG-2000 gap from **+4.6 pp** to **+1.6 pp**. The distribution
confound was real and is now removed.

The PNG row is **not** evidence about anything: ffmpeg's PNG encoder is not the
reference implementation (libpng/zopfli), and a 30 pp gap from an encoder difference is
expected. WebP and JPEG-2000 are the two codecs whose encoders are comparable, and both
now land within a few points of the paper.

**The model, 5 images**, official channel-wise convention:

| image | rate (%) |
|---|---|
| alejandro-escamilla-6 | 12.1318 |
| ales-krivec-15949 | 13.0406 |
| alberto-montalesi-176097 | 14.8875 |
| allef-vinicius-109434 | 17.2361 |
| alexander-shustov-73 | **28.6858** |
| **weighted** | **17.3212** |
| median | 14.8875 |
| **weighted, excluding the outlier** | **14.3779** (4 of 5) |

Three of five images land between 12.1 % and 14.9 %, i.e. **on** the published 14.62 %.
The 28.69 % image is genuinely high-detail content, and the baselines agree: WebP needs
**50.5 %** on that same image against an average of 39.4 %. The weighted mean is pulled
up by it because weighting by samples gives the largest, hardest image the most influence
-- which is the correct convention for a corpus rate, but a poor estimator on n = 5.

**What may and may not be claimed.**

* The model reaches the published regime on this corpus: median 14.89 %, and 14.38 %
  excluding the outlier, against 14.62 %. That is a **2.4 pp** improvement over the
  aerial-photograph measurement (15.51 %), and the previous gap was the corpus.
* With **n = 5** the corpus mean is not pinned down: including the outlier the weighted
  figure is 17.32 %. The paper averages 197 images; we average 5. Stating "14.4 %" or
  "17.3 %" as *the* reproduction would both overstate what 5 images can support.
* This does **not** remove the convention problem of sections 4.4.2-4.4.3: the numerator
  is still the code length of a 32x32, 512-colour token grid while the denominator counts
  source samples. Nor does it remove the row-`t` optimism of section 4.4.5. It removes
  only the **corpus** confound.

### 4.5 Video [M]

Source: **`Netflix_Aerial_4096x2160_60fps_8bit_420.y4m`**, from the Xiph.org collection
the paper draws on -- 4096x2160 @60 fps, aerial footage, **10 frames** streamed over HTTP
Range (133 MB of a 15.9 GB file). Measured on an A100 via
`colab/Video_ChannelWise_Colab.ipynb`.

Frame count and geometry were verified by parsing the Y4M header and frame markers
directly, and the channel-wise unit count (8640 per channel, **25 920 per frame**) was
checked against the official loader's arithmetic.

**The paper's own method, channel-wise.** The paper deliberately uses no inter-frame
information, so iGPT codes each frame independently while the classical codecs below may
exploit temporal redundancy.

| quantity | value |
|---|---|
| frames | **9 distinct** (10 rows, 2 of them identical -- see below) |
| units per frame | **25 920** (8 640 chunks x 3 channels) |
| samples per frame | **26 542 080** |
| **weighted rate, duplicates removed** | **18.1512 %** (18.1508 % as emitted, before de-duplication) |
| per-frame median | 18.1499 % |
| frame-to-frame range | **0.0219 pp** (std 0.00654) |
| published, SI Table 4 high-res | 15.31 % |
| **delta** | **+2.84 pp** |
| cost per frame | 478 s model + 18 s tokenise |

The frame-to-frame spread is **0.02 pp**, so ten frames pin the mean to about
0.007 pp and the +2.84 pp gap is not sampling noise on this clip.

**Classical baselines on the identical frames**, run through the same harness:

| codec | ours (%) | published (%) | delta |
|---|---|---|---|
| H.265 crf 0 (x265 veryslow) | **26.467** | 21.69 | +4.78 |
| H.264 crf 0 (x264 veryslow) | **34.087** | 37.42 | -3.33 |
| FFV1 level 3 | **40.114** | 40.16 | **-0.05** |
| zstd --ultra -22 over raw YUV | **46.292** | 51.04 | -4.75 |

Two results of note. First, **the ordering reproduces exactly**
(H.265 < H.264 < FFV1 < zstd), which is the check that the harness is measuring what the
paper measured. Second, **the model beats the best codec by 1.46x** (18.15 % against
26.47 %), which is the paper's headline qualitative claim for video and it holds here
independently of the corpus.

**What the +2.84 pp gap is, and is not.** The baselines themselves move by up to 4.8 pp
against their published values on this clip (H.265 +4.78, zstd -4.75), so clip-to-clip
variation of several points is established by our own controls. Our local clip -- a
different aerial shot -- measured **16.7162 %** on its first frame, 1.4 pp closer to the
published value than this one. The honest reading is that the model lands in the
published regime and that the residual is dominated by which clip is used, which
`n = 10` frames of one clip cannot resolve.

**Withdrawn numbers.** The previously reported **0.0042 %** and an exact-reconstruction
completion of **73.05 %** came from tokenising whole RGB frames rather than channels, a
defect recorded in section 4.4.4. They are void; the corrected figure differs by three
orders of magnitude.

**Two defects in the Colab output, both recorded rather than smoothed over.**

1. **A duplicate row.** Ten rows were written but only **nine distinct `coded_bits`**
   values appear: `frame_000000` and `frame_000001` are bit-identical
   (\num{38532200.173828} both), so one frame was scored and written twice -- most
   likely across a session restart. De-duplicating gives **18.1512 %** against
   18.1508 % as emitted, a 0.0004 pp difference, so no conclusion depends on it, but
   the correct frame count is **9**, not 10.
2. **A mislabelled column.** Each row recorded the *aggregate* `coded_bytes` rather
   than that frame's own bytes, so the column looks identical across rows while
   `coded_bits` differs. Every rate is computed from `coded_bits` and is correct;
   only that redundant column is wrong. Fixed in the notebook.

Neither moves the measured rate beyond the fourth decimal place, but a reader
reconciling the rows would have been misled by both.

## 5. Text, and the fine-tuning result

### 5.1 The recipe, recovered [M]

The paper's supplementary §S1.4 gives LoRA `α = 32, rank = 8`, batch 1, gradient
accumulation 8, 10 epochs, LR 1e-4, warmup "first 10 %", base model in 8-bit,
trained in bfloat16 [P]. The capsule's own artefacts supply what the paper omits:

| Item | Value | Source |
|---|---|---|
| LoRA dropout | 0.05 | `adapter_config.json` |
| target modules | q, k, v, o, gate, up, down_proj — all 32 layers | `adapter_config.json` |
| trainable parameters | 20 971 520 (≈ 21.0 M) | sum of adapter tensor shapes |
| LR scheduler | linear | `trainer_state.json` |
| warmup steps | 400 of 5 120 = **7.8 %**, not the stated 10 % | `trainer_state.json` |
| optimizer | `adamw_torch`, weight decay 0 | `trainer_state.json` |
| total steps | 5 120 (= 10 epochs × 512) | `trainer_state.json` |
| parallelism | DeepSpeed ZeRO-2, 8 pipeline stages, 1 GPU | checkpoint file naming |
| training corpus | first 64 MB train / next 16 MB validation / rest test [P, arXiv §3.4] | |

### 5.2 The decisive evidence [M]

The shipped `trainer_state.json` logs every 20 steps. Extracting the evaluation
curve:

| step | epoch | eval_loss |
|---|---|---|
| 20 | 0.04 | 2.7491 |
| **1 020** | **≈ 2.0** | **2.4708 (best)** |
| 5 120 | 10.0 | **2.6728** |

**C3 [M].** The best evaluation loss occurs at roughly **2 epochs**, after which 8
further epochs raise it by **8.2 %**. Training loss meanwhile continues to fall
(2.8866 → 1.9826). This is textbook overfitting, and the checkpoint distributed
with the paper is the **final, most overfitted** one.

**Consistency with our independent measurement [M].** Our own result is that this
exact checkpoint compresses *worse* than the base model (§4.1). The training log
and the compression measurement agree, which is strong mutual corroboration: two
independent signals, from different artefacts, both say the authors' adapter is
past its optimum.

### 5.3 Why the gap is domain match, not training quantity [I]

Three pieces of evidence:

1. **The paper's own numbers.** MeDAL: 9.08 % → 9.06 % (relative −0.2 %, i.e.
   inside noise). Pile of Law / eurlex: 5.86 % → 4.90 % (relative **−16.4 %**)
   [P, Suppl. Table 6]. The same recipe, the same base model, the same training
   quantity — wildly different gains. The variable that differs is the match
   between the fine-tuning corpus and the evaluation domain.
2. **The paper's own training curve** shows the optimum at ~2 epochs, so *more*
   training than that is actively harmful — the opposite of a quantity story
   (§5.2).
3. **Our measurement** finds no gain at all on MeDAL, and a small loss.

**Therefore [I]:** the claim "the only difference is training quantity, so more
training would close the gap" is **contradicted by the paper's own training log**.
The defensible claim is that the gain is governed by domain match and early
stopping.

### 5.4 Why the text arms were measured at context 4096 [M]

A reviewer asked the obvious question: this audit names context length as the largest
confound in the fine-tuning result, the paper's own Supplementary Table 9 quantifies it
at about **0.3 pp** -- the same magnitude as the effect being reported -- so why was the
experiment not run at 8192 from the start?

**The two plausible excuses do not hold, and one of them I initially got wrong.**

| excuse | verdict |
|---|---|
| "4096 was all the 12 GB card could hold" | **Not supported as stated, and the precise reason matters.** The paper reports peak GPU memory of **15.49 GB (14.43 GiB) for all three** context lengths, so context length does not drive the footprint -- but the footprint it reports is *8-bit*, and 8-bit is unavailable here: this machine's `bitsandbytes` has no CUDA binary. At bf16 the 8 B model is **14.96 GiB of weights alone**, against **11.94 GiB** of VRAM, so an unquantised model does not fit *at any context length*. A context-driven memory limit is therefore real, but it applies to 4096 just as much as to 8192. |
| "a longer context is slower, and the budget was tight" | **false, and backwards.** The paper's own timings *fall* as context grows: **7323 s at 4096, 6855 s at 6144, 3072 s at 8192** for the same corpus. |

What actually happened: the port was written with a 4096 default, the text arms were
measured with that default, and the confound was only recognised later, when Table 9 was
read for an unrelated reason. The consequence is that this reproduction **cannot separate
the adapter effect from the context effect**, and must carry that caveat wherever the
penalty is quoted.

The fix is a paired run at 8192. Two facts make it tractable: our three test files
total only **57 547 tokens**, so the 15 windows at 4096 tokens become 9 at 8192 with the same
token count; and the paper's own timing for 8192 (3071.86 s) is **faster** than for 4096
(7323.03 s). What it does need is a machine where the model fits at the published 8-bit
precision, which this one cannot do.

**A correction to this section as first written.** The first version of the table above
called the memory excuse "false" outright. That was too strong, and wrong in a way worth
recording: the paper's 15.49 GB peak is an *8-bit* figure, and 8-bit does not work here,
so the effective footprint on this machine is 14.96 GiB at bf16 against 11.94 GiB of
VRAM. Memory is therefore a genuine constraint -- it simply applies equally to 4096 and
to 8192, which is why it cannot justify the choice between them. Correcting one's own
overstatement in place is cheaper than letting a reviewer find it.

Recorded rather than quietly patched because a reader is entitled to know whether a
confound was an unavoidable constraint or an oversight. It was an oversight.

### 5.5 The two "context lengths" were never the same quantity [M]

Found while building the paired notebook, and it changes how every text number here may
be compared with the paper.

**The reference windows by characters.** `compress_with_bgpt_or_llama.py` builds the text
corpus as a string and splits it:

```python
def split(s, max_length):
    return [s[i : i + max_length] for i in range(0, len(s), max_length)]
...
split(self.data[0], context_window_size)      # self.data[0] is text_str, a str
```

`--context_window 8192` therefore means **8192 characters**, not 8192 tokens.

**Our port windows by tokens.** `uic.data.text_units` tokenises the whole text, and the
predictor chunks the resulting id list at `--context-window`. Measured on the same
corpus, the two axes are:

| setting | window unit | windows | mean tokens per window |
|---|---|---|---|
| paper 4096 | characters | 78 | **739** |
| paper 6144 | characters | 51 | **1129** |
| paper 8192 | characters | 39 | **1476** |
| our 4096 | tokens | 15 | **3836** |
| our 8192 | tokens | 9 | **6394** |

**Our "context 4096" carried 2.6x more context than the paper's largest setting.**

**What this invalidates and what it does not.**

* It does **not** invalidate the measurement: base 9.0410 % and +LoRA 9.3529 % are correct
  for what they actually are, a 4096-token window.
* It **invalidates the comparison to the paper's 9.36 % row**. That row is a 739-token
  context; ours was 3836. Comparing them was apples to oranges, and the direction of the
  error is knowable: more context compresses better, so our 9.0410 % should sit *below*
  the 739-token figure, and it does.
* It also **explains an otherwise suspicious coincidence**: our 9.0410 % landed near the
  paper's 8192 row (9.06 %) rather than its 4096 row (9.36 %), which looked like a good
  reproduction but was really a longer context happening to land in the same place.
* The adapter penalty (+0.312 pp at our settings) is unaffected as a measurement, but its
  comparability to the paper's LoRA claim is now doubly limited: different window unit and
  different effective context.

**A second discrepancy this reopens.** The published main text says the text context is
**1024**, while SI Table 9 reports 8192 / 6144 / 4096. Under character units a 1024-char
window is only ~190 tokens, which would compress far worse than 9.06 %; so "1024" cannot
be a character window consistent with Table 9. It is either a token count, a different
experiment, or an error. This audit cannot resolve it and does not guess, but it now has
the unit information needed to state the question precisely.

**The fix** is `colab/Text_Paired_Context_Colab.ipynb`, which measures **both axes** --
character windows (4096/6144/8192, directly comparable with Table 9) and token windows
(4096/8192, comparable with our port) -- crossed with both arms. Ten cells, and the
character axis carries its own independent check: the authors' unmodified harness at 8192
characters reported base **8.9253 %** / +LoRA **9.5154 %** on this corpus.

### 5.6 The paired run at ten cells: confound resolved, absolute level not [M]

The ten-cell run (`results/text_paired.json`, Colab A100,
`colab/Text_Paired_Context_Colab.ipynb`) measured both arms on both window axes.

**Setup checks, all passing** -- these are what make the run trustworthy as a *difference*:

| check | value | expected |
|---|---|---|
| adapter parameters | 20 971 520 | 20 971 520 |
| corpus bytes | 312 053 | 312 053 |
| character windows | 78 / 51 / 39 | 78 / 51 / 39 |
| token windows | 15 / 9 | 15 / 9 |
| dtype | torch.bfloat16 | bf16 |
| peak VRAM | 20.44 GiB | consistent with an 8 B model in bf16 |

**Rates (%)**

| arm | A: 4096 c | A: 6144 c | A: 8192 c | B: 4096 t | B: 8192 t |
|---|---|---|---|---|---|
| base | 6.5322 | 6.3927 | 6.3321 | 6.1657 | 6.1023 |
| +LoRA | 6.9319 | 6.7958 | 6.7308 | 6.6107 | 6.5751 |

**The adapter penalty**

| context | penalty (pp) |
|---|---|
| A 4096 chars | **+0.3997** |
| A 6144 chars | **+0.4031** |
| A 8192 chars | **+0.3987** |
| B 4096 tokens | **+0.4450** |
| B 8192 tokens | **+0.4728** |

Positive at **every** context, range only **0.0741 pp**. The context effect over the same
span is at most **0.2001 pp** (character axis) and **0.0634 pp** (token axis).

**So the confound is resolved.** Context length is *smaller* than the penalty and the
penalty is present regardless of it. The earlier caveat -- "the +0.312 pp and the ~0.30 pp
context effect cannot be separated" -- is now a measurement rather than an open question:
the adapter costs **0.40-0.47 pp** here, and context can account for at most **0.20 pp**.
The direction also agrees with the authors' own harness, where the penalty at 8192
characters was **+0.5901 pp** -- larger than, not smaller than, our 4096-token figure.

**An anomaly that blocks the absolute numbers.** Every cell is 2.6-3.0 pp *better* than
three independent references:

| measurement | rate | instrument |
|---|---|---|
| notebook, base, A@8192 chars | **6.3321 %** | this run, A100, bf16 |
| notebook, base, B@4096 tokens | **6.1657 %** | this run |
| paper, SI Table 9 @8192 | 9.06 % | published |
| authors' unmodified harness @8192 chars | **8.9253 %** | reference script |
| our port @4096 tokens | **9.0410 %** | `uic` |

In bits per token: **2.65-3.01** here against **~3.9** for all three others (perplexity
6.3-8.0 against ~15). The setup checks above rule out the mundane explanations -- wrong
corpus, wrong tokenizer, wrong window counts, a small model, an unloaded adapter, a
mis-shifted scorer (the notebook asserts the scorer against `output.loss` to 1e-4 and it
matched to 1e-8).

Two candidate explanations remain and this audit does not pick between them:

1. **The other three share a degradation.** The paper states the base model is loaded in
   **8-bit** for text. If that path is worse than nominal, or if the reference script's
   CPU/RAM path behaves differently, then paper, harness and our port would all be
   pessimistic together -- which is exactly the pattern in the table.
2. **This run is optimistic by some information leak not covered by the checks.** The
   scorer is verified, but a leak could sit elsewhere, e.g. in how the window tensor is
   presented to the model.

**The discriminating test** is cheap and specific: run the authors' unmodified harness on
the same A100, in the same bf16, on the same HF weights as this notebook. If it then
reports ~6.3 %, source 1 is confirmed and the notebook is right; if it still reports
~8.9 %, source 2 is confirmed and this run has a defect. Until that is done, **no absolute
text rate from this audit should be quoted**, and the report says so.

The penalty finding is unaffected: a constant offset does not change the sign or the size
of a difference, which is why it can be stated firmly while the level cannot.

### 5.7 RETRACTED: our text code length is ~48 % too large [M]

**What was measured.** `tools/isolate_harness.py`, one 4096-token window of
`medal_102285.csv`, on the capsule's own Llama-3-8B, CPU, bf16:

| quantity | bits | bits/token |
|---|---|---|
| plain HuggingFace forward, causal shift | **10 516.775** | **2.5682** |
| the same via `output.loss * (L-1)` | 10 516.776 | 2.5682 |
| relative difference | **5.6e-08** | -- |
| our port's recorded result, same file | **73 410.306** (19 333 scored) | **3.797** |

The first two agreeing to 6e-08 makes the scorer correct *by construction* -- it equals the
model's own loss. And the weights are not in question: the capsule checkpoint is
**byte-identical** to the public one (`tools/compare_llama_weights.py` checked
`model.norm.weight` and the 1 051 MB `model.embed_tokens.weight`; both hashes matched).

**So on identical weights our port reports 48 % more bits than the model's loss implies**
(3.797 / 2.5682 = 1.478).

**What is withdrawn.**

1. **The text base rate of 9.0410 % is wrong.** A correct causal code length on this
   corpus is about **6.0 %**, which is what the paired A100 run measured (6.1657 % at
   4096 tokens, 6.3321 % at 8192 characters).
2. **The claim that text base "reproduces the paper to 0.04 pp" is withdrawn.** It was
   two different inflations landing near the same place, which is the least informative
   kind of agreement.
3. **The specific adapter penalty of +0.312 pp is replaced.** The paired run measures
   **+0.3987 to +0.4728 pp**, positive at all five contexts, on the correct harness. The
   qualitative finding survives; the number does not.
4. **The paper's 9.06-9.36 % cannot be a plain causal code length on this model either.**
   The reference's own `forward_next` is correctly shifted (`shifted_logits =
   logits[:, :-1, :]` against `shifted_input_ids = input_ids[:, 1:]`), so a run of that
   code on this corpus should also land near 6 %, not 9 %. Either the paper's numbers come
   from a different corpus or setting, or the 8.9253 % previously attributed to the
   authors' harness is not measuring what we assumed. This audit does not guess which.

**The defect is bounded but not yet located.** The ratio 1.478 sits close to
`log2(e) = 1.4427` but is not equal to it, so this is **not** a simple natural-log/log2
confusion; a 2.4 % gap remains that a log-base error would not produce. The port's
`iter_steps` was driven directly for this test and raised `IndexError: list index out of
range` at `self._all_tokens[doc_start + i]` because `_all_tokens` is populated by the
runner, so the port arm of the isolation probe did not complete -- the comparison above
rests on the port's *recorded* aggregate, not on a fresh port run. Locating the defect
needs one more pass over `LlamaTokenPredictor.iter_steps`.

**How this survived until now.** The port agreed with the authors' harness to 0.0622 pp,
and both agreed with the paper. Three-way agreement was read as validation. It was in fact
one shared convention being counted three times -- the same failure mode this audit already
named once in section 4.1 ("we searched for an artefact, searched in the wrong place, and
reported the absence of evidence as evidence"). The A100 run is the first measurement in
this project that was validated against the model's own loss rather than against another
implementation of the same idea.

### 5.9 What kind of difference is the text level? An enumerated analysis [M]

An earlier draft of this audit called the paper's text figure "pessimistic". That was too
loose to be useful, and a reader was right to ask whether the difference is one of
**convention**, **formula** or **code**. This section replaces the adjective with four
enumerated candidates and eliminates three of them by measurement.

| # | candidate | what it would mean |
|---|---|---|
| A | **convention** | a different definition of what is counted: the denominator, which distribution codes which token, what is excluded at window edges |
| B | **formula** | the same convention evaluated by a different expression (a mean where a sum is needed, a logarithm base, a mask applied at the wrong point) |
| C | **code** | an implementation defect in one of the two code paths |
| D | **corpus** | a different MeDAL subset, which genuinely compresses differently |

#### A is eliminated for the shift

The reference's `forward_next` uses

```python
shifted_logits     = logits[:, :-1, :]
shifted_input_ids  = input_ids[:, 1:]
```

which is the correct causal alignment -- position *j* scores token *j+1*. And an
independent gather over one window reproduces the model's own loss,
`output.loss * (L-1) = 10 516.776` bits against a computed `10 516.775`, a relative
difference of **5.6e-08**. So neither side is mis-shifted.

#### A is eliminated for the denominator

Container vs extracted payload differ by **0.39 %** (313 272 vs 312 053 bytes); payload
characters vs payload bytes differ by **0.14 %**. A 2.73 pp gap cannot come from either.

#### D is eliminated by measuring the corpus distribution

`tools/test_corpus_subset.py` runs brotli q11 and lzma 9e at the paper's 8192-byte window
over **all 100** CSVs in the sample:

| codec, window | corpus min | corpus mean | corpus max | our three | our mean |
|---|---|---|---|---|---|
| brotli, 8192 | 30.26 | 31.53 | 32.99 | 31.73 / 31.77 / 32.01 | **31.84** |
| brotli, inf | 26.96 | 28.47 | 29.56 | 28.61 / 28.78 / 29.02 | **28.80** |
| lzma, 8192 | 38.72 | 40.68 | 42.40 | 41.09 / 41.36 / 41.86 | **41.44** |
| lzma, inf | 29.55 | 31.25 | 32.41 | 31.39 / 31.51 / 32.15 | **31.68** |

The whole corpus spans only **2.7 pp** for a classical compressor, and our three files are
**harder** than the corpus mean by 0.31-0.76 pp, not easier. A harder subset would push the
model's rate **up**, whereas the gap to be explained has our rate **below** the paper's.
**Corpus choice is therefore eliminated as an explanation of the 2.73 pp gap.**

#### That leaves B and C, and bounds them

Both the reference harness (**8.9253 %**) and our port (**9.0410 %**) report roughly
**1.48x** the code length implied by the model's own loss on identical weights -- 3.87 and
3.80 bits/token against 2.5682 -- while a direct forward pass reports 1.00x.

Two independently written pipelines landing on the same inflation factor is the signature
of a **shared environmental cause**, not of two coincident bugs in two different codebases.
And the only thing that can raise cross-entropy by 1.3 bits/token *without changing the
formula* is the model being in a different state.

The leading candidate is therefore **precision**, not arithmetic: the paper states its base
model is loaded in **8-bit**, this machine cannot do 8-bit at all (bitsandbytes ships no
CUDA binary here), and a quantised 8B model on out-of-domain medical text plausibly loses
on the order of a bit per token. That single hypothesis explains the 1.48x without
requiring either implementation to contain a bug.

**The test that discriminates it is cheap on an A100**: one window, three precisions
(bf16 / 8-bit / 4-bit), and see which reproduces 3.87 bits/token. If 8-bit lands near 3.87,
the difference is a genuine precision effect and *both* the paper's figure and our port's
retracted figure are honest measurements of a quantised model; if 8-bit stays near 2.6, then
both harnesses share a computation defect and locating it becomes mandatory.

**What may not be claimed until that test is run**: neither "the paper is pessimistic" nor
"our port had a bug in the text path". What may be claimed is the elimination above --
shift, denominator and corpus are all excluded by measurement, which already narrows the
question to precision or shared computation.

### 5.8 The training-amount ablation (still not run)

The paper contains **no ablation over training amount** [U]. The inference in
§5.3 is therefore provisional. The experiment that would convert it into a
measurement:

* **Corpus:** eurlex (the domain where the paper reports a large gain), so the
  effect is not floored by a domain mismatch.
* **Sweep:** epochs ∈ {1, 2, 5, 10}, everything else held at the paper's recipe.
* **Two ways to obtain the checkpoints:** (a) train four times from scratch; or
  (b) train once for 10 epochs with `save_strategy="epoch"` and evaluate each
  saved checkpoint — cheaper and holds the trajectory fixed, which is the
  cleaner control.
* **Measure:** compression rate on the held-out test split (everything after the
  first 80 MB), at the paper's context length 8 192, with the base model as the
  zero-epoch point.
* **Decision rule:** if rate falls monotonically with epochs, the quantity story
  survives. If it is U-shaped with a minimum near 2 epochs, §5.3 is confirmed and
  the paper's fine-tuning claim should be restated as being about domain match and
  early stopping.
* **Implementation:** `colab/LMCompress_Colab_H100.ipynb`, Part C, with per-epoch
  checkpointing so a disconnect costs at most one epoch.

**Cost estimate [D]:** 64 MB of training text, batch 1 × grad-accum 8, 10 epochs at
context 2 048 ≈ 5 120 steps. An H100 should complete this in 1.5–3 h.

---

## 10. MAJOR CORRECTION: a log-base error in our own image, video and text rates [M]

**Read this before any number in chapters 4, 5 or 7.**

`torch.log_softmax` returns **natural** logarithms. Every rate in this project computed as
`-log_softmax(logits).gather(target)` and then treated as bits is in **nats**, and must be
multiplied by `log2(e) = 1.4427` to become bits. Verified directly:

| quantity | values |
|---|---|
| `-log_softmax(x)` | 2.40761, 1.40761, 0.40761 |
| `-log2(softmax(x))` | 3.47344, 2.03075, 0.58805 |
| ratio, every element | **0.693147 = ln(2)** |

### Which numbers move

| quantity | as reported (nats) | corrected (bits) | paper | delta |
|---|---|---|---|---|
| video iGPT | 18.1512 | **26.1866** | 15.31 | **+10.88** |
| image CLIC, weighted excl outlier | 14.3779 | **20.7429** | 14.62 | **+6.12** |
| image CLIC, weighted incl outlier | 17.3212 | **24.9892** | 14.62 | +10.37 |
| image CLIC, median | 14.8875 | **21.4781** | 14.62 | +6.86 |
| text A@8192 chars | 6.3321 | **9.1353** | 9.06 | **+0.08** |
| text B@4096 tokens | 6.1657 | **8.8952** | 9.36 | -0.46 |

### Which numbers do not move

* the port's `engine.measure_stream`, which uses `math.log2` -- **true bits**;
* every classical baseline, which emits real bytes;
* the official iGPT rate **15.5144 %**, produced by the reference's real arithmetic coder.

### Consequences

1. **The paper's text figure reproduces.** The port's 9.0410 % against the paper's 9.08 % is
   **-0.04 pp**, and the A100 paired run corrected is 9.1353 % against SI Table 9's 9.06 %
   (**+0.08 pp**). Section 5.7's retraction was itself the error: the port agrees with the
   model's own loss to **2.5 %** (3.797 vs 3.705 bits/token), not 48 %.
2. **The writing agent's 26.19 % was correct**; our 18.1512 % was wrong. The correction
   notice sent earlier asked for a right number to be replaced by a wrong one and is
   withdrawn.
3. **Video is worse than the paper, not better.** 26.19 % against 15.31 % is **+10.88 pp**,
   and against the same-frame H.265 baseline (26.467 %) the model is **1.01x** better, not
   1.46x. The "1.417 vs 1.458 ratio agreement" argument in section 4.5 was built on the
   wrong figure and is withdrawn.
4. **Image is worse than the paper**: 20.74 % against 14.62 % is **+6.12 pp**. The earlier
   "reproduces within 0.24 pp" was an artefact.
5. **The 1.48x "inflation" does not exist.** It was 1.4427 (this unit error) times about
   1.025 (window aggregation). The authors' harness, our port and the paper all agree once
   units are consistent, so the precision test in
   `colab/Precision_Test_Colab.ipynb` has nothing left to explain; it is retained only as a
   sanity check.

### Why this happened, and what it says about the audit

Section 5.7 named the failure mode "we searched for an artefact, searched in the wrong place,
and reported the absence of evidence as evidence". This is a different but related failure:
**a value correct in one frame of reference reused in another without conversion.** The audit
had already flagged exactly this pattern four times in external work -- the writing agent's
`x log2(e)` scalings, the character-vs-token window units, the container-vs-payload
denominators, and the row-`t` convention -- and then committed it itself.

The one thing that would have caught it earlier: every scorer should assert against
`output.loss` **in the same base as the reported number**. The check that existed compared
nats with nats and so passed trivially, which is why the error survived a cross-check
designed to catch exactly this.

## 6. Threats to validity — what a reviewer should attack

Listed deliberately, in descending order of how much damage each would do.

1. **Sample size (all modalities).** 3 text files, 1 audio file, 5 CLIC images, 9 video
   frames. Absolute
   agreement with the paper is not tested; ordering and like-for-like deltas are.
   The one place this bites hardest is §4.1: the LoRA-vs-base gap of 0.312 pp on 3
   files has no error bar. *Mitigation available:* run more files; the CLI is
   resumable and the cost is ~5 min/file locally.

2. **Quantisation, context length and window unit differ from the paper (text).**
   Three separate differences, each now resolved or bounded:

   * **Quantisation.** We did **not** use 4-bit: bitsandbytes has no CUDA binary on this
     machine, so no text run here was quantised (see the retraction in section 4.1). The
     base model ran in bf16 where it fit and with CPU offload where it did not. The paper
     used 8-bit. The paired run of section 5.6 removes this variable.
   * **Context length.** Resolved rather than caveated: section 5.6 measures the adapter
     penalty at **five** contexts on two axes and finds it positive at every one
     (+0.3987 to +0.4728 pp, range 0.0741 pp), while the context effect over the same span
     is at most 0.2001 pp. Context cannot account for the penalty.
   * **Window unit.** Still a genuine comparability limit: the reference windows the text
     by **characters**, our port by **tokens**, so "context 4096" is 739 tokens in one and
     3836 in the other (section 5.5). The absolute rates are therefore not
     interchangeable even though the penalty is.
   * **Unresolved, and larger than any of the above.** Every cell of the paired run is
     2.6-3.0 pp better than the paper, the authors' harness and our own port. Section 5.6
     states the contradiction and the single test that discriminates its two
     explanations. Until that test is run, **no absolute text rate from this audit should
     be quoted**; the penalty figure may be.

3. **The audio conclusion rests on one file.** The entropy argument (§4.3.1) is
   file-specific, and the scoring-misalignment argument (§4.3.3) is
   implementation-level and therefore file-independent. The second is the
   stronger of the two. *Mitigation:* repeat §4.3.3 on more LibriSpeech/LJSpeech
   files; the probe script is parameterised.

4. **The image magnitude rests on five images.** The *structural* claim — the
   official path quantises each 32×32 patch to 512 colours and codes only the
   quantised grid, with no side information — is established by instrumenting the
   official code and is therefore **image-independent**. The *magnitude*
   (14.38 % excluding one outlier, 17.32 % including it, against 14.62 % published) is
   not: with n = 5 the corpus mean is not pinned down. *Mitigation partly taken:* the
   baselines were re-measured on 12 CLIC images and now close to 1.6-3.8 pp (section
   4.4.6), so the earlier corpus confound is gone; the model side still needs more images.

5. **The "reference has the same coder hole" claim (C8) is structural, not
   demonstrated.** We proved the hole exists in our own first implementation and
   fixed it; we argued by inspection that theirs is exposed to the same failure
   mode. We have **not** produced a concrete input that breaks their coder. It
   should be read as a risk note.

5b. **We committed the same class of defect we attribute to the reference.** This
   port's text path had a one-position target lag — scoring each token against the
   distribution for its predecessor — and it survived several rounds of inspection
   because it produced plausible numbers. It was found only when a reviewer
   objected to the *explanation* of a symptom. Two implications worth weighing:
   (a) our own pre-fix numbers were wrong in exactly the silent way we criticise,
   so any measurement of this pipeline should be treated as suspect until it
   carries an explicit alignment assertion (this port now has one, and it fails
   loudly); (b) it is evidence that the mistake is *easy*, which supports the
   plausibility of C9 — while also meaning C9 should be read as "this class of
   error is present in their index arithmetic" (which we derive) rather than as an
   accusation of carelessness.

6. **The video rate is for quantised frames, and it rests on one clip.** Now measured
   (9 distinct frames, 18.1512 %, against 15.31 % published), so "unmeasured" no longer
   applies. Two limits remain. First, the §4.4.2 result carries over: every frame is
   quantised to a 32x32 grid of 512 colours before coding, so the rate describes
   quantised frames rather than the source video, and no decoder can recover what the
   quantiser discarded. Second, it is **one clip** -- an aerial shot -- and our own
   controls show clip-to-clip movement of up to 4.8 pp in the baselines, which is larger
   than the 2.84 pp gap to the paper. *Mitigation:* more clips; the notebook is
   parameterised and streams frames by HTTP Range.

7. **Our rolling-window design differs from the reference (text).** The reference
   re-tokenises each window independently and discards context between windows; we
   carry a rolling window with `prior` context. Our text numbers are therefore
   mildly *better* than the reference's would be at the same settings — but since
   the base-model figure matches the paper to 0.04 pp, the difference is evidently
   not material here. This should still be checked by running the reference's
   scheme side by side.

8. **Container-vs-payload denominator (text baselines).** §4.2 discloses that the
   classical baselines are measured on the CSV container while the model is
   measured on the extracted payload, a 0.13 % difference. It shifts every
   baseline row by ≤ 0.13 pp, so it cannot affect any conclusion, but it is a real
   asymmetry and is reported rather than smoothed over.

9. **The per-window alignment is now asserted, and that matters because we got it
   wrong once.** An earlier revision of this port had a one-position target lag —
   every token scored against the distribution for its predecessor — which produced
   plausible numbers and survived several inspections. It is fixed, and
   `iter_steps` now raises on any misalignment. Residual risk: the assertion covers
   the target side; the *scope* side is verified only by the tiling identity
   (`coded_symbols == symbols`). A reviewer wanting to probe this should perturb
   the offset and confirm the assertion fires.

---

## 7. Raw data and artefacts

Everything below is in `repro_understanding_is_compression/`.

### 7.1 Per-unit measurements (JSONL, one object per unit)

| File | Contents |
|---|---|
| `results/text_base_v2.jsonl` | base Llama, 3 files — the C1/C2 evidence (post-alignment-fix) |
| `results/text_lora_v2.jsonl` | + the authors' published LoRA, same 3 files, same code path |
| `results/igpt_grayscale_collisions.json` | **the full 256 → 48 collision table** for C7 |
| `results/official_image_accounting.json` | **the final official image totals** for C7 |
| `results/baselines_text_medal_3.jsonl` | zlib/lzma/brotli/zstd on the same 3 files |
| `results/baselines_text_medal_raw_3.jsonl` | same, with the raw-text denominator |
| `results/baselines_audio_wsl.jsonl` | FLAC / WavPack / OptimFROG |
| `results/baselines_audio_python.jsonl` | zlib/lzma/brotli/zstd on the WAV |
| `results/baselines_image_wsl.jsonl` | PNG / WebP / JPEG-2000 |
| `results/preflight.json` | environment + asset inventory |

Each model row carries `coded_bits` (the reported code length), `raw_bytes` (the
denominator), `coded_symbols` / `symbols` (which must be equal — the alignment
assertion), `windows`, and `wall_seconds`.

### 7.2 Audit and probe scripts (the evidence for §3 and §4.3/§4.4)

**C7's load-bearing evidence.** These four are the artefacts a reviewer must be
able to re-run to check the image conclusion; they are the ones added in the
second and third review rounds.

| Script | Output | Establishes |
|---|---|---|
| `tools/test_igpt_injectivity.py` | `results/igpt_grayscale_collisions.json` | **the non-injectivity proof**: all 256 grayscale values `(v,v,v)` through the official processor → 48 token ids, 585 colliding pairs, plus real-patch collisions |
| `tools/probe_official_image_accounting.py` | `results/official_image_accounting.json` | the **true** official totals after generator exhaustion: 5 733 chunks, raw 5 870 592, compressed 910 788, rate 15.5144 % |
| `tools/probe_official_image_path.py`, `tools/probe_official_image_preproc.py` | stdout traces | the official path's shape chain `(32,32,1) → repeat → (32,32,3) → (1,1024)`, the 512-cluster processor, the single-code-stream fact, and that the official log's printing is truncated |
| `tools/probe_loader_volume.py`, `tools/probe_denominator_reconcile.py` | stdout traces | the loader census (5 733 chunks) and the 1 572 samples lost to floor division |

**Supporting and other-modality evidence.**

| Script | Establishes |
|---|---|
| `tests/test_coder.py` | coder correctness: 8 tests incl. exact 8.000 bits/byte for uniform bytes and round-trip over 30+ adversarial distributions |
| `tools/audit_bgpt_audio.py` | the audio entropy comparison (§4.3.1) and the target-set enumeration |
| `tools/probe_bgpt_semantics.py` | current-byte vs next-byte argmax agreement (§4.3.3) |
| `tools/probe_bgpt_alignment.py`, `probe_bgpt_context.py`, `probe_reference_selection.py` | the offset sweep and the reference's scored-target enumeration (§3.2) |
| `tools/compare_reference_bgpt.py` | runs the *reference* code and ours on identical bytes, printing both code lengths |
| `tools/probe_igpt_quantization.py` | that `centres * 0.5 + 0.5` inverts the quantiser for **100 %** of pixels |
| `tools/baselines_linux.sh` | classical baselines using the capsule's own Linux binaries |

*`tools/audit_igpt_image.py` is retained but is **no longer evidence for C7**: it
implements the superseded whole-image path and produced the withdrawn 67.43 %
figure (§4.4.3).*

### 7.3 Reproduction of the checkpoint-level claims

`data/model/finetuned_llama/Llama-3-8B_lora/checkpoint-5120/trainer_state.json`
is the only input needed for §5.2; `adapter_config.json` and the safetensors
header for §5.1. All three ship with the capsule and were read, not modified.

---

## 8. What we are *not* claiming

To bound the conclusions:

* We are **not** claiming the paper's text results are wrong — we reproduce them.
* We are **not** claiming the paper's audio or image results were fabricated. We
  claim that (a) the shipped code plus the shipped checkpoint do not reproduce the
  audio number, and that the released checkpoint's argmax matches the byte *at* a
  slot better than the byte *after* it, so the figure it reports does not
  correspond to next-byte prediction of those bytes; and (b) the shipped image path
  quantises each 32×32 channel patch to 512 colours and codes only the resulting
  token ids with no side information, and the quantiser is **provably non-injective
  on that input domain by exhaustive enumeration**, so the path is lossless with
  respect to the token stream and not with respect to the source samples.
* We are **not** claiming that either defect is deliberate, nor that the
  denominator of the image ratio is dishonest — it is internally consistent and
  simply measures a different object than the source image.
* We are **not** claiming the model families are weak. On correctly aligned
  scoring, bGPT, iGPT and Llama all behave competently on the tokens they are given.
* We are **not** claiming the fine-tuning inference (§5.3) is established — it is
  marked [I] and §5.4 specifies the experiment that would settle it.

---

## 9. The one-paragraph version

The text modality reproduces the paper to **0.04 pp** on the base model
(9.041 % vs 9.08 %), which validates the reimplementation; on that same code path
the authors' own published LoRA adapter makes compression **0.31 pp worse**, and the
adapter's own training log independently shows it is the most overfitted
checkpoint of its run (best eval loss at ~2 epochs, +8.2 % by epoch 10). The audio
and image results are also not reproducible as stated, but **for two different
reasons, and the distinction matters**:

* **Audio remains unreconciled, not a demonstrated scoring defect.** The earlier
  alignment mechanism was retracted: the reference keeps 16 rows from each
  17-slot patch, which is the checkpoint's offset-16 pairing. The released
  invocation fails on the bundled file's final non-patch-aligned chunk, so the
  released 1.9376-bpb value has not been independently reproduced end to end.

* **Image is a tokenisation defect.** The official path — instrumented end to end,
  not inferred — splits the source channel-wise into 32×32 uint8 patches, repeats
  each to RGB, maps every sample through the 512-cluster ImageGPT colour quantiser,
  and arithmetic-codes only the resulting token ids with no residual or side
  stream. Exhaustive enumeration of the input domain (all 256 grayscale values)
  shows the quantiser is **not injective**: 256 values collapse to 48 token ids,
  with 585 colliding pairs, and the same collisions occur on real patches (patch
  #116: 110 distinct source values → 32 tokens). The denominator here is
  *internally consistent* — it counts the samples actually processed, and on that
  basis the official path lands **numerically close** to the published value
  (15.51 % measured against 14.62 %) — so nothing about the ratio is wrong; what is
  wrong is that the object being coded is a non-invertible surrogate rather than the
  source image. Per the wording caution in §4.4.2, this validates the **accounting
  convention**, not the dataset-level result.

Both paths are lossless with respect to the object they actually code, and neither
is lossless with respect to the medium it claims to compress.


