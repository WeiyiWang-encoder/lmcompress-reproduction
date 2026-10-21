# Response to review — corrections and new evidence

This document responds point by point to the external review of
`AUDIT_FOR_REVIEW.md`. Three of the five criticisms are accepted outright, one is
accepted with a correction, and one led to finding and fixing a real defect in the
implementation — which changes the headline text numbers.

**Summary of what changed**

| Review point | Verdict | Action |
|---|---|---|
| 1. C7 attribution gap: "you only proved *your* path" | **accepted, and now closed** | drove the official capsule path under instrumentation; the gap is filled and the conclusion changes shape |
| 2. 67.43 % over-claimed as "true lossless rate" | **accepted, then withdrawn entirely** | it came from a superseded whole-image pipeline; the main audit marks it Withdrawn and it is not evidence for C7 |
| 3. 3× denominator inconsistency | **accepted, in the opposite direction to expected** | the official denominator is *not* the source image; both bases now reported |
| 4. order-1 entropy is not a valid lower bound | **accepted** | C5 restated; the entropy argument is now a corroborating signal, not the proof |
| 5. "the last token cannot be encoded" is wrong | **accepted — it was a real bug** | target-slice off-by-one fixed; alignment assertion added; text numbers re-measured |

---

## 1. The image claim: the attribution gap is closed, and the finding inverts

### 1.1 What was done

The criticism was that §4.4.2 measured *our* processor path and inferred the
paper's. That was fair. So the capsule's own image path was driven directly —
`compress_with_igpt.py` → `compressor.py` → `language_model.compress` — with the
capsule's own data loader, and the ImageGPT processor was instrumented at the
**class** level (an earlier instance-level attempt silently failed because
`_retrieve_model_params` re-fetches its own processor from `from_pretrained`,
which is itself a small finding about the code's structure).

Scripts:
`tools/probe_official_image_path.py` (full run),
`tools/probe_official_image_preproc.py` (single-chunk, exact accounting).

### 1.2 Result — the official path, measured

```
=== official loader chunk ===
  shape=(32, 32, 1)  dtype=uint8  range=[0, 38]  pixels=1024

=== running the OFFICIAL language_model.compress() on it ===
  official coder produced 165 bytes
  log_probs shape (1024, 512)

=== processor calls during that call: 1 ===
  call 0: input=(32, 32, 3) uint8 range=(0.0, 38.0) -> tokens=(1, 1024)
          unique_palette_ids=9

=== accounting ===
  loader chunk pixels (official denominator h*w) = 1024
  processor tokens                               = 1024
  symbols actually coded                         = 1024
  official coder bytes                           = 165
  official ratio for this chunk                  = 16.1133 %
  source image = 1714x1142 = 1957388 pixels = 5872164 RGB bytes
```

And the full official run over the whole image. **Note the log is truncated** — the
official evaluator prints only every 100th chunk, so both totals below stop short
of the end (see §1.7); the true finals are in §1.4.

```
raw_length: 5836800          <- last PRINTED value, not the final one
compressed length: 906459    <- likewise
compression rate: 0.1555
compression factor: 6.44
```

### 1.3 What this establishes, and how it differs from both prior positions

The chain, now grounded in the official code rather than in ours:

1. `data_loaders.get_imagenet_iterator(is_channel_wised=True, is_seq=True,
   is_seq=True)` calls `_extract_image_sequence`, which splits each colour plane
   into **32×32 patches** — confirmed: every loader chunk is `(32, 32, 1)`,
   `h*w = 1024`.
2. `language_model.compress` receives one such patch. Because its last axis is 1
   rather than 3, it takes the `np.repeat(..., 3, axis=2)` branch and calls
   `image_processor` on the result — confirmed: **exactly one processor call**, with
   input shape `(32, 32, 3)`, returning `(1, 1024)` tokens.
3. The processor is the standard ImageGPT one: `do_resize=True`, `size=32×32`,
   `do_color_quantize=True`, **512 clusters**. It quantises each pixel's 24-bit
   RGB to one of 512 ids.
4. `compress()` then arithmetic-codes **only those 1024 symbols**:
   `for pdf, symbol in zip(raw_probs, sequence_array)`. There is no second
   stream, no residual, no original-pixel channel.
5. The official ratio denominator accumulates `raw_length += h * w` where `data`
   is the **patch** — confirmed by the observed increment
   `102400 = 100 patches × 1024`.

**Therefore, on the official path:**

* the coded object is a 32×32, 512-colour-quantised token grid, **not** the
  source image;
* ~~per patch, `24 bits/pixel × 1024 pixels = 24 576 bits` of source information is
  represented by `1024 × 9 bits = 9216 bits` of palette indices — a **many-to-one**
  map, so no inverse exists without side information.~~
  **SUPERSEDED BY §1.6.** This cardinality argument is **invalid** and must not be
  relied on: the official path feeds the processor `(v, v, v)` built from a single
  8-bit channel value, so the domain is 256 and the codomain is 512 — cardinality
  proves nothing. §1.6 replaces it with exhaustive domain enumeration, which does
  establish non-injectivity (256 → 48 ids, 585 colliding pairs). This bullet is
  retained only to record the review history;
* the reported rate is `compressed bytes ÷ (Σ patch areas)`, i.e. it compares the
  code length of the quantised grid against **the pixel count of that same grid**,
  so the ratio is internally consistent — but it is a rate for a *lossy
  pre-processed surrogate*, not for the image.

**Crucially, my original audit was wrong about the denominator.** I wrote that the
ratio divides by the original pixel count (1 957 388). It does not — it divides by
the patch-area sum. The review predicted a 3× discrepancy from the opposite
assumption; the actual convention is a third thing again. Both bases are now
reported explicitly.

### 1.4 The honest comparison

> **Correction (second review round).** The rates in this subsection were
> originally wrong: the percentages omitted the ×100, and they carried a stale
> **330-byte** numerator from the superseded whole-image experiment instead of the
> then-observed last-printed value 906 459 bytes; the final exhausted-generator
> total is **910 788 bytes**. The corrected figures come from
> `results/official_image_accounting.json` and the numbers below supersede both
> earlier versions.

| Basis | Official measure | Paper (CLIC2019) |
|---|---|---|
| ÷ processed-channel-sample count (official convention) | **15.5144 %** | 14.62 % |
| ÷ source RGB bytes (5 872 164) | 15.5102 % | — |
| ÷ source pixels (1 957 388) — mixes units | 46.53 % | — |

On the official basis the path **lands close to the published figure** — 15.51 %
against 14.62 % — which is far better evidence than the 0.017 % reported in the
first revision. It is **not** a like-for-like reproduction: this is a different
image from CLIC2019, and §3a of the comparison tables shows the sample compresses
very differently from CLIC under classical codecs (PNG 84.4 % versus 47.1 %
published). The agreement validates the *accounting convention*, not the
dataset-level result. That figure was simply the right numerator over the wrong
denominator, compounded by the ×100 omission — three errors that happened to
produce a number (0.017 %) implausible enough to be worth chasing. The corrected
values all cluster near 15 %, which is the consistency check the mistake should
have failed.

The substantive finding is unchanged and is now demonstrated rather than inferred:
**the pipeline is lossless with respect to the quantised token grid and lossy with
respect to the image** — and §1.6 shows the quantiser is provably non-injective on
the domain this path actually feeds it.

### 1.5 C7 restated

> **C7 (revised).** The shipped reference image path converts each 32×32 source
> patch to a 512-colour ImageGPT token grid, and arithmetic-codes only that grid;
> no side information sufficient to invert the quantisation is emitted. The path is
> therefore lossless with respect to the token sequence and lossy with respect to
> the source image, and its reported rate is the rate of the quantised grid
> measured against the same grid's processed-sample count. This is established by
> driving the capsule's own code, not by inference.

Evidence class: **[M]** for every step, with the instrumentation in §7.

### 1.6 The injectivity test — the second review's required check

The second review made a sharp and correct objection to the *proof* in §1.5 as
originally written. I had argued:

> "24 bits/pixel × 1024 = 24 576 source bits are represented by at most
> 1024 × 9 = 9216 token bits, hence many-to-one."

**That argument is invalid**, and the review is right about why. The official path
splits the source **channel-wise** first, so `language_model.compress` receives a
`(32, 32, 1)` patch and then does `np.repeat(..., 3, axis=2)`. Every processor input
is therefore `(v, v, v)` built from **one** 8-bit channel value. The domain is 256
values, the codomain is 512 ids, and 512 > 256 — so a cardinality argument proves
nothing, and the map could in principle have been injective.

The only rigorous approach is to enumerate the domain. Done
(`tools/test_igpt_injectivity.py`, results in
`results/igpt_grayscale_collisions.json`), pushing all 256 values `(v,v,v)` through
the **official processor**:

| Quantity | Value |
|---|---|
| domain size | 256 |
| image size (distinct token ids) | **48** |
| injective | **False** |
| colliding source-value pairs | **585** |
| largest collision class | token 391 ← **7 distinct values** (78…84) |

Concrete collisions, reproducible from the JSON:

```
token  71  <-  grayscale values [0, 1, 2, 3]
token  59  <-  grayscale values [8, 9, 10, 11]
token  48  <-  grayscale values [169, 170, 171, 172, 173, 174, 175]
token 391  <-  grayscale values [78, ..., 84]
```

And the same collisions occur on **real source patches**, so this is not an
artefact of synthetic constant inputs — patch #116 contains 110 distinct source
values mapping to 32 distinct tokens, with 29 token ids receiving more than one
source value:

```
token 143  <-  source values [35, 36, 37, 38, 39]
token 341  <-  source values [30, 31, 32, 33]
token 500  <-  source values [128, 129, 131, 132, 133]
```

**C7 is therefore proved by exhaustive enumeration rather than by cardinality:**
there exist `a ≠ b` in `{0,…,255}` with `token(a,a,a) == token(b,b,b)`. The coder
emits only that id and no residual, so no decoder can distinguish the two source
values. This is the clean mathematical statement the review asked for, and it does
not depend on any claim about the source image's colour depth.

### 1.7 The volume question — answered, with a correction

The review also asked whether the official denominator's deficit against the source
(`5 836 800` vs `5 872 164` = 35 364 samples, 0.602 %) indicated dropped edge data.
Counting the loader exactly (`tools/probe_loader_volume.py`,
`tools/probe_denominator_reconcile.py`):

| Quantity | Value |
|---|---|
| loader chunks to exhaustion (channel-wise, is_seq) | **5 733** |
| Σ h·w over those chunks | **5 870 592** |
| source RGB channel samples | 5 872 164 |
| dropped by loader floor division | 1 572 (524 per channel) |
| last `raw_length` **printed** by the official run | 5 836 800 |

So there **are** two real volume losses — the loader's floor division drops 524
samples per channel — but they are much smaller than the printed figure suggests.

**The 35 364 gap is a print artifact, not dropped data.** The official evaluator
gates its printing on `if print_count % 100 == 0`, so the last `raw_length` in its
log corresponds to an earlier chunk than the final one; the tail of the run is
never reflected. This is also why the log's own last `compression rate` (0.1553) is
not the true final rate: its numerator and denominator are both truncated, at
different points.

Consequence for this audit: **no rate should be quoted from the official log.** The
correct final figures are being computed directly by
`tools/probe_official_image_accounting.py`, which reimplements the official
accumulation over the same generator and reports it after exhaustion. **Result: 5 733 chunks, raw_length 5 870 592, compressed 910 788, rate 15.5144 %, ratio 6.4456×.** That also
means the review's suggestion that the deficit might be a *second* independent
non-losslessness argument is correct in principle but smaller than the raw gap
implies: ~0.027 % of samples (1 572 / 5 872 164), not 0.602 %.

---

## 2. The 67.43 % figure: WITHDRAWN (superseded by the third review round)

> **Status change.** The second review round asked for this figure to be demoted
> from a rate to an illustrative completion. The third round went further, and is
> right: because the 330-byte numerator came from a **different, whole-image**
> pipeline of our own construction, 67.43 % is not a completion for the official
> path at all. The main audit now marks the whole section **Withdrawn** (§4.4.3).
> It is retained below only as review history, and it is **not evidence for C7**.
> C7 rests entirely on the exhaustive injectivity test (§1.6).

Accepted. Calling it a "true lossless rate" was over-strong: it is one particular
exact-reconstruction completion, not a bound. A better residual model could do
better, and the review is right that our own WebP result (48.28 %) already proves
67.43 % cannot be a lower bound.

**Restated:**

> As an *illustrative* exact-reconstruction completion, coding the residual with
> zlib −9 requires 67.43 % of the raw RGB size on this image. This is **not a
> lower bound** and not a property of the method; it demonstrates only that a
> 330-byte token stream omits a large amount of information required for exact
> reconstruction.

**Full construction, so the number is independently reproducible:**

1. Take the 32×32 token grid the processor returns.
2. Map each token id through the palette `clusters * 0.5 + 0.5`, giving RGB
   floats in `[0, 1]` (validated: nearest-centre lookup against this palette
   reproduces the processor's ids for **100 %** of pixels —
   `tools/probe_igpt_quantization.py`).
3. Convert to `uint8` with `round(x * 255)`, giving a 32×32×3 image.
4. Upscale that to the original size with **PIL `Image.BILINEAR`**.
5. Residual = `(original_rgb.astype(int16) - upscaled.astype(int16))`, then cast
   to `int8` (the difference fits in ±127 on this image).
6. `zlib.compress(residual.tobytes(), 9)` over the **whole image at once**;
   the denominator is `1714 × 1142 × 3 = 5 872 164`.

Note the interface detail that changes the number: step 5 uses the **signed
difference**, not `abs()`. An earlier version of `audit_igpt_image.py` coded
`abs(err)` per plane and produced 3 959 410 bytes / 67.43 %; the run reported in
§4.4.2 of the audit used the signed residual at the whole-image level. Both are
defensible; the signed form is what the CLI implements. The distinction is
recorded here so the two figures can be reconciled.

---

## 3. Denominator consistency: both bases now reported

Accepted. §4.4.1 measured classical codecs against `5 872 164` RGB bytes while
§4.4.2 measured the model against `1 957 388` pixels. That is now tabulated on
three bases (§1.4), and the reason the paper (and the official code) uses a
pixel-like basis is stated: the 32×32 grid's natural unit is a pixel, and the
official denominator is the patch-area sum.

Also corrected: the phrase "paper's convention" was being applied to a convention
I had inferred. It is now attributed to the specific line of official code that
implements it, and quoted.

---

## 4. The entropy argument: reclassified

Accepted, and the caveat that was buried in a parenthesis is promoted to the main
statement.

> **C5 (restated).** The reference implementation reports 1.94 bits/byte for a WAV
> whose order-1 entropy is 3.17 bits/byte. The empirical order-1 entropy is **not**
> a lower bound for a model with longer context, so this discrepancy is not by
> itself a proof. It is, however, a strong indicator, and it is **corroborated
> independently** by the direct scoring test in §4.3.3: the released checkpoint's
> argmax matches the byte *at* a slot better than the byte *after* it at every
> window length tested (0.415–0.756 vs 0.111–0.551). The scoring test is the
> load-bearing evidence; the entropy comparison is supporting.

The word "proof" has been removed from C5 throughout.

---

## 5. The "last token" claim — accepted, and it was a genuine bug

The review is right, and this is the most valuable correction in the response.

### 5.1 The error

I wrote that the final token of each document was uncodeable "because a causal
model cannot predict anything from the final position". That is wrong twice over:

* a causal LM predicts token `j` from position `j-1`, so the **last** token is
  predicted normally, by the second-to-last position;
* what a causal LM cannot do is predict a token **after** the end of the sequence.

The missing token was an artefact of my window bookkeeping, and the wording
mis-attributed it to the model. Worse, the review's scrutiny of that wording
exposed the underlying defect.

### 5.2 The actual defect

The target slice was one position late:

```python
scope   = logits[:, offset     : offset + n_target, :]
targets = ids[   :, offset + 1 : offset + 1 + n_target]   # <- one late
```

Because position `p` predicts token `p+1`, the prediction for the token at index
`offset + 1` lives at `logits[offset]` — so pairing `scope[0]` with `targets[0]`
scores the token at `offset+1` against the distribution for the token at `offset`.
**Every token was scored against the distribution for its predecessor.** A
compensating "drop the first prediction" then removed the spurious symbol, which
is why the totals nearly balanced and why the bug was invisible: the run produced
plausible numbers.

This is the same *class* of defect I had attributed to the reference
implementation's audio path (§3.2/C9). Finding it in my own code is uncomfortable
and worth recording explicitly: a one-position lag produces a plausible-looking
compression rate, so it survives casual inspection.

### 5.3 The fix, and the guard that now prevents recurrence

The scored region now carries one token of context on **each** side:

```
feed = [ predecessor , scored tokens ... , successor ]
```

which reduces to `[scored tokens, successor]` on the first window and
`[predecessor, scored tokens]` on the last. Neither the predecessor nor the
successor is scored; the scope starts at `offset - 1` and targets at `offset`.

A hard assertion now compares every target against the document token it must
equal, so a lag of any size fails loudly instead of producing a plausible number:

```python
for i in range(n_target):
    expected = int(self._all_tokens[doc_start + i])
    if int(targets[0, i]) != expected:
        raise RuntimeError(f"target misalignment at {i}: ...")
```

Window tiling is also now exact for every payload length, including `n = 1`:
`[4094, 4094, 4094, 4094, 2962]` sums to 19 338. **No token is dropped**, and the
`uncoded_symbols` field is consequently 0.

### 5.4 Effect on the reported numbers

The corrected alignment does not flatter the result — it makes it slightly worse
on every file, which is the expected direction if the old slice was scoring an
easier target. Both text arms were re-measured:

| File | raw bytes | base (buggy) | base (corrected) | +LoRA (buggy) | +LoRA (corrected) |
|---|---|---|---|---|---|
| medal_102285.csv | 103 534 | 8.8534 % | **8.8631 %** | 9.1180 % | **9.1269 %** |
| medal_103851.csv | 104 332 | 9.2331 % | **9.2365 %** | 9.5476 % | **9.5506 %** |
| medal_108122.csv | 104 187 | 9.0120 % | **9.0222 %** | 9.3703 % | **9.3794 %** |
| **total** | **312 053** | **9.0333 %** | **9.0410 %** | **9.3459 %** | **9.3529 %** |

Ratios: base **11.0607×**, LoRA **10.6919×**.

Two consequences:

1. **C1 improves.** The base-model figure moves from 0.05 pp to **0.04 pp** from
   the paper's published 9.08 %. The corrected code is a closer reproduction, not
   a worse one — which is what one expects when a scoring lag is removed: the
   model's own probabilities are now used for the right tokens.

2. **C2 is unchanged in direction and magnitude.** The LoRA penalty is
   +0.312 pp corrected versus +0.313 pp before. This is exactly as predicted: the
   alignment defect was common to both arms, so it largely cancels in a paired
   comparison, and the conclusion — the authors' published adapter compresses MeDAL *worse*
   than the base model — survives the fix.

3. **No token is dropped any more.** `coded_symbols == symbols == 57 547`, so the
   `uncoded_symbols` field is 0 and the §4.1.1 disclosure is obsolete.

---

## 6. What this response does *not* change

* **C1** (text reproduces the paper) — unaffected in substance; numbers refreshed.
* **C2** (the authors' published LoRA is worse on MeDAL) — unaffected in direction.
* **C3** (the shipped checkpoint is the most overfitted one) — unaffected; it
  rests on the adapter's own training log.
* **C6** (the audio checkpoint reconstructs rather than predicts) — unaffected;
  it is a direct measurement and does not depend on my alignment.
* **C9** (the reference's audio scoring is misaligned) — unaffected, and now more
  interesting: the same failure mode was independently present in my own code,
  which is evidence that it is an easy mistake to make rather than an exotic one.

---

## 7. Evidence added by this response

| Artefact | Establishes |
|---|---|
| `tools/probe_official_image_path.py` | drives the official image pipeline; records its denominator and reveals that its log printing is truncated |
| `tools/probe_official_image_preproc.py` | records the processor call `(32,32,3) → (1,1024)`, the 512-entry palette, the 165-byte per-patch code length, and that the denominator is the patch area |
| `tools/test_igpt_injectivity.py` | **the exhaustive injectivity test**: all 256 grayscale values → 48 token ids, 585 colliding pairs, plus real-data collisions |
| `tools/probe_official_image_accounting.py` | the true final official accounting: 5 733 chunks, 5 870 592 raw, 910 788 compressed, **15.5144 %** |
| `tools/probe_loader_volume.py`, `tools/probe_denominator_reconcile.py` | loader census (5 733 chunks) and the 1 572-sample floor-division loss |
| `tools/probe_igpt_quantization.py` | that `clusters*0.5+0.5` inverts the quantiser for 100 % of pixels |
| `src/uic/models.py` (`iter_steps`) | the corrected alignment plus the assertion that prevents recurrence |
| `results/text_base_v2.jsonl`, `results/text_lora_v2.jsonl` | re-measured text arms under the corrected alignment |

---

## 8. Revised confidence

| Claim | Before | After | Basis for change |
|---|---|---|---|
| C7 (image) | medium-high with attribution gap | **high** | official path instrumented end to end, and the quantiser's non-injectivity proved by exhaustive domain enumeration |
| C5 (audio) | high (called a proof) | **high**, but on the scoring test alone | entropy argument demoted to corroboration |
| C1, C2 (text) | high | **high** | re-measured under corrected alignment; direction unchanged |
| residual 67.43 % | presented as a rate | **withdrawn** | came from the wrong (whole-image) pipeline; not evidence for C7 |
