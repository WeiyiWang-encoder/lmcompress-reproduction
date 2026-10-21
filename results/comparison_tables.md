# Reproduction comparison tables

Metric convention: **compression rate = compressed / original, in percent, lower
is better** — this is the paper's own metric (NMI Methods). `ratio` is the
reciprocal and is shown for readability.

Every measured row states its denominator explicitly, because two of the paper's
modalities compare against bytes the codec never actually coded:

* **text** — the paper's container is a CSV whose size differs slightly from the
  UTF-8 payload fed to the model, so both denominators are reported;
* **image** — the paper's pipeline runs every image through ImageGPT's processor,
  which **resizes to 32×32 and quantises colours to 512 clusters** before the
  "lossless" coder sees it, yet the ratio divides by the *original* pixel count.
  That is flagged, not silently reproduced.

Local subsets:

| Modality | Files measured | Total |
|---|---|---|
| text (MeDAL) | 3 sample CSVs | 0.31 MB |
| text (Pile of Law / eurlex) | 3 sample CSVs | 0.35 MB |
| audio | 1 WAV (`LJ001-0004.wav`) | 82 220 PCM bytes |
| image | 1 PNG (1714×1142) | 5 872 164 RGB bytes |
| video | 4k Y4M, 60 frames (4096×2160) | 1 592 MB raw |

The paper's numbers come from ~100 MB–180 MB per dataset (~128 MB for images), so
absolute agreement is expected to be loose and is not the point; the point is
whether the ordering, the model-vs-model gap and the model-vs-baseline gap
reproduce.

---

## 1. Text — MeDAL

Denominator: UTF-8 bytes of the sample CSV files (the same bytes the model was
given).

| Method | measured rate | measured ratio | paper rate | Δ (pp) |
|---|---|---|---|---|
| LMCompress\* (raw Llama-3-8B, no fine-tuning) | **9.041 %** | 11.06× | 9.08 % | **−0.04** |
| LMCompress (Llama-3-8B + the authors' published LoRA) | **9.353 %** | 10.69× | 9.06 % | **+0.29** |
| brotli q11 | 28.805 % | 3.47× | 27.93 % | +0.88 |
| lzma preset 9e (7z-equivalent) | 31.726 % | 3.15× | 30.25 % (7z) | +1.48 |
| zstd --ultra -22 | 31.994 % | 3.13× | 30.45 % | +1.54 |
| zlib 9 | 35.018 % | 2.86× | — | — |

Per-file, same files and identical code path (the only difference is whether the
LoRA adapter is attached):

| File | raw Llama-3-8B | + authors' LoRA | Δ |
|---|---|---|---|
| medal_102285.csv | 8.863 % | 9.127 % | **+0.264** |
| medal_103851.csv | 9.237 % | 9.551 % | **+0.314** |
| medal_108122.csv | 9.022 % | 9.379 % | **+0.357** |
| **weighted mean** | **9.041 %** | **9.353 %** | **+0.312** |

### What this table says

1. **The model reproduces.** The raw model — which is the paper's LMCompress\* —
   lands **0.04 pp** from the published 9.08 %. That is essentially a
   like-for-like match given 4-bit versus 8-bit and 4096 versus 8192 context.
2. **The baselines reproduce in ordering and rough magnitude.** brotli is the
   strongest classical method in both, zstd and 7z sit close together, zlib is
   worst. The systematic +1 to +1.5 pp offset is expected: these are 0.31 MB
   samples versus the paper's ~180 MB, and every classical codec improves with
   more input.
3. **The headline claim holds**: the large model is ≈3.2× better than the best
   classical baseline (9.03 % vs 28.81 %), which is the paper's "nearly
   one-third of zpaq" claim in the same regime.
4. **The fine-tuning claim does NOT hold on this sample.** The paper reports
   9.08 % → 9.06 % (0.02 pp, i.e. nothing) on MeDAL. We measure the
   authors' adapter as **worse than the base model by 0.312 pp**, consistently
   across all three files. §6 of the report explains why, from the checkpoint's
   own training log: it is the *most overfitted* checkpoint of the run.

---

## 2. Audio — LJSpeech sample

Denominator: raw PCM payload bytes (82 220).

| Method | measured rate | measured ratio | paper rate (LJSpeech) |
|---|---|---|---|
| optimfrog --preset max | 27.311 % | 3.66× | 25.42 % |
| wavpack -hh -x3 | 35.303 % | 2.83× | 33.15 % |
| flac -compression_level 12 | 45.334 % | 2.21× | 41.26 % |
| zlib 9 | 62.627 % | 1.60× | — |
| **bGPT (reference code + released weights)** | **not reported by this port** | **not reported by this port** | 18.26 % |

The classical audio baselines reproduce in **both ordering and magnitude**:
OptimFROG < WavPack < FLAC exactly as the paper reports, and each lands within
2–4 pp of the published number, which is the right size of gap for a single 82 KB
file versus the paper's ~100 MB. This is independent evidence that the local
measurement setup is sound — which matters, because it is the control that makes
the bGPT row credible.

**The bGPT result remains unresolved.** The released code plus weights report
**1.9376 bpb** on the bundled sample, whereas the paper reports 18.26 % on
LJSpeech. The two use different evaluation material, and this reproduction does
not establish a mechanism for their difference. The former port result has been
removed because it used the withdrawn contiguous-logit mapping.

> **WITHDRAWN / 已撤回**: the numbers 13.12, 13.1233, 14.1804 and 15.7844 bits/byte,
> and any "misalignment" or "6.29x" explanation, came from an index error in **our**
> probe (we used flat row `16+k` where the grid has 17 slots per patch, so the correct
> row is `17*(k//16) + (k%16)`). They are **void**. The verified position: the
> reference's pairing is the byte the model is most confident about (0.3052 at offset
> 16 vs 0.0804 at offset 1), the scheme is causal (Δ = 0 under a future-byte
> perturbation), and 1.9376 bpb is its genuine code length. See
> `docs/REVIEW_GUIDE_FOR_AGENT.md` §5.2.


---

## 3. Image — one 1714×1142 PNG

Denominator: raw RGB pixels (5 872 164) for the classical codecs; the original
pixel count (1 957 388) for the ImageGPT row, which is the paper's convention.

### 3a. Classical baselines

| Method | measured rate | measured ratio | paper rate (CLIC2019) |
|---|---|---|---|
| webp -lossless -q100 | 48.279 % | 2.07× | 35.56 % |
| jpeg-2000 lossless | 53.252 % | 1.88× | 48.64 % |
| png (ffmpeg cl100) | 84.381 % | 1.19× | 47.08 % |

Ordering reproduces (WebP clearly best, JPEG-2000 ≈ PNG class). Absolute values
are much worse than the paper's because this test image is 1714×1142 of
high-detail aerial content, which resists lossless coding far more than CLIC's
professional photographs — PNG at 84 % versus the paper's 47 % is the clearest
signal of that. **A like-for-like image comparison needs CLIC2019 or ILSVRC
images.**

### 3b. ImageGPT on the **official** path, instrumented end to end

> **Superseded figures.** An earlier version of this section reported 0.017 % and a
> "true lossless rate" of 67.43 %, both derived from a whole-image experiment of
> *our own* construction. That was the wrong pipeline **and** the wrong
> denominator. The official path is patch-wise and channel-wise, and the figures
> below come from driving the capsule's own code.

Measured by driving `compress_with_igpt.py` → `language_model.compress` with the
capsule's own loader (`tools/probe_official_image_accounting.py`), after the
generator is exhausted — the official log's own totals are truncated because it
prints only every 100th chunk:

| Quantity | Value |
|---|---|
| chunks processed (32×32, channel-wise) | **5 733** |
| `raw_length` (final) | **5 870 592** |
| `compressed_length` (final) | **910 788** |
| **rate = comp / raw** | **15.5144 %** (ratio 6.4456×) |
| source RGB channel samples | 5 872 164 |
| samples the loader never processes (floor division) | 1 572 (0.027 %) |
| paper, CLIC2019 | 14.62 % |

**On the official convention this lands close to the published figure** (15.51 % vs
14.62 %), which validates the accounting convention. It is *not* a like-for-like
reproduction: this is a different image from CLIC2019, and §3a shows this sample
compresses very differently from CLIC under classical codecs.

The path is nevertheless not lossless with respect to the source image, and this is
now proved by **exhaustive enumeration** rather than by a cardinality argument
(`tools/test_igpt_injectivity.py`): every processor input along this path is
`(v, v, v)` built from **one** 8-bit channel value, so the domain is 256 values —
and the 512-cluster quantiser maps them onto only **48** distinct token ids, with
**585 colliding pairs**:

```
token  71  <-  grayscale values [0, 1, 2, 3]
token  48  <-  grayscale values [169, 170, 171, 172, 173, 174, 175]
token 391  <-  grayscale values [78, ..., 84]     (largest class, 7 values)
```

The same collisions occur on real patches: patch #116 holds 110 distinct source
values which map to 32 tokens, with 29 token ids receiving more than one value.
Since the coder emits only the token id and no residual, no decoder can recover
which source value it was. **Lossless w.r.t. the token stream, lossy w.r.t. the
image** — with the denominator itself internally consistent, so this is a
*tokenisation* defect, not a denominator mismatch.

---

## 4. Video — 4K Y4M (4096×2160, 60 frames, raw 1 592 MB)

Not measured locally; the lossless H.264/H.265/FFV1 passes are queued. The Y4M
container is parsed in-process (`uic/y4m.py`) with no ffmpeg dependency, and the
video path reuses the image path frame by frame, exactly as the paper does (it
deliberately ignores interframe information). The `iGPT` caveat of §3b applies
unchanged to every frame.

---

## 5. Headline summary

| Modality | This port reproduces the paper? | Evidence |
|---|---|---|
| text | **yes, closely** | raw 9.041 % vs 9.08 %; LoRA 9.353 % vs 9.06 % |
| audio | **unresolved** | the released materials report 1.9376 bpb, but the supplied invocation fails on the final non-patch-aligned chunk |
| image | baselines yes; official model rate reproduces (15.51 % vs 14.62 %) but the path is not source-lossless | §3b: quantiser maps 256 grayscale values onto 48 tokens |
| video | pending | iGPT caveat applies to every frame |

Two findings matter most for the report:

1. **The text modality is solid**, and the fine-tuning gain on MeDAL does not
   exist — the authors' published adapter makes compression worse by 0.31 pp on every file
   tested, because it is the most overfitted checkpoint of the run (best eval loss
   at ~2 epochs, then 8 more epochs of degradation). See `docs/REPORT_CN.md` §6.

2. **The audio and image findings have different status**, and the distinction
   matters for how the paper should be read:
   * **audio — unresolved reproducibility.** The earlier scoring-alignment
     explanation is withdrawn. The reference's 17-slot selection matches the
     checkpoint's offset-16 pairing, but its supplied invocation cannot complete
     the bundled file without a tail-padding repair.
   * **image — a tokenisation defect.** The denominator is internally consistent
     (it counts the samples the pipeline actually processed) and the rate
     lands close to the published figure, validating the accounting convention. What
     fails is that the preprocessing is **non-injective**
     — the 512-cluster quantiser maps 256 distinct grayscale values onto 48 token
     ids — so the object coded is a surrogate from which the source cannot be
     recovered.

   In both cases the model families themselves (bGPT, iGPT, Llama) behave
   competently on the object they are actually given.
