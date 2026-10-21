# Reproduction notes — LMCompress ("Lossless data compression by large models")

Working folder: `repro_understanding_is_compression/`
Reference capsule (read-only): `<capsule>`

---

## 1. What was built

A clean-room implementation of the paper's pipeline — tokenise → large model →
per-symbol probability → arithmetic coding → lossless bitstream — in
`src/uic/`:

| Module | Role |
|---|---|
| `bitio.py` | bit packing / unpacking |
| `coder.py` | **own** integer arithmetic coder + probability quantisation |
| `engine.py` | resumable measurement driver, crash-safe progress records |
| `models.py` | the three predictors: bGPT (bytes), ImageGPT (pixels), Llama (BPE) |
| `data.py` | four modality loaders (audio / image / video / text) |
| `runner.py` | window planning, per-unit batch running, aggregation |
| `y4m.py` | self-contained YUV4MPEG2 reader (no ffmpeg needed) |
| `preflight.py` | environment + asset verification |
| `cli.py` | `python -m uic.cli {audio,image,video,text,preflight}` |

Power-loss protection, as required by the environment:

* progress is committed to disk **after every forward pass** (`ChunkCursor`),
  so at most one window is ever recomputed;
* every result file is written through an atomic temp-and-`os.replace`;
* per-unit results are appended to JSONL, so a run killed at 90 % still yields
  numbers for the 90 %;
* re-running the identical command skips completed work.

---

## 2. A real bug found in the arithmetic coder, and fixed

The first coder implementation passed the uniform-byte test (exactly 8.000
bits/byte, perfect round trip) but **corrupted the stream on rare symbols**.

Range coders subdivide the current interval as
`low + width * cdf_lo // FULL`. If a symbol is quantised down to an interval of
width 1 while the current width has already fallen below `FULL = 2**32`, then
`width * cdf_hi // FULL == 0`, so `high` can end up *below* `low`. The coder then
emits a stream that decodes to a **wrong symbol** — silently, without an
exception. This was caught by a round-trip test over deliberately skewed
distributions, at a symbol whose true probability was `9.06e-11`.

Fix: `min_interval_width()` reserves `FULL / 2**20` per symbol, making the coder
exact. The encoder now also asserts against interval collapse instead of corrupting
output.

**A correction to the framing of this fix.** An earlier revision said the reserve
"cannot distort a reported ratio because 20 bits exceeds the 8 bits/byte ceiling of
a byte alphabet". That is wrong: 256 symbols does not bound the surprisal of the
true byte, so a model assigning the true symbol `p < 2^-20` *is* affected by the
floor. The reserve does not touch the headline numbers for a different reason —
**the reported rate is never computed from the coder's quantised probabilities**;
it is `Σ -log₂ q(true symbol)` on the model's own float32 values, computed before
any quantisation. The quantised cdf affects only the arithmetic-coded bitstream
length, which is tracked separately.

**On the reference implementation** we note only a *structural* exposure:
`normalize_pdf_for_arithmetic_coding` adds a machine-epsilon term and renormalises,
which leaves the same failure mode conceivable for very small probabilities. We
have **not** constructed a concrete input that makes their coder fail, so this is a
risk note, not a demonstrated defect.

---

## 3. Findings that affect how the paper's numbers should be read

### 3.1 The reference `audio` invocation crashes

```
python compress_with_bgpt_or_llama.py --data_type wav ...
```
imports `data/model/bgpt/utils.py`, which requires the declared dependency
`samplings>=0.1.7`. It is listed in `pyproject.toml` and locked in `uv.lock`; it
must be installed in the active environment. An earlier review incorrectly called
this a missing capsule module after using an interpreter outside the project
environment; that conclusion is withdrawn.

There is also a second boundary failure in the documented `--context_window
8160` invocation. The 82,220-byte bundled WAV ends with a 620-byte chunk;
`byte_collate()` adds two 16-byte structural patches but does not pad that chunk
to a 16-byte boundary. The bGPT forward pass then fails reshaping 652 values into
16-byte patches. Thus this invocation cannot produce a full-file reference
result without an additional padding policy.

`bits2bytes` is not present in the capsule. The undefined name
`compression_utils` occurs only in the commented-out `encoder_decoder()` path.

### 3.2 The text path cannot run on a 12 GB card as shipped

`compress_with_bgpt_or_llama.py` loads Llama-3-8B with

```python
AutoModelForCausalLM.from_pretrained(model_path)   # no dtype, no quantisation
self.model.to(self.device)                          # whole 8B model onto the GPU
```

with no `torch_dtype`, so it materialises **float32** — about **32 GB** of
weights. That is an instant OOM on any consumer GPU. Loading in 4-bit (NF4,
double quantisation) takes the same model to **5.71 GB and 18 s**, which is what
this port does; the LoRA adapter is then merged so the coding loop is not slowed
by per-layer adapter dispatch.

### 3.3 Audio: the released full-file result remains unreconciled

The bundled sample `data/test_audios/LJ001-0004.wav` is 16 kHz mono **8-bit**
PCM with 82,220 payload bytes. The reference pairs its first 16 byte-level rows
from each 17-slot patch with bytes one patch ahead. The checkpoint is most
confident about exactly that offset-16 byte (0.3052 mean probability versus
0.0804 at offset 1), so the earlier claim of a scoring misalignment is withdrawn.
The code and weights nevertheless do not supply an end-to-end reproducible run:
the final 620-byte chunk is not padded to a 16-byte boundary.

> **WITHDRAWN / 已撤回**: the numbers 13.12, 13.1233, 14.1804 and 15.7844 bits/byte,
> and any "misalignment" or "6.29x" explanation, came from an index error in **our**
> probe (we used flat row `16+k` where the grid has 17 slots per patch, so the correct
> row is `17*(k//16) + (k%16)`). They are **void**. The verified position: the
> reference's pairing is the byte the model is most confident about (0.3052 at offset
> 16 vs 0.0804 at offset 1), and the supplied full-file invocation remains
> unreproducible until its missing module and tail padding are repaired. See
> `docs/REVIEW_GUIDE_FOR_AGENT.md` §5.2.


The reference keeps the first 16 rows of each 17-slot patch and pairs those rows
with payload positions `16 .. 16+n-1`. In the untrimmed logit tensor this is the
non-contiguous row sequence `17*(k//16)+(k%16)`, not the contiguous slice `16+k`.
The port now implements that exact selection.

**Consequence for the report:** the released materials state **1.9376 bpb
(≈4.13×)** on the bundled sample, while the paper reports **18.26 % (≈1.46 bpb)**
on LJSpeech. This reproduction does not independently reproduce the released
full-file number and does not reconcile the two.

**We do not know why.** The most likely explanation is that the released
`weights-audio.pth` is a **different training run** from the one behind 18.26 %.
Testing that needs either the authors' configuration or more of their data, and is
not something this reproduction establishes. Any claim of a *mechanism* is
withdrawn — see below.

**A note on which number is which**, because earlier revisions conflated them:

| Quantity | Value | What it is |
|---|---|---|
| paper headline (LJSpeech) | 18.26 % | the published result; **we do not claim it is wrong** |
| released materials' bundled-sample value | 1.9376 bpb = 4.13× | reported, but not independently reproduced end to end |

### 3.4 The image result is not source-image lossless compression

> **Rewritten.** Earlier revisions of these notes described a *whole-image* path of
> our own construction (one image → one 32×32 token grid → 1 024 tokens) and
> reported a "true lossless rate" of 67.43 % from a residual. Both are
> **superseded**: that is not the reference's path, and the 67.43 % figure is
> **withdrawn**. See `AUDIT_FOR_REVIEW.md` §4.4.2–§4.4.3.

The reference image path, instrumented end to end
(`tools/probe_official_image_path.py`, `probe_official_image_preproc.py`):

1. `get_imagenet_iterator(is_channel_wised=True, is_seq=True)` splits each colour
   plane into **32×32 single-channel patches** — every loader chunk is
   `(32, 32, 1)`, `h*w = 1024`.
2. `language_model.compress` receives one such patch; because its last axis is 1, it
   takes the `np.repeat(..., 3, axis=2)` branch and calls the ImageGPT processor —
   **one call**, input `(32, 32, 3)`, output `(1, 1024)` tokens.
3. The processor is the standard one: `do_resize=True`, `size=32×32`,
   `do_color_quantize=True`, **512 clusters**.
4. Only those 1 024 token ids are arithmetic-coded. **No residual, no original
   pixels, no side stream.**
5. The ratio denominator is `Σ h*w` over the chunks actually processed.

Measured over the whole image after the generator is exhausted
(`tools/probe_official_image_accounting.py`; note the official log's own totals are
truncated because it prints only every 100th chunk):

| Quantity | Value |
|---|---|
| chunks | **5 733** |
| processed samples (`raw_length`) | **5 870 592** |
| code length | **910 788 bytes** |
| rate | **15.5144 %** (ratio 6.4456×) |
| source RGB channel samples | 5 872 164 |
| dropped by the loader's floor division | **1 572** (524 per channel) |
| paper, CLIC2019 | 14.62 % |

**Why it is not lossless** — proved by exhaustive enumeration, not by cardinality
(`tools/test_igpt_injectivity.py` → `results/igpt_grayscale_collisions.json`). Every
processor input along this path is `(v, v, v)` built from **one** 8-bit channel
value, so the domain is 256 — and the quantiser maps those onto only **48 distinct
token ids**, with **585 colliding pairs**:

```
token  71  <-  grayscale values [0, 1, 2, 3]
token 391  <-  grayscale values [78, ..., 84]     (largest class, 7 values)
```

The same collisions occur on real patches (patch #116: 110 distinct source values →
32 tokens). Since the coder emits only the token id, no decoder can recover which
source value it was: **lossless w.r.t. the token stream, lossy w.r.t. the image.**
The denominator is *internally consistent*, so this is a **tokenisation** defect,
not a denominator mismatch.

15.51 % against the published 14.62 % is numerically close and shows the official
*accounting convention* explains the paper's magnitude — it is **not** a
dataset-level reproduction (this image is high-detail aerial content, not CLIC).

---

## 4. Status

| Modality | Model | State |
|---|---|---|
| text | Llama-3-8B (+LoRA) | **working; reproduces the paper's base-model figure** |
| audio | bGPT | pipeline runs, round-trip verified; shipped code+checkpoint do not reproduce the paper's headline (§3.3) |
| image | ImageGPT | official path instrumented; not source-image lossless, proved by exhaustive enumeration (§3.4) |
| video | ImageGPT | **unmeasured**; Y4M reader working (4096×2160, 60 frames, 0.55 s for 2 frames) |

Tests: `tests/test_coder.py` — 8 tests, all passing, including exhaustive
round-trip over skewed/near-degenerate distributions.

### 4.1 Text result

Three MeDAL files, 312 053 UTF-8 bytes total, 57 547 BPE tokens, Llama-3-8B
(± the LoRA published by the authors with their capsule), 4-bit NF4, context window 4 096, rolling window. Measured after
the target-alignment fix (§4.1.1), with every token coded exactly once
(`coded_symbols == symbols == 57 547`):

| Variant | compression rate | ratio | paper |
|---|---|---|---|
| base Llama-3-8B (= LMCompress\*) | **9.0410 %** | 11.0607× | 9.08 % |
| + authors' published LoRA (= LMCompress) | **9.3529 %** | 10.6919× | 9.06 % |
| Δ (LoRA − base) | **+0.312 pp** (9.353 % vs 9.041 %) | | paper: −0.02 pp |

The base model lands **0.04 pp** from the published LMCompress\* figure, which is
the primary evidence that the port is faithful. The adapter makes compression
*worse* on all three files.

**Two caveats that must travel with these numbers**, both raised by review:

* **Context length is a first-order confound, not "noise".** The paper's own
  Supplementary Table 9 reports MeDAL at ctx 8 192 / 6 144 / 4 096 as
  9.06 / 9.16 / 9.36 %. That is a ~0.3 pp spread — **the same magnitude as the
  0.312 pp LoRA penalty being measured**. So part or all of the apparent penalty
  could be a context-length interaction rather than the adapter. Re-running both
  arms at ctx 8 192 is required to separate them (Part A of the Colab notebook).
* **n = 3.** No error bar; the paper's own claimed MeDAL gain (0.02 pp) is smaller
  than any single-file variation here.

### 4.1.1 The alignment defect that was found and fixed

An earlier revision scored every token against the model's distribution for its
**predecessor**: the target slice began one position late
(`ids[:, offset+1 : offset+1+n]` paired with `logits[:, offset : offset+n]`), and a
compensating "drop the first prediction" both masked the error and silently
discarded the last token of every document. It was caught because a reviewer
objected to the *explanation* of that missing token (which had been mis-attributed
to causal LMs). `iter_steps` now asserts every target against the document token it
must equal, so a lag of any size fails loudly. Pre-fix values were 9.0333 % /
9.3459 %; the fix moved both slightly *up* and made the base figure *closer* to the
paper.

---

## 5. Running the whole thing: cost model

Measured throughput, so runs can be planned rather than guessed:

| Workload | Rate | Local data volume | Extrapolated |
|---|---|---|---|
| text, Llama-3-8B 4-bit, ctx 4 096, RTX 5070 | ≈ 66 tokens/s | 100 MeDAL files (10.4 MB) | ≈ 8 h |
| audio, bGPT, 8 160-byte windows | ≈ 3.4 kB/s | 1 file (82 kB) | 25 s |

The text throughput is the binding constraint. The full paper-scale text
evaluation needs Colab, not for memory (4-bit fits in 5.71 GB) but for time.

