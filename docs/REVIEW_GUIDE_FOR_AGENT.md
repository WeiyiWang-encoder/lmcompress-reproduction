# Independent Review Guide — adversarial verification

**Audience:** an autonomous reviewing agent (or a human reviewer) whose job is to
**falsify**, not to agree.

**Artifact under review:** a clean-room reproduction of
*Lossless data compression by large models* (NMI 7:794–799, 2025; preprint
*Understanding is Compression*, arXiv:2407.07723), Code Ocean capsule 4957991.

**Why this document exists.** The reproduction reports several findings that are
critical of the published implementation. Because those findings are intended to
appear in a capstone thesis, they must be independently checkable. **Do not accept
any claim here because it is stated confidently.** Every claim below has (a) a
verification procedure, (b) a falsification attempt that should break it, and (c) an
explicit statement of what is **not** being claimed.

---

## 0. Ground rules for the reviewer

1. **You have two codebases.** Distinguish them constantly:
   * `<capsule>\` — the **authors' reference implementation**
     (read-only; treat as evidence, never modify).
   * `<repo>\` — the
     **clean-room reproduction** written for this thesis (the artifact under review).
   A defect in the first is a *finding*; a defect in the second is a *bug*.
2. **Both can be wrong.** The reproduction's author already found and fixed two bugs
   in his own code (Claims 1 and 8). Assume more exist.
3. **Report the strongest counter-argument you can construct** for each claim, even if
   you ultimately agree. A claim that survives your best attack is worth more than one
   you merely failed to break.
4. **Prefer running code to reading it.** Where an automated check exists, a result
   that disagrees with this document means *this document is wrong* until you prove
   otherwise.

---

## 1. Claims under review

Confidence is the reproduction's own assessment. `[M]` = measured, `[D]` = derived.

| # | Claim | Key number | Conf. | Severity if true |
|---|---|---|---|---|
| 1 | The text path's target slice was off by one (a bug in **this** port, fixed) | 9.0333 → **9.0410 %** | [M] | affects our own credibility |
| ~~2~~ | ~~The reference's bGPT-audio scoring pairs each prediction with a byte 16 positions away~~ **— RETRACTED, see §5.2** | — | **withdrawn** | — |
| 3 | The reference image path is **non-injective**: 256 grayscale values → **48** token ids | 585 colliding pairs | [M] | high |
| 4 | The reference image loader drops **1 572** channel samples via floor division | 5 872 164 → 5 870 592 | [M] | medium |
| 5 | The reference image log's own totals are **stale** (prints every 100th chunk) | logged 906 459 vs true **910 788** | [M] | medium |
| 6 | The published LoRA adapter is **worse** than base on all 3 files | **+0.312 pp** | [M] | medium (confounded — see §5.6) |
| 7 | `bits2bytes` does not exist; the text failure is an fp32 OOM | ~32 GB vs 5.71 GB | [M] | low |
| ~~10~~ | ~~The capsule omits a `samplings` module, blocking the reference code~~ **— RETRACTED**: `samplings` is an installed PyPI package in `.venv`; the unmodified capsule imports and runs | — | **withdrawn** | — |
| 8 | This port's arithmetic coder silently collapsed intervals (bug, fixed) | p = 9.06e-11 | [M] | affects our own credibility |
| 9 | Audio claim 2 was **falsely asserted** and is retracted | see §5.2 | [M] | **read this first** |

---

## 2. What is **not** being claimed

Read this before assessing tone. The reproduction deliberately does **not** claim:

* **That the paper's results are fabricated, or that the authors acted in bad faith.**
  Every surviving finding is consistent with ordinary implementation slips.
* **That the paper's audio number (18.26 % on LJSpeech) is wrong.** We claim only that
  the released code plus released weights report 1.9376 bpb on the bundled sample where
  the paper reports 18.26 % (= 1.46 bpb). **The reason is unknown** — a different
  training run, a different sample, or something else. An earlier revision of this
  project claimed a mechanism ("scoring misalignment"); **that claim was wrong and has
  been retracted (§5.2).**
* **That image/video rates are "miscalculated".** Their denominator is *internally
  consistent* — it counts the samples the pipeline actually processed. The defect is
  that the pipeline is **lossy with respect to the source**, so the rate describes a
  quantised surrogate, not the source. An earlier audit revision got this wrong and it
  was corrected; see `docs/REVIEW_RESPONSE.md`.
* **That the models are incapable.** On the contrary: the text model reproduces to
  **0.04 pp**, and all four video baselines reproduce to within 1.17 pp. This is
  evidence the models and the concept are sound.
* **That any modality was reproduced at dataset scale.** All runs are small samples.

---

## 3. Code map

### 3.1 Authors' implementation (evidence)

| Path | Lines | Relevance |
|---|---|---|
| `compress_with_bgpt_or_llama.py` | 486 | **Claim 2.** Audio scoring. See lines 226–243 and 262–268. |
| `compress_with_igpt.py` | 162 | **Claims 3–5.** Image path. |
| `language_model.py` | 214 | Image `compress`; the `np.repeat(...,3)` branch. |
| `data_loaders.py` | 173 | **Claims 3–4.** `get_imagenet_iterator`; the floor division. |
| `utils/arithmetic_coder.py` | 320 | **Claim 8.** `normalize_pdf_for_arithmetic_coding`. |
| `data/model/bgpt/config.py` | — | PATCH_SIZE=16, PATCH_LENGTH=512, HIDDEN_SIZE=768 |
| `data/model/bgpt/utils.py` | — | `bGPTLMHeadModel`; the mask left-shift |

**The central citation for Claim 2** (`compress_with_bgpt_or_llama.py:226–243`):

```python
if self.data_type == "wav" or self.data_type == "video" or self.data_type == "bmp":
    # logits: (511, 17, 257)
    logits = logits[:-1, :-1, :]
    logits = logits.reshape(1, -1, 257)        # (1, 510*16, 257)
    input_ids = input_ids[:, PATCH_SIZE:-PATCH_SIZE]   # (1, 510*16)
    attention_mask = attention_mask.repeat_interleave(PATCH_SIZE, dim=1)
    attention_mask = attention_mask[:, PATCH_SIZE:-PATCH_SIZE]
```

then `probs = shifted_logits.softmax(-1)`, `pd = gather(probs, shifted_input_ids)`.

### 3.2 Reproduction (artifact under review)

| Path | Relevance |
|---|---|
| `src/uic/models.py` | `LlamaTokenPredictor.iter_steps` (Claim 1), `bgpt_prediction_indices` (Claim 2) |
| `src/uic/data.py` | image/audio loaders |
| `src/uic/coder.py` | Claims 1, 8 |
| `tools/test_igpt_injectivity.py` | Claim 3 |
| `tools/probe_loader_volume.py`, `probe_denominator_reconcile.py` | Claim 4 |
| `tools/probe_official_image_accounting.py` | Claim 5 |
| `tools/identify_reference_rows.py`, `prove_alignment_mechanism.py` | Claim 2 |
| `tools/check_doc_consistency.py` | documentation hygiene (not a scientific claim) |

---

## 4. Machine-checkable harness

Run this first. It re-derives the numbers from **the authors' own files** where
possible, and prints PASS/FAIL against the values claimed here.

```powershell
cd <repo>
..\.venv\Scripts\python.exe tools\review_harness.py            # all fast checks
..\.venv\Scripts\python.exe tools\review_harness.py --list     # list checks
```

**A check that fails, or that passes for a reason other than the one described, is a
reportable finding.** The harness is written to be adversarial: it prints the raw
quantities it measured, not just a verdict, so you can disagree with its arithmetic.

---

## 5. Per-claim verification

### Claim 1 — the text target slice was off by one *(our bug, fixed)*

**Where.** `src/uic/models.py`, `LlamaTokenPredictor.iter_steps`; history in
`docs/REVIEW_RESPONSE.md` §5.

**Verify.**
1. Read `iter_steps`. Confirm it now builds a window `[predecessor][scored][successor]`,
   scores `n_target` tokens at `offset`, and **asserts** each target equals the document
   token it should be (`"target misalignment at {i}"`).
2. Confirm `results/text_base_v2.jsonl` and `text_lora_v2.jsonl` have
   `coded_symbols == symbols == 57547` (no silently dropped token).
3. Recompute the weighted means: base **9.0410 %**, LoRA **9.3529 %**, Δ **+0.312 pp**.

**Falsify.** Reintroduce a one-position lag (`offset + 1` in the target slice) and
confirm the assertion fires instead of silently producing a number. If it does not
fire, the guard is not real and Claim 1's fix is unverified.

**Not claimed.** That the *reference* text path has this bug — we never ran it to
completion (see Claim 7).

---

### ~~Claim 2~~ — RETRACTED before review: an off-by-`p` index in *our* probe

**Read this first.** This was the project's headline audio finding across several
revisions, marked "high confidence" and already written into the audit, when the
harness written to test it destroyed it. It is documented in full because a thesis
that cites it would be wrong.

**What was claimed.** That `compress_with_bgpt_or_llama.py` pairs each prediction with
a byte one patch away from the byte it predicts, inflating its apparent compression by
**6.29×** (2.2561 vs 14.1804 bpb).

**Why it was wrong.** A patch row carries **17** slot predictions, not 16. The flat
logit row for payload position `k` is therefore

```
17 * (k // 16) + (k % 16)          <-- correct
16 + k                             <-- what our probes used
```

These differ by `k // 16`, so they drift apart by one per patch. Every "honest"
audio figure this project published — 13.1219, 14.1804, 15.7844 bpb — came from the
wrong rows and is withdrawn.

**What is actually true**, each step measured:

1. The reference pairs row `k = 16p + j` with `payload[k]` = `grid[16p+j+16]`.
2. The model's row `(p, j)` **is** most confident about exactly that byte: mean
   probability **0.3052** at offset `L = 16`, versus **0.0804** at `L = 1`, and ≤ 0.19
   at every other offset in −2…19 (`tools/measure_row_target_offset.py`).
3. The scheme is **causal**: perturbing bytes strictly after a target changes the row
   by `max |Δ log-prob| = 0.000000e+00`, while the control (perturbing bytes before it)
   changes it by **9.44** (`tools/test_audio_causality.py`).
4. The arithmetic coder really emits **900 016 bits = 112 502 bytes** for an
   82 220-byte payload — realized output, not a scoring artefact.

**So: the reference's audio scoring is internally coherent, causal, and its reported
figure is a genuine code length for its own scheme.**

**What survives:** only the bare, mechanism-free observation that the released code
plus weights report **1.9376 bpb** on the bundled sample while the paper reports
**18.26 %** (= 1.46 bpb) on LJSpeech. **Why they differ is unknown.**

**Task for you, specifically.** This episode is the best available evidence about how
this project fails, so please check whether the *same* class of error exists elsewhere:

* Are there other places where a flat index into a 17-slot-per-patch row grid is
  computed as if it were 16-per-patch? Search the `tools/` directory for `16 +`,
  `+ 16`, and `PATCH_SIZE` arithmetic applied to logits.
* Does the text path (`LlamaTokenPredictor`) have an analogous row/target index bug?
  Its offsets are token-level, not patch-level, so the failure mode would look
  different — check the asserts actually cover the scored range.
* Are any *surviving* claims downstream of the withdrawn probe? C6 was re-scoped; check
  nothing else quietly depends on it.

**Tone note.** This retraction is included deliberately. If the guide had simply
deleted Claim 2, a reviewer would have had no way to judge whether *other* claims rest
on similarly thin derivations. Their credibility now rests on your assessment, not on
our confidence.

---

### Claim 3 — the image path is non-injective *(strongest surviving finding)*

**Where.** `language_model.py` (`np.repeat(..., 3, axis=2)` branch),
`data_loaders.py` (`get_imagenet_iterator`); test in `tools/test_igpt_injectivity.py`;
data in `results/igpt_grayscale_collisions.json`.

**The argument.** The loader splits the image **channel-wise**, so
`language_model.compress` receives `(32, 32, 1)`. Because the last axis is 1, it takes
the repeat branch and calls the ImageGPT processor with `(32, 32, 3)`, i.e. `(v, v, v)`
for a **single** 8-bit channel value. The processor quantises to 512 clusters and emits
one token id. The coder transmits **only that id**, with no residual.

Therefore: if two distinct values `a ≠ b` give the same token id, no decoder can
recover which was sent. This is a statement about the *composition* of the official
preprocessing, and it is decided by **enumerating all 256 possible inputs**.

**Result (exhaustive):** domain 256 → image **48** distinct token ids; **585** colliding
pairs; largest class 7 values.

**Verify.**
1. Run `tools/test_igpt_injectivity.py`. It enumerates all 256 values through the
   **actual** `ImageGPTImageProcessor` from `data/model/imagegpt-small/` (not a
   reimplementation) and writes `results/igpt_grayscale_collisions.json`.
2. Independently confirm two examples from the JSON:
   `token 71 ← [0,1,2,3]` and `token 391 ← [78..84]`.
3. Confirm collisions occur on **real** patches too (the JSON records a real patch with
   110 distinct source values → 32 tokens), so this is not an artefact of constant
   inputs.

**Falsify (please attempt).**
* Show the processor is injective on this domain, i.e. reproduce the enumeration and get
  256 distinct ids. **This single result would destroy the claim.**
* Show that the official path passes something *other* than `(v, v, v)` — e.g. that the
  last axis is 3 before the processor call, so the domain is 2^24 rather than 256. Read
  the call site and print the tensor shape.
* Show the decoder receives side information (a residual, a palette index, the original
  channel) that disambiguates.

**Not claimed.** That the *denominator* is wrong. It is internally consistent
(see §2). Also not claimed: that a 32×32 thumbnail is a useless representation — only
that a rate for it is not a rate for the image.

---

### Claim 4 — the loader drops 1 572 channel samples

**Where.** `data_loaders.py`; verification `tools/probe_loader_volume.py`.

**Verify.** Confirm chunks = **5 733**, processed samples = **5 870 592**, source RGB
channel samples = **5 872 164**, difference **1 572** = 524 × 3 (one per channel).

**Falsify.** Show the source count is not 5 872 164 (1 714 × 1 142 × 3), or that the
missing samples are processed elsewhere.

**Not claimed.** That this materially changes the headline rate — it does not (0.027 %).
It is offered as *independent* evidence that the path is not source-lossless, in case a
reviewer disputes Claim 3.

---

### Claim 5 — the reference log's own totals are stale

**Where.** `compress_with_igpt.py` (printing gated on every 100th chunk);
`tools/probe_official_image_accounting.py`; `results/official_image_accounting.json`.

**Verify.** Read the print guard in the authors' file, confirm the last *printed*
`compressed_length` is 906 459, then run the accounting probe to exhaustion and confirm
the true total is **910 788** and the rate **15.5144 %**.

**Falsify.** Show the log is printed unconditionally, or that 906 459 is the true total.

---

### Claim 6 — the published LoRA adapter is worse than base *(confounded)*

**Where.** `results/text_base_v2.jsonl`, `results/text_lora_v2.jsonl`;
training evidence in `data/model/finetuned_llama/Llama-3-8B_lora/checkpoint-5120/trainer_state.json`.

**Verify.**
1. Both arms used an **identical code path and identical windowing**; only the adapter
   differs. Confirm this in the runner (cursor keys are tagged per variant).
2. Per-file: 8.863→9.127, 9.237→9.551, 9.022→9.379. All three worsen.
3. Read `trainer_state.json`: eval loss best **2.4708 at step 1020** (≈2 epochs), rising
   to **2.6728 at step 5120**. The shipped adapter is the step-5120 one.

**Falsify.** Show the two arms differ in something other than the adapter (e.g. window
boundaries, context, prompt) — that would invalidate the comparison.

**Known weakness — do not let the thesis overstate this.** The measured penalty
(0.312 pp) is the **same magnitude** as the context-length effect implied by the paper's
own Supplementary Table 9 (MeDAL at ctx 8192/6144/4096 = 9.06/9.16/9.36 %). We measured
at ctx 4096, the paper at 8192. **The adapter effect and the context effect are not
separated.** Also n = 3, no error bars. Treat as suggestive, not established.

---

### Claim 7 — `bits2bytes` does not exist; the text failure is an fp32 OOM

**Verify.** Ripgrep the authors' repository (excluding `.venv` and our folder) for
`bits2bytes`. Confirm **zero** hits, and that the undefined name is `compression_utils`
inside a function that is commented out at its call site. Then confirm the model is
loaded without `torch_dtype` (⇒ fp32, ≈32 GB for an 8B model).

**Falsify.** Produce a `bits2bytes` definition anywhere in the authors' code.

---

### Claim 8 — this port's arithmetic coder silently collapsed intervals *(our bug)*

**Where.** `src/uic/coder.py`; tests `tests/test_coder.py`.

**Verify.** Run `tests/test_coder.py`. Confirm the interval-collapse assertion exists
and that `min_interval_width()` reserves a nonzero minimum. Confirm the exact-8.000
bits/byte test for uniform bytes passes.

**Note for the reviewer.** An earlier revision of our audit justified the minimum width
with an **invalid** argument ("20 bits exceeds the 8 bits/byte ceiling of a byte
alphabet"). That was wrong and has been corrected; the correct justification is that the
**reported rate is never computed from the coder's quantised probabilities**. If you see
the old argument anywhere, it is stale text — report it.

---

## 6. Cross-cutting things to attack

1. **Documentation drift.** This project has repeatedly shipped stale numbers across
   documents. Run `tools/check_doc_consistency.py` (expect exit 0), then **try to break
   its exemptions** — over-broad retraction matching could hide real residue. The
   checker's own rules are as much a claim as the numbers.
2. **The reported-rate convention.** Claim: the reported figure is
   `Σ −log₂ q(true symbol)` on float32 model probabilities, *not* the coder's output.
   Verify in `src/uic/engine.py` / `runner.py` and confirm the two are reported
   separately (the repo tracks `coded_bits` and `ac_bits` distinctly).
3. **Denominator hygiene.** Every rate should state its denominator. Check
   `results/comparison_tables.md` and `results/video_tables.md` for any rate whose
   denominator is ambiguous — particularly the text container-vs-payload pair
   (28.805 vs 28.6187 %) and video raw-YUV-vs-RGB.
4. **Baseline reproduction as the control.** If the classical baselines did *not*
   reproduce, the model comparisons would be uninterpretable. Verify the video numbers
   independently (FFV1 40.167 vs 40.16; zstd 51.035 vs 51.04) — they are the tightest
   agreements and therefore the best test of the harness.
5. **Our own two bugs.** Claims 1 and 8 are self-reported defects. Check whether the
   fixes are complete or merely patch the symptoms.

---

## 7. Reporting format requested

For each claim, return:

```
CLAIM <n>: <agree | partially agree | disagree | cannot verify>
EVIDENCE: <what you ran/read, with exact commands and observed output>
STRONGEST COUNTER-ARGUMENT: <the best case against the claim>
RESIDUAL UNCERTAINTY: <what would still need to be true for the claim to fail>
SEVERITY FOR A THESIS: <safe to state | state with caveat | do not state>
```

Then a top-level verdict:

* Which claims are **safe to publish** as stated.
* Which require **hedging**, and the exact hedging language you recommend.
* Any **new** defect found in either codebase.
* Whether the *tone* of the reproduction is defensible: specifically, does it avoid
  implying misconduct while still reporting a real discrepancy?

---

## 8. What would change our conclusions

State these plainly in your report — a finding that cannot be overturned is suspect.

| Claim | Would be overturned by |
|---|---|
| 2 | Finding a semantically correct pairing that also yields ~2.26 bpb; or the authors' script printing a value far from 1.94 on the bundled sample |
| 3 | Reproducing the enumeration and getting 256 distinct ids; or showing the path passes RGB (not `(v,v,v)`) |
| 4 | A source channel-sample count other than 5 872 164 |
| 5 | An unconditional print in the authors' log |
| 6 | Any asymmetry between the two text arms other than the adapter; or a ctx-8192 rerun showing the penalty vanish |
| 1, 8 | (our bugs) an unguarded path that still produces a number |

---

## 9. Files you will need

| Path | Purpose |
|---|---|
| `docs/AUDIT_FOR_REVIEW.md` | the full audit (primary artefact) |
| `docs/REVIEW_RESPONSE.md` | point-by-point responses to three prior review rounds, including where we were **wrong** |
| `docs/REPORT_CN.md` | the Chinese report (thesis-facing) |
| `docs/REPRODUCTION_NOTES.md` | technical notes |
| `results/*.jsonl`, `results/*.json` | all raw measurements |
| `results/comparison_tables.md`, `results/video_tables.md` | cross-modality tables |
| `tools/review_harness.py` | the automated checks |

**Prior review history matters.** Three earlier review rounds already rejected several
of our positions (a cardinality argument for the image claim, an "8 bits/byte ceiling"
justification, an entropy "proof" for audio, a withdrawn 67.43 % figure). Those
retractions are recorded rather than hidden. Check that we have not quietly
re-introduced any of them.
