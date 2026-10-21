# Independent reproduction of *Lossless data compression by large models* (LMCompress)

A from-scratch reimplementation of the LMCompress pipeline across four modalities — text,
audio, image and video — together with the classical compression baselines it is compared
against, and an audit of whether the published procedure is in fact **lossless**.

The implementation does not call the authors' compression routines. The authors' released
Code Ocean capsule is used only for (a) reading the algorithm, (b) the released model
weights, (c) reference data, and (d) running unmodified as a cross-check. Nothing large is
committed here: datasets and weights live in the capsule and are located at run time
(see [`data/README.md`](data/README.md)).

Reference: Li T. *et al.* Lossless data compression by large models.
*Nature Machine Intelligence* **7**, 794–799 (2025). DOI
[10.1038/s42256-025-01033-7](https://doi.org/10.1038/s42256-025-01033-7).

---

## Headline results

Rates are compressed bits ÷ (8 × payload symbols), so **lower is better**; the ratio is the
reciprocal. Model code lengths are `sum(-log2 q(true symbol))` in **bits**, computed from the
model's float32 probabilities, never from the arithmetic coder's quantised distribution.

| Modality | What was measured | Ours | Paper | Verdict |
|---|---|---|---|---|
| Text | Llama-3-8B base, MeDAL sample | **9.0410 %** | 9.08 % (SI Table 6) | reproduces to −0.04 pp |
| Text | base, A100 paired run, 8 192-char window | **9.1353 %** | 9.06 % (SI Table 9) | reproduces to +0.08 pp |
| Text | + authors' released LoRA | **+0.58 … +0.68 pp worse** across five window settings | −0.02 pp (claimed gain) | **direction differs** |
| Audio | bGPT, LJSpeech clip, 82 220 B | **19.82 %** (reference: 19.94 %) | 18.26 % | same payload, 0.6 % apart |
| Image | iGPT-small, 5 CLIC images | **24.9892 %** weighted, 21.4781 % median | 14.62 % | **does not match**; and not lossless |
| Video | iGPT-small per frame, 9 frames | **26.1866 %** | 15.31 % | **does not match**; and not lossless |

Classical baselines were reproduced independently in every modality, and they are what
validates the measurement apparatus:

* **Video** (same 10 frames): H.265 26.467 %, H.264 34.087 %, FFV1 40.114 %,
  zstd 46.292 % — codec ordering reproduces exactly.
* **Text** (the paper's own fixed-window protocol, SI Table 11): 7z at 8 192 / 16 384 /
  32 768 B gives 41.44 / 38.40 / 35.78 % against the paper's 42.02 / 38.21 / 35.26 % —
  three of four columns within 0.6 pp.
* **Image** (12 CLIC images): WebP 39.362 %, JPEG-2000 50.215 %, PNG 76.848 %.
* **Audio**: OptimFROG 27.311 %, WavPack 35.303 %, FLAC 45.334 %, zlib 62.627 %.

## The two findings worth reading

**1. The image and video path is not a lossless compressor.** iGPT's processor quantises a
32×32 patch onto a 512-colour palette and codes only the resulting token grid; the palette
mapping is not injective. Exhaustive enumeration of all 256 single-channel values shows they
map onto only **48** distinct token IDs, forming **585** colliding pairs, and the same
many-to-one mapping occurs on real images. No residual is transmitted, so the source pixels
cannot be recovered. The published claim that "the tokens of iGPT are greyscale values" is
also incorrect: all 512 palette entries are RGB triples, and none is exactly grey.

**2. The released LoRA adapter makes MeDAL worse, not better.** Under five window settings,
and independently in the authors' own unmodified harness (+0.5901 pp at 8 192 characters),
the adapter raises the rate. The adapter's `trainer_state.json` shows it is the step-5 120
checkpoint, whose evaluation loss is 8.2 % above the best value reached at step 1 020.

## Repository layout

```
src/uic/            the reimplementation
  coder.py            integer range coder (base 2, 32-bit, guaranteed non-zero intervals)
  engine.py           measurement engine: streaming, resumable, atomic checkpointing
  runner.py           corpus traversal and per-unit bookkeeping
  models.py           predictors: Llama-3-8B (+LoRA), bGPT, ImageGPT
  data.py             per-modality corpus readers and tokenisation
  y4m.py              Y4M parsing and frame extraction
  paths.py            locates the capsule and the results directory
tests/              coder round-trip and boundary tests
tools/              one script per experiment; each writes an artefact under results/
results/            the raw measurement files quoted in the report
docs/               audit, threat model, and the correction log
colab/              notebooks for the runs that need an A100
report/             LaTeX source, figures and compiled PDFs
scripts/            repository checks and helpers (not experiment code)
data/README.md      how to obtain the datasets and weights (nothing is committed)
```

## Getting started

### 1. Prerequisites

Python 3.10+, PyTorch, and — for the video and audio baselines — `ffmpeg`, `zstd` and the
statistical audio codecs supplied inside the capsule's `third_party/`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Point the code at the capsule

The datasets, the released weights and the reference implementation are **not** part of this
repository. Obtain the capsule, then either place it next to this repository as
`capsule-4957991-code/` (the default) or set an environment variable:

```bash
export LMCOMPRESS_CAPSULE=/path/to/capsule-4957991-code
```

`src/uic/paths.py` resolves everything else from there and raises a message naming the
variable if it cannot find the capsule. No absolute path is hard-coded anywhere in this
repository; `python scripts/verify_repo.py` enforces that.

### 3. Smoke test

```bash
python tests/test_coder.py           # coder round-trip, no data needed
python scripts/verify_repo.py        # repo hygiene: no absolute paths, no huge files
```

## Reproducing the tables

Each command is standalone and writes a fresh artefact under `results/`. Heavy runs print
per-unit progress and resume from a checkpoint, so an interruption costs at most one unit.

```bash
# text: base vs LoRA over five window settings (needs a GPU with ~20 GiB for context 8192)
python tools/text_paired_local.py          # or use colab/Text_Paired_Context_Colab.ipynb

# text baselines, including the paper's fixed-window grid (CPU only, seconds)
python tools/baselines_text_windowed.py

# audio: our port vs the reference implementation on one clip
python tools/remeasure_audio_nextbyte.py

# image: CLIC images, per-channel, official convention
python tools/run_image_igpt_clic.py --limit 5

# image: palette injectivity, exhaustive
python tools/test_igpt_injectivity.py

# video: iGPT per frame (or colab/Video_ChannelWise_Colab.ipynb on an A100)
python tools/run_video_igpt_channelwise.py --limit 10

# video/audio/image classical baselines
bash tools/baselines_video.sh
bash tools/baselines_audio_wsl.sh
bash tools/baselines_image_clic.sh
```

## Conventions, and two traps

**Rate versus ratio.** The paper's main text quotes the *ratio* (original ÷ compressed) while
its supplementary tables quote the *rate*. Mixing them silently inverts every comparison.
This repository always states which one it means.

**Log base.** `torch.log_softmax` returns **natural** logarithms. A code length computed as
`-log_softmax(...)` is in nats, not bits, and must be divided by `ln 2` (multiplied by
1.4427) before it is called a rate. This repository computes code lengths with `math.log2`,
and `docs/AUDIT_FOR_REVIEW.md` records a case where the two were confused while this
reproduction was being written — including the wrong conclusions that followed and how they
were caught. If you write your own scorer, assert it against `output.loss * (L-1)` in the
same log base you report.

**Denominators.** For images and video the numerator counts palette tokens while the
denominator counts source channel samples. Both are reported explicitly, because they
describe different objects.

## Verification and honesty

`docs/AUDIT_FOR_REVIEW.md` is the long-form record: what was measured, what was retracted,
and why. It deliberately keeps the retractions — a mis-aligned audio probe, a claimed
missing dependency, a denominator confusion, and the log-base error above — because each of
them identifies a way this kind of reproduction can go wrong. `docs/REVIEW_GUIDE_FOR_AGENT.md`
is an adversarial reading guide pointing at the weakest parts first.

Nothing in `results/` is hand-entered: every figure in the report can be traced to a file
there, and the appendix of the report names the file for each number.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use the datasets or model weights, cite the
original authors as well.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The capsule's data and weights keep their own terms;
the report and the audit text are provided as-is for academic review.
