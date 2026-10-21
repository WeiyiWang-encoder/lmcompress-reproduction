# Measurement artefacts

Every number in the report traces to a file here. Nothing is hand-entered: the tools in
`../tools/` write these files, and the report appendix names the file behind each figure.

## Naming

| Prefix / suffix | Meaning |
|---|---|
| `baselines_*` | classical-compressor runs (rate comparisons) |
| `image_*`, `video_*`, `text_*`, `audio_*` | model-side measurements |
| `_*.txt`, `*.out`, `*.err` | raw stdout/stderr of a run, kept as evidence |
| `*.json` | one result object, usually with per-item detail |
| `*.jsonl` | one JSON object per line, append-only, so an interrupted run stays usable |

## The files the report appendix points at

| File | What it contains |
|---|---|
| `text_paired.json` | the ten-cell paired run: base and +LoRA at five window settings, with dtype and peak VRAM for each cell |
| `audio_bgpt_rows17_tailpad_v1.jsonl` | audio port: coded bits, real coder output, bits per byte |
| `image_igpt_clic.json` | CLIC image rates, per image and aggregated |
| `igpt_grayscale_collisions.json` | exhaustive palette injectivity check: 256 values → 48 token IDs, 585 colliding pairs |
| `video_channelwise_summary.json` | video rates over nine distinct frames, plus the classical baselines on the same clip |
| `baselines_video_wsl.jsonl` | H.265 / H.264 / FFV1 on the shared frames |
| `baselines_video_zstd.jsonl` | zstd over raw YUV on the same frames |

## Two conventions to keep in mind when reading them

**Code lengths are in bits** (`math.log2`), and the rate is `bits / (8 × payload symbols)`.
A scorer built from `torch.log_softmax` returns **nats**; those numbers must be multiplied
by `log2(e) = 1.4427` before they are comparable. `../docs/AUDIT_FOR_REVIEW.md` records a
case during this reproduction where the two were confused.

**Denominators differ by modality.** For images and video the numerator counts palette
tokens while the denominator counts source channel samples, so the rate describes a
quantised representation rather than the source pixels. Both are stated in the JSON.

## Path handling

Raw logs originally echoed absolute paths from the machine the runs happened on. Those
prefixes have been replaced with `<capsule>`, `<repo>` and `<tmp>`; the numbers are
unchanged. Input manifests were removed rather than genericised, because they are pure input
lists and `../tools/make_manifests.py` regenerates them from the capsule.
