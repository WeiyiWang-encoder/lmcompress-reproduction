# Data and model weights

Nothing large is committed to this repository. Everything below is resolved at run time
through `src/uic/paths.py`, which looks for the authors' Code Ocean capsule — either in a
sibling directory named `capsule-4957991-code/`, or wherever `LMCOMPRESS_CAPSULE` points.

```bash
export LMCOMPRESS_CAPSULE=/path/to/capsule-4957991-code
python -c "from uic.paths import capsule_root; print(capsule_root())"
```

## What the capsule supplies

| Path inside the capsule | What it is | Used by |
|---|---|---|
| `data/model/imagegpt-small/` | iGPT-small checkpoint (~293 MB) and its `preprocessor_config.json`, which holds the 512-entry palette | image, video |
| `data/model/bgpt/` | three byte-level bGPT checkpoints (`weights-audio.pth`, `weights-image.pth`, `weights-text.pth`, ~1.26 GB each) | audio |
| `data/model/finetuned_llama/Llama-3-8B/` | Llama-3-8B in four safetensors shards (~15.3 GB) | text |
| `data/model/finetuned_llama/Llama-3-8B_lora/` | the released LoRA adapter (~40 MB) plus `checkpoint-5120/trainer_state.json` | text |
| `data/text/MEDAL_sample/` | 108 MeDAL CSVs; this reproduction uses three of them (312 053 B of payload) | text |
| `data/test_audios/LJ001-0004.wav` | 82 220 B, 16 kHz mono 8-bit PCM | audio |
| `data/test_images/` | one 1714×1142 image used for the first image probe | image |
| `data/test_videos/4k_aerial_first_second.y4m` | 4096×2160 @60 fps, 60 frames, 759 MB | video |
| `third_party/` | static `ffmpeg`/`ffprobe` and the OptimFROG / WavPack binaries the audio baselines need | baselines |

## What is downloaded instead

| Item | Where from | Size |
|---|---|---|
| CLIC professional validation set | `downloads.compression.cc/clic2020_professional_valid.zip` (public) | 128.6 MiB, 41 images |
| `Netflix_Aerial_4096x2160_60fps_8bit_420.y4m` | `media.xiph.org/video/derf/y4m/` (public, HTTP Range supported) | 15.9 GB total; the notebooks fetch only the first frames |
| Llama-3-8B weights | the capsule, or the public mirror `unsloth/llama-3-8b` | 15.3 GB |

The video and CLIC downloads are performed by the tools themselves — see
`tools/baselines_image_clic.sh` and `colab/Video_ChannelWise_Colab.ipynb`.

## Verified provenance

The capsule's Llama-3-8B is **byte-identical** to the public checkpoint: both
`model.norm.weight` and the 1 051 MB `model.embed_tokens.weight` hash the same
(`tools/compare_llama_weights.py` re-checks this using HTTP Range, so only a few kilobytes
are transferred).

## Licence notes

The datasets and the released model weights keep the terms they were distributed under.
Nothing here redistributes them. If you only want to run the coder and the corpus readers,
`tests/test_coder.py` and the text baselines need no model weights at all.
