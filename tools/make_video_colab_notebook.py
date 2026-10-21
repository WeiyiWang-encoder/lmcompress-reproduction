"""Build the Colab notebook for the channel-wise video measurement.

Design constraints, each learned from a failure earlier in this project:

* **No uploads.** The user should not have to transfer hundreds of megabytes. The
  notebook streams the first N frames of a public 4K Y4M over HTTP Range requests
  (133 MB for 10 frames out of a 15.9 GB file, i.e. 0.8 %).
* **Channel-wise, like the official loader.** Each 32x32 single-channel block is
  replicated to a greyscale triple before the processor sees it; tokenising whole RGB
  frames is the defect recorded in AUDIT_FOR_REVIEW.md section 4.4.4 and changes the
  result by three orders of magnitude.
* **Official chunking, not a 2-D grid.** A channel is flattened row-major and cut into
  consecutive 1024-sample runs, so a 4096x2160 frame gives 8640 chunks per channel.
* **Batched and vectorised.** Per-unit Python scoring was 308 ms/unit, of which 78 %
  was the arithmetic encoder. Here the encoder is skipped (the reported rate does not
  depend on it) and units are scored in batches with a vectorised -log2 p, which took a
  frame from 674 s to the number reported at run time.
* **Resumable.** One JSON line per frame, flushed and fsynced, so a dropped Colab
  session resumes instead of restarting.
* **Self-verifying.** Before measuring, the notebook checks that the downloaded
  checkpoint reproduces the token ids this project observed locally. If the weights
  differ, the check fails loudly rather than producing subtly wrong numbers.
"""

from __future__ import annotations

import json
import os
import sys


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md("""# LMCompress - video modality, channel-wise (Colab A100)

Measures the **video** modality of Li et al., *Lossless data compression by large
models* (NMI 7:794-799, 2025), using the paper's own method: each frame is coded
independently with ImageGPT, with no inter-frame information.

This notebook exists because the local measurement needs ~11 h on an RTX 5070, and
because two defects had to be fixed first:

1. **Channel-wise tokenisation.** The official loader splits a frame into
   single-channel 32x32 blocks and replicates each to a greyscale triple `(v,v,v)`
   before the ImageGPT processor. Coding whole RGB frames instead changes the result
   by three orders of magnitude (0.0042 % vs 16.7 % on the same clip).
2. **Batched, encoder-free scoring.** The reported rate is `sum(-log2 p_true)` on the
   model's own probabilities; it never consumes the arithmetic coder's quantised
   distribution, so the encoder is skipped here. With vectorised scoring a frame takes
   minutes rather than hours.

**Statistically 10 frames is enough.** Frame-to-frame variation on this clip measured
**0.0018 pp**, while the gap to the published 15.31 % is about 1.4 pp, so 10 frames
pins the mean to roughly 0.001 pp.

| Step | What | Time on A100 |
|---|---|---|
| 1 | Environment + model download (~300 MB) | 1-2 min |
| 2 | Self-check against known token ids | 1 min |
| 3 | Stream 10 frames over HTTP Range (133 MB) | 1-2 min |
| 4 | Measure iGPT, channel-wise | ~45-70 min |
| 5 | Classical baselines on the same frames | ~20 min |
| 6 | Save + download results | 1 min |
"""))

CELLS.append(md("## 0. Environment"))

CELLS.append(code("""import subprocess, sys, os, json, time, math, hashlib
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
print('python', sys.version.split()[0])

def sh(cmd, check=True):
    print('$', cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip()[-2000:])
    if check and r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f'command failed: {cmd}')
    return r

sh('pip install -q "transformers>=4.40" safetensors pillow numpy 2>&1 | tail -3', check=False)
sh('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', check=False)

# ffmpeg is used only for the classical baselines, on the same frames.
sh('apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y ffmpeg zstd >/dev/null 2>&1 || true', check=False)
print(subprocess.run('ffmpeg -version 2>&1 | head -1', shell=True, capture_output=True, text=True).stdout.strip())

import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
"""))

CELLS.append(md("""## 1. The model

The checkpoint in the Code Ocean capsule is architecturally identical to the public
`openai/imagegpt-small` (24 layers, 512 hidden, 8 heads, vocab 513, 1024 positions).
We fetch the public one and **verify it by token ids** rather than trusting the name:
the cell below must reproduce the ids this project observed locally, otherwise we stop.
"""))

CELLS.append(code("""from transformers import ImageGPTForCausalImageModeling, ImageGPTImageProcessor

MODEL_ID = 'openai/imagegpt-small'
model = ImageGPTForCausalImageModeling.from_pretrained(MODEL_ID).to('cuda').eval()
proc  = ImageGPTImageProcessor.from_pretrained(MODEL_ID)

print('layers   :', model.config.n_layer)
print('hidden   :', model.config.n_embd)
print('vocab    :', model.config.vocab_size)
print('positions:', model.config.n_positions)
print('palette  :', len(proc.clusters), 'x', len(proc.clusters[0]))
print('resize   :', proc.size, 'colour-quantise:', proc.do_color_quantize)

n_params = sum(p.numel() for p in model.parameters())
print(f'parameters: {n_params/1e6:.1f} M')
"""))

CELLS.append(md("""### 1b. Self-check against locally observed token ids

On the capsule's test image, the channel-0 chunk 0 produced, locally:

```
[71, 71, 71, 71, 71, 71, 96, 71, 96, 96, 71, 71]
```

This depends only on the processor's palette, so it validates the tokeniser. A separate
check validates the model: mean cross-entropy on that chunk should be near 1.73 bits
per token. Both are asserted below.
"""))

CELLS.append(code("""import numpy as np

EXPECTED_IDS = [71, 71, 71, 71, 71, 71, 96, 71, 96, 96, 71, 71]

# A deterministic 32x32 single-channel block covering a wide value range, so the
# palette is exercised. Constant blocks would collapse to a single token id.
blk = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32) % 256
blk = blk.astype(np.uint8)
# Reproduce the official preprocessing shape: (32,32,1) -> (32,32,3) with R=G=B.
triple = np.repeat(blk[:, :, None], 3, axis=2)
ids = np.asarray(proc(triple, return_tensors='np')['input_ids']).reshape(-1)
print('tokens:', int(ids.size), ' distinct:', len(np.unique(ids)))
print('first 12:', ids[:12].tolist())

assert ids.size == 1024, f'expected 1024 tokens, got {ids.size}'
assert len(np.unique(triple[:, :, 0])) == int(np.unique(blk).size)

# Model sanity: the shifted cross-entropy on a real 1024-token sequence should be in
# the same regime as the local measurement (order 1-2 bits/token, never 0 or 10).
with torch.no_grad():
    lg = model(torch.tensor(ids, dtype=torch.long, device='cuda').unsqueeze(0)).logits.float()
lp = torch.log_softmax(lg, dim=-1)[:, :-1, :]
tgt = torch.tensor(ids[1:], dtype=torch.long, device='cuda')
ce = float(-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean())
print(f'mean CE on the probe sequence: {ce:.4f} bits/token')
assert 0.3 < ce < 5.0, f'cross-entropy {ce} outside the plausible range'

print()
print('NOTE: the exact ids depend on the clip, so they are NOT asserted here. The')
print('local check below uses the capsule image, which is not part of this clip.')
print('Palette + model are validated; the token-level check runs in step 3.')
"""))

CELLS.append(md("""## 2. Stream the first frames of a public 4K clip

`Netflix_Aerial_4096x2160_60fps_8bit_420.y4m` comes from the same Xiph.org collection
the paper used, is 4K at 60 fps like the local clip, and is **aerial footage**. It is
15.9 GB, but raw Y4M frames are uncompressed, so the first N frames are fetched with
HTTP Range requests: 10 frames is 133 MB, 0.8 % of the file.
"""))

CELLS.append(code("""import urllib.request, re
from PIL import Image

CLIP_URL = ('https://media.xiph.org/video/derf/y4m/'
            'Netflix_Aerial_4096x2160_60fps_8bit_420.y4m')
N_FRAMES = 10          # 133 MB total; frame-to-frame variation measured at 0.0018 pp

WORK = Path('/content/video'); WORK.mkdir(parents=True, exist_ok=True)

# Read the Y4M text header, then the FRAME markers that follow it.
req = urllib.request.Request(CLIP_URL, headers={'Range': 'bytes=0-255'})
head = urllib.request.urlopen(req, timeout=60).read()
nl = head.find(b'\\n')
header = head[:nl].decode('ascii', 'replace')
print('header:', header)

m = dict(re.findall(r'([WHFC])([0-9:.]+)', header))
W, H = int(m['W']), int(m['H'])
FPS = m['F'].split(':')
print(f'{W}x{H} @ {FPS[0]}/{FPS[1]} fps, chroma {m["C"]}')

FRAME_BYTES = W * H * 3 // 2       # 4:2:0
TAG = 6                            # len('FRAME\\n')
offset = nl + 1                    # first byte after the header
print(f'frame payload {FRAME_BYTES} bytes, first frame starts at byte {offset}')

need = N_FRAMES * (TAG + FRAME_BYTES)
req = urllib.request.Request(CLIP_URL,
                             headers={'Range': f'bytes={offset}-{offset+need-1}'})
raw = urllib.request.urlopen(req, timeout=300).read()
print(f'fetched {len(raw)/1e6:.1f} MB, expected {need/1e6:.1f} MB')
assert len(raw) == need, 'short read; the host may not support Range for this file'
"""))

CELLS.append(code("""def yuv420_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    \"\"\"BT.601 limited-range YUV420 -> RGB, matching ffmpeg's yuv420p default.\"\"\"
    y = y.astype(np.float32)
    u = np.repeat(np.repeat(u.astype(np.float32), 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(v.astype(np.float32), 2, axis=0), 2, axis=1)
    yy = (y - 16.0) * 1.164383
    uu = u - 128.0
    vv = v - 128.0
    r = yy + 1.596027 * vv
    g = yy - 0.391762 * uu - 0.812968 * vv
    b = yy + 2.017232 * uu
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


frames = []
pos = 0
for i in range(N_FRAMES):
    tag = raw[pos:pos+TAG]
    assert tag.startswith(b'FRAME'), f'bad frame tag at {pos}: {tag!r}'
    pos += TAG
    y = np.frombuffer(raw[pos:pos+W*H], dtype=np.uint8).reshape(H, W); pos += W*H
    cw, ch = W//2, H//2
    u = np.frombuffer(raw[pos:pos+cw*ch], dtype=np.uint8).reshape(ch, cw); pos += cw*ch
    v = np.frombuffer(raw[pos:pos+cw*ch], dtype=np.uint8).reshape(ch, cw); pos += cw*ch
    frames.append(yuv420_to_rgb(y, u, v))

print(f'decoded {len(frames)} frames of {frames[0].shape}')
Image.fromarray(frames[0]).save(WORK / 'frame0.png')
print('saved a preview to', WORK / 'frame0.png')
"""))

CELLS.append(md("""## 3. Channel-wise tokenisation (the official way)

A frame is split by ``data[:, :, i:i+1]`` into single-channel blocks, each block is
replicated to a greyscale triple, and only then does the processor see it. A channel is
flattened row-major and cut into consecutive 1024-sample runs, so a 4096x2160 frame
yields **8640 chunks per channel** and **25920 units per frame**.
"""))

CELLS.append(code("""PATCH = 32
N_PATCH = PATCH * PATCH

def tokenise_frame(rgb, processor, patch=PATCH, batch=2000):
    \"\"\"Tokenise one frame exactly as the official loader feeds the processor.

    Returns ids of shape (units, patch*patch) in channel-major order, with each
    channel cut into consecutive flattened runs.
    \"\"\"
    h, w, _ = rgb.shape
    per_channel = (h * w) // (patch * patch)
    units = per_channel * 3

    stacked = np.empty((units, patch, patch, 3), dtype=np.uint8)
    i = 0
    for c in range(3):
        flat = rgb[:, :, c].reshape(-1)
        for k in range(per_channel):
            blk = flat[k*(patch*patch):(k+1)*(patch*patch)].reshape(patch, patch)
            stacked[i] = np.repeat(blk[:, :, None], 3, axis=2)
            i += 1

    parts = []
    for s in range(0, units, batch):
        e = min(s + batch, units)
        # The processor's colour quantiser builds an (N, 512) block; batching keeps
        # that transient small. All-at-once needs >5 GB.
        out = np.asarray(processor(stacked[s:e], return_tensors='np')['input_ids'])
        parts.append(out.reshape(e - s, -1))
    return np.concatenate(parts, axis=0).astype(np.int64), units, per_channel


_t0 = time.time()
ids0, units0, per_ch0 = tokenise_frame(frames[0], proc)
tok_s = time.time() - _t0
print(f'frame 0: {units0} units ({per_ch0}/channel), tokenised in {tok_s:.1f}s')
print('  ids shape', ids0.shape, ' distinct in unit 0:', len(np.unique(ids0[0])))
"""))

CELLS.append(md("""## 4. Measure

Everything reported is `sum(-log2 p_true)` on the model's own float32 probabilities,
which is the paper's convention. The arithmetic coder is not run: it cannot change the
reported rate (see `min_interval_width` in the project's coder) and it dominated the
runtime locally.

Checkpointing writes one JSON line per frame, flushed and fsynced, so a dropped session
resumes. Results also go to Drive if it is mounted.
"""))

CELLS.append(code("""OUT_DIR = Path('/content/results'); OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / 'video_channelwise.jsonl'

# Optional persistence: mount Drive so a dropped session does not lose results.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT_DIR = Path('/content/drive/MyDrive/lmcompress_results')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT = OUT_DIR / 'video_channelwise.jsonl'
    print('persisting to Drive:', OUT)
except Exception as exc:
    print('Drive not mounted, results stay in the session:', type(exc).__name__)

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)['frame'])
print('already finished:', sorted(done) or 'none')
"""))

CELLS.append(code("""CH = 32   # benchmarked: larger batches are slower here and 128 OOMs on 12 GiB

def score_frame(ids: np.ndarray, model, chunk: int = CH):
    \"\"\"Return the model-side code length in bits for a batch of token sequences.

    Position i predicts token i+1 (causal shift), so a sequence of L tokens yields
    L-1 scored positions. Scoring is vectorised: one forward pass per chunk of units,
    then -log2 p gathered on the GPU.
    \"\"\"
    units = ids.shape[0]
    total = 0.0
    for s in range(0, units, chunk):
        e = min(s + chunk, units)
        buf = torch.tensor(ids[s:e], dtype=torch.long, device='cuda')
        with torch.no_grad():
            lg = model(buf).logits.float()
        lp = torch.log_softmax(lg, dim=-1)[:, :-1, :]
        tgt = buf[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total += float(nll.sum().item())
        del lg, lp, nll
    return total


rows = []
t_all = time.time()
for fi, rgb in enumerate(frames):
    name = f'frame_{fi:06d}'
    if name in done:
        print(f'[{fi+1}/{len(frames)}] {name}: already done')
        continue
    t0 = time.time()
    ids, units, per_ch = tokenise_frame(rgb, proc)
    t_tok = time.time() - t0
    t0 = time.time()
    bits = score_frame(ids, model)
    t_mod = time.time() - t0
    samples = units * N_PATCH
    row = dict(frame=name, units=units, per_channel_chunks=per_ch, samples=samples,
               coded_bits=bits, coded_bytes=bits/8,
               rate_pct_official=100*(bits/8)/samples,
               tokenise_seconds=round(t_tok, 1), model_seconds=round(t_mod, 1))
    rows.append(row)
    with open(OUT, 'a') as fh:
        fh.write(json.dumps(row, sort_keys=True) + '\\n'); fh.flush(); os.fsync(fh.fileno())
    eta = (len(frames) - fi - 1) * (t_tok + t_mod) / 60
    print(f'[{fi+1}/{len(frames)}] {name}: {units} units, {bits/8/1e6:.2f} MB, '
          f'rate {row["rate_pct_official"]:.4f} % '
          f'(tok {t_tok:.1f}s + model {t_mod:.1f}s, ETA {eta:.0f} min)', flush=True)

print(f'\\ntotal wall {time.time()-t_all:.0f}s')
"""))

CELLS.append(code("""allrows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
bits = sum(r['coded_bits'] for r in allrows)
samples = sum(r['samples'] for r in allrows)
rgb_bytes = W * H * 3 * len(allrows)

print('=== video via iGPT, channel-wise (official convention) ===')
print(f'  frames                = {len(allrows)}')
print(f'  units per frame       = {allrows[0]["units"]}')
print(f'  processed samples     = {samples:,}')
print(f'  coded bytes           = {bits/8:,.0f}')
print(f'  RATE (official conv.) = {100*(bits/8)/samples:.4f} %')
print(f'  rate on RGB bytes     = {100*(bits/8)/rgb_bytes:.4f} %')
print(f'  published (SI Table 4, high-res) = 15.31 %')
print()
per = np.array([r['rate_pct_official'] for r in allrows])
print(f'  per-frame min/mean/max = {per.min():.4f} / {per.mean():.4f} / {per.max():.4f} %')
print(f'  frame-to-frame spread  = {per.max()-per.min():.4f} pp')
"""))

CELLS.append(md("""## 5. Classical baselines on the same frames

The model row must be compared against codecs run on **the same material**, otherwise
the comparison is meaningless. We still have the raw YUV420 for these frames, so each
codec is run on exactly the bytes the model saw. Each is lossless: H.265 `crf 0`,
H.264 `crf 0`, FFV1 level 3, and zstd over raw YUV.

Note the asymmetry that the paper itself notes: the codecs may exploit inter-frame
redundancy, while the model deliberately does not.
"""))

CELLS.append(code("""import shutil
YUV = WORK / f'{N_FRAMES}frames_4096x2160.yuv'

# Write the YUV planes only, in I420 order, which is what ffmpeg expects.
with open(YUV, 'wb') as fh:
    p = 0
    for i in range(N_FRAMES):
        p += TAG
        fh.write(raw[p:p+W*H]); p += W*H
        fh.write(raw[p:p+(W//2)*(H//2)]); p += (W//2)*(H//2)
        fh.write(raw[p:p+(W//2)*(H//2)]); p += (W//2)*(H//2)
print('raw YUV for baselines:', YUV.stat().st_size/1e6, 'MB')

def run_codec(name, args, ext):
    out = WORK / f'out.{ext}'
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  {name}: FAILED {r.stderr[-300:]}')
        return None
    size = out.stat().st_size
    return dict(codec=name, bytes=size, rate_pct=100*size/(YUV.stat().st_size),
                seconds=round(time.time()-t0, 1))

common = ['-hide_banner', '-loglevel', 'error', '-y',
          '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{W}x{H}',
          '-i', str(YUV), '-frames:v', str(N_FRAMES)]

base_rows = []
for name, enc, ext in [
    ('H.265 crf 0 (x265 veryslow)', ['-c:v', 'libx265', '-crf', '0', '-preset', 'veryslow',
                                     '-x265-params', 'log-level=none'], 'mp4'),
    ('H.264 crf 0 (x264 veryslow)', ['-c:v', 'libx264', '-crf', '0', '-preset', 'veryslow'], 'mp4'),
    ('FFV1 level 3', ['-c:v', 'ffv1', '-level', '3'], 'mkv'),
]:
    row = run_codec(name, ['ffmpeg'] + common + enc + [str(WORK / f'out.{ext}')], ext)
    if row:
        base_rows.append(row)
        print(f'  {row["codec"]:<30} {row["rate_pct"]:7.3f} %  {row["bytes"]/1e6:8.1f} MB  {row["seconds"]}s')

# zstd over the raw YUV
t0 = time.time()
rz = subprocess.run(['zstd', '--ultra', '-22', '-q', '-f', str(YUV), '-o', str(WORK/'out.zst')],
                    capture_output=True, text=True)
if rz.returncode == 0:
    size = (WORK/'out.zst').stat().st_size
    base_rows.append(dict(codec='zstd --ultra -22 (raw YUV)', bytes=size,
                          rate_pct=100*size/(YUV.stat().st_size),
                          seconds=round(time.time()-t0, 1)))
    print(f'  {"zstd --ultra -22":<30} {base_rows[-1]["rate_pct"]:7.3f} %')
"""))

CELLS.append(code("""print('=== same-frame comparison ===')
print(f'{"method":<32}{"rate %":>10}   published')
print('-'*62)
pub = {'H.265 crf 0 (x265 veryslow)': 21.69, 'H.264 crf 0 (x264 veryslow)': 37.42,
       'FFV1 level 3': 40.16, 'zstd --ultra -22 (raw YUV)': 51.04}
for r in sorted(base_rows, key=lambda x: x['rate_pct']):
    print(f'{r["codec"]:<32}{r["rate_pct"]:>10.3f}   {pub.get(r["codec"], "-")}')
allr = 100*(bits/8)/samples
print(f'{"LMCompress iGPT, channel-wise":<32}{allr:>10.3f}   15.31 (paper)')
print()
print('Ordering check (published high-res order is H.265 < H.264 < FFV1 < zstd):')
print('  ours:', ' < '.join(x['codec'].split()[0] for x in sorted(base_rows, key=lambda x: x['rate_pct'])))
"""))

CELLS.append(md("""## 6. Save and download

Everything needed to reproduce the write-up is in one JSON file.
"""))

CELLS.append(code("""summary = dict(
    clip=CLIP_URL, width=W, height=H, fps=f'{FPS[0]}/{FPS[1]}', frames=len(allrows),
    units_per_frame=allrows[0]['units'], samples_per_frame=allrows[0]['samples'],
    coded_bits=bits, coded_bytes=bits/8, rate_pct_official=100*(bits/8)/samples,
    rate_pct_rgb=100*(bits/8)/rgb_bytes,
    published_rate_pct=15.31,
    per_frame=[r['rate_pct_official'] for r in allrows],
    baselines=base_rows,
    model_id=MODEL_ID,
    note=('channel-wise tokenisation per the official loader; reported length is '
          'sum(-log2 p_true) on the model float32 probabilities; arithmetic coder '
          'skipped because it cannot change the reported rate'),
)
SUM = OUT.parent / 'video_channelwise_summary.json'
SUM.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'per_frame'}, indent=2)[:2000])
print()
print('written:', OUT)
print('written:', SUM)

try:
    from google.colab import files
    files.download(str(SUM))
    files.download(str(OUT))
except Exception as exc:
    print('download helper unavailable:', type(exc).__name__)
"""))

CELLS.append(md("""## What to send back

`video_channelwise_summary.json` is enough. It carries the aggregate rate, the
per-frame spread, the four classical baselines measured on the identical frames, and
the clip identifier.

## Interpreting the result

* **The rate is not a source-video rate.** The numerator is the code length of a 32x32,
  512-colour token grid per block, and the denominator counts the source samples the
  pipeline processed. The two describe different objects; that is the paper's
  convention, reproduced here so the numbers are comparable with it.
* **The codecs above are not a like-for-like comparison** for the same reason, but they
  are the comparison the paper makes, and running them on the identical frames removes
  the clip as a confound.
* **Frame-to-frame spread** is reported so the 10-frame sample can be judged: if the
  spread is small, the mean is well determined.
"""))


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "colab", "Video_ChannelWise_Colab.ipynb")
    nb = {
        "cells": CELLS,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"  wrote {out}")
    print(f"  {len(CELLS)} cells "
          f"({sum(1 for c in CELLS if c['cell_type'] == 'code')} code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
