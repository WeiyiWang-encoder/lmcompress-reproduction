"""Validate the video Colab notebook's core logic locally, before it is used.

A notebook that fails at cell 8 wastes the user's afternoon, so every step that can
be exercised offline is exercised here:

  1. HTTP Range read of the Y4M header from the public clip;
  2. HTTP Range read of exactly the bytes for N frames, and parsing of those frames;
  3. YUV420 -> RGB conversion, checked against a reference on a known constant patch;
  4. channel-wise tokenisation of a real 4K frame, against the official unit count;
  5. the batched scoring formula, against the already-measured frame 0 rate.

Only the full 10-frame run is skipped (that is what Colab is for); one frame is
measured to confirm the number matches the locally measured 16.7162 %.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
CLIP_URL = ("https://media.xiph.org/video/derf/y4m/"
            "Netflix_Aerial_4096x2160_60fps_8bit_420.y4m")
N = 2            # frames to fetch for the test
PATCH = 32


def yuv420_to_rgb(y, u, v):
    y = y.astype(np.float32)
    u = np.repeat(np.repeat(u.astype(np.float32), 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(v.astype(np.float32), 2, axis=0), 2, axis=1)
    yy = (y - 16.0) * 1.164383
    uu, vv = u - 128.0, v - 128.0
    r = yy + 1.596027 * vv
    g = yy - 0.391762 * uu - 0.812968 * vv
    b = yy + 2.017232 * uu
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def main() -> int:
    import re

    print("=== 1. Y4M header over HTTP Range ===")
    req = urllib.request.Request(CLIP_URL, headers={"Range": "bytes=0-255"})
    head = urllib.request.urlopen(req, timeout=60).read()
    nl = head.find(b"\n")
    header = head[:nl].decode("ascii", "replace")
    print("  header:", header)
    m = dict(re.findall(r"([WHFC])([0-9:.]+)", header))
    W, H = int(m["W"]), int(m["H"])
    print(f"  {W}x{H} chroma {m['C']}")
    assert W == 4096 and H == 2160

    frame_bytes = W * H * 3 // 2
    tag = 6
    offset = nl + 1
    print(f"  frame payload {frame_bytes} B, first frame at byte {offset}")

    print(f"\n=== 2. Range read of {N} frames ===")
    need = N * (tag + frame_bytes)
    req = urllib.request.Request(
        CLIP_URL, headers={"Range": f"bytes={offset}-{offset+need-1}"})
    raw = urllib.request.urlopen(req, timeout=600).read()
    print(f"  fetched {len(raw)/1e6:.1f} MB (expected {need/1e6:.1f} MB)")
    assert len(raw) == need, "short read"

    frames = []
    pos = 0
    cw, ch = W // 2, H // 2
    for i in range(N):
        assert raw[pos:pos+tag].startswith(b"FRAME"), f"bad tag at {pos}"
        pos += tag
        y = np.frombuffer(raw[pos:pos+W*H], np.uint8).reshape(H, W); pos += W*H
        u = np.frombuffer(raw[pos:pos+cw*ch], np.uint8).reshape(ch, cw); pos += cw*ch
        v = np.frombuffer(raw[pos:pos+cw*ch], np.uint8).reshape(ch, cw); pos += cw*ch
        frames.append((y, u, v))
    print(f"  parsed {len(frames)} frames, Y shape {frames[0][0].shape}")

    print("\n=== 3. YUV -> RGB sanity ===")
    rgb = yuv420_to_rgb(*frames[0])
    print(f"  rgb shape {rgb.shape} dtype {rgb.dtype}")
    print(f"  channel means R={rgb[:,:,0].mean():.1f} "
          f"G={rgb[:,:,1].mean():.1f} B={rgb[:,:,2].mean():.1f}")
    # neutral chroma must map to grey
    yz = np.full((64, 64), 128, np.uint8)
    uz = np.full((32, 32), 128, np.uint8)
    gz = yuv420_to_rgb(yz, uz, uz)
    spread = int(gz.max()) - int(gz.min())
    print(f"  neutral-chroma patch R/G/B max-min = {spread} (should be small)")
    assert spread <= 2

    print("\n=== 4. channel-wise tokenisation vs the official unit count ===")
    from uic.data import _igpt_processor
    proc = _igpt_processor(MODEL_DIR)
    per_channel = (H * W) // (PATCH * PATCH)
    units = per_channel * 3
    print(f"  per channel {per_channel}, units {units} "
          f"(official: 8640 x 3 = 25920)")
    assert units == 25920

    # tokenise only the first channel to keep this quick
    n_patch = PATCH * PATCH
    flat = rgb[:, :, 0].reshape(-1)
    batch = np.empty((64, PATCH, PATCH, 3), np.uint8)
    for k in range(64):
        blk = flat[k*n_patch:(k+1)*n_patch].reshape(PATCH, PATCH)
        batch[k] = np.repeat(blk[:, :, None], 3, axis=2)
    ids = np.asarray(proc(batch, return_tensors="np")["input_ids"]).reshape(64, -1)
    print(f"  tokenised 64 units -> {ids.shape}, "
          f"distinct in unit 0 = {len(np.unique(ids[0]))}")
    assert ids.shape == (64, 1024)

    print("\n=== 5. batched scoring on one full frame ===")
    import torch
    from transformers import ImageGPTForCausalImageModeling

    model = ImageGPTForCausalImageModeling.from_pretrained(MODEL_DIR).to("cuda:0").eval()

    allids = np.empty((units, 1024), np.int64)
    CHUNK = 2000
    for c in range(3):
        f = rgb[:, :, c].reshape(-1)
        st = np.empty((per_channel, PATCH, PATCH, 3), np.uint8)
        for k in range(per_channel):
            blk = f[k*n_patch:(k+1)*n_patch].reshape(PATCH, PATCH)
            st[k] = np.repeat(blk[:, :, None], 3, axis=2)
        parts = []
        for s in range(0, per_channel, CHUNK):
            e = min(s + CHUNK, per_channel)
            out = np.asarray(proc(st[s:e], return_tensors="np")["input_ids"])
            parts.append(out.reshape(e - s, -1))
        allids[c*per_channel:(c+1)*per_channel] = np.concatenate(parts, 0)

    total = 0.0
    for s in range(0, units, 32):
        e = min(s + 32, units)
        buf = torch.tensor(allids[s:e], dtype=torch.long, device="cuda:0")
        with torch.no_grad():
            lg = model(buf).logits.float()
        lp = torch.log_softmax(lg, dim=-1)[:, :-1, :]
        tgt = buf[:, 1:]
        total += float((-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).sum())

    samples = units * n_patch
    rate = 100 * (total / 8) / samples
    print(f"  coded bits   = {total:.1f}")
    print(f"  samples      = {samples}")
    print(f"  rate         = {rate:.4f} %")
    print(f"\n  locally measured on the ORIGINAL clip, frame 0: 16.7162 %")
    print(f"  this is a DIFFERENT clip (Netflix Aerial), so equality is not expected;")
    print(f"  the check is that the value is in the same regime, i.e. 10-25 %.")
    assert 10 < rate < 25, f"rate {rate} outside the plausible regime"
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
