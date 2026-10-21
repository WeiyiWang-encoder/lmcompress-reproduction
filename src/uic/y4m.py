"""Minimal YUV4MPEG2 (``.y4m``) reader.

The paper's video data is raw Y4M from Xiph.org (``media.xiph.org/video/derf``),
and the capsule relies on ffmpeg to turn it into PNG frames.  ffmpeg is not
always available, and shelling out for a 4K clip is wasteful anyway, so this
module parses the container directly.  It supports the 8-bit 4:2:0 and 4:4:4
layouts, which covers the Xiph test set.

Y4M layout::

    YUV4MPEG2 W<w> H<h> F<num>:<den> I<p|t|b|m> A<num>:<den> C<chroma>[ tags]\n
    FRAME[ tags]\n
    <Y plane><U plane><V plane>

All planes are 8-bit.  ``C420jpeg`` means BT.601 with the *full* 0-255 luma
range (JFIF style), which is the case for the clip shipped with the capsule, so
the conversion below uses the JPEG variant of BT.601 rather than the studio-swing
one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

CHROMA_PLANES = {
    "420": 1,  # U and V at half resolution in both axes
    "420jpeg": 1,
    "420mpeg2": 1,
    "420paldv": 1,
    "422": 1,  # handled via explicit widths below
    "444": 0,
    "444alpha": 0,
    "mono": 0,
}


@dataclass
class Y4MHeader:
    width: int
    height: int
    fps_num: int
    fps_den: int
    chroma: str
    interlacing: str
    tags: str

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den if self.fps_den else 0.0

    def plane_sizes(self) -> tuple[int, int, int]:
        h, w = self.height, self.width
        if self.chroma.startswith("mono"):
            return h * w, 0, 0
        if self.chroma.startswith("444"):
            return h * w, h * w, h * w
        if self.chroma.startswith("422"):
            return h * w, h * (w // 2), h * (w // 2)
        # default / 420*
        return h * w, (h // 2) * (w // 2), (h // 2) * (w // 2)

    @property
    def frame_bytes(self) -> int:
        return sum(self.plane_sizes())


def read_header(fh) -> Y4MHeader:
    line = fh.readline()
    if not line.startswith(b"YUV4MPEG2"):
        raise ValueError("not a Y4M stream")
    parts = line.decode("ascii", "replace").split()
    w = h = None
    num, den, chroma, inter = 25, 1, "420", "p"
    for token in parts[1:]:
        if not token:
            continue
        key, value = token[0], token[1:]
        if key == "W":
            w = int(value)
        elif key == "H":
            h = int(value)
        elif key == "F":
            num, den = (int(x) for x in value.split(":"))
        elif key == "C":
            chroma = value
        elif key == "I":
            inter = value
    if w is None or h is None:
        raise ValueError("Y4M header missing frame size")
    return Y4MHeader(w, h, num, den, chroma, inter, "")


def iter_frames(path: str, limit: Optional[int] = None) -> Iterator[np.ndarray]:
    """Yield RGB ``uint8`` arrays of shape ``(h, w, 3)``."""
    with open(path, "rb") as fh:
        header = read_header(fh)
        sizes = header.plane_sizes()
        count = 0
        while limit is None or count < limit:
            line = fh.readline()
            if not line:
                return
            if not line.startswith(b"FRAME"):
                return
            raw = fh.read(header.frame_bytes)
            if len(raw) < header.frame_bytes:
                return
            yield _to_rgb(np.frombuffer(raw, dtype=np.uint8), header, sizes)
            count += 1


def _to_rgb(buf: np.ndarray, header: Y4MHeader, sizes) -> np.ndarray:
    h, w = header.height, header.width
    y = buf[: sizes[0]].reshape(h, w).astype(np.float32)
    if header.chroma.startswith("mono") or sizes[1] == 0:
        grey = np.clip(y, 0, 255).astype(np.uint8)
        return np.stack([grey, grey, grey], axis=-1)

    if header.chroma.startswith("422"):
        cw, ch = w // 2, h
    elif header.chroma.startswith("444"):
        cw, ch = w, h
    else:
        cw, ch = w // 2, h // 2

    u = buf[sizes[0] : sizes[0] + sizes[1]].reshape(ch, cw).astype(np.float32)
    v = buf[sizes[0] + sizes[1] :].reshape(ch, cw).astype(np.float32)

    if (cw, ch) != (w, h):
        ry = max(1, h // ch)
        rx = max(1, w // cw)
        u = np.repeat(np.repeat(u, ry, axis=0), rx, axis=1)[:h, :w]
        v = np.repeat(np.repeat(v, ry, axis=0), rx, axis=1)[:h, :w]

    # JFIF-range BT.601 (C420jpeg uses full-range luma)
    cb = u - 128.0
    cr = v - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def probe(path: str) -> dict:
    with open(path, "rb") as fh:
        header = read_header(fh)
    return {
        "width": header.width,
        "height": header.height,
        "fps": header.fps,
        "chroma": header.chroma,
        "interlacing": header.interlacing,
        "frame_bytes": header.frame_bytes,
        "approx_frames": os.path.getsize(path) // max(1, header.frame_bytes + 6),
    }
