"""Modality loaders.

Every loader yields :class:`Unit` objects.  A unit is the smallest thing the
runner reports a compression number for, and the runner always reports *both*:

* the metric the paper uses (compressed bytes / original bytes), and
* a **true-lossless** metric that charges for any lossy pre-processing the
  pipeline performs.

The second one matters.  The paper's image and video pipeline runs images
through ImageGPT's processor, which resizes them to 32x32 and quantises colours
to 512 clusters before the "lossless" arithmetic coder ever sees them.  A ratio
computed against the original pixel count is therefore not the ratio of a
lossless codec.  Rather than silently reproduce that, the loaders hand the
runner the residual so it can be charged explicitly.
"""

from __future__ import annotations

import glob
import os
import wave
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class Unit:
    """One measurable item.

    ``payload`` is whatever the predictor consumes: a list of byte values for
    bGPT, a list of token ids for ImageGPT / Llama.

    ``original_bytes`` is the denominator of the compression rate and is always
    the size of the *real* data, not of a downscaled stand-in.
    """

    name: str
    payload: Sequence[int]
    original_bytes: int
    meta: dict = field(default_factory=dict)

    @property
    def n_symbols(self) -> int:
        return len(self.payload)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #


def wav_pcm_bytes(path: str) -> tuple[bytes, dict]:
    """Raw PCM frames of a WAV file plus its format metadata.

    The paper standardises audio to 16 kHz / mono / 8-bit.  We reproduce the
    8-bit reduction when the source is 16-bit (see
    :func:`to_8bit_unsigned`) but never resample: the ratio is computed against
    the file's own PCM payload, so no information can leak away unnoticed.
    """
    with wave.open(path, "rb") as fh:
        params = fh.getparams()
        frames = fh.readframes(params.nframes)
    meta = {
        "channels": params.nchannels,
        "sampwidth": params.sampwidth,
        "framerate": params.framerate,
        "nframes": params.nframes,
    }
    return frames, meta


def to_8bit_unsigned(frames: bytes, sampwidth: int) -> bytes:
    """Convert signed PCM to the unsigned 8-bit form the paper's models see.

    For 16-bit sources this is the lossy step, and the loader records it so the
    runner can either charge for it or ignore it (paper convention).
    """
    if sampwidth == 1:
        return frames
    if sampwidth == 2:
        import numpy as np

        arr = np.frombuffer(frames, dtype="<i2").astype(np.int32)
        quant = ((arr + 32768) >> 8).astype(np.uint8)
        return quant.tobytes()
    raise ValueError(f"unsupported sample width {sampwidth}")


def audio_units(
    directory: str,
    limit: Optional[int] = None,
    use_8bit: bool = False,
) -> List[Unit]:
    """Load ``*.wav`` files as byte payloads (the paper's audio representation)."""
    paths = sorted(glob.glob(os.path.join(directory, "*.wav")))
    if limit is not None:
        paths = paths[:limit]
    units: List[Unit] = []
    for path in paths:
        frames, meta = wav_pcm_bytes(path)
        payload = frames
        charged = 0
        if use_8bit:
            payload = to_8bit_unsigned(frames, meta["sampwidth"])
            charged = len(frames)
        units.append(
            Unit(
                name=os.path.basename(path),
                payload=list(payload),
                original_bytes=len(frames),
                meta={
                    **meta,
                    "extension": "wav",
                    "quantised_bytes_charged": charged,
                },
            )
        )
    return units


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def _igpt_processor(model_dir: str):
    from transformers import ImageGPTImageProcessor

    return ImageGPTImageProcessor.from_pretrained(model_dir)


def igpt_palette(processor) -> np.ndarray:
    """RGB palette ``(512, 3)`` in ``0..1`` for the ImageGPT colour clusters.

    transformers 5.x dropped ``ImageGPTImageProcessor.decode_color_quantization``,
    so the inverse quantisation is reconstructed here.  The cluster centres are
    stored on ``[-1, 1]`` -- they are applied after ``normalize(mean=0.5,
    std=0.5)`` -- so they map back to ``[0, 1]`` as ``c * 0.5 + 0.5``.

    Verified empirically in ``tools/probe_igpt_quantization.py``: nearest-centre
    lookup against this palette reproduces the processor's token ids for **100 %**
    of pixels, so it is the exact inverse of the quantisation step.
    """
    clusters = np.asarray(processor.clusters, dtype=np.float32)
    return clusters * 0.5 + 0.5


def image_units(
    directory: str,
    model_dir: str,
    limit: Optional[int] = None,
    extensions: Sequence[str] = (".png", ".bmp", ".jpg", ".jpeg"),
    include_residual: bool = True,
    patch: int = 32,
) -> List[Unit]:
    """Quantise images exactly the way the official loader does: **channel-wise**.

    This function previously tokenised the whole RGB image in a single processor
    call. That was wrong. The official path is:

      1. ``get_imagenet_iterator(is_channel_wised=True)`` splits the image with
         ``data[:, :, i:i+1]``, so every patch it yields has last axis **1**;
      2. ``language_model.compress`` therefore sees ``shape[-1] == 1``, skips its
         ``shape[-1] == 3`` branch, and takes ``np.repeat(..., 3, axis=2)``;
      3. the processor consequently receives a **greyscale triple** ``(v, v, v)``
         built from one channel value -- not a colour patch.

    Verified live: for one 32x32 patch the official path yields 1024 tokens with
    9 distinct ids, while the previous RGB call yielded 1024 tokens with 101
    distinct ids, differing from the very first token. See ``AUDIT_FOR_REVIEW.md``
    section 4.4.4.

    Both differences matter:

    * **What reaches the processor.** ``(v, v, v)`` confines the input to the 256
      values of a single channel; a colour patch spans all 2^24 triples. Only the
      former matches the released code, and only the former is what makes the
      preprocessing non-injective in the way the audit proves exhaustively.
    * **The denominator.** The official accounting adds ``h * w`` per *patch*,
      which over a whole image equals the source pixel count -- not three times it.

    A unit here is therefore one (patch, channel) pair, matching the official
    granularity.
    """
    from PIL import Image

    paths: List[str] = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    paths = sorted(set(paths))
    if limit is not None:
        paths = paths[:limit]

    processor = _igpt_processor(model_dir)
    palette = igpt_palette(processor)
    units: List[Unit] = []

    for path in paths:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        h, w, _ = rgb.shape
        base = os.path.basename(path)

        # Official chunking: the channel is flattened row-major and cut into
        # consecutive `patch*patch` runs, NOT into a 2-D grid of blocks.  The two
        # differ whenever the dimensions are not multiples of 32, because a
        # flattened cut may span a row boundary.  `_extract_image_sequence` does
        # exactly this and discards the trailing remainder, so we do too:
        #   total_chunks = (h*w) // (patch*patch)
        n_patch = patch * patch
        per_channel = (h * w) // n_patch

        for channel in range(3):
            flat = rgb[:, :, channel].reshape(-1)
            for k in range(per_channel):
                block = flat[k * n_patch:(k + 1) * n_patch].reshape(patch, patch)
                # (32,32,1) -> (32,32,3) with R = G = B, mirroring
                # np.repeat(data[:, :, i:i+1], 3, axis=2) in the official code.
                triple = np.repeat(block[:, :, None], 3, axis=2)
                out = processor(triple, return_tensors="np")["input_ids"]
                tokens = np.asarray(out).reshape(-1)

                residual_bytes = b""
                if include_residual:
                    # What the token stream does not carry: the source block minus
                    # its de-quantised reconstruction.
                    low = palette[tokens].reshape(patch, patch, 3)
                    recon = (low[:, :, channel] * 255.0).round()
                    src = block.astype(np.float64)
                    residual_bytes = np.clip(
                        src - recon, -128, 127).astype(np.int8).tobytes()

                units.append(
                    Unit(
                        name=f"{base}#ch{channel}k{k:05d}",
                        payload=[int(t) for t in tokens],
                        original_bytes=n_patch,
                        meta={
                            "source": base,
                            "channel": channel,
                            "chunk": k,
                            "height": h,
                            "width": w,
                            "patch": patch,
                            "token_grid": f"{patch}x{patch}",
                            "residual_bytes": len(residual_bytes),
                            "residual": residual_bytes,
                        },
                    )
                )
    return units


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #


def extract_frames(video_path: str, out_dir: str, limit: Optional[int] = None) -> List[str]:
    """Extract frames to PNG.

    Prefers the built-in Y4M reader (the paper's video data is raw Y4M from
    Xiph.org) and only falls back to ffmpeg for other containers.
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
    if existing:
        return existing[:limit] if limit else existing

    if video_path.lower().endswith((".y4m", ".yuv")):
        from . import y4m

        for idx, rgb in enumerate(y4m.iter_frames(video_path, limit=limit)):
            Image.fromarray(rgb).save(os.path.join(out_dir, f"frame_{idx:06d}.png"))
        frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
        if frames:
            return frames

    return _extract_frames_ffmpeg(video_path, out_dir, limit)


def _extract_frames_ffmpeg(video_path: str, out_dir: str, limit: Optional[int]) -> List[str]:
    import subprocess

    from .ffmpeg_tools import find_ffmpeg

    ffmpeg = find_ffmpeg()
    pattern = os.path.join(out_dir, "frame_%06d.png")
    cmd = [ffmpeg, "-y", "-i", video_path]
    if limit:
        cmd += ["-frames:v", str(limit)]
    cmd += [pattern]
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
    return frames[:limit] if limit else frames


def video_units(
    video_path: str,
    model_dir: str,
    work_dir: str,
    frame_limit: Optional[int] = None,
    include_residual: bool = False,
) -> List[Unit]:
    """Treat each frame as an image and code it independently (paper, Fig. 4).

    The paper deliberately does not use interframe information, so this simply
    delegates to the image path frame by frame.
    """
    frames = extract_frames(video_path, work_dir, limit=frame_limit)
    units: List[Unit] = []
    for idx, frame in enumerate(frames):
        for unit in _single_image_units(frame, model_dir, include_residual):
            unit.name = f"frame{idx:06d}#ch{unit.meta['channel']}"
            unit.meta["frame_index"] = idx
            unit.meta["source"] = os.path.basename(video_path)
            units.append(unit)
    return units


def _single_image_units(path: str, model_dir: str, include_residual: bool) -> List[Unit]:
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="uic_frame_")
    try:
        target = os.path.join(tmp, os.path.basename(path))
        if os.path.abspath(path) != os.path.abspath(target):
            shutil.copyfile(path, target)
        return image_units(
            tmp, model_dir, limit=1, extensions=(".png",), include_residual=include_residual
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #


def read_text_csv(path: str) -> str:
    """Reproduce the paper's text container: CSV with a single ``TEXT`` column."""
    import pandas as pd

    df = pd.read_csv(path)
    if "TEXT" in df.columns:
        return "".join(str(x) for x in df["TEXT"].tolist())
    # EURLEX-derived files keep the same column name; be explicit if not.
    raise KeyError(f"{path} has no TEXT column (columns={list(df.columns)})")


def text_units(
    directory: str,
    tokenizer=None,
    limit: Optional[int] = None,
    extensions: Sequence[str] = (".csv",),
) -> List[Unit]:
    """Load text files; if a tokenizer is given, units carry token ids."""
    paths: List[str] = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    paths = sorted(paths)
    if limit is not None:
        paths = paths[:limit]

    units: List[Unit] = []
    for path in paths:
        text = read_text_csv(path)
        raw = text.encode("utf-8")
        if tokenizer is not None:
            payload: Sequence[int] = tokenizer.encode_text(text)
        else:
            payload = list(raw)
        units.append(
            Unit(
                name=os.path.basename(path),
                payload=payload,
                # denominator is the UTF-8 byte length of the *original* text,
                # which is what the paper's CSV container holds.
                original_bytes=len(raw),
                meta={"chars": len(text), "extension": "csv"},
            )
        )
    return units
