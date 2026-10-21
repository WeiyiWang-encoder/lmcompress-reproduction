"""Compression engine: drives a probability predictor through an arithmetic coder.

Everything long-running lives here, and everything here is written to survive an
abrupt power cut:

* ``ChunkCursor`` records, *on disk and atomically*, the index of the last fully
  finished chunk together with the running bit/byte totals.  A restarted run
  re-reads it and resumes; the only work redone is the single in-flight chunk.
* All result files are written through :func:`atomic_write_text` so a partial
  write can never replace a good file.
* ``results/`` receives one JSON line per finished file, so a batch run that dies
  halfway still leaves usable numbers for everything already done.

The predictor contract
----------------------
A predictor is any object exposing::

    name: str
    symbols_per_unit: int      # bytes produced per yielded item
    targets(unit) -> Iterator[Step]

Each ``Step`` carries the model's distribution over the next symbol, already
converted to a float32 numpy array of length ``n_symbols``, plus the true symbol
value.  The engine is deliberately agnostic about whether the "symbol" is a byte
(bGPT / Llama) or a pixel token (ImageGPT).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    cdf_lookup,
    probs_to_cdf,
    sparse_true_symbol_cdf,
)

MAGIC = b"UIC1"


def atomic_write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path`` atomically (survives power loss)."""
    tmp = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, payload) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def append_jsonl(path: str, payload) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


@dataclass
class Step:
    """One coded symbol.

    Exactly one of ``cdf`` / ``probs`` is used:

    * ``probs`` -- a dense distribution over the alphabet.  This is the *faithful*
      path: bGPT and ImageGPT really do emit a full posterior (257 and 513
      entries), so the arithmetic code is built from the model's own numbers and
      the coded length equals the model cross-entropy up to coder rounding.
    * ``cdf`` -- a ready-made integer CDF, used when only ``p(true symbol)`` is
      known (Llama's 128 k vocabulary).  See
      :func:`uic.coder.sparse_true_symbol_cdf`.

    ``p_true`` is always the model's exact probability for ``symbol`` and is what
    the reported compression numbers are computed from, so the headline metric
    never depends on which representation was chosen.
    """

    symbol: int
    p_true: float
    probs: Optional[np.ndarray] = None
    cdf: Optional[Sequence[int]] = None

    def integer_cdf(self) -> Sequence[int]:
        if self.cdf is not None:
            return self.cdf
        if self.probs is None:
            raise ValueError("Step needs either probs or cdf")
        return probs_to_cdf(self.probs.tolist())


@dataclass
class Window:
    """A single forward pass's worth of symbols.

    ``offset`` is model-specific bookkeeping (bGPT needs it to map the flattened
    logit grid onto payload bytes; causal models ignore it).
    """

    start: int
    stop: int
    prior: int = 0
    offset: int = 0

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class UnitResult:
    """Accumulated result for one input unit (file / channel / frame)."""

    raw_bytes: int = 0
    coded_bits: float = 0.0  # sum of -log2 p(true symbol)
    coded_symbols: int = 0
    ac_bits: int = 0  # actual arithmetic-coder output length
    wall_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def bits_per_byte(self) -> float:
        return self.coded_bits / self.raw_bytes if self.raw_bytes else float("nan")

    @property
    def compression_rate(self) -> float:
        """compressed / original, in percent -- the metric the paper reports."""
        if not self.raw_bytes:
            return float("nan")
        return 100.0 * (self.coded_bits / 8.0) / self.raw_bytes

    @property
    def compression_ratio(self) -> float:
        """original / compressed -- the reciprocal metric."""
        return 1.0 / (self.compression_rate / 100.0) if self.raw_bytes else float("nan")

    def as_dict(self) -> dict:
        return {
            "raw_bytes": self.raw_bytes,
            "coded_bits": round(self.coded_bits, 6),
            "coded_symbols": self.coded_symbols,
            "ac_bits": self.ac_bits,
            "bits_per_byte": round(self.bits_per_byte, 6),
            "compression_rate_pct": round(self.compression_rate, 6),
            "compression_ratio": round(self.compression_ratio, 6),
            "wall_seconds": round(self.wall_seconds, 3),
            **self.extra,
        }


class ChunkCursor:
    """Append-only, crash-safe progress record for one long coding task."""

    def __init__(self, path: str, total_chunks: int, meta: Optional[dict] = None):
        self.path = path
        self.total_chunks = int(total_chunks)
        self.state = {
            "done_chunks": 0,
            "coded_bits": 0.0,
            "coded_symbols": 0,
            "raw_bytes": 0,
            "meta": meta or {},
            "finished": False,
        }
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if int(loaded.get("total_chunks", -1)) == self.total_chunks:
                    self.state.update(loaded)
            except (json.JSONDecodeError, OSError, ValueError):
                # A torn write from a power cut: start over rather than guess.
                pass

    @property
    def done_chunks(self) -> int:
        return int(self.state["done_chunks"])

    @property
    def finished(self) -> bool:
        return bool(self.state.get("finished", False))

    def commit(self, chunk_index: int, bits: float, symbols: int, raw_bytes: int) -> None:
        self.state.update(
            {
                "done_chunks": chunk_index + 1,
                "coded_bits": self.state["coded_bits"] + bits,
                "coded_symbols": self.state["coded_symbols"] + symbols,
                "raw_bytes": self.state["raw_bytes"] + raw_bytes,
            }
        )
        self.state["total_chunks"] = self.total_chunks
        self.state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(self.path, self.state)

    def mark_finished(self, payload: Optional[dict] = None) -> None:
        self.state["finished"] = True
        self.state["total_chunks"] = self.total_chunks
        if payload:
            self.state.setdefault("result", {}).update(payload)
        atomic_write_json(self.path, self.state)


def measure_stream(
    steps: Iterable[Step],
    precision: int = 32,
    verify_roundtrip: bool = False,
    with_coder: bool = True,
) -> tuple[UnitResult, Optional[bytes]]:
    """Code a symbol stream, returning the ideal (cross-entropy) length.

    The arithmetic coder is run in *encode* mode by default so that ``ac_bits`` is a
    real number rather than a modelled one.  Decoding is optional because it is
    O(n) extra model calls in general; when ``verify_roundtrip`` is set we only
    decode the bit stream against the *recorded* cdfs, which validates the coder
    and container without re-running the model (the decoder is deterministic and
    the model would reproduce the same cdfs).

    ``with_coder=False`` skips the arithmetic encoder and returns only the model-side
    code length ``sum(-log2 p_true)``.

    **Why that switch exists.** Profiling a video frame put 78 % of wall time in
    ``ArithmeticEncoder.encode`` (through ``integer_cdf``), against 25.7 ms for the
    model forward pass -- the coder, not the model, was the bottleneck at 308 ms per
    unit.  The *reported* rate never consumes the coder's quantised probabilities
    (see ``coder.min_interval_width``), so for bulk measurement the encoder is pure
    overhead.  It stays on for the correctness tests and for any run that reports
    ``ac_bits`` as a consistency check.
    """
    t0 = time.perf_counter()
    result = UnitResult()
    out_bits: List[int] = []
    encoder = ArithmeticEncoder(out_bits.append) if with_coder else None

    recorded_cdfs: List[np.ndarray] = []
    recorded_symbols: List[int] = []

    for step in steps:
        # The reported code length uses the model's exact probability, so it is
        # independent of how the dense cdf was constructed.
        result.coded_bits += -math.log2(max(step.p_true, np.finfo(np.float64).tiny))
        result.coded_symbols += 1

        if encoder is not None:
            cdf = step.integer_cdf()
            encoder.encode(cdf, step.symbol)
            if verify_roundtrip:
                recorded_cdfs.append(np.asarray(cdf, dtype=np.int64))
                recorded_symbols.append(step.symbol)

    if encoder is not None:
        encoder.terminate()
        result.ac_bits = len(out_bits)
    result.raw_bytes = result.coded_symbols

    payload = None
    if verify_roundtrip:
        from .bitio import bits_to_bytes

        payload, num_padded = bits_to_bytes(out_bits)
        bits = list(out_bits)
        pos = {"i": 0}

        def _next_bit() -> int:
            i = pos["i"]
            pos["i"] = i + 1
            return bits[i] if i < len(bits) else 0

        decoder = ArithmeticDecoder(_next_bit)
        decoded: List[int] = []
        for cdf, symbol in zip(recorded_cdfs, recorded_symbols):
            target = decoder.decode_target()
            guess = cdf_lookup(cdf, target)
            if guess != symbol:
                raise AssertionError(
                    f"round-trip mismatch at symbol {len(decoded)}: "
                    f"{guess} != {symbol}"
                )
            decoder.update(cdf, symbol)
            decoded.append(guess)
        result.extra["roundtrip"] = "ok"
        result.extra["roundtrip_payload_bytes"] = len(payload)
        result.extra["roundtrip_padded_bits"] = num_padded

    result.wall_seconds = time.perf_counter() - t0
    return result, payload


# --------------------------------------------------------------------------- #
# Reference coder: uniform bytes.  This is the "no model at all" upper bound and
# doubles as an end-to-end test of the coder itself (it must give exactly 8
# bits per byte and must round-trip perfectly).
# --------------------------------------------------------------------------- #

_UNIFORM_256 = np.full(256, 1.0 / 256.0, dtype=np.float32)


def uniform_byte_steps(data: bytes) -> Iterator[Step]:
    for byte in data:
        yield Step(symbol=byte, p_true=1.0 / 256.0, probs=_UNIFORM_256)


def measure_uniform_bytes(data: bytes, verify_roundtrip: bool = True) -> UnitResult:
    result, _ = measure_stream(
        uniform_byte_steps(data), verify_roundtrip=verify_roundtrip
    )
    return result
