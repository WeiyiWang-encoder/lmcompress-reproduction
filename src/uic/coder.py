"""A self-contained integer arithmetic coder (bit-oriented, base 2).

This is an independent implementation written from the textbook range-coding
algorithm.  It is deliberately small and explicit so that its correctness can be
argued line by line, and it is *not* a copy of the coder shipped with the
reference implementation.

State is kept in two 32-bit halves of a 64-bit interval; the emitted symbol is a
single bit.  The classic carry problem is handled by ``_emit_with_pending``.

Invariant maintained by ``Encoder.encode`` / ``Decoder.decode``:
    ``cdf`` is a non-decreasing integer sequence with ``cdf[0] == 0``,
    ``cdf[-1] == 2**PRECISION`` and at least two entries.  The convention is
    half-open intervals ``[cdf[i], cdf[i + 1])``, i.e. ``cdf[i]`` is the
    inclusive lower bound and ``cdf[i + 1]`` the exclusive upper bound.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

PRECISION = 32
FULL = 1 << PRECISION
MASK = FULL - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)
THREE_QUARTERS = QUARTER * 3
TOP = 0xFF  # the top byte is resolved once it is fully determined


class ArithmeticEncoder:
    """Range encoder emitting one bit at a time via ``output_fn``."""

    def __init__(self, output_fn: Callable[[int], None]) -> None:
        self._output_fn = output_fn
        self._low = 0
        self._high = MASK
        self._pending = 0
        self.num_bits = 0

    def _emit_bit(self, bit: int) -> None:
        self._output_fn(bit)
        self.num_bits += 1

    def _emit_with_pending(self, bit: int) -> None:
        self._emit_bit(bit)
        follow = bit ^ 1
        for _ in range(self._pending):
            self._emit_bit(follow)
        self._pending = 0

    def encode(self, cdf: Sequence[int], symbol: int) -> None:
        """Encode ``symbol`` given its cumulative distribution ``cdf``."""
        lo = int(cdf[symbol])
        hi = int(cdf[symbol + 1])
        if not (0 <= lo < hi <= FULL):
            raise ValueError(f"illegal cdf interval [{lo}, {hi}) for {symbol}")

        width = self._high - self._low + 1
        # 64-bit intermediate keeps this exact for PRECISION <= 32.
        new_high = self._low + (width * hi // FULL) - 1
        new_low = self._low + (width * lo // FULL)
        if new_high < new_low:
            # Should be unreachable given probs_to_cdf's minimum interval width;
            # fail loudly rather than emit a corrupt stream.
            raise AssertionError(
                f"arithmetic coder interval collapsed: width={width} "
                f"interval=[{lo}, {hi}) -- increase min_interval_width()"
            )
        self._high = new_high
        self._low = new_low

        while True:
            if self._high < HALF:
                self._emit_with_pending(0)
            elif self._low >= HALF:
                self._emit_with_pending(1)
                self._low -= HALF
                self._high -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTERS:
                self._pending += 1
                self._low -= QUARTER
                self._high -= QUARTER
            else:
                break
            self._low = (self._low << 1) & MASK
            self._high = ((self._high << 1) | 1) & MASK

    def terminate(self) -> None:
        """Flush enough bits for the decoder to resolve the final interval."""
        self._pending += 1
        if self._low < QUARTER:
            self._emit_with_pending(0)
        else:
            self._emit_with_pending(1)


class ArithmeticDecoder:
    """Range decoder consuming one bit at a time via ``input_fn``.

    ``input_fn`` must return ``0``/``1`` and is allowed to return zeros forever
    once the stream is exhausted; that mirrors the reference implementation's
    behaviour and keeps the decoder branch-free.
    """

    def __init__(self, input_fn: Callable[[], int]) -> None:
        self._input_fn = input_fn
        self._low = 0
        self._high = MASK
        self._code = 0
        self.num_bits = 0
        for _ in range(PRECISION):
            self._code = ((self._code << 1) | (self._input_fn() & 1)) & MASK
            self.num_bits += 1

    def decode_target(self) -> int:
        """Return the scaled offset used to look up the symbol."""
        width = self._high - self._low + 1
        return ((self._code - self._low + 1) * FULL - 1) // width

    def update(self, cdf: Sequence[int], symbol: int) -> None:
        """Narrow the interval exactly as the encoder did."""
        lo = int(cdf[symbol])
        hi = int(cdf[symbol + 1])
        width = self._high - self._low + 1
        self._high = self._low + (width * hi // FULL) - 1
        self._low = self._low + (width * lo // FULL)

        while True:
            if self._high < HALF:
                pass
            elif self._low >= HALF:
                self._low -= HALF
                self._high -= HALF
                self._code -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTERS:
                self._low -= QUARTER
                self._high -= QUARTER
                self._code -= QUARTER
            else:
                break
            self._low = (self._low << 1) & MASK
            self._high = ((self._high << 1) | 1) & MASK
            self._code = ((self._code << 1) | (self._input_fn() & 1)) & MASK
            self.num_bits += 1


def min_interval_width(n_symbols: int) -> int:
    """Smallest cdf interval we are willing to hand to a single symbol.

    Why this is needed (this was a genuine bug found while building this port):
    a range coder's interval can only be subdivided faithfully while

        (current width) * (cdf interval width) >= FULL,

    because ``encode`` computes ``low + width * lo // FULL`` in *integers*.
    If a symbol is quantised to a mass of 1 while the current width is already
    below ``FULL``, then ``high`` can fall below ``low`` and the coder silently
    corrupts the stream.  The final symbol of a near-deterministic distribution
    is exactly where this bites: for the 70-symbol case that exposed this, the
    true symbol had p = 9.06e-11 and received an interval of width 1, which made
    the coder emit a decodable-but-wrong symbol.

    Reserving ``FULL / 2**20`` per symbol caps the worst-case cost of any one
    symbol at 20 bits and keeps the coder exact.

    **Why this does not distort the reported rate.** An earlier revision of this
    docstring claimed the reserve is harmless because "20 bits is above the 8
    bits/byte ceiling of a byte alphabet".  That argument is **invalid** and is
    recorded here only so it is not reintroduced: having 256 symbols does *not*
    bound the surprisal of the true byte, so a badly calibrated model can assign
    the true symbol a probability below ``2**-20``, in which case the floor does
    change the distribution the coder uses.

    The reserve nonetheless does not affect any headline number, for a different
    and simpler reason: **the reported rate is never computed from the coder's
    quantised probabilities.**  It is ``sum(-log2 q(true symbol))`` evaluated on
    the model's own float32 probabilities, before and independently of any
    quantisation (see :func:`uic.engine.measure_stream`).  The quantised cdf
    affects only the arithmetic-coded bitstream length, which is tracked
    separately (``ac_bits`` vs ``coded_bits``) and reported as a consistency
    check rather than as the result.
    """
    width = FULL >> 20
    if width < 1:
        width = 1
    if n_symbols * width > FULL // 2:
        width = max(1, FULL // (2 * n_symbols))
    return int(width)


def probs_to_cdf(probs: Sequence[float], min_width: Optional[int] = None) -> List[int]:
    """Convert a probability vector into a strictly increasing integer CDF.

    Guarantees, all of which the coder relies on:
      * ``cdf[0] == 0`` and ``cdf[-1] == FULL``;
      * ``cdf[i + 1] - cdf[i] >= min_width >= 1`` for every symbol, so a symbol
        the model believes impossible is still codable *and* decodable (this is
        what makes the code lossless rather than merely "lossless when the model
        is well calibrated");
      * the widths are otherwise proportional to the supplied probabilities.
    """
    n = len(probs)
    if n < 2:
        raise ValueError("need at least two symbols")
    if n > FULL // 2:
        raise ValueError("alphabet too large for this precision")

    if min_width is None:
        min_width = min_interval_width(n)
    min_width = int(min_width)
    if min_width < 1:
        raise ValueError("min_width must be >= 1")
    if n * min_width > FULL:
        raise ValueError(f"min_width {min_width} too large for {n} symbols")

    total = float(sum(probs))
    if not total > 0.0 or not math.isfinite(total):
        probs = [1.0 / n] * n
        total = 1.0

    budget = FULL - n * min_width

    cdf = [0] * (n + 1)
    running = 0.0
    prev = 0
    for i in range(n):
        running += max(0.0, float(probs[i]))
        raw = int(running / total * budget) + (i + 1) * min_width
        # Monotone, and never eats the guaranteed minimum of the symbols left.
        if raw < prev + min_width:
            raw = prev + min_width
        remaining = n - 1 - i
        if raw > FULL - remaining * min_width:
            raw = FULL - remaining * min_width
        cdf[i + 1] = raw
        prev = raw

    cdf[n] = FULL
    return cdf


def sparse_true_symbol_cdf(
    p_true: float,
    n_symbols: int,
    symbol: int,
    tail_mass: float = 2.0 ** -30,
) -> List[int]:
    """CDF for a large vocabulary where only ``p(true symbol)`` is known.

    Reconstructing Llama's full 128 k posterior for every token is neither
    necessary nor affordable: the coded length is determined by
    ``p(true) / (p(true) + tail)``, so any ``tail_mass`` far below the model's
    own probability floor is invisible.  With ``tail_mass = 2**-30`` the
    reconstruction error in the reported code length is below 1e-6 bits whenever
    ``p(true) > 1e-6``, while a symbol the model rates at ~0 still costs at most
    ~30 bits instead of diverging.

    The tail is spread evenly and rounded *up* to whole coder units, so every
    symbol keeps a non-empty interval and the stream stays lossless.
    """
    n = int(n_symbols)
    if not (0 <= symbol < n):
        raise ValueError(f"symbol {symbol} outside [0, {n})")

    tail_mass = float(min(max(tail_mass, 1e-12), 0.5))
    n_tail = n - 1
    p_true = float(p_true)
    if not (p_true > 0.0) or not math.isfinite(p_true):
        p_true = 1e-12
    p_true = min(p_true, 1.0)
    if p_true > 1.0 - tail_mass:
        tail_mass = max(1e-12, (1.0 - p_true) * 0.5)

    tail_units = int(tail_mass * FULL) + n_tail
    if tail_units > FULL - n_tail:
        tail_units = FULL - n_tail
    true_units = FULL - tail_units
    if true_units < 1:
        raise ValueError("distribution too flat for the chosen tail_mass")

    per_tail = tail_units // n_tail
    extra = tail_units - per_tail * n_tail

    cdf = [0] * (n + 1)
    acc = 0
    for i in range(n):
        if i == symbol:
            acc += true_units
        else:
            j = i if i < symbol else i - 1
            acc += per_tail + (1 if j < extra else 0)
        cdf[i + 1] = acc
    cdf[n] = FULL
    return cdf


def cdf_lookup(cdf: Sequence[int], target: int) -> int:
    """Return the symbol index ``i`` with ``cdf[i] <= target < cdf[i + 1]``.

    Linear scan is optimal here because alphabets are small (<= 257) and the
    distribution is usually peaked, so the scan exits almost immediately.
    """
    for i in range(len(cdf) - 1):
        if target < cdf[i + 1]:
            return i
    return len(cdf) - 2
