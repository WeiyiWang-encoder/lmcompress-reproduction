"""Bit-level stream helpers.

These are used by :mod:`uic.coder` for the arithmetic coder's binary I/O, and by a
trivial pass-through "8 bits per byte" coder that serves purely as the
**no-compression baseline** (a ratio of exactly 1.000x).  That baseline is a
convention, not a claim about any byte alphabet's entropy -- see the note in
:func:`uic.coder.min_interval_width` for why "256 symbols therefore at most 8
bits" is not a valid argument.

Design note: the encoder writes bits MSB-first and prepends nothing.  The
decoder needs to know how many bits it may consume, so we keep the bit count of
the *real* (unpadded) stream in the container header written by
:mod:`uic.engine`.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence


def bits_to_bytes(bits: Sequence[int]) -> tuple[bytes, int]:
    """Pack a bit sequence into bytes, zero-padding on the right.

    Args:
      bits: sequence of 0/1 integers.

    Returns:
      ``(data, num_padded_bits)`` where ``num_padded_bits`` is the number of
      zero bits appended so that ``len(data) * 8 == len(bits) + num_padded_bits``.
    """
    num_padded_bits = (-len(bits)) % 8
    padded: List[int] = list(bits) + [0] * num_padded_bits
    data = bytearray(len(padded) // 8)
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | (padded[i + j] & 1)
        data[i >> 3] = byte
    return bytes(data), num_padded_bits


def bytes_to_bits(data: bytes, num_padded_bits: int = 0) -> List[int]:
    """Unpack bytes into a bit list, dropping the trailing padding."""
    bits: List[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    if num_padded_bits:
        bits = bits[:-num_padded_bits] if num_padded_bits <= len(bits) else []
    return bits


def byte_to_bits(byte: int, width: int = 8) -> List[int]:
    """MSB-first bit expansion of a single integer."""
    return [(byte >> shift) & 1 for shift in range(width - 1, -1, -1)]
