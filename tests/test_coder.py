"""Coder self-tests.

The most important property of a lossless compressor is that it is *actually*
lossless, and the most important property of a *measurement* harness is that its
numbers are not fabricated.  The uniform-byte test pins both down at once: with
a uniform distribution over 256 symbols the ideal code length must be exactly
8 bits per byte, so any deviation means the harness is lying.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from uic.coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    cdf_lookup,
    probs_to_cdf,
    sparse_true_symbol_cdf,
)
from uic.engine import measure_stream, measure_uniform_bytes, Step
from uic.models import bgpt_prediction_indices


def test_bgpt_rows_skip_each_cross_patch_slot():
    """Reference bGPT removes the 17th row of every patch before flattening."""
    assert bgpt_prediction_indices(0) == []
    assert bgpt_prediction_indices(16) == list(range(16))
    assert bgpt_prediction_indices(32) == list(range(16)) + list(range(17, 33))
    assert bgpt_prediction_indices(8160)[-1] == 8668


def test_probs_to_cdf_is_valid():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 300))
        probs = rng.random(n)
        probs /= probs.sum()
        cdf = probs_to_cdf(probs.tolist())
        assert cdf[0] == 0
        assert cdf[-1] == 1 << 32
        assert len(cdf) == n + 1
        assert all(b > a for a, b in zip(cdf, cdf[1:])), "empty interval!"


def test_probs_to_cdf_survives_zeros():
    """A symbol the model thinks is impossible must still be codable."""
    probs = [0.0] * 256
    probs[7] = 1.0
    cdf = probs_to_cdf(probs)
    assert cdf[8] > cdf[7], "zero-probability symbol got no mass"
    p = [0.0] * 256
    p[200] = 1.0
    cdf2 = probs_to_cdf(p)
    assert cdf2[201] > cdf2[200]


def test_uniform_byte_coder_is_exactly_8_bits_per_byte():
    rng = np.random.default_rng(1)
    for size in (1, 2, 7, 8, 9, 255, 4096):
        data = bytes(rng.integers(0, 256, size=size, dtype=np.uint8).tolist())
        res = measure_uniform_bytes(data, verify_roundtrip=True)
        assert res.coded_symbols == size
        assert abs(res.coded_bits - 8.0 * size) < 1e-6, res.coded_bits
        assert res.extra["roundtrip"] == "ok"


def test_roundtrip_against_adversarial_distributions():
    """Encode/decode with deliberately skewed and near-degenerate pdfs."""
    rng = np.random.default_rng(7)
    for trial in range(20):
        n = int(rng.integers(2, 260))
        steps = []
        for _ in range(64):
            probs = rng.random(n) ** (trial + 1)
            probs /= probs.sum()
            probs = probs.astype(np.float32)
            symbol = int(rng.integers(0, n))
            steps.append(Step(symbol=symbol, p_true=float(probs[symbol]), probs=probs))
        res, _ = measure_stream(steps, verify_roundtrip=True)
        assert res.extra["roundtrip"] == "ok"


def test_roundtrip_with_sparse_true_symbol_cdf():
    """The Llama path: only p(true symbol) is known."""
    rng = np.random.default_rng(11)
    n = 128256
    steps = []
    for _ in range(48):
        symbol = int(rng.integers(0, n))
        p = float(10.0 ** rng.uniform(-9, -0.05))
        steps.append(
            Step(symbol=symbol, p_true=p, cdf=sparse_true_symbol_cdf(p, n, symbol))
        )
    res, _ = measure_stream(steps, verify_roundtrip=True)
    assert res.extra["roundtrip"] == "ok"
    # reconstructed length must track the true -log2 p closely
    expected = sum(-np.log2(s.p_true) for s in steps)
    assert abs(res.coded_bits - expected) < 1e-3 * max(1.0, expected), (
        res.coded_bits,
        expected,
    )


def test_sparse_cdf_is_valid_and_monotone():
    n = 32000
    for symbol in (0, 1, n // 2, n - 2, n - 1):
        cdf = sparse_true_symbol_cdf(0.5, n, symbol)
        assert cdf[0] == 0 and cdf[-1] == 1 << 32
        assert len(cdf) == n + 1
        assert all(b > a for a, b in zip(cdf, cdf[1:]))
        # the true symbol must own almost all of the range
        assert cdf[symbol + 1] - cdf[symbol] > 0.4 * (1 << 32)


def test_peaked_distribution_costs_almost_nothing():
    probs = np.full(256, 1e-12, dtype=np.float32)
    probs[42] = 1.0
    probs /= probs.sum()
    steps = [Step(symbol=42, p_true=float(probs[42]), probs=probs) for _ in range(1000)]
    res, _ = measure_stream(steps, verify_roundtrip=True)
    assert res.coded_bits < 1.0, res.coded_bits
    assert res.ac_bits <= 64, res.ac_bits


def test_cdf_lookup_boundaries():
    cdf = [0, 5, 10, 100]
    assert cdf_lookup(cdf, 0) == 0
    assert cdf_lookup(cdf, 4) == 0
    assert cdf_lookup(cdf, 5) == 1
    assert cdf_lookup(cdf, 9) == 1
    assert cdf_lookup(cdf, 10) == 2
    assert cdf_lookup(cdf, 99) == 2


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:  # noqa: PERF203
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)
