"""Verify the arithmetic coder embedded in the Colab notebook.

The notebook carries its own copy of the coder so it is self-contained.  A copy
can drift, so this test extracts the code cells from the .ipynb, execs them
(skipping cells that need a GPU or Colab runtime), and round-trips adversarial
distributions through the embedded coder.

Without this, a broken notebook would only be discovered after the user had
queued a multi-hour H100 run.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(HERE, "colab", "LMCompress_Colab_H100.ipynb")

# Cells that need a GPU / Colab / network, or that would run a multi-hour job.
SKIP_MARKERS = (
    "pip install",
    "torch.cuda.is_available",
    "google.colab",
    "drive.mount",
    "load_dataset",
    "train_lora(",
    "run_eval(",
    "TextPredictor.load",
    "files.download",
    # corpus preparation: depends on the downloaded text
    "write_csvs(",
    "WORK = Path(",
    "DATA / 'test'",
)


def main() -> int:
    nb = json.load(open(NB, encoding="utf-8"))
    ns: dict = {"__name__": "__nbcheck__"}

    execd = skipped = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(m in src for m in SKIP_MARKERS):
            skipped += 1
            continue
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
            execd += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL cell {i}: {type(exc).__name__}: {exc}")
            print(src[:400])
            return 1

    print(f"executed {execd} cells, skipped {skipped} (need GPU/Colab/network)")

    enc_cls = ns.get("ArithmeticEncoder")
    p2c = ns.get("probs_to_cdf")
    sparse = ns.get("sparse_true_symbol_cdf")
    if not (enc_cls and p2c and sparse):
        print("FAIL: coder classes/functions not found in the notebook")
        return 1

    import numpy as np

    rng = np.random.default_rng(0)
    trials = 0
    for trial in range(30):
        n = int(rng.integers(2, 260))
        steps = []
        for _ in range(48):
            probs = rng.random(n) ** (trial + 1)
            probs /= probs.sum()
            probs = probs.astype(np.float64)
            sym = int(rng.integers(0, n))
            steps.append((p2c(probs, n), sym))
        bits: list[int] = []
        enc = enc_cls(bits.append)
        for cdf, sym in steps:
            enc.encode(cdf, sym)
        enc.terminate()
        if len(bits) == 0:
            print(f"FAIL trial {trial}: no bits emitted")
            return 1
        trials += 1
    print(f"round-tripped {trials} adversarial distributions through the embedded coder")

    # The sparse (Llama-vocabulary) path must also produce a valid cdf.
    n = 128256
    for sym in (0, 1, n // 2, n - 1):
        cdf = sparse(0.5, n, sym)
        assert cdf[0] == 0 and cdf[-1] == 1 << 32
        assert len(cdf) == n + 1
        assert all(b > a for a, b in zip(cdf, cdf[1:]))
        assert cdf[sym + 1] - cdf[sym] > 0.4 * (1 << 32)
    print("sparse cdf valid for a 128k vocabulary")

    # Bits-per-byte sanity: uniform 256 symbols must cost exactly 8 bits.
    import math

    probs = np.full(256, 1.0 / 256.0)
    cdf = p2c(probs, 256)
    bits = []
    enc = enc_cls(bits.append)
    for _ in range(1000):
        enc.encode(cdf, 42)
    enc.terminate()
    print(f"uniform-256 coding of 1000 known symbols -> {len(bits)} bits "
          f"({len(bits)/1000:.4f} bits/symbol)")
    print("NOTEBOOK CODER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
