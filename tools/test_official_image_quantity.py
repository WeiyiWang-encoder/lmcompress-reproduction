"""What quantity does the official image metric actually measure?

Established from reading ``language_model.compress``:

  * ``raw_probs`` is ``(1024, 512)``: the softmax of the model's logits, one row per
    position, where row ``t`` is produced **after** consuming tokens ``0..t``.
  * the encoding loop is ``for pdf, symbol in zip(raw_probs, sequence_array)``, i.e.
    row ``t`` codes token ``t``.  So token 0 is coded with row 0 -- the model's
    *prior* over the first token -- and every later token is coded with a
    distribution computed from itself and its predecessors.  That is not a
    next-token predictive distribution for token ``t``.
  * the function returns ``compressed_bytes``, i.e. **the arithmetic coder's real
    emitted output**, not a theoretical ``-log2 q`` length.

Our port measures ``sum(-log2 q_true)``, a theoretical code length for genuine
next-token prediction.  Those are different quantities, and the 1.54x gap between
our image figure and the official one is consistent with that.

This script measures the official quantity directly: build one chunk, run the
official probability path, drive the official arithmetic coder, and compare the
emitted size against both the official whole-image rate and our own figure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
sys.path.insert(0, REPO)

MODEL_DIR = os.path.join(REPO, "data", "model", "imagegpt-small")
IMG_DIR = os.path.join(REPO, "data", "test_images")
PATCH = 32
OFFICIAL_BITS_PER_SAMPLE = 1.241153


def main() -> int:
    from PIL import Image

    from uic.data import _igpt_processor

    path = os.path.join(IMG_DIR, sorted(os.listdir(IMG_DIR))[0])
    rgb = np.asarray(Image.open(path).convert("RGB"))
    n = PATCH * PATCH
    flat = rgb[:, :, 0].reshape(-1)[:n]
    block = flat.reshape(PATCH, PATCH)
    triple = np.repeat(block[:, :, None], 3, axis=2)
    ids = np.asarray(_igpt_processor(MODEL_DIR)(triple, return_tensors="np")
                     ["input_ids"]).reshape(-1)
    print(f"chunk: {n} samples, {ids.size} tokens, "
          f"{len(np.unique(ids))} distinct")

    # ---- the official probability path ------------------------------------ #
    import language_model as lm
    params, img_proc, tok = lm._retrieve_model_params("igpt", MODEL_DIR)
    predict_fn = lm._retrieve_predict_fn(params, img_proc, tok, is_lvm=False)
    probs = np.asarray(predict_fn(np.asarray(ids)))
    probs = probs.reshape(-1, probs.shape[-1])
    print(f"official raw_probs shape = {probs.shape}")

    # row t codes token t, per the official zip()
    p_official = probs[np.arange(ids.size), ids]
    bits_official_loop = float(-np.log2(np.clip(p_official, 1e-300, None)).sum())
    print(f"\n=== scoring row t against token t (the official zip) ===")
    print(f"  bits    = {bits_official_loop:.2f}")
    print(f"  bpb     = {bits_official_loop/ids.size:.4f}")
    print(f"  official whole-image bpb = {OFFICIAL_BITS_PER_SAMPLE:.6f}")
    print(f"  ratio   = {bits_official_loop/ids.size/OFFICIAL_BITS_PER_SAMPLE:.4f}")

    # for contrast: proper causal scoring (row t-1 for token t)
    p_shift = probs[np.arange(ids.size - 1), ids[1:]]
    bits_shift = float(-np.log2(np.clip(p_shift, 1e-300, None)).sum())
    print(f"\n=== proper causal scoring (row t-1 for token t, plus row 0 for token 0) ===")
    bits_causal = float(-np.log2(np.clip(probs[0, ids[0]], 1e-300, None))) + bits_shift
    print(f"  bits    = {bits_causal:.2f}   bpb = {bits_causal/ids.size:.4f}")

    # ---- drive the official arithmetic coder ------------------------------ #
    print("\n=== official arithmetic coder on this chunk ===")
    try:
        import constants
        import utils.arithmetic_coder as ac

        out = []
        enc = ac.Encoder(base=constants.ARITHMETIC_CODER_BASE,
                         precision=constants.ARITHMETIC_CODER_PRECISION,
                         output_fn=out.append)
        for pdf, symbol in zip(probs, ids):
            enc.encode(lm.normalize_pdf_for_arithmetic_coding(pdf), int(symbol))
        enc.terminate()
        bits_str = "".join(map(str, out))
        from utils.ac_utils import bits_to_bytes
        payload, padded = bits_to_bytes(bits_str)
        print(f"  emitted bytes = {len(payload)}")
        print(f"  emitted bits  = {8*len(payload)}")
        print(f"  bits/sample   = {8*len(payload)/ids.size:.4f}")
        print(f"  official whole-image bits/sample = {OFFICIAL_BITS_PER_SAMPLE:.6f}")
        print(f"  ratio         = "
              f"{8*len(payload)/ids.size/OFFICIAL_BITS_PER_SAMPLE:.4f}")
    except Exception as exc:
        print(f"  could not drive the official coder: {type(exc).__name__}: {exc}")

    print("\n=== our port, for comparison ===")
    from uic.engine import measure_stream
    from uic.models import ImageGPTPredictor
    pred = ImageGPTPredictor.load(MODEL_DIR, device="cuda:0")
    try:
        res, _ = measure_stream(pred.iter_steps([int(x) for x in ids]))
    finally:
        pred.release()
    print(f"  coded bits = {res.coded_bits:.2f}  n={res.coded_symbols}"
          f"  bpb={res.coded_bits/res.coded_symbols:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
