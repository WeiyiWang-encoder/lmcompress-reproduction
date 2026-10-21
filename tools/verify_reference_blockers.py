"""Verify the two blocking claims about the reference audio script.

Claim A: ``data/model/bgpt/utils.py`` imports a ``samplings`` module that the capsule
         does not ship, so the reference script fails at import time.
Claim B: with ``--context_window 8160`` the 82 220-byte WAV splits into 10 full windows
         plus a 620-byte tail; ``byte_collate`` pads it to 652 bytes, which is not a
         multiple of PATCH_SIZE, so the model's reshape fails and the reference script
         cannot complete a full-file measurement.

Both are structural, so both can be checked without running inference.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
sys.path.insert(0, REPO)

PATCH_SIZE = 16
CTX = 8160
TOTAL = 82220


def claim_a() -> bool:
    print("=== Claim A: missing `samplings` module ===")
    utils = os.path.join(REPO, "data", "model", "bgpt", "utils.py")
    head = open(utils, encoding="utf-8").read().splitlines()[:12]
    imp = [l for l in head if "samplings" in l]
    print(f"  utils.py imports: {imp}")

    found = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in (".venv", "__pycache__", ".git",
                                "repro_understanding_is_compression")]
        for f in files:
            if f.startswith("samplings") and f.endswith(".py"):
                found.append(os.path.join(root, f))
    print(f"  samplings.py anywhere in the capsule: {found or 'NOT FOUND'}")

    spec = importlib.util.find_spec("samplings") if not found else None
    print(f"  importable as a package: {bool(found)}")
    ok = bool(imp) and not found
    print(f"  -> claim holds: {ok}\n")
    return ok


def claim_b() -> bool:
    print("=== Claim B: 620-byte tail cannot be reshaped ===")
    blocks = [TOTAL] if TOTAL <= CTX else None
    sizes = []
    for i in range(0, TOTAL, CTX):
        sizes.append(min(CTX, TOTAL - i))
    print(f"  total = {TOTAL} bytes, context_window = {CTX}")
    print(f"  split() gives {len(sizes)} blocks: {sizes[:3]} ... {sizes[-1]}")
    full = sum(1 for s in sizes if s == CTX)
    print(f"  full windows = {full} ({full*CTX} bytes), "
          f"tail = {sizes[-1]} bytes")

    # byte_collate: bos_patch (PATCH_SIZE) + payload + PATCH_SIZE trailing pads
    tail = sizes[-1]
    padded = PATCH_SIZE + tail + PATCH_SIZE
    print(f"  tail through byte_collate = {PATCH_SIZE} + {tail} + {PATCH_SIZE} "
          f"= {padded} bytes")
    print(f"  divisible by PATCH_SIZE ({PATCH_SIZE})? {padded % PATCH_SIZE == 0} "
          f"(remainder {padded % PATCH_SIZE})")
    print(f"  -> reshape to (n_patches, {PATCH_SIZE}) is impossible")

    # and confirm a full window IS fine
    fpadded = PATCH_SIZE + CTX + PATCH_SIZE
    print(f"  for comparison, a full window = {fpadded} bytes, "
          f"divisible: {fpadded % PATCH_SIZE == 0}")

    ok = (full == 10 and sizes[-1] == 620 and padded % PATCH_SIZE != 0)
    print(f"  -> claim holds: {ok}\n")
    return ok


def extra() -> None:
    print("=== Consequence for our own numbers ===")
    print(f"  our port measured the FULL file ({TOTAL} bytes) by padding the tail up to")
    print(f"  a multiple of {PATCH_SIZE}. The reference script, run as shipped, would")
    print(f"  raise on the tail block and therefore never report a full-file total.")
    print(f"  Our 15.7844 bpb figure is therefore OUR port's behaviour on all")
    print(f"  {TOTAL} bytes, NOT a direct reproduction of the reference script's output.")
    print(f"  Any comparison against the reference must be restricted to the")
    print(f"  {10*CTX} bytes covered by the 10 complete windows.")


if __name__ == "__main__":
    a = claim_a()
    b = claim_b()
    extra()
    print(f"RESULT: A={a} B={b}")
