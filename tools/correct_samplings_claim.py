"""Correct the false "missing samplings dependency" claim.

WHAT WE GOT WRONG
-----------------
We asserted that ``data/model/bgpt/utils.py:9`` imports a ``samplings`` module the capsule
does not ship, so the reference code "fails at import time".  A reviewer agreed.

That is **false**.  ``samplings`` is a real PyPI package (by Sander Wood) and it is
**already installed** in the capsule's virtualenv:

    .venv/Lib/site-packages/samplings/__init__.py   (21 316 bytes)

It is correctly *not* vendored inside the capsule source — it is a dependency, which is
exactly what a virtualenv is for.  Our search walked the capsule directory while
excluding ``.venv``, found nothing, and we drew the wrong conclusion from that.

The tell we ignored: importing the reference module from the **unmodified** capsule
succeeds.  We had run that check and got a clean exit, then explained it away instead of
following it.  Two independent confirmations are recorded below.

The practical consequence is large: the reference script runs, unmodified, on the
original capsule.  Its audio result is reproducible by anyone:

    python compress_with_bgpt_or_llama.py \\
      --data_path data/test_audios --data_type wav \\
      --model_path data/model/bgpt/weights-audio.pth \\
      --context_window 8160 --batch_size 1

    Theoretical compression ratio: 5.016113
    Total size: 82264
    compressed size: 16399.949935913086
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    print("=== Confirmation 1: where does `samplings` actually come from? ===")
    spec = importlib.util.find_spec("samplings")
    print(f"  find_spec('samplings') -> {spec.origin if spec else None}")
    print(f"  is it inside the capsule source? "
          f"{bool(spec and 'site-packages' not in spec.origin)}")
    print(f"  is it an installed package?      "
          f"{bool(spec and 'site-packages' in spec.origin)}")

    print("\n=== Confirmation 2: does the UNMODIFIED capsule import? ===")
    py = os.path.join(REPO, ".venv", "Scripts", "python.exe")
    # run from a clean cwd so nothing of ours shadows anything
    proc = subprocess.run(
        [py, "-c", "import compress_with_bgpt_or_llama as C; print('IMPORT OK')"],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    out = (proc.stdout + proc.stderr).strip().splitlines()
    print(f"  exit code: {proc.returncode}")
    for line in out[-4:]:
        print(f"  {line}")

    ok = bool(spec and "site-packages" in spec.origin) and proc.returncode == 0
    print(f"\n  => the 'missing dependency' claim is FALSE: {ok}")

    print("\n=== What this invalidates ===")
    print("  * docs/AUDIT_FOR_REVIEW.md  : any statement that samplings blocks the run")
    print("  * docs/REVIEW_GUIDE_FOR_AGENT.md : Claim about missing dependency")
    print("  * replacement_modules/samplings.py : unnecessary; the real package exists")
    print("  * tools/verify_reference_blockers.py : Claim A is wrong as written")
    print()
    print("  What SURVIVES: the tail-block arithmetic is still correct (620-byte tail")
    print("  pads to 652, not a multiple of 16). But since the reference run completes")
    print("  and reports Total size 82264, the predicted crash does not happen either.")
    print("  Both blockers are therefore withdrawn; the reference simply works.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
