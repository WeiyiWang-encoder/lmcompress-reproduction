"""Compare planned vs actual scored-token counts per Llama window."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

REPO = _capsule_root()
MODEL = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B")
LORA = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora")


def main() -> int:
    from uic.models import LlamaTokenPredictor
    from uic.runner import plan_windows

    p = LlamaTokenPredictor.load(
        MODEL, lora_path=LORA, device="cuda:0", load_in_4bit=True, context_window=4096
    )

    n_items = 19338
    payload = list(range(1000, 1000 + n_items))
    p.set_document(payload)
    windows = plan_windows(n_items, p.window_capacity, p.name, has_bos=bool(p.bos_id))
    print(f"{len(windows)} windows for {n_items} tokens")
    total = 0
    for i, w in enumerate(windows):
        got = sum(1 for _ in p.iter_steps(payload[w.start : w.stop], prior=w.prior))
        covered = w.stop - w.start
        print(
            f"  window {i}: start={w.start:>6} stop={w.stop:>6} covered={covered:>5} "
            f"prior={w.prior:>6} yielded={got:>5}"
        )
        total += got
    print(f"total yielded = {total} (payload {n_items})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
