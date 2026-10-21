"""Compare the reference bGPT scorer with the clean-room row selector.

The reference removes slot 16 from every 17-slot patch before flattening.  This
tool calculates its score over every window and compares it with the clean-room
selector on the same logits and targets.
"""

from __future__ import annotations

import os
import sys
import time

REPO = _capsule_root()
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "repro_understanding_is_compression", "src"))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from functools import partial  # noqa: E402

import compress_with_bgpt_or_llama as ref  # noqa: E402

from uic.data import audio_units  # noqa: E402
from uic.models import BGPTBytePredictor, build_bgpt  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8160


def main() -> int:
    unit = audio_units(os.path.join(REPO, "data", "test_audios"), limit=1)[0]
    data = list(unit.payload)
    ext = [ord(c) for c in "wav"]

    # ---- reference path -------------------------------------------------- #
    model = build_bgpt(
        REPO, os.path.join(REPO, "data", "model", "bgpt", "weights-audio.pth"), "cuda:0"
    )
    reference_bits = 0.0
    port_bits = 0.0
    scored = 0
    for number, chunk in enumerate(ref.split(data, N), 1):
        loader = DataLoader(
            [chunk], batch_size=1, shuffle=False,
            collate_fn=partial(ref.byte_collate, appendix={"ext": ext}, device="cuda:0"),
        )
        batch = next(iter(loader))
        try:
            with torch.no_grad():
                logits = model(**batch).logits
        except RuntimeError as exc:
            print(f"[reference] cannot score window {number}: {exc}")
            print("[reference] the final chunk is not patch-aligned; no full-file "
                  "reference result is available from this invocation.")
            break

        ref_logits = logits[:-1, :-1, :].reshape(1, -1, 257)
        ref_ids = batch["patches"][:, 16:-16]
        ref_mask = batch["masks"].repeat_interleave(16, dim=1)[:, 16:-16]
        ref_probs = ref_logits.softmax(dim=-1)
        ref_p = torch.gather(ref_probs, -1, ref_ids.unsqueeze(-1)).squeeze(-1).add_(6e-8)
        reference_bits += float(((-torch.log2(ref_p)) * ref_mask).sum())
        active = int(ref_mask.sum())

        grid = batch["patches"].reshape(1, -1)
        indices = [17 * (k // 16) + (k % 16) for k in range(active)]
        rows = logits.reshape(1, -1, 257).index_select(
            1, torch.tensor(indices, device=logits.device)
        )
        targets = grid[:, 16:16 + active]
        port_p = torch.softmax(rows.float(), dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        port_bits += float((-torch.log2(port_p.clamp_min(1e-38))).sum())
        scored += active
        print(f"window {number:>2}: active={active:>4} reference_bpb="
              f"{float(((-torch.log2(ref_p)) * ref_mask).sum()) / active:.4f} "
              f"port_bpb={float((-torch.log2(port_p.clamp_min(1e-38))).sum()) / active:.4f}")

    print(f"[reference, complete windows only] bytes={scored} bits={reference_bits:.1f} bpb={reference_bits/scored:.4f} "
          f"ratio={8*scored/reference_bits:.4f}")
    print(f"[port, same complete windows]      bytes={scored} bits={port_bits:.1f} bpb={port_bits/scored:.4f} "
          f"ratio={8*scored/port_bits:.4f}")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    raise SystemExit(main())
