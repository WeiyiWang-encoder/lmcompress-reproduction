"""Add the adapter-provenance note to the English audit, next to the text results."""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")

ANCHOR = (
    "Identical code path for both rows; the only difference is whether the LoRA adapter\n"
    "is attached. Code length is `\u03a3 -log\u2082 q(true token)`."
)

NOTE = """> **Adapter provenance — do not misread this section.** The two arms compared here
> are (a) raw Llama-3-8B and (b) **the LoRA adapter that the paper's authors
> published inside their Code Ocean capsule**
> (`data/model/finetuned_llama/Llama-3-8B_lora/`). It is *their* fine-tuning, not
> ours. Evidence: its `adapter_config.json` records `base_model_name_or_path` as a
> path on their machine (a home directory of the form `/home/<user>/...`), and the
> same directory still carries their DeepSpeed 8-way pipeline optimiser shards
> (`bf16_zero_pp_rank_0..7_*_optim_states.pt`). This section therefore answers
> *"is the authors' published adapter better than the base model?"* — and **not**
> *"how good is our own fine-tuning?"*, which is the job of §5.4 and Colab
> Parts B-C. No adapter was trained during this reproduction.

Identical code path for both rows; the only difference is whether the LoRA adapter
is attached. Code length is `\u03a3 -log\u2082 q(true token)`."""


def main() -> int:
    text = open(AUDIT, encoding="utf-8").read()
    if ANCHOR not in text:
        print("!! anchor not found; nothing changed")
        return 1
    if "Adapter provenance" in text:
        print("-- note already present")
        return 0
    text = text.replace(ANCHOR, NOTE, 1)
    with open(AUDIT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("ok  provenance note added to AUDIT_FOR_REVIEW.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
