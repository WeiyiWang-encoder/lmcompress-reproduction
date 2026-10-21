"""Disambiguate the provenance of the LoRA adapter across all documents.

The adapter in ``data/model/finetuned_llama/Llama-3-8B_lora/`` was trained by the
**paper's authors** and shipped inside their Code Ocean capsule.  It was NOT
supplied by the user, and the user has not fine-tuned anything.  Phrasings like
"capsule's LoRA" are technically true but read as "the LoRA you gave me", which
inverts the contribution: the whole point is that the *authors'* fine-tuned adapter
is being evaluated, not one of ours.

This script normalises the wording and lets the consistency checker enforce it.
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (path, old, new) -- applied wherever the exact string occurs.
REPLACEMENTS = [
    # --- Chinese report ------------------------------------------------------
    (
        os.path.join(HERE, "docs", "REPORT_CN.md"),
        "**设置**：Llama-3-8B ± capsule 提供的 LoRA；",
        "**设置**：Llama-3-8B ± **论文作者随 capsule 发布的、他们自己微调的 LoRA 适配器**"
        "（不是我们训练的，也不是本次复现的产物）；",
    ),
    (
        os.path.join(HERE, "docs", "REPORT_CN.md"),
        "| LMCompress（+ capsule 提供的 LoRA） |",
        "| LMCompress（+ 作者发布的 LoRA） |",
    ),
    (
        os.path.join(HERE, "docs", "REPORT_CN.md"),
        "| + 发布的 LoRA（step 5120） |",
        "| + 作者发布的 LoRA（step 5120） |",
    ),
    (
        os.path.join(HERE, "docs", "REPORT_CN.md"),
        "| 文本 | + 发布的 LoRA |",
        "| 文本 | + 作者发布的 LoRA |",
    ),
    (
        os.path.join(HERE, "docs", "REPORT_CN.md"),
        "论文正文没有这些，全部来自 capsule 的 `trainer_state.json` / `adapter_config.json`：",
        "论文正文没有这些，全部来自 **作者随 capsule 发布的** `trainer_state.json` / "
        "`adapter_config.json`（即他们自己那次训练的产物）：",
    ),
    # --- English audit -------------------------------------------------------
    (
        os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md"),
        "| C2 | The shipped LoRA adapter makes MeDAL compression **worse**",
        "| C2 | The LoRA adapter **published by the authors with their capsule** "
        "(their own fine-tuning, not ours) makes MeDAL compression **worse**",
    ),
    (
        os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md"),
        "# text: same files, shipped LoRA attached",
        "# text: same files, the AUTHORS' published LoRA attached",
    ),
    (
        os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md"),
        "independent signals, from different artefacts, both say the shipped adapter is",
        "independent signals, from different artefacts, both say the authors' adapter is",
    ),
    (
        os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md"),
        "| `results/text_lora_v2.jsonl` | + shipped LoRA, same 3 files, same code path |",
        "| `results/text_lora_v2.jsonl` | + the authors' published LoRA, same 3 files, same code path |",
    ),
    (
        os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md"),
        "the paper's shipped LoRA adapter makes compression **0.31 pp worse**",
        "the authors' own published LoRA adapter makes compression **0.31 pp worse**",
    ),
    # --- comparison tables ---------------------------------------------------
    (
        os.path.join(HERE, "results", "comparison_tables.md"),
        "| LMCompress (Llama-3-8B + shipped LoRA) |",
        "| LMCompress (Llama-3-8B + the authors' published LoRA) |",
    ),
    (
        os.path.join(HERE, "results", "comparison_tables.md"),
        "| File | raw Llama-3-8B | + shipped LoRA | Δ |",
        "| File | raw Llama-3-8B | + authors' LoRA | Δ |",
    ),
    (
        os.path.join(HERE, "results", "comparison_tables.md"),
        "shipped adapter as **worse than the base model by 0.312 pp**",
        "authors' adapter as **worse than the base model by 0.312 pp**",
    ),
    (
        os.path.join(HERE, "results", "comparison_tables.md"),
        "the shipped adapter makes compression worse by 0.31 pp on every file",
        "the authors' published adapter makes compression worse by 0.31 pp on every file",
    ),
    # --- reproduction notes --------------------------------------------------
    (
        os.path.join(HERE, "docs", "REPRODUCTION_NOTES.md"),
        "(± shipped LoRA), 4-bit NF4",
        "(± the LoRA published by the authors with their capsule), 4-bit NF4",
    ),
    (
        os.path.join(HERE, "docs", "REPRODUCTION_NOTES.md"),
        "| + shipped LoRA (= LMCompress) |",
        "| + authors' published LoRA (= LMCompress) |",
    ),
    # --- review response -----------------------------------------------------
    (
        os.path.join(HERE, "docs", "REVIEW_RESPONSE.md"),
        "the shipped adapter compresses MeDAL *worse*",
        "the authors' published adapter compresses MeDAL *worse*",
    ),
    (
        os.path.join(HERE, "docs", "REVIEW_RESPONSE.md"),
        "* **C2** (the shipped LoRA is worse on MeDAL)",
        "* **C2** (the authors' published LoRA is worse on MeDAL)",
    ),
]


def main() -> int:
    applied = 0
    for path, old, new in REPLACEMENTS:
        text = open(path, encoding="utf-8").read()
        if old not in text:
            print(f"  -- no match in {os.path.basename(path)}: {old[:60]!r}")
            continue
        text = text.replace(old, new)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        applied += 1
        print(f"  ok  {os.path.basename(path)}: {old[:55]!r}")
    print(f"\n{applied} replacement(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
