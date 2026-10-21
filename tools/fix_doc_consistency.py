"""Apply cross-consistency corrections to the audit and response documents.

Written as a script rather than a series of ad-hoc edits, because three review
rounds have shown that **silent version residue across documents** is this
project's dominant error mode.  The script therefore enforces two invariants:

1. every ``old`` string must occur exactly the expected number of times -- a
   replacement that silently matches nothing is reported as an error, not treated
   as a no-op;
2. each removal and insertion is echoed, so the diff is auditable.

Run with ``--check`` to report without writing.
"""

from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
RESPONSE = os.path.join(HERE, "docs", "REVIEW_RESPONSE.md")

EDITS = [
    # ---- audit §2.4: the protocol must not point at withdrawn experiments ----
    (
        AUDIT,
        "python -m uic.cli image --limit 1 --charge-residual",
        "# image: the C7 evidence is the four probes listed in section 7.2, in this order\n"
        "python tools/test_igpt_injectivity.py\n"
        "python tools/probe_official_image_accounting.py\n"
        "python tools/probe_official_image_path.py\n"
        "python tools/probe_loader_volume.py",
        1,
    ),
    (
        AUDIT,
        "python tools/audit_igpt_image.py",
        "# (tools/audit_igpt_image.py is SUPERSEDED -- whole-image path, NOT C7 evidence)",
        1,
    ),
    # ---- audit: "close reproduction" must match the caution just above it ----
    (
        AUDIT,
        "CLIC2019, so the official convention on this single image lands at 15.51 % — a\nclose reproduction.",
        "CLIC2019, so the official convention on this single image lands at 15.51 % —\n"
        "numerically close to the published value, which validates the accounting\n"
        "convention rather than reproducing the dataset-level result (see the wording\n"
        "caution in §4.4.2).",
        1,
    ),
    # ---- response §1.4: name the stale numerator correctly -------------------
    (
        RESPONSE,
        "> official run's **906 459 bytes**. The corrected figures come from",
        "> then-observed last-printed value 906 459 bytes; the final exhausted-generator\n"
        "> total is **910 788 bytes**. The corrected figures come from",
        1,
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    by_file: dict[str, list] = {}
    for path, old, new, expect in EDITS:
        by_file.setdefault(path, []).append((old, new, expect))

    applied = skipped = 0
    for path, edits in by_file.items():
        text = open(path, encoding="utf-8").read()
        original = text
        name = os.path.basename(path)
        for old, new, expect in edits:
            n = text.count(old)
            if n != expect:
                print(f"  !! SKIP [{name}] found {n}, expected {expect}: {old[:64]!r}")
                skipped += 1
                continue
            text = text.replace(old, new)
            applied += 1
            print(f"  ok  [{name}] replaced {n}x {old[:60]!r}")
        if not args.check and text != original:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print(f"  -> wrote {name}")

    print(f"\n{applied} applied, {skipped} skipped")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
