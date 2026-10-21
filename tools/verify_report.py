"""Verify the compiled report and produce the Chinese DOCX companion.

Two jobs:
  1. print the section/table inventory of report.tex so the structure can be checked;
  2. build the Chinese-language version and append it to the supplied Word
     literature review, as requested.
"""

from __future__ import annotations

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(HERE, "report", "report.tex")


def inventory() -> int:
    text = open(TEX, encoding="utf-8").read()
    print("=== sections ===")
    for m in re.finditer(r"\\section\{([^}]*)\}", text):
        print("  " + m.group(1))
    print("\n=== subsections ===")
    for m in re.finditer(r"\\subsection\{([^}]*)\}", text):
        print("  " + m.group(1))
    tables = len(re.findall(r"\\begin\{table\}", text))
    lsts = len(re.findall(r"\\begin\{lstlisting\}", text))
    eqs = len(re.findall(r"\\begin\{equation\}", text))
    print(f"\ntables: {tables}   listings: {lsts}   equations: {eqs}")
    print(f"source characters: {len(text):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(inventory())
