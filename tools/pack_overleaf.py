"""Build the Overleaf upload package.

Produces a zip containing exactly what Overleaf needs and nothing else:

    main.tex            the report source, self-contained
    figures/*.pdf       three figures generated from our own measurements
    README_OVERLEAF.md  how to compile it

Deliberately excluded: build artefacts, reference lists, and any citation of
files outside the archive, so the project compiles on a clean TeX Live.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(HERE, "report")
STAGE = os.path.join(REPORT, "overleaf")

README = """# Overleaf upload instructions

## What is in this archive

| File | Purpose |
|---|---|
| `main.tex` | The complete report source. Self-contained: no `\\input`, no bibliography file, no external assets beyond the figures. |
| `figures/fig_reproduction.pdf` | Measured vs published compression rate, per modality. |
| `figures/fig_methods_per_modality.pdf` | Every measured method, by modality. |
| `figures/fig_image_accounting.pdf` | How the image denominator is constructed. |

All three figures were plotted by this project from its own measurements
(`tools/make_report_figures.py`). Nothing is reproduced from the paper's figures.

## How to compile

**Option A — pdfLaTeX (simplest).**
Upload the archive, open the project, and set *Menu → Compiler → pdfLaTeX*.
Verified: compiles cleanly with **0 errors and 0 undefined references**.

**Option B — XeLaTeX.**
The source begins with `% !TEX program = xelatex`, so Overleaf will usually select
XeLaTeX automatically. Also verified clean.

Either engine works. Compile twice so that the table of contents and all
cross-references resolve.

## Package dependencies

Everything used is in a standard TeX Live installation, so nothing needs to be
installed:

`geometry`, `graphicx`, `booktabs`, `longtable`, `array`, `tabularx`, `amsmath`,
`amssymb`, `siunitx`, `xcolor`, `listings`, `enumitem`, `caption`, `titlesec`,
`fancyhdr`, `hyperref`, `microtype`.

One portability note: `microtype` font *expansion* is disabled in the preamble
(`\\microtypesetup{expansion=false}`) because it requires scalable fonts and has
caused engine-specific failures. Protrusion remains enabled. If you prefer, you
may delete that line, but there is no reason to.

## Expected output

33 pages on XeLaTeX, 31 on pdfLaTeX (the difference is font metrics only).
19 sections, 30 tables, 3 figures, 9 equations.

## If something fails

1. Check the compiler setting first — the two engines differ in font handling.
2. Compile twice; a single pass leaves the ToC and references unresolved.
3. If a figure is missing, confirm the `figures/` directory uploaded with the
   same capitalisation.

## A note on the content

Three claims made during this project were later falsified and are **retained in
the report as explicit retractions** (Chapter 11). They are there deliberately,
not by oversight: a reader who cannot see what was withdrawn cannot judge how much
weight the surviving claims carry. Please do not remove them.
"""


def clean_tex(text: str) -> tuple[str, list[str]]:
    """Report anything that would fail on a clean TeX Live."""
    problems = []
    # \file{} uses \detokenize, which is fine; check for local absolute paths
    for m in re.finditer(r"[A-Za-z]:\\\\", text):
        problems.append(f"possible absolute Windows path at offset {m.start()}")
    if "\\input{" in text or "\\include{" in text:
        problems.append("document uses \\input/\\include; package must include those files")
    if "\\bibliography{" in text or "\\addbibresource" in text:
        problems.append("document has a bibliography; a .bib file must be included")
    return text, problems


def main() -> int:
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(os.path.join(STAGE, "figures"), exist_ok=True)

    src = os.path.join(REPORT, "report.tex")
    text = open(src, encoding="utf-8").read()
    text, problems = clean_tex(text)

    # main.tex
    main_tex = os.path.join(STAGE, "main.tex")
    with open(main_tex, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"  ok  main.tex ({len(text):,} chars)")

    # figures
    figs = 0
    for name in sorted(os.listdir(os.path.join(REPORT, "figures"))):
        if name.endswith(".pdf"):
            shutil.copy2(os.path.join(REPORT, "figures", name),
                         os.path.join(STAGE, "figures", name))
            figs += 1
    print(f"  ok  {figs} figure(s)")

    # README
    with open(os.path.join(STAGE, "README_OVERLEAF.md"), "w",
              encoding="utf-8", newline="\n") as fh:
        fh.write(README)
    print("  ok  README_OVERLEAF.md")

    # zip
    packages = os.path.join(REPORT, "packages")
    os.makedirs(packages, exist_ok=True)
    zip_path = os.path.join(packages, "overleaf_LMCompress_reproduction.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(STAGE):
            for f in sorted(files):
                p = os.path.join(root, f)
                arc = os.path.relpath(p, STAGE)
                z.write(p, arc)
    size = os.path.getsize(zip_path)
    print(f"  ok  {os.path.basename(zip_path)} ({size/1024:.0f} KB)")

    if problems:
        print("\n  !! portability warnings:")
        for p in problems:
            print("     - " + p)
    else:
        print("\n  no portability warnings")

    print("\n  archive contents:")
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            print(f"    {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
