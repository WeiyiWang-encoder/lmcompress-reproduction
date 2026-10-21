"""Measure how many pages chapter 2 occupies, then find a size/spacing that fits 7-8.

Content is never touched: the only edits are the document class option, a spacing
command in the preamble, and two temporary labels used purely for measurement.

Usage:
  tune_ch2_pages.py measure            -> report the current chapter-2 page span
  tune_ch2_pages.py sweep              -> try a grid of (class size, stretch) and report
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(HERE, "report")
SRC = os.path.join(REPORT, "CN_translated_en.tex")
TMP = os.path.join(REPORT, "_tune.tex")


def build(cls_size: int, stretch: float, cls: str = "article") -> str:
    t = open(SRC, encoding="utf-8").read()
    # document class: use extarticle for sizes outside 10/11/12
    repl = "\\documentclass[" + str(cls_size) + "pt,a4paper]{" + cls + "}"
    t = re.sub(r"\\documentclass\[[^\]]*\]\{article\}", lambda m: repl, t, count=1)
    t = t.replace(r"\usepackage{microtype}",
                  "\\usepackage{microtype}\n\\usepackage{setspace}\n"
                  f"\\setstretch{{{stretch}}}", 1)
    # measurement labels, harmless in the output
    t = t.replace(r"\section{Literature review}",
                  "\\section{Literature review}\\label{ch:lit-start}", 1)
    t = t.replace(r"\section{Method}", "\\label{ch:lit-end}\\section{Method}", 1)
    t = t.replace(r"\end{document}",
                  "\\typeout{LITSPAN \\pageref{ch:lit-start} \\pageref{ch:lit-end}}\n"
                  "\\typeout{ALLPAGES \\pageref{LastPage}}\n\\end{document}", 1)
    t = t.replace(r"\usepackage[hidelinks]{hyperref}",
                  "\\usepackage[hidelinks]{hyperref}\n\\usepackage{lastpage}", 1)
    open(TMP, "w", encoding="utf-8", newline="\n").write(t)
    return TMP


def compile_once(tex: str) -> tuple[int, int, int]:
    """Return (lit_start_page, lit_end_page, total_pages).

    Page numbers are read from the .aux because \\typeout does not expand \\pageref.
    """
    base = os.path.splitext(tex)[0]
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", os.path.basename(tex)],
                       cwd=REPORT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    aux_path = base + ".aux"
    log_path = base + ".log"
    if not os.path.exists(aux_path):
        return -1, -1, -1
    aux = open(aux_path, encoding="utf-8", errors="replace").read()
    log = open(log_path, encoding="utf-8", errors="replace").read() \
        if os.path.exists(log_path) else ""
    errs = len(re.findall(r"(?m)^!", log))
    if errs:
        return -1, -1, errs

    def page_of(label: str) -> int:
        m = re.search(r"\\newlabel\{" + label + r"\{\{[^}]*\}\{(\d+)\}", aux)
        return int(m.group(1)) if m else -1

    total = -1
    m = re.search(r"lastpage@lastpage\{(\d+)\}", aux)
    if m:
        total = int(m.group(1))
    return page_of("ch:lit-start"), page_of("ch:lit-end"), total


def cleanup() -> None:
    for ext in (".aux", ".log", ".out", ".toc", ".pdf"):
        f = os.path.join(REPORT, "_tune" + ext)
        if os.path.exists(f):
            os.remove(f)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "measure"

    if mode == "measure":
        build(11, 1.0)
        a, b, n = compile_once(TMP)
        print(f"  chapter 2 spans pages {a}-{b}  ->  {b - a + 1} pages")
        print(f"  whole document: {n} pages")
        cleanup()
        return 0

    print(f"  {'cls':>10}{'size':>6}{'stretch':>9}{'ch2 pages':>11}{'total':>7}")
    print("  " + "-" * 44)
    results = []
    for cls_size, stretch, cls in [
        (12, 1.5, "article"), (12, 1.8, "article"),
        (13, 1.5, "extarticle"), (13, 1.9, "extarticle"),
        (14, 1.4, "extarticle"), (14, 1.6, "extarticle"),
        (15, 1.4, "extarticle"),
    ]:
        build(cls_size, stretch, cls)
        a, b, n = compile_once(TMP)
        if a < 0:
            print(f"  {cls:>10}{cls_size:>6}{stretch:>9}   compile failed")
            continue
        pages = b - a + 1
        flag = "  <== target" if 7 <= pages <= 8 else ""
        results.append((pages, n, cls_size, stretch, cls))
        print(f"  {cls:>10}{cls_size:>6}{stretch:>9.1f}{pages:>11}{n:>7}{flag}")

    hit = [r for r in results if 7 <= r[0] <= 8]
    print()
    if hit:
        print("  candidates that put chapter 2 at 7-8 pages:")
        for pages, n, size, st, cls in sorted(hit):
            print(f"    {cls} {size}pt, stretch {st}  -> ch2 {pages} pages, "
                  f"whole document {n} pages")
    else:
        print("  none of the tried settings landed in 7-8 pages")
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
