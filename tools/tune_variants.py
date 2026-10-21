"""Generate and read back page-count variants for the CN translation.

Deliberately avoids subprocess: under the current sandbox the PDF engine cannot be
launched with piped stdio, so compilation is driven from PowerShell and this script only
writes the .tex variants and reads the resulting .aux files.

Usage:
  tune_variants.py make      -> write report/_v0.tex .. _v6.tex
  tune_variants.py read      -> print chapter-2 page span and total for each
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
REPORT = os.path.join(HERE, "report")
SRC = os.path.join(REPORT, "CN_translated_en.tex")

VARIANTS = [
    ("v0", 11, 1.0, "article"),      # current baseline
    ("v1", 12, 1.5, "article"),
    ("v2", 12, 1.8, "article"),
    ("v3", 13, 1.5, "extarticle"),
    ("v4", 13, 1.9, "extarticle"),
    ("v5", 14, 1.4, "extarticle"),
    ("v6", 14, 1.6, "extarticle"),
    ("v7", 15, 1.4, "extarticle"),
]


def make() -> int:
    src = open(SRC, encoding="utf-8").read()
    for name, size, stretch, cls in VARIANTS:
        t = re.sub(r"\\documentclass\[[^\]]*\]\{article\}",
                   lambda m, s=size, c=cls: "\\documentclass[" + str(s)
                   + "pt,a4paper]{" + c + "}", src, count=1)
        t = t.replace("\\usepackage{microtype}",
                      "\\usepackage{microtype}\n\\usepackage{setspace}\n"
                      "\\setstretch{" + str(stretch) + "}", 1)
        t = t.replace("\\section{Literature review}",
                      "\\section{Literature review}\\label{ch:lit-start}", 1)
        t = t.replace("\\section{Method}", "\\label{ch:lit-end}\\section{Method}", 1)
        t = t.replace("\\usepackage[hidelinks]{hyperref}",
                      "\\usepackage[hidelinks]{hyperref}\n\\usepackage{lastpage}", 1)
        out = os.path.join(REPORT, "_" + name + ".tex")
        open(out, "w", encoding="utf-8", newline="\n").write(t)
        print(f"  wrote _{name}.tex  ({cls} {size}pt, stretch {stretch})")
    return 0


def read() -> int:
    print(f"  {'var':>4}{'cls':>12}{'size':>6}{'stretch':>9}{'ch2 start':>11}"
          f"{'ch2 end':>9}{'ch2 pages':>11}{'total':>7}")
    print("  " + "-" * 70)
    for name, size, stretch, cls in VARIANTS:
        aux_p = os.path.join(REPORT, "_" + name + ".aux")
        if not os.path.exists(aux_p):
            print(f"  {name:>4}  (no aux - compile failed)")
            continue
        aux = open(aux_p, encoding="utf-8", errors="replace").read()

        def pg(label: str) -> int:
            m = re.search(r"\\newlabel\{" + label + r"\{\{[^}]*\}\{(\d+)\}", aux)
            return int(m.group(1)) if m else -1

        a, b = pg("ch:lit-start"), pg("ch:lit-end")
        m = re.search(r"lastpage@lastpage\{(\d+)\}", aux)
        total = int(m.group(1)) if m else -1
        pages = b - a + 1 if a > 0 and b > 0 else -1
        flag = "   <== 7-8 pages" if 7 <= pages <= 8 else ""
        print(f"  {name:>4}{cls:>12}{size:>6}{stretch:>9.1f}{a:>11}{b:>9}"
              f"{pages:>11}{total:>7}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(make() if (len(sys.argv) > 1 and sys.argv[1] == "make") else read())
