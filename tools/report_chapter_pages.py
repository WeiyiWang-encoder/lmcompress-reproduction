"""Report the page span of each chapter in the compiled translation.

Heading detection must be line-exact: a bare substring search matches ordinary words in
the body text ("method", "results"), which is why the first attempt mislocated the
chapters.
"""

import re
import sys

import pypdf

PDF = "report/CN_translated_en.pdf"

# (display name, regex that must match a whole extracted line)
HEADS = [
    ("Abstract", r"abstract"),
    ("1 Introduction", r"1\s+introduction"),
    ("2 Literature review", r"2\s+literature\s+review"),
    ("3 Method", r"3\s+method"),
    ("4 Results and analysis", r"4\s+results\s+and\s+analysis"),
    ("Appendix A", r"a\s+translator'?s\s+notes.*"),
]


def main() -> int:
    r = pypdf.PdfReader(PDF)
    n = len(r.pages)
    lines_by_page = []
    for p in r.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        lines_by_page.append([re.sub(r"\s+", " ", l).strip().lower()
                              for l in t.splitlines()])

    print(f"  {PDF}: {n} pages")
    print(f"  {'chapter':<26}{'starts':>8}{'pages':>8}")
    print("  " + "-" * 42)

    found = []
    for name, pat in HEADS:
        hit = None
        for i, lines in enumerate(lines_by_page):
            if any(re.fullmatch(pat, l) for l in lines):
                hit = i + 1
                break
        if hit is None:
            print(f"  {name:<26}{'not found':>16}")
            continue
        found.append((name, hit))

    for i, (name, start) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else n + 1
        print(f"  {name:<26}{start:>8}{end - start:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
