"""Extract the CN docx into a structured text file for translation."""

from __future__ import annotations

import re
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

SRC = r"results\_CN.docx"


def main() -> int:
    z = zipfile.ZipFile(SRC)
    xml = z.read("word/document.xml").decode("utf-8")
    print(f"  tables: {xml.count('<w:tbl>')}")
    print(f"  paragraphs (raw): {xml.count('<w:p ')}")

    # walk body sequentially so table cells keep their order relative to paragraphs
    body = xml[xml.find("<w:body>"):]
    blocks = re.findall(r"<w:tbl>.*?</w:tbl>|<w:p[ >].*?</w:p>", body, re.S)

    out: list[str] = []
    n_tbl = 0
    for b in blocks:
        if b.startswith("<w:tbl>"):
            rows = re.findall(r"<w:tr[ >].*?</w:tr>", b, re.S)
            cells: list[str] = []
            for r in rows:
                cs = re.findall(r"<w:tc>.*?</w:tc>", r, re.S)
                vals = []
                for c in cs:
                    t = "".join(re.sub(r"<[^>]+>", "", m)
                                for m in re.findall(r"<w:t[^>]*>(.*?)</w:t>", c, re.S))
                    vals.append(re.sub(r"\s+", " ", t).strip())
                cells.append(" | ".join(vals))
            if cells:
                n_tbl += 1
                out.append(f"@@TABLE {n_tbl}@@")
                out.extend("  ROW: " + c for c in cells)
                out.append("@@ENDTABLE@@")
        else:
            t = "".join(re.sub(r"<[^>]+>", "", m)
                        for m in re.findall(r"<w:r[ >].*?</w:r>", b, re.S))
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                out.append(t)

    text = "\n".join(out)
    open("results/_CN_extracted.txt", "w", encoding="utf-8").write(text)
    print(f"  blocks written: {len(out)}   chars: {len(text):,}")
    print(f"  tables captured: {n_tbl}")
    print()
    for i, line in enumerate(out[:30]):
        print(f"  {i:>3} | {line[:95]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
