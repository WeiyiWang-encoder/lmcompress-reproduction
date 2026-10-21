"""Dump the appendix of the review docx, which lists the script paths for GitHub."""

from __future__ import annotations

import re
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

SRC = r"report\capstone\deliverables\LMCompress_中文版审阅稿_实验结果更新版.docx"


def blocks(path: str) -> list[str]:
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    body = xml[xml.find("<w:body>"):]
    out = []
    for b in re.findall(r"<w:tbl>.*?</w:tbl>|<w:p[ >].*?</w:p>", body, re.S):
        if b.startswith("<w:tbl>"):
            for r in re.findall(r"<w:tr[ >].*?</w:tr>", b, re.S):
                cells = []
                for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S):
                    t = "".join(re.sub(r"<[^>]+>", "", m) for m in
                                re.findall(r"<w:t[^>]*>(.*?)</w:t>", c, re.S))
                    cells.append(re.sub(r"\s+", " ", t).strip())
                if any(cells):
                    out.append(" | ".join(cells))
        else:
            t = "".join(re.sub(r"<[^>]+>", "", m) for m in
                        re.findall(r"<w:r[ >].*?</w:r>", b, re.S))
            t = re.sub(r"\s+", " ", t).strip()
            if t:
                out.append(t)
    return out


def main() -> int:
    bs = blocks(SRC)
    print(f"blocks: {len(bs)}")
    # find the appendix
    start = None
    for i, t in enumerate(bs):
        if re.search(r"附录|Appendix", t) and len(t) < 60:
            start = i
            print(f"appendix heading at {i}: {t}")
            break
    if start is None:
        for i, t in enumerate(bs):
            if "附录" in t:
                print(f"first mention at {i}: {t[:120]}")
        return 0
    print("-" * 70)
    for t in bs[start:]:
        print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
