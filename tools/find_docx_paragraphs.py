"""Locate the paragraphs that still need a red note, and show their numbers."""

from __future__ import annotations

import os
import re
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


def main() -> int:
    path = sys.argv[1]
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    paras = re.split(r"(?<=</w:p>)", xml)
    texts = []
    for p in paras:
        t = "".join(re.sub(r"<[^>]+>", "", m)
                    for m in re.findall(r"<w:r[ >].*?</w:r>", p, re.S))
        if t.strip():
            texts.append(t)

    nums = re.compile(r"\d{1,3}\.\d{1,4}")

    for key in ("6.53", "0.13"):
        print(f"\n########## paragraphs containing {key} ##########")
        for i, t in enumerate(texts):
            if re.search(rf"(?<![\d.]){re.escape(key)}(?![\d])", t):
                print(f"[{i}] {t[:400]}")
                print(f"     numbers here: {nums.findall(t)}")

    print("\n########## paragraphs mentioning 视频 with numbers ##########")
    for i, t in enumerate(texts):
        if "视频" in t and nums.search(t):
            print(f"[{i}] {t[:360]}")
            print(f"     numbers: {nums.findall(t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
