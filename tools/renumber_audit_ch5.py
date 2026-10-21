"""Repair chapter 5 of the audit: missing chapter heading, mixed levels, bad numbering.

Three defects accumulated from repeated insertions:

  * chapter 5 has **no level-2 heading at all** -- the document jumps from
    `## 4. Measurements` to `## 6. Threats to validity`, with 5.x sections floating
    between them;
  * `## 5.1` is level 2 while every sibling is level 3;
  * the numbers are `5.2, 5.5, 5.4, 5.3, 5.4` -- two 5.4s, and 5.3 after 5.4.

This rewrites the chapter in logical order with a proper heading and consecutive numbers.
"""

from __future__ import annotations

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")

CHAPTER_TITLE = "## 5. Text, and the fine-tuning result"

ORDER = [
    ("5.1 The recipe, recovered", "5.1"),
    ("5.2 The decisive evidence", "5.2"),
    ("5.3 Why the gap is domain match", "5.3"),
    ("5.4 Why the text arms were measured", "5.4"),
    ("5.5 The two \"context lengths\"", "5.5"),
    ("5.4 The experiment that would settle it", "5.6"),
]


def main() -> int:
    text = io.open(AUDIT, encoding="utf-8").read()

    # region spans from the 5.1 heading to the next level-2 heading after it
    start = text.find("## 5.1 ")
    if start < 0:
        start = text.find("### 5.1 ")
    if start < 0:
        print("  !! 5.1 heading not found")
        return 1
    nxt = text.find("\n## ", start + 3)
    if nxt < 0:
        print("  !! could not find the end of chapter 5")
        return 1
    region = text[start:nxt]
    print(f"  chapter region: {len(region)} chars")

    # normalise every 5.x heading to level 3, then split into blocks
    region = re.sub(r"(?m)^#{2,3}\s+5\.\d+\s", "### ", region)
    parts = re.split(r"(?m)^### ", region)
    blocks = ["### " + p.rstrip() + "\n" for p in parts[1:]]
    print(f"  {len(blocks)} subsections found:")
    for b in blocks:
        print("   ", b.splitlines()[0][:70])

    used = [False] * len(blocks)
    ordered = []
    for prefix, newnum in ORDER:
        hit = None
        for i, b in enumerate(blocks):
            if not used[i] and b[4:].startswith(prefix.replace("5.1 ", "")
                                               .replace("5.2 ", "")
                                               .replace("5.3 ", "")
                                               .replace("5.4 ", "")
                                               .replace("5.5 ", "")):
                hit = i
                break
        if hit is None:
            print(f"  !! unmatched: {prefix!r}")
            return 1
        used[hit] = True
        first, rest = blocks[hit].split("\n", 1)
        title = first[4:].strip()
        ordered.append(f"### {newnum} {title}\n{rest}")

    for i, b in enumerate(blocks):
        if not used[i]:
            print(f"  note: keeping unmatched block as-is: {b.splitlines()[0][:60]}")
            ordered.append(b)

    new_region = CHAPTER_TITLE + "\n\n" + "\n".join(ordered)
    out = text[:start] + new_region + text[nxt:]

    out = out.replace(
        "so ten windows at 4096 become three at 8192 with the same token count",
        "so the 15 windows at 4096 tokens become 9 at 8192 with the same token count", 1)

    io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(out)

    print("\n  resulting level-2 headings:")
    for m in re.finditer(r"(?m)^## .*$", out):
        print("   ", m.group(0)[:66])
    print("\n  resulting 5.x headings:")
    for m in re.finditer(r"(?m)^### 5\.\d+ .*$", out):
        print("   ", m.group(0)[:70])
    return 0


if __name__ == "__main__":
    sys.exit(main())
