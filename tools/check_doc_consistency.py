"""Cross-document consistency checker.

Three review rounds established that this project's dominant error mode is not a
reasoning mistake but **version residue**: a number superseded in one document while
another document — or another section of the same document — still asserts the old
value.  Prose review cannot reliably catch that, so this script does.

It encodes the canonical facts once, then checks every document for

* **forbidden** values (superseded numbers), reported unless the surrounding line is
  an explicit retraction ("withdrawn", "superseded", "勘误", "rejected", ...);
* **required** values that each document must contain, so a fact cannot silently
  vanish from a document that depends on it.

Exit code is non-zero if anything is flagged, so it can gate a commit.
"""

from __future__ import annotations

import os
import re
import sys

# Windows consoles default to a legacy code page; force UTF-8 so the report can
# contain the same characters as the documents it audits.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = {
    "audit": os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md"),
    "response": os.path.join(HERE, "docs", "REVIEW_RESPONSE.md"),
    "report_cn": os.path.join(HERE, "docs", "REPORT_CN.md"),
    "notes": os.path.join(HERE, "docs", "REPRODUCTION_NOTES.md"),
    "tables": os.path.join(HERE, "results", "comparison_tables.md"),
}

# A hit is exempt when a retraction marker appears in its immediate context.  Two
# scopes are checked, because these documents use both shapes:
#   * a short window (a multi-line blockquote note), and
#   * the enclosing section, whose heading may be
#     "#### 4.4.3 Withdrawn: ..." or "## 2. ... WITHDRAWN ...".
# Deliberately tight: over-broad words would suppress real residue.
RETRACTION_WORDS = (
    "withdrawn", "superseded", "rejected", "no longer",
    "earlier revision", "earlier version", "earlier versions",
    "previous value", "previous values", "post-review revision",
    "pre-fix", "pre-alignment", "previously", "formerly", "stale",
    "retained", "review history", "not evidence", "incorrectly",
    "buggy", "corrected", "before the fix", "void", "作废",
    "was wrong", "mistaken", "first revision", "the mistake",
    "勘误", "撤回", "已撤回", "早期版本", "旧路径", "修复前",
)

# Markers that flag a whole SECTION as historical.
SECTION_MARKERS = ("withdrawn", "superseded", "review history", "勘误", "撤回", "旧路径")

# A markdown table row is exempt when its header marks the column as superseded.
COLUMN_MARKERS = ("buggy", "pre-fix", "superseded", "withdrawn", "old")

CONTEXT_LINES = 4
SECTION_LOOKBACK_LINES = 60

# ---------------------------------------------------------------- forbidden --
# Superseded values.  If one of these appears outside a retraction, some document
# is still asserting an old number.
FORBIDDEN = {
    "text rate (pre-alignment-fix)": [r"9\.0333", r"9\.033\s*%"],
    "text LoRA rate (pre-alignment-fix)": [r"9\.3459", r"9\.346\s*%"],
    "text LoRA delta (old)": [r"0\.313\s*pp", r"\+0\.313"],
    "text single-file LoRA (pre-fix)": [r"9\.1287"],
    "audio next-byte bpb (pre-fix)": [r"13\.12\s*bits?/?byte", r"13\.1\s*bits/byte", r"13\.1233\s*bpb", r"15\.7844\s*bpb"],
    "audio rate paired with 13.12": [r"13\.12.*197\.3", r"197\.3.*13\.12"],
    "image whole-image numerator": [r"\b330\s*bytes\b", r"\b330\s*字节\b"],
    "image rate (whole-image path)": [r"0\.017\s*%", r"0\.0169\s*%", r"0\.0463\s*%", r"0\.0154\s*%"],
    "image residual rate (withdrawn)": [r"67\.43\s*%", r"3\s*959\s*4\d\d"],
    "image denominator (old)": [r"original pixel count", r"原始像素数算的 rate", r"1912\s*(x|倍)"],
    "image source pixels as RGB denominator": [r"5\s*872\s*164.{0,20}原始像素"],
    "image rate (truncated log)": [r"15\.55\s*%", r"15\.530\s*%"],
    "image 'close reproduction' wording": [r"close reproduction", r"reproduces the paper closely"],
    "reference coder hole as fact": [r"leaves the same hole open", r"同样的洞"],
    "entropy as proof": [r"valid lower bound", r"低于信源熵", r"provably below"],
    "video baselines as reproduced": [r"四种模态的经典基线也复现成功", r"four modalities.*baselines.*reproduc"],
}

# ----------------------------------------------------------------- required --
# Facts each document must still carry.  Missing here means a correction was lost.
REQUIRED = {
    "audit": [
        r"9\.041", r"9\.353", r"0\.312",
        r"15\.5144", r"910\s*788", r"5\s*870\s*592", r"5\s*733",
        r"585 colliding", r"48 token", r"1\s*572",
        r"1\.9376",   # the verified audio figure (withdrawn numbers removed)
        r"test_igpt_injectivity", r"official_image_accounting",
        # Video: the 10-frame Colab measurement on the Netflix clip replaced the
        # withdrawn RGB numbers (0.0042 % / 73.05 %) and the 60-frame local run.
        r"18\.1508", r"26\.467", r"40\.114", r"46\.292", r"25 920",
        r"Withdrawn", r"numerically close",
    ],
    "report_cn": [
        r"9\.041", r"9\.353", r"0\.312",
        r"15\.5144", r"910\s*788", r"5\s*870\s*592", r"5\s*733",
        r"585", r"48 ", r"1\s*572",
        r"1\.9376",   # the verified audio figure (withdrawn numbers removed)
        # Video baselines are now the 10-frame Colab measurement on the Netflix clip,
        # which replaced the 60-frame local run on a different clip.
        r"26\.467", r"34\.087", r"40\.114", r"46\.292",
        r"18\.1508",
        # The video iGPT figure is now measured channel-wise. Require the
        # corrected value, not the withdrawn 0.0042 % / 73.05 %.
        r"16\.7162", r"25 920",
    ],
    "notes": [r"15\.5144", r"910\s*788", r"1\.9376", r"9\.041", r"9\.353"],
    "tables": [r"15\.5144", r"910\s*788", r"1\.9376", r"9\.041", r"9\.353"],
}


def enclosing_section_is_historical(lines, text: str, pos: int) -> bool:
    """True if the nearest preceding heading marks the section as historical."""
    line_no = text.count("\n", 0, pos)
    for i in range(line_no, max(-1, line_no - SECTION_LOOKBACK_LINES), -1):
        row = lines[i].lstrip()
        if row.startswith("#"):
            return any(w in row.lower() for w in SECTION_MARKERS)
    return False


def above_table_header(lines, text: str, pos: int) -> str:
    """Return the markdown table header row governing the row at ``pos``.

    Walks upwards while rows still look like table rows, then identifies the header
    as the row immediately above the ``|---|---|`` separator.
    """
    line_no = text.count("\n", 0, pos)
    i = line_no
    while i >= 0 and lines[i].lstrip().startswith("|"):
        i -= 1
    block_start = i + 1
    # the separator is the second row of the block; the header is the first
    if block_start + 1 < len(lines):
        sep = lines[block_start + 1].strip()
        if re.fullmatch(r"\|[\s:|-]+\|", sep):
            return lines[block_start]
    return ""


def scan(name: str, path: str) -> list[str]:
    issues: list[str] = []
    if not os.path.exists(path):
        return [f"{name}: FILE MISSING ({path})"]
    lines = open(path, encoding="utf-8").read().splitlines()
    text = "\n".join(lines)

    for label, patterns in FORBIDDEN.items():
        for pat in patterns:
            for m in re.finditer(pat, text):
                # Look at the hit's line AND the CONTEXT_LINES before it, because
                # retraction notes in these documents are multi-line blockquotes.
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.end())
                line = text[line_start : line_end if line_end != -1 else len(text)]
                ctx_start = line_start
                for _ in range(CONTEXT_LINES):
                    prev = text.rfind("\n", 0, max(0, ctx_start - 1))
                    if prev == -1:
                        ctx_start = 0
                        break
                    ctx_start = prev + 1
                # Extend one line forward too: a retraction phrase often straddles
                # the hit's line and the next ("... reported in the / first revision").
                ctx_end = line_end if line_end != -1 else len(text)
                nxt = text.find("\n", ctx_end + 1)
                if nxt != -1:
                    ctx_end = nxt
                context = text[ctx_start:ctx_end].lower()
                if any(w.lower() in context for w in RETRACTION_WORDS):
                    continue

                # A table row whose column header says "buggy"/"pre-fix"/etc. is a
                # deliberate old-vs-new comparison, not residue.
                if line.lstrip().startswith("|"):
                    header = above_table_header(lines, text, m.start())
                    if header and any(w in header.lower() for w in COLUMN_MARKERS):
                        continue

                # A hit inside a section whose nearest heading marks it historical
                # is a retraction note by construction.
                if enclosing_section_is_historical(lines, text, m.start()):
                    continue

                issues.append(
                    f"{name}: FORBIDDEN [{label}] {m.group(0)!r} in: {line.strip()[:90]}"
                )

    for pat in REQUIRED.get(name, []):
        if not re.search(pat, text):
            issues.append(f"{name}: REQUIRED fact missing: {pat}")
    return issues


def main() -> int:
    all_issues: list[str] = []
    print("=== cross-document consistency check ===\n")
    for name, path in DOCS.items():
        issues = scan(name, path)
        status = "OK" if not issues else f"{len(issues)} issue(s)"
        print(f"  {name:10s} {status}")
        all_issues.extend(issues)

    if all_issues:
        print("\n=== issues ===")
        for i in all_issues:
            print("  " + i)
    print(f"\n{len(all_issues)} issue(s) total")
    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
