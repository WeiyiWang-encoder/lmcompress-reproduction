"""Review the numbers in the user's docx and annotate errors in RED, leaving text intact.

Correction set (all in true bits, i.e. log2, after the log-base fix in AUDIT chapter 10):

  video iGPT            18.1512 nats -> 26.1866 bits   (paper 15.31)
  image CLIC weighted   17.3212 -> 24.9892             (paper 14.62)
  image CLIC excl outlr 14.3779 -> 20.7429
  image CLIC median     14.8875 -> 21.4781
  text A@8192           6.3321 -> 9.1353               (paper 9.06)
  text B@4096           6.1657 -> 8.8952               (paper 9.36)

Usage:
  review_docx_numbers.py <docx>            -> audit table only
  review_docx_numbers.py <docx> --write    -> also emit an annotated copy

Annotation method: a red run is inserted immediately after the run whose text contains the
wrong value. The original text is never modified, so the author can accept or reject each
note. Run-level insertion keeps each note next to the number it refers to.
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

F = 1.4426950408889634


def n(v: float) -> str:
    return f"{v * F:.2f}"


# value as printed in the docx -> correction note (None means "this one is right")
WRONG: dict[str, str] = {}


def add(wrong: float, right_text: str, why: str) -> None:
    for v in (f"{wrong:.4f}", f"{wrong:.2f}", f"{wrong:.1f}", f"{wrong}"):
        WRONG.setdefault(v, f"应为 {right_text}（{why}）")


add(18.1512, "26.19 %", "log_softmax 返回自然对数，须 ×log2(e)=1.4427")
add(17.3212, "24.99 %", "同上，×1.4427")
add(14.8875, "21.48 %", "同上，×1.4427")
add(14.3779, "20.74 %", "同上，×1.4427")
add(12.1318, "17.50 %", "同上，×1.4427")
add(13.0406, "18.82 %", "同上，×1.4427")
add(17.2361, "24.87 %", "同上，×1.4427")
add(28.6858, "41.39 %", "同上，×1.4427")
add(6.3321, "9.14 %", "同上，×1.4427；修正后与论文 9.06 % 吻合")
add(6.5322, "9.43 %", "同上，×1.4427")
add(6.3927, "9.22 %", "同上，×1.4427")
add(6.1657, "8.90 %", "同上，×1.4427")
add(6.1023, "8.80 %", "同上，×1.4427")
add(6.9319, "10.00 %", "同上，×1.4427")
add(6.7958, "9.81 %", "同上，×1.4427")
add(6.7308, "9.71 %", "同上，×1.4427")
add(6.6107, "9.54 %", "同上，×1.4427")
add(6.5751, "9.49 %", "同上，×1.4427")
add(0.3997, "0.58 pp", "惩罚值同比例换算 ×1.4427")
add(0.4031, "0.58 pp", "同上")
add(0.3987, "0.58 pp", "同上")
add(0.4450, "0.64 pp", "同上")
add(0.4728, "0.68 pp", "同上")
add(2.73, "应说 9.14 % 对 9.06 %，即我们高 0.08 pp（不是低 2.73）", "单位修正后结论反转")
add(5.0, "3.48×（9.135 % vs brotli 31.84 %）", "模型率修正后倍数随之改变")
add(3.20, "约 3.1×", "brotli 实为 28.6187 %，模型率需 ×1.4427")
add(28.918, "28.6187 %", "该值在任何结果行中都不存在")

# values that are RIGHT and must not be flagged
RIGHT = {"26.19", "26.1866", "9.06", "9.08", "15.31", "14.62", "9.16", "9.36",
         "18.26", "19.82", "19.94", "8.9253", "9.5154", "23.89", "42.02",
         "38.21", "35.26", "39.54", "26.467", "34.087", "40.114", "46.292",
         "39.362", "50.215", "76.848", "15.5144", "21.69", "37.42", "40.16",
         "51.04", "35.56", "48.64", "47.08", "31.84", "41.44", "30.26", "32.99"}


def paragraphs(xml: str) -> list[str]:
    return re.split(r"(?<=</w:p>)", xml)


def run_texts(p: str) -> list[str]:
    return [re.sub(r"<[^>]+>", "", m) for m in
            re.findall(r"<w:r[ >].*?</w:r>", p, re.S)]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: review_docx_numbers.py <docx> [--write]")
        return 2
    src = sys.argv[1]
    write = "--write" in sys.argv

    z = zipfile.ZipFile(src)
    xml = z.read("word/document.xml").decode("utf-8")
    paras = paragraphs(xml)
    print(f"docx: {os.path.basename(src)}   paragraphs {len(paras)}")

    pat = re.compile(r"\d{1,3}\.\d{1,4}")
    found_bad: list[tuple[int, str, str]] = []
    unknown: dict[str, int] = {}
    n_right = 0
    for i, p in enumerate(paras):
        for rt in run_texts(p):
            for m in pat.finditer(rt):
                tok = m.group(0)
                if tok in RIGHT:
                    n_right += 1
                    continue
                if tok in WRONG:
                    found_bad.append((i, tok, WRONG[tok]))
                else:
                    unknown[tok] = unknown.get(tok, 0) + 1

    print(f"\n=== correctly valued numbers: {n_right} ===")
    print(f"=== numbers needing a red note: {sum(1 for _ in found_bad)} occurrences, "
          f"{len({t for _, t, _ in found_bad})} distinct ===")
    seen = set()
    for i, tok, why in found_bad:
        if tok in seen:
            continue
        seen.add(tok)
        print(f"  {tok:>10}  ->  {why}")
    print(f"\n=== unrecognised numbers ({len(unknown)} distinct) ===")
    for tok, c in sorted(unknown.items())[:40]:
        print(f"  {tok:>10}  x{c}")

    if not write:
        print("\n(dry run; pass --write to produce the annotated copy)")
        return 0

    # ---- annotate: insert a red run after the run containing each wrong number ---- #
    out_paras = []
    inserted = 0
    for p in paras:
        runs = re.findall(r"<w:r[ >].*?</w:r>", p, re.S)
        if not runs:
            out_paras.append(p)
            continue
        new_p = p
        for r in runs:
            txt = re.sub(r"<[^>]+>", "", r)
            hit = None
            for m in pat.finditer(txt):
                if m.group(0) in WRONG:
                    hit = m.group(0)
                    break
            if hit is None:
                continue
            note = (f"【审阅更正：{hit} → {WRONG[hit]}】")
            note = (note.replace("&", "&amp;").replace("<", "&lt;")
                    .replace(">", "&gt;"))
            red = ('<w:r><w:rPr><w:b/><w:color w:val="FF0000"/></w:rPr>'
                   f'<w:t xml:space="preserve">{note}</w:t></w:r>')
            new_p = new_p.replace(r, r + red, 1)
            inserted += 1
        out_paras.append(new_p)

    new_xml = "".join(out_paras)
    dst = os.path.splitext(src)[0] + "_数字审阅批注.docx"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = new_xml.encode("utf-8")
            zout.writestr(item, data)

    print(f"\n  inserted {inserted} red notes")
    print(f"  wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
