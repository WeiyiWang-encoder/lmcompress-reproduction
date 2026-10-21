"""Audit every percentage in the user's docx against this project's measured values.

The docx claims video 26.19 %, image weighted 25.80 %, image median 22.33 %. None of those
appear anywhere in this project's documents, while the baselines quoted in the *same
sentence* (H.265 26.467, H.264 34.087, FFV1 40.114, zstd 46.292) are verbatim ours.

So this script extracts every percentage from the docx and classifies it as
  MATCH      -- a value this project measured or published
  DERIVED    -- ours multiplied by a constant factor, i.e. a transformation of our number
  UNKNOWN    -- not ours and not a simple transform

so the user can see exactly which figures need correcting and whether they came from a
different source or from arithmetic on ours.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCX = os.path.join(HERE, "results", "_user_doc.docx")

# Values this project measured or has published, with provenance.
OURS = {
    18.1512: "video, iGPT channel-wise, 9 frames (results/video_channelwise_summary.json)",
    26.467: "video baseline H.265 crf 0, same 10 frames",
    34.087: "video baseline H.264 crf 0, same 10 frames",
    40.114: "video baseline FFV1 level 3, same 10 frames",
    46.292: "video baseline zstd -22 raw YUV, same 10 frames",
    17.3212: "image CLIC, 5 images weighted including the outlier",
    14.3779: "image CLIC, 4 images weighted excluding the outlier",
    14.8875: "image CLIC, median",
    12.1318: "image CLIC, alejandro-escamilla-6",
    13.0406: "image CLIC, ales-krivec-15949",
    17.2361: "image CLIC, allef-vinicius-109434",
    28.6858: "image CLIC, alexander-shustov-73 (outlier)",
    15.31: "paper, video high-res (SI Table 4)",
    14.62: "paper, image CLIC2019 (SI Table 3)",
    21.69: "paper, video H.265",
    37.42: "paper, video H.264",
    40.16: "paper, video FFV1",
    51.04: "paper, video zstd",
    35.56: "paper, image WebP",
    48.64: "paper, image JPEG-2000",
    47.08: "paper, image PNG",
    9.06: "paper, text ctx 8192 (SI Table 9)",
    9.16: "paper, text ctx 6144",
    9.36: "paper, text ctx 4096",
    9.08: "paper, text (SI Table 6)",
    19.82: "audio, our port",
    19.94: "audio, reference implementation",
    18.26: "paper, audio",
    8.9253: "text, authors' harness base @8192 chars",
    9.5154: "text, authors' harness +LoRA @8192 chars",
    6.1657: "text, paired A100 base @4096 tokens",
    6.3321: "text, paired A100 base @8192 chars",
    6.6107: "text, paired A100 +LoRA @4096 tokens",
    6.7308: "text, paired A100 +LoRA @8192 chars",
    9.0410: "text, RETRACTED port figure",
    9.3529: "text, RETRACTED port LoRA figure",
    39.362: "image baseline WebP on 12 CLIC images",
    50.215: "image baseline JPEG-2000 on 12 CLIC images",
    76.848: "image baseline PNG on 12 CLIC images",
}

# Factors that would indicate a units error rather than a different source.
FACTORS = {
    1.4426950408889634: "log2(e)  <- bits converted as if they were nats",
    1.5: "1.5",
    0.6931471805599453: "ln(2)",
    1.5422: "the image whole-image cheat ratio",
    1.352: "the image per-chunk cheat ratio",
}


def docx_paragraphs(path: str) -> list[str]:
    z = zipfile.ZipFile(path)
    xml = z.read("word/document.xml").decode("utf-8", "replace")
    out = []
    for p in re.split(r"</w:p>", xml):
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
        t = (t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&#39;", "'"))
        if t.strip():
            out.append(t)
    return out


def main() -> int:
    if not os.path.exists(DOCX):
        print(f"missing {DOCX}")
        return 1
    paras = docx_paragraphs(DOCX)
    print(f"docx paragraphs: {len(paras)}")

    # every "NN.NN%" or "NN.NN %" occurrence, with its paragraph
    pat = re.compile(r"(\d{1,3}\.\d{1,4})\s*%")
    seen: dict[float, list[int]] = {}
    for i, t in enumerate(paras):
        if "<w:" in t:          # raw-xml paragraphs, not real text
            continue
        for m in pat.finditer(t):
            v = float(m.group(1))
            seen.setdefault(v, []).append(i)

    print(f"distinct percentages in the docx: {len(seen)}\n")

    match, derived, unknown = [], [], []
    for v in sorted(seen):
        if v in OURS:
            match.append(v)
            continue
        hit = None
        for ours, prov in OURS.items():
            for f, fname in FACTORS.items():
                if abs(ours * f - v) < 0.006:
                    hit = (ours, prov, f, fname)
                    break
            if hit:
                break
        if hit:
            derived.append((v, hit))
        else:
            unknown.append(v)

    print(f"=== {len(match)} values that are ours verbatim ===")
    for v in match:
        print(f"  {v:>8.4f} %   {OURS[v]}")

    print(f"\n=== {len(derived)} values that are OURS multiplied by a constant ===")
    for v, (ours, prov, f, fname) in derived:
        print(f"  {v:>8.4f} %  =  {ours:.4f} x {f:.6f}   [{fname}]")
        print(f"              source: {prov}")

    print(f"\n=== {len(unknown)} values with no trace in this project ===")
    for v in unknown:
        print(f"  {v:>8.4f} %")

    print("\n=== where the disputed figures sit ===")
    for v in (26.19, 25.80, 22.33):
        tag = "ours" if v in OURS else (
            "derived from ours" if any(d[0] == v for d in derived) else "unknown")
        print(f"  {v:>7.2f} %  ->  {tag}")

    # the exact arithmetic for the three disputed values
    print("\n=== reproducing the three disputed values from our data ===")
    vid = 18.1512
    print(f"  video : {vid:.4f} x log2(e) = {vid*1.4426950408889634:.4f} %  "
          f"(docx says 26.19)")
    per = [12.1318, 13.0406, 14.8875, 17.2361, 28.6858]
    simple_mean = sum(per) / len(per)
    print(f"  image : simple mean of the 5 per-image rates = {simple_mean:.4f} %")
    print(f"          {simple_mean:.4f} x 1.5 = {simple_mean*1.5:.4f} %  "
          f"(docx says 25.80)")
    print(f"  image : median 14.8875 x 1.5 = {14.8875*1.5:.4f} %  "
          f"(docx says 22.33)")
    print(f"  our weighted (bits-weighted) = 17.3212 %, excluding outlier 14.3779 %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
