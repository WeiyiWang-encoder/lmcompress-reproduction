"""Audit every percentage in a docx against this project's measured values.

Usage: audit_docx_v2.py <path-to.docx>

Matching is tolerant of the docx rounding to two decimals, and it separates three cases:

  OK        -- within 0.005 of a value this project measured (i.e. rounded)
  SCALED    -- equal to one of our values times a constant, which means a unit or
               aggregation error rather than a different source
  UNKNOWN   -- neither, so it needs tracing before it goes in a paper
"""

from __future__ import annotations

import os
import re
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every figure this project measured or has published, with provenance.
OURS: dict[float, str] = {
    # video, iGPT channel-wise, 9 distinct frames
    18.1512: "video iGPT channel-wise (9 frames)",
    18.1508: "video iGPT as emitted (10 rows, one duplicated)",
    26.467: "video baseline H.265 crf 0",
    34.087: "video baseline H.264 crf 0",
    40.114: "video baseline FFV1 level 3",
    46.292: "video baseline zstd -22",
    22.862: "video baseline H.265, 60-frame local clip",
    36.866: "video baseline H.264, 60-frame local clip",
    40.167: "video baseline FFV1, 60-frame local clip",
    51.035: "video baseline zstd, 60-frame local clip",
    # image, CLIC professional
    17.3212: "image CLIC 5, weighted incl outlier",
    14.3779: "image CLIC 4/5, weighted excl outlier",
    14.8875: "image CLIC median",
    12.1318: "image CLIC alejandro-escamilla-6",
    13.0406: "image CLIC ales-krivec-15949",
    17.2361: "image CLIC allef-vinicius-109434",
    28.6858: "image CLIC alexander-shustov-73",
    15.5144: "image, official path on the capsule image",
    23.9257: "image, our port channel-wise on the capsule image",
    39.362: "image baseline WebP, 12 CLIC images",
    50.215: "image baseline JPEG-2000, 12 CLIC images",
    76.848: "image baseline PNG, 12 CLIC images",
    48.279: "image baseline WebP, aerial photo",
    53.252: "image baseline JPEG-2000, aerial photo",
    84.381: "image baseline PNG, aerial photo",
    # text
    9.0410: "text base, RETRACTED port figure",
    9.3529: "text +LoRA, RETRACTED port figure",
    6.5322: "text A@4096 chars base",
    6.3927: "text A@6144 chars base",
    6.3321: "text A@8192 chars base",
    6.1657: "text B@4096 tokens base",
    6.1023: "text B@8192 tokens base",
    6.9319: "text A@4096 chars +LoRA",
    6.7958: "text A@6144 chars +LoRA",
    6.7308: "text A@8192 chars +LoRA",
    6.6107: "text B@4096 tokens +LoRA",
    6.5751: "text B@8192 tokens +LoRA",
    8.9253: "authors' harness base @8192 chars",
    9.5154: "authors' harness +LoRA @8192 chars",
    # text baselines, bytes-weighted
    28.6187: "text baseline brotli q11 (10 units, payload)",
    31.4479: "text baseline lzma 9e / 7z-equivalent (10 units)",
    31.6755: "text baseline zstd -22 (10 units)",
    34.7582: "text baseline zlib 9 (10 units)",
    28.672: "text baseline brotli q11 (3 raw files)",
    31.525: "text baseline lzma 9e (3 raw files)",
    31.649: "text baseline zstd (3 raw files)",
    34.673: "text baseline zlib (3 raw files)",
    28.918: "text baseline brotli, WRONG value in our docs (should be 28.62-28.67)",
    # audio
    19.82: "audio, our port",
    19.94: "audio, reference implementation",
    20.801474: "audio, superseded intermediate",
    1.9376: "audio, bpb",
    # paper
    15.31: "paper video high-res",
    21.69: "paper video H.265",
    37.42: "paper video H.264",
    40.16: "paper video FFV1",
    51.04: "paper video zstd",
    23.89: "paper 7z unbounded",
    42.02: "paper 7z window 8192",
    38.21: "paper 7z window 16384",
    35.26: "paper 7z window 32768",
    39.54: "paper zstd window 8192",
    14.62: "paper image CLIC2019",
    35.56: "paper image WebP",
    48.64: "paper image JPEG-2000",
    47.08: "paper image PNG",
    9.08: "paper text SI Table 6",
    9.06: "paper text ctx 8192",
    9.16: "paper text ctx 6144",
    9.36: "paper text ctx 4096",
    18.26: "paper audio",
}

FACTORS = {
    1.4426950408889634: "log2(e) -- bits treated as nats",
    0.6931471805599453: "ln(2)",
    1.5: "1.5",
    1.5422: "image whole-image cheat ratio",
    1.352: "image per-chunk cheat ratio",
}


def paragraphs(path: str) -> list[str]:
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
    if len(sys.argv) < 2:
        print("usage: audit_docx_v2.py <path-to.docx>")
        return 2
    path = sys.argv[1]
    paras = paragraphs(path)
    print(f"docx: {os.path.basename(path)}")
    print(f"paragraphs: {len(paras)}")

    pat = re.compile(r"(\d{1,3}(?:\.\d{1,6})?)\s*%")
    seen: dict[float, list[int]] = {}
    for i, t in enumerate(paras):
        if "<w:" in t:
            continue
        for m in pat.finditer(t):
            seen.setdefault(float(m.group(1)), []).append(i)

    print(f"distinct percentages: {len(seen)}\n")
    ok, scaled, unknown = [], [], []
    for v in sorted(seen):
        near = [o for o in OURS if abs(o - v) <= 0.005]
        if near:
            ok.append((v, near[0]))
            continue
        hit = None
        for o, prov in OURS.items():
            for f, fname in FACTORS.items():
                if abs(o * f - v) <= 0.006:
                    hit = (o, prov, f, fname)
                    break
            if hit:
                break
        if hit:
            scaled.append((v, hit))
        else:
            unknown.append((v, seen[v]))

    print(f"=== OK: {len(ok)} values within rounding of ours ===")
    for v, o in ok:
        print(f"  {v:>10.4f} %  ~= {o:<10.4f}  {OURS[o]}")

    print(f"\n=== SCALED: {len(scaled)} values equal to ours x a constant ===")
    for v, (o, prov, f, fname) in scaled:
        print(f"  {v:>10.4f} %  =  {o:.4f} x {f:.6f}   [{fname}]")
        print(f"                 source: {prov}")

    print(f"\n=== UNKNOWN: {len(unknown)} values needing a trace ===")
    for v, ps in unknown:
        ctx = ""
        for i in ps:
            if "<w:" not in paras[i]:
                ctx = paras[i][:110]
                break
        print(f"  {v:>10.4f} %   para {ps[:3]}   {ctx}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
