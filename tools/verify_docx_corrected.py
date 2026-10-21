"""Verify the docx against the corrected (log2) value set, and list what is left."""

from __future__ import annotations

import os
import re
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

F = 1.4426950408889634

# every figure that SHOULD appear once the log-base fix is applied
EXPECT = {
    # text, paired A100 (nats x log2e)
    "9.43": "text A@4096 chars base (6.5322 x 1.4427)",
    "9.22": "text A@6144 chars base",
    "9.14": "text A@8192 chars base",
    "8.90": "text B@4096 tokens base",
    "8.80": "text B@8192 tokens base",
    "10.00": "text A@4096 chars +LoRA",
    "9.81": "text A@6144 chars +LoRA",
    "9.71": "text A@8192 chars +LoRA",
    "9.54": "text B@4096 tokens +LoRA",
    "9.49": "text B@8192 tokens +LoRA",
    "...": "",
    # penalties
    "0.5766": "A@4096 penalty pp", "0.5816": "A@6144", "0.5752": "A@8192",
    "0.6420": "B@4096", "0.6821": "B@8192",
    # port text, unaffected (math.log2)
    "9.04": "port text base", "9.08": "paper SI Table 6",
    "8.9253": "authors harness base", "9.5154": "authors harness +LoRA",
    # video
    "26.19": "video iGPT (corrected)", "15.31": "paper video",
    "26.467": "H.265", "34.087": "H.264", "40.114": "FFV1", "46.292": "zstd",
    # image
    "24.99": "image CLIC weighted incl outlier",
    "20.74": "image CLIC weighted excl outlier",
    "21.48": "image CLIC median",
    "17.50": "alejandro", "18.82": "ales", "24.87": "allef", "41.39": "alexander",
    "14.62": "paper image", "15.5144": "official iGPT rate",
    # baselines
    "39.362": "WebP 12 CLIC", "50.215": "JPEG-2000 12 CLIC", "76.848": "PNG 12 CLIC",
    "31.84": "brotli window 8192", "41.44": "lzma window 8192",
    "28.6187": "brotli whole file",
    # audio
    "19.82": "audio port", "19.94": "audio reference", "18.26": "paper audio",
}

WRONG_STILL = {
    "6.53": "9.43 %（×1.4427）",
    "6.33": "9.14 %（×1.4427）",
    "6.17": "8.90 %（×1.4427）",
    "18.15": "26.19 %（×1.4427）",
    "17.32": "24.99 %（×1.4427）",
    "14.89": "21.48 %（×1.4427）",
    "14.38": "20.74 %（×1.4427）",
    "2.73": "应为我们高 0.08 pp",
    "28.918": "28.6187 %",
    "0.13": "应为 0.389 %（分母体积）/ 0.13 pp（码率）",
}


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
    blob = "\n".join(texts)
    print(f"docx {os.path.basename(path)}   paragraphs {len(texts)}")

    print("\n=== corrected values that are PRESENT ===")
    for v, why in EXPECT.items():
        if v == "...":
            continue
        c = blob.count(v)
        mark = "ok " if c else "MISS"
        print(f"  {mark} {v:>8}  x{c:<3} {why}")

    print("\n=== corrected values that are ABSENT (expected after the fix) ===")
    for v, why in EXPECT.items():
        if v == "...":
            continue
        if blob.count(v) == 0:
            print(f"  {v:>8}  {why}")

    print("\n=== numbers that would still be WRONG if present ===")
    any_bad = False
    for v, fix in WRONG_STILL.items():
        # require a word boundary-ish match so 6.53 != 16.53
        hits = re.findall(rf"(?<![\d.]){re.escape(v)}(?![\d])", blob)
        if hits:
            any_bad = True
            print(f"  {v:>8}  x{len(hits)}   -> {fix}")
    if not any_bad:
        print("  none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
