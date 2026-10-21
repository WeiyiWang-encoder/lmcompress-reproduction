"""Build the comparison tables for the reproduction report.

Pulls together three sources of numbers:

* ``results/*.jsonl``       -- this port's model measurements
* ``results/baselines_*.jsonl`` -- classical baselines measured locally
* ``configs/paper_results.yaml`` -- the paper's published numbers

and emits a Markdown table per modality plus a machine-readable JSON summary.

The key discipline this module enforces: **a ratio is only meaningful if the
numerator and denominator describe the same bytes.**  The paper's image pipeline
downsamples to 32x32 before coding yet divides by the original pixel count, and
its text container is a CSV whose size differs from the UTF-8 payload actually
fed to the model.  Both are reported explicitly here rather than silently
normalised away.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = _capsule_root()

# Published numbers, from the paper's Supplementary Tables 3-6.
# Units: compression rate in percent (compressed / original, lower is better).
PAPER = {
    "text": {
        "dataset": "MeDAL",
        "baselines": {"zstd --ultra -22": 30.45, "7z -mx9": 30.25, "zpaq -method 5": 25.11, "brotli q11": 27.93},
        "raw_llama": 9.08,          # LMCompress* (no fine-tuning)
        "lora_llama": 9.06,         # LMCompress (LoRA fine-tuned)
    },
    "text_pileoflaw": {
        "dataset": "Pile of Law (eurlex)",
        "baselines": {"zstd --ultra -22": 20.13, "7z -mx9": 19.42, "zpaq -method 5": 15.94, "brotli q11": 18.02},
        "raw_llama": 5.86,
        "lora_llama": 4.90,
    },
    "audio": {
        "dataset": "LJSpeech",
        "baselines": {"flac -compression_level 12": 41.26, "optimfrog --preset max": 25.42, "wavpack -hh -x3": 33.15},
        "model": 18.26,             # LMCompress-bgpt
    },
    "image": {
        "dataset": "CLIC2019",
        "baselines": {"png": 47.08, "jpeg-xl": 31.16, "webp": 35.56, "jpeg-2000": 48.64},
        "bgpt": 28.57,
        "igpt": 14.62,
    },
    "video": {
        "dataset": "low-res static",
        "baselines": {"zstd --ultra -22": 50.47, "ffv1": 38.02, "h.264 crf 0": 29.58, "h.265 crf 0": 21.05},
        "model": 13.05,
    },
}

# Map our codec labels onto the paper's rows.
CODEC_ALIASES = {
    "zstd --ultra -22": "zstd --ultra -22",
    "lzma preset 9e (7z-equivalent)": "7z -mx9",
    "brotli q11": "brotli q11",
    "zlib 9": None,
    "flac -compression_level 12": "flac -compression_level 12",
    "wavpack -hh -x3": "wavpack -hh -x3",
    "optimfrog --preset max": "optimfrog --preset max",
    "png (ffmpeg cl100)": "png",
    "jpeg-xl -distance 0": "jpeg-xl",
    "webp -lossless -q100": "webp",
    "jpeg-2000 lossless": "jpeg-2000",
    "ffv1 level 3": "ffv1",
    "h.264 crf 0 (x264 veryslow)": "h.264 crf 0",
    "h.265 crf 0 (x265 veryslow)": "h.265 crf 0",
}


def read_jsonl(pattern: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(os.path.join(HERE, "results", pattern))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_source"] = os.path.basename(path)
                    rows.append(row)
    return rows


def aggregate(rows: Iterable[dict], key: str = "codec") -> Dict[str, dict]:
    """Sum bytes across units so the ratio is a proper pooled ratio."""
    bucket: Dict[str, dict] = defaultdict(lambda: {"orig": 0, "comp": 0, "secs": 0.0, "units": 0})
    for row in rows:
        b = bucket[row[key]]
        b["orig"] += int(row["original_bytes"])
        b["comp"] += int(row["compressed_bytes"]) if "compressed_bytes" in row else int(round(row["coded_bits"] / 8))
        b["secs"] += float(row.get("seconds", row.get("wall_seconds", 0.0)))
        b["units"] += 1
    out = {}
    for name, b in bucket.items():
        if not b["orig"] or not b["comp"]:
            continue
        out[name] = {
            "original_bytes": b["orig"],
            "compressed_bytes": b["comp"],
            "compression_rate_pct": 100.0 * b["comp"] / b["orig"],
            "compression_ratio": b["orig"] / b["comp"],
            "seconds": b["secs"],
            "units": b["units"],
        }
    return out


def md_table(title: str, measured: Dict[str, dict], paper: Dict[str, float], unit_note: str) -> str:
    lines = [f"### {title}", "", unit_note, "",
             "| Method | measured rate | measured ratio | paper rate | Δ (pp) |",
             "|---|---|---|---|---|"]
    for name, vals in sorted(measured.items(), key=lambda kv: kv[1]["compression_rate_pct"]):
        alias = CODEC_ALIASES.get(name, name)
        p = paper.get(alias) if alias else None
        delta = f"{vals['compression_rate_pct'] - p:+.2f}" if p is not None else "n/a"
        pr = f"{p:.2f}%" if p is not None else "—"
        lines.append(
            f"| {name} | **{vals['compression_rate_pct']:.2f}%** | {vals['compression_ratio']:.2f}× | {pr} | {delta} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "comparison_tables.md"))
    args = ap.parse_args()

    sections: List[str] = ["# Reproduction comparison tables", ""]
    summary: dict = {}

    # ---------------- text ----------------
    base = aggregate(read_jsonl("baselines_text_medal_3.jsonl"))
    raw = aggregate(read_jsonl("baselines_text_medal_raw_3.jsonl"))
    base = {k: v for k, v in base.items() if CODEC_ALIASES.get(k) is not None}

    model_rows = read_jsonl("text_medal_*.jsonl")
    # Deduplicate by (unit, configuration) keeping the newest file per unit name.
    by_unit: Dict[str, dict] = {}
    for row in model_rows:
        by_unit.setdefault(row["unit"], row)

    lora_bits = sum(int(round(r["coded_bits"])) for r in by_unit.values())
    lora_orig = sum(int(r["original_bytes"]) for r in by_unit.values())
    measured = dict(base)
    if lora_orig:
        measured["LMCompress (this port, LoRA)"] = {
            "original_bytes": lora_orig,
            "compressed_bytes": lora_bits // 8,
            "compression_rate_pct": 100.0 * (lora_bits / 8) / lora_orig,
            "compression_ratio": lora_orig / (lora_bits / 8),
            "seconds": 0.0,
            "units": len(by_unit),
        }
    sections.append(
        md_table(
            "Text - MeDAL (CSV container denominator)",
            measured,
            PAPER["text"]["baselines"] | {"LMCompress (this port, LoRA)": PAPER["text"]["lora_llama"]},
            "Denominator: UTF-8 bytes of the CSV file, which is what the model is fed.",
        )
    )

    # Raw-text denominator view
    if raw:
        lines = ["### Text - MeDAL (raw UTF-8 text denominator)", "",
                 "Denominator: the concatenated `TEXT` column, i.e. the actual payload without CSV quoting.", "",
                 "| Method | rate (container denom) | rate (raw text denom) |", "|---|---|---|"]
        for name in sorted(base):
            alias = CODEC_ALIASES.get(name)
            a = base[name]["compression_rate_pct"]
            b = raw.get(name, {}).get("compression_rate_pct")
            lines.append(f"| {name} | {a:.2f}% | {b:.2f}% |" if b else f"| {name} | {a:.2f}% | — |")
        if lora_orig:
            raw_bytes = sum(os.path.getsize(os.path.join(HERE, "artifacts", "text_raw", os.path.basename(u).replace(".csv", ".txt")))
                            for u in by_unit if os.path.exists(os.path.join(HERE, "artifacts", "text_raw", os.path.basename(u).replace(".csv", ".txt"))))
            if raw_bytes:
                lines.append(
                    f"| LMCompress (this port, LoRA) | {100.0*(lora_bits/8)/lora_orig:.2f}% | "
                    f"{100.0*(lora_bits/8)/raw_bytes:.2f}% |"
                )
        lines.append("")
        sections.append("\n".join(lines))

    # ---------------- audio ----------------
    audio = aggregate(read_jsonl("baselines_audio*.jsonl"))
    sections.append(
        md_table(
            "Audio - LJSpeech sample",
            audio,
            PAPER["audio"]["baselines"],
            "Denominator: raw PCM payload bytes. BMCompress rows are in "
            "`results/audio_*.jsonl`; see the audio section of the report for why "
            "that number is not comparable.",
        )
    )

    # ---------------- image / video ----------------
    for modality, pattern in (("image", "baselines_image*.jsonl"), ("video", "baselines_video*.jsonl")):
        rows = aggregate(read_jsonl(pattern))
        if rows:
            paper = PAPER[modality]["baselines"]
            sections.append(md_table(f"{modality.title()} baselines (measured locally)", rows, paper, ""))

    text = "\n".join(sections)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
