"""Summarise the video iGPT per-frame measurements."""

from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(HERE, "results", "video_igpt.jsonl")

rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
b = [r["coded_bytes"] for r in rows]
u = [r["unique_tokens"] for r in rows]
px = sum(r["pixels"] for r in rows)
rgb = sum(r["rgb_bytes"] for r in rows)
bits = sum(r["coded_bits"] for r in rows)
res = sum(r["residual_zlib9"] for r in rows)

print(f"frames                 = {len(rows)}")
print(f"code bytes/frame       = min {min(b):.1f}, max {max(b):.1f}, mean {sum(b)/len(b):.1f}")
print(f"palette ids/frame      = min {min(u)}, max {max(u)}")
print(f"total pixels           = {px:,}")
print(f"total RGB bytes        = {rgb:,}")
print(f"model code total       = {bits/8:,.1f} bytes")
print(f"rate, paper convention = {100*(bits/8)/px:.6f} %")
print(f"rate on RGB bytes      = {100*(bits/8)/rgb:.6f} %")
print(f"exact-completion rate  = {100*((bits/8)+res)/rgb:.4f} %   (residual {res:,} B)")
print()
print("per-frame rate spread  = "
      f"{min(r['rate_pct_paper_convention'] for r in rows):.6f} % … "
      f"{max(r['rate_pct_paper_convention'] for r in rows):.6f} %")
print()
print("cross-check: 8640x reduction implies code/pixel ratio")
print(f"  pixels per frame {rows[0]['pixels']:,} / tokens {rows[0]['tokens']} = "
      f"{rows[0]['pixels']//rows[0]['tokens']}x")
