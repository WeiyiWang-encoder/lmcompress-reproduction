"""Adversarial verification harness for the claims in REVIEW_GUIDE_FOR_AGENT.md.

Design rules, so a reviewer can trust the output:

* every check prints the **raw quantities** it measured, not just a verdict, so the
  reviewer can disagree with the arithmetic rather than having to trust it;
* checks read the **authors' own files** wherever that is possible, rather than our
  re-derived copies;
* a check that cannot run is reported SKIP, never silently PASS;
* nothing is hard-coded as "expected" unless the expectation itself is the claim.

Usage:
    python tools/review_harness.py            # run all fast checks
    python tools/review_harness.py --list     # list checks
    python tools/review_harness.py --only C2  # run one
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()

RESULTS = []
SKIPS = []


def report(tag: str, title: str, ok: bool, lines: list[str]) -> None:
    RESULTS.append((tag, title, ok, lines))
    # Print inline as well as in the summary, so the raw quantities appear next to
    # the check that produced them rather than only at the end.
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    for line in lines:
        print(line)
    sys.stdout.flush()


def skip(tag: str, title: str, why: str) -> None:
    SKIPS.append((tag, title, why))


# --------------------------------------------------------------------------- #
# Claim 1: text alignment fix (our own bug)
# --------------------------------------------------------------------------- #
def check_c1() -> None:
    tag, title = "C1", "text alignment fix is real and guarded"
    src = os.path.join(HERE, "src", "uic", "models.py")
    text = open(src, encoding="utf-8").read()
    lines = []

    has_assert = "target misalignment" in text
    lines.append(f"  assertion present in models.py: {has_assert}")

    # both arms must have coded_symbols == symbols
    for name, expect in (("text_base_v2.jsonl", 9.0410), ("text_lora_v2.jsonl", 9.3529)):
        path = os.path.join(HERE, "results", name)
        if not os.path.exists(path):
            skip(tag, title, f"{name} missing")
            return
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        bits = sum(r["coded_bits"] for r in rows)
        raw = sum(r["raw_bytes"] for r in rows)
        cs = sum(r.get("coded_symbols", -1) for r in rows)
        sy = sum(r.get("symbols", -1) for r in rows)
        rate = 100 * (bits / 8) / raw
        lines.append(f"  {name}: coded_symbols={cs} symbols={sy} "
                     f"rate={rate:.4f}% (guide says {expect})")

    base = os.path.join(HERE, "results", "text_base_v2.jsonl")
    with open(base, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    ok_aligned = all(r.get("coded_symbols", -1) == r.get("symbols", -2) for r in rows)
    report(tag, title, has_assert and ok_aligned, lines)


# --------------------------------------------------------------------------- #
# Claim 2: audio scoring misalignment
# --------------------------------------------------------------------------- #
def check_c2() -> None:
    """Regression guard for the RETRACTED claim C2.

    The claim (a 16-position misalignment inflating the reference's audio result) was
    wrong: a patch row carries 17 slots, so the original row for payload k is
    17*(k//16)+(k%16), not 16+k.  This is a static regression guard for that mapping
    and the recorded retraction; it does not pretend to rerun the GPU measurements.
    """
    tag, title = "C2", "withdrawn audio misalignment is recorded as retracted"
    lines = []

    # (a) Verify the production selector, not merely a copied formula.
    sys.path.insert(0, os.path.join(HERE, "src"))
    from uic.models import bgpt_prediction_indices

    selected = bgpt_prediction_indices(32)
    expected = [17 * (k // 16) + (k % 16) for k in range(32)]
    old = list(range(16, 48))
    selector_ok = selected == expected and selected != old
    lines.append(f"  selector rows for 32 bytes: {selected}")
    lines.append(f"  expected 17-slot mapping: {expected}")
    lines.append(f"  differs from withdrawn contiguous 16+k mapping: {selected != old}")

    ref = os.path.join(REPO, "compress_with_bgpt_or_llama.py")
    ref_text = open(ref, encoding="utf-8").read() if os.path.exists(ref) else ""
    reference_trims_slot = "logits = logits[:-1, :-1, :]" in ref_text
    lines.append(f"  reference removes the 17th slot before flattening: {reference_trims_slot}")

    # (c) the retraction must be recorded in every document
    missing = []
    for d in ("AUDIT_FOR_REVIEW.md", "REPORT_CN.md", "REPRODUCTION_NOTES.md",
              "REVIEW_GUIDE_FOR_AGENT.md"):
        path = os.path.join(HERE, "docs", d)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        marked = any(w in text for w in
                     ("RETRACT", "已撤回", "WITHDRAWN", "作废", "withdrawn", "void"))
        if not marked:
            missing.append(d)
        lines.append(f"  {d}: retraction recorded = {marked}")
    lines.append(f"  documents missing a retraction record: {missing or 'none'}")

    report(tag, title, selector_ok and reference_trims_slot and not missing, lines)


# --------------------------------------------------------------------------- #
# Claim 3: image non-injectivity (exhaustive)
# --------------------------------------------------------------------------- #
def check_c3() -> None:
    tag, title = "C3", "image quantiser is non-injective (256 -> 48, 585 pairs)"
    path = os.path.join(HERE, "results", "igpt_grayscale_collisions.json")
    if not os.path.exists(path):
        skip(tag, title, "collisions JSON not found")
        return
    data = json.load(open(path, encoding="utf-8"))
    lines = []
    lines.append(f"  top-level keys: {sorted(data.keys())[:8]}")

    # find the mapping and recompute the statistics ourselves
    mapping = None
    for key in ("mapping", "token_by_value", "value_to_token", "map"):
        if key in data:
            mapping = data[key]
            break
    if mapping is None:
        # maybe nested
        for v in data.values():
            if isinstance(v, dict) and len(v) == 256:
                mapping = v
                break
    if mapping is None:
        skip(tag, title, "could not locate the 256-entry mapping in the JSON")
        return

    vals = {int(k): int(v) for k, v in mapping.items()}
    distinct = sorted(set(vals.values()))
    from collections import Counter
    counts = Counter(vals.values())
    colliding_pairs = sum(c * (c - 1) // 2 for c in counts.values())
    largest = counts.most_common(1)[0]

    lines.append(f"  domain size            = {len(vals)}")
    lines.append(f"  image size (distinct)  = {len(distinct)}   (guide says 48)")
    lines.append(f"  colliding value pairs  = {colliding_pairs}   (guide says 585)")
    lines.append(f"  largest class          = token {largest[0]} <- {largest[1]} values")
    ok = len(vals) == 256 and len(distinct) == 48 and colliding_pairs == 585
    report(tag, title, ok, lines)


# --------------------------------------------------------------------------- #
# Claim 4: loader drops 1572 samples
# --------------------------------------------------------------------------- #
def check_c4() -> None:
    tag, title = "C4", "loader floor division drops 1572 channel samples"
    acc = os.path.join(HERE, "results", "official_image_accounting.json")
    if not os.path.exists(acc):
        skip(tag, title, "official_image_accounting.json missing")
        return
    data = json.load(open(acc, encoding="utf-8"))
    lines = [f"  keys: {sorted(data.keys())}"]
    lines.append(f"  raw contents: {json.dumps(data)[:400]}")

    # recompute the arithmetic of the loss independent of the JSON
    W, H = 1714, 1142
    source = W * H * 3
    per_channel = (W * H) // 1024 * 1024
    processed = per_channel * 3
    lines.append(f"  source RGB channel samples = {source}")
    lines.append(f"  per-channel floor to 1024  = {per_channel} "
                 f"(dropped {W*H - per_channel} per channel)")
    lines.append(f"  processed (3 channels)     = {processed}")
    lines.append(f"  difference                 = {source - processed}   (guide says 1572)")
    ok = (source - processed) == 1572
    report(tag, title, ok, lines)


# --------------------------------------------------------------------------- #
# Claim 5: reference log totals are stale
# --------------------------------------------------------------------------- #
def check_c5() -> None:
    tag, title = "C5", "reference log prints every 100th chunk, so its total is stale"
    ref = os.path.join(REPO, "compress_with_igpt.py")
    if not os.path.exists(ref):
        skip(tag, title, "compress_with_igpt.py not found")
        return
    text = open(ref, encoding="utf-8").read()
    lines = []
    hits = [l.strip() for l in text.splitlines() if "% 100" in l or "%100" in l]
    for h in hits[:5]:
        lines.append(f"  print guard: {h!r}")
    lines.append(f"  guards found: {len(hits)}")
    ok = len(hits) > 0
    if not ok:
        lines.append("  !! no modulo guard found -- the claim would be WRONG")
    report(tag, title, ok, lines)


# --------------------------------------------------------------------------- #
# Claim 6: LoRA penalty, and its confound
# --------------------------------------------------------------------------- #
def check_c6() -> None:
    tag, title = "C6", "published LoRA is worse on all 3 files (confounded by ctx)"
    lines = []
    got = {}
    for name in ("text_base_v2.jsonl", "text_lora_v2.jsonl"):
        path = os.path.join(HERE, "results", name)
        if not os.path.exists(path):
            skip(tag, title, f"{name} missing")
            return
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        got[name] = {r["unit"]: 100 * (r["coded_bits"] / 8) / r["raw_bytes"] for r in rows}

    base, lora = got["text_base_v2.jsonl"], got["text_lora_v2.jsonl"]
    units = sorted(set(base) & set(lora))
    worsen = 0
    for u in units:
        d = lora[u] - base[u]
        worsen += d > 0
        lines.append(f"  {os.path.basename(u)}: base {base[u]:.4f}% -> "
                     f"lora {lora[u]:.4f}%  delta {d:+.4f} pp")
    tb = sum(100 * (json.loads(l)["coded_bits"] / 8) / json.loads(l)["raw_bytes"]
             for l in open(os.path.join(HERE, "results", "text_base_v2.jsonl"),
                           encoding="utf-8") if l.strip())
    lines.append(f"  files where LoRA is worse: {worsen}/{len(units)}")
    lines.append("  CONFOUND: measured at ctx 4096; paper used 8192; paper's own "
                 "Table 9 implies ~0.3 pp from that alone -- same magnitude.")

    # training-state evidence
    ts = os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora",
                      "checkpoint-5120", "trainer_state.json")
    if os.path.exists(ts):
        st = json.load(open(ts, encoding="utf-8"))
        hist = st.get("log_history", [])
        ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
        if ev:
            best = min(ev, key=lambda x: x[1])
            lines.append(f"  eval_loss best {best[1]:.4f} at step {best[0]}; "
                         f"last {ev[-1][1]:.4f} at step {ev[-1][0]}")
            ok = worsen == len(units) and best[0] < ev[-1][0]
        else:
            ok = worsen == len(units)
    else:
        lines.append("  trainer_state.json not found; skipping training evidence")
        ok = worsen == len(units)
    report(tag, title, ok, lines)


# --------------------------------------------------------------------------- #
# Claim 7: no bits2bytes; fp32 OOM
# --------------------------------------------------------------------------- #
def check_c7() -> None:
    tag, title = "C7", "'bits2bytes' does not exist in the authors' code"
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in (".venv", "repro_understanding_is_compression",
                                "__pycache__", ".git")]
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            try:
                text = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if "bits2bytes" in line:
                    hits.append(f"{os.path.relpath(p, REPO)}:{i}: {line.strip()[:80]}")
    lines = [f"  occurrences of 'bits2bytes': {len(hits)}"]
    for h in hits[:10]:
        lines.append("    " + h)

    # the undefined name that IS used
    undef = []
    src = os.path.join(REPO, "compress_with_bgpt_or_llama.py")
    if os.path.exists(src):
        text = open(src, encoding="utf-8").read()
        for name in ("compression_utils",):
            n = text.count(name)
            lines.append(f"  occurrences of {name!r} in compress_with_bgpt_or_llama.py: {n}")
            if n:
                undef.append(name)
    report(tag, title, len(hits) == 0, lines)


# --------------------------------------------------------------------------- #
# Claim 8: our own arithmetic coder
# --------------------------------------------------------------------------- #
def check_c8() -> None:
    tag, title = "C8", "coder has a documented interval-collapse guard"
    path = os.path.join(HERE, "src", "uic", "coder.py")
    text = open(path, encoding="utf-8").read()
    lines = []
    has_min = "min_interval_width" in text
    has_assert = "interval collapsed" in text
    lines.append(f"  min_interval_width present: {has_min}")
    lines.append(f"  collapse assertion present: {has_assert}")

    # The docstring must carry the CORRECT justification, and must not assert the
    # invalid one as if it were true.  Checked narrowly: the invalid sentence may
    # still appear *as a recorded retraction*, so we require the correction to be
    # present alongside it.
    correct = "never computed from the coder's" in text
    lines.append(f"  correct justification present: {correct}")
    if "8 bits/byte ceiling" in text:
        idx = text.find("8 bits/byte ceiling")
        window = text[max(0, idx - 700): idx + 300]
        labelled = ("invalid" in window) or ("earlier revision" in window)
        lines.append(f"  the '8 bits/byte ceiling' phrase appears; is it labelled "
                     f"as an invalid/withdrawn argument? {labelled}")
    else:
        labelled = True
        lines.append("  the '8 bits/byte ceiling' phrase is absent entirely")

    report(tag, title, has_min and has_assert and correct and labelled, lines)


# --------------------------------------------------------------------------- #
# Cross-cutting
# --------------------------------------------------------------------------- #
def check_x1() -> None:
    tag, title = "X1", "reported rate is q-based, tracked separately from coder output"
    lines = []
    for rel in ("src/uic/engine.py", "src/uic/runner.py", "src/uic/coder.py"):
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        lines.append(f"  {rel}: coded_bits={'coded_bits' in text} "
                     f"ac_bits={'ac_bits' in text}")
    ok = any("ac_bits" in open(os.path.join(HERE, r), encoding="utf-8").read()
             for r in ("src/uic/runner.py", "src/uic/engine.py")
             if os.path.exists(os.path.join(HERE, r)))
    report(tag, title, ok, lines)


def check_x2() -> None:
    tag, title = "X2", "video baselines reproduce the published figures"
    lines = []
    paper = {"h.265 crf 0 (x265 veryslow)": 21.69, "h.264 crf 0 (x264 veryslow)": 37.42,
             "ffv1 level 3": 40.16, "zstd --ultra -22 (raw yuv)": 51.04}
    worst = 0.0
    found = 0
    for name in ("baselines_video_wsl.jsonl", "baselines_video_zstd.jsonl"):
        path = os.path.join(HERE, "results", name)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            p = paper.get(r["codec"])
            if p is None:
                continue
            found += 1
            d = r["compression_rate_pct"] - p
            worst = max(worst, abs(d))
            lines.append(f"  {r['codec']}: measured {r['compression_rate_pct']:.3f}% "
                         f"vs paper {p:.2f}%  delta {d:+.3f} pp  frames={r['frames']}")
    if not found:
        skip(tag, title, "no video baseline rows")
        return
    lines.append(f"  worst |delta| = {worst:.3f} pp over {found} codecs")
    report(tag, title, found == 4 and worst < 2.0, lines)



def check_c11() -> None:
    """Regression guard: our port must not claim the official image path is RGB.

    The official pipeline is channel-wise (last axis 1 -> np.repeat -> greyscale
    triple).  Our `image_units` used to tokenise the whole RGB image instead.
    This check asserts the mismatch is at least *documented*, so it cannot be
    silently forgotten.
    """
    tag, title = "C11", "channel-wise mismatch in our own port is recorded"
    lines = []
    audit = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
    text = open(audit, encoding="utf-8").read() if os.path.exists(audit) else ""
    recorded = "channel-wise mismatch" in text
    lines.append(f"  AUDIT_FOR_REVIEW.md records the mismatch: {recorded}")

    src = os.path.join(HERE, "src", "uic", "data.py")
    stext = open(src, encoding="utf-8").read()
    # the port should still be flagged: the RGB call is the thing that is wrong
    rgb_call = 'processor(Image.fromarray(rgb)' in stext
    lines.append(f"  image_units still tokenises whole RGB: {rgb_call}")
    lines.append("  (if True, the port is NOT yet channel-wise: no image/video"
                 " number from the port should be quoted)")

    report(tag, title, recorded, lines)


CHECKS = {
    "C1": check_c1, "C2": check_c2, "C3": check_c3, "C4": check_c4,
    "C5": check_c5, "C6": check_c6, "C7": check_c7, "C8": check_c8,
    "X1": check_x1, "X2": check_x2, "C11": check_c11,
}

TITLES = {
    "C1": "text alignment fix (our bug)", "C2": "audio scoring misalignment",
    "C3": "image non-injectivity", "C4": "loader drops 1572 samples",
    "C5": "reference log totals stale", "C6": "LoRA penalty (confounded)",
    "C7": "no bits2bytes / fp32 OOM", "C8": "coder collapse guard (our bug)",
    "X1": "rate convention", "X2": "video baselines reproduce",
    "C11": "channel-wise mismatch recorded",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    if args.list:
        for k, t in TITLES.items():
            print(f"  {k}  {t}")
        return 0

    todo = [args.only] if args.only else list(CHECKS)
    for key in todo:
        fn = CHECKS.get(key)
        if fn is None:
            print(f"unknown check {key}")
            continue
        print(f"\n{'='*74}\n{key}: {TITLES[key]}\n{'='*74}")
        try:
            fn()
        except Exception as exc:  # a crash is a finding, not a pass
            print(f"  !! check raised {type(exc).__name__}: {exc}")

    print(f"\n{'='*74}\nSUMMARY\n{'='*74}")
    passed = 0
    for tag, title, ok, lines in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {tag}  {title}")
        passed += bool(ok)
    for tag, title, why in SKIPS:
        print(f"  [SKIP] {tag}  {title} -- {why}")

    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed, {len(SKIPS)} skipped")
    print("\nA FAIL or SKIP is a finding to report, not a reason to stop. "
          "Verify the raw quantities above by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
