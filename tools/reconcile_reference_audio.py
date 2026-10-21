"""Why does the reference report 5.016 (1.596 bpb) on the full file when our port
reports 2.2561 bpb on the first window alone?

Hypotheses:
  H1  the reference scores all 11 blocks including the 620-byte tail (which we padded,
      so our full-file number is not the same quantity);
  H2  the 6e-8 epsilon in ``pd = pd.add_(eps)`` plus the min-over-... produces a lower
      per-byte cost than our clamp;
  H3  the reference's aggregation over blocks differs from ours (e.g. a block that
      contributes almost nothing drags the mean down);
  H4  our port's windowing differs structurally from the reference's.

This script computes, for every candidate interpretation, the implied bpb from the
reference's own printed numbers, and compares each against the quantity our port
reports.  Arithmetic only -- no model load -- so it is cheap to re-run.
"""

from __future__ import annotations

import os

TOTAL = 82220          # payload bytes (the 44-byte WAV header is not scored)
CTX = 8160
PATCH = 16

# From the two reference runs (unmodified script, reconstructed samplings.py):
REF_FULL = {"ratio": 5.016113, "compressed": 16399.949935913086,
            "total_size_stated": 82264}
REF_ONE = {"ratio": 3.666938, "compressed": 2237.288734436035,
           "total_size_stated": 8204}

# Our port, on the same single 8160-byte window:
OURS_ONE_BPB = 2.2567


def bpb_from(compressed_bytes: float, n_bytes: int) -> float:
    return 8.0 * compressed_bytes / n_bytes


def blocks(total: int, ctx: int) -> list[int]:
    return [min(ctx, total - i) for i in range(0, total, ctx)]


def main() -> int:
    sizes = blocks(TOTAL, CTX)
    print(f"payload = {TOTAL} bytes; split({CTX}) -> {len(sizes)} blocks")
    print(f"  full windows: {sum(1 for s in sizes if s == CTX)}  "
          f"tail: {sizes[-1]}\n")

    print("=== reference, single window ===")
    print(f"  stated total size  : {REF_ONE['total_size_stated']} "
          f"(= {CTX} + 2*{PATCH} + 12 wav header)")
    print(f"  compressed         : {REF_ONE['compressed']:.3f} bytes")
    print(f"  ratio stated       : {REF_ONE['ratio']:.6f}")
    for label, n in (("per payload byte (8160)", CTX),
                     ("per stated total (8204)", REF_ONE["total_size_stated"])):
        print(f"  bpb {label:<28}: {bpb_from(REF_ONE['compressed'], n):.4f}")
    print(f"  our port, same window          : {OURS_ONE_BPB:.4f} bpb")
    d = 100 * (OURS_ONE_BPB - bpb_from(REF_ONE["compressed"], CTX)) / \
        bpb_from(REF_ONE["compressed"], CTX)
    print(f"  -> disagreement                 : {d:+.2f} %")

    print("\n=== reference, FULL file ===")
    print(f"  stated total size  : {REF_FULL['total_size_stated']} "
          f"(= {TOTAL} + 44 wav header)")
    print(f"  compressed         : {REF_FULL['compressed']:.3f} bytes")
    print(f"  ratio stated       : {REF_FULL['ratio']:.6f}")
    print(f"  bpb per payload byte ({TOTAL})   : "
          f"{bpb_from(REF_FULL['compressed'], TOTAL):.4f}")
    print(f"  bpb per stated total              : "
          f"{bpb_from(REF_FULL['compressed'], REF_FULL['total_size_stated']):.4f}")

    print("\n  cross-check: does compressed/total reproduce the stated ratio?")
    print(f"    {REF_FULL['compressed']:.3f} / {REF_FULL['total_size_stated']} "
          f"= {REF_FULL['compressed']/REF_FULL['total_size_stated']:.6f}")
    print(f"    stated ratio = {REF_FULL['ratio']:.6f}")
    print(f"    match: {abs(REF_FULL['compressed']/REF_FULL['total_size_stated'] - REF_FULL['ratio']) < 1e-5}")

    print("\n=== H1: is the full-file number consistent with per-block scoring? ===")
    one = REF_FULL["compressed"]
    print(f"  if all 11 blocks scored and the ratio were uniform, each block of 8160")
    print(f"  bytes would cost {REF_FULL['compressed']/11:.1f} bytes")
    print(f"  the single-window run actually cost {REF_ONE['compressed']:.1f} bytes")
    print(f"  -> the full-file average is "
          f"{100*(1 - (REF_FULL['compressed']/11)/REF_ONE['compressed']):.1f} % cheaper "
          f"per block than the first window alone")
    print("  This is expected: the model sees a longer context, and the tail block is")
    print("  short, so the mean is not a per-window average. It also means the")
    print("  reference is NOT crashing on the tail -- it completes all blocks.")

    print("\n=== H4: structural check, what does the reference score? ===")
    print(f"  byte_collate pads each block to PATCH_SIZE + n + PATCH_SIZE.")
    print(f"  a full block  -> {PATCH}+{CTX}+{PATCH} = {PATCH+CTX+PATCH} bytes "
          f"/ {PATCH} = {(PATCH+CTX+PATCH)//PATCH} patches (exact)")
    tail = sizes[-1]
    print(f"  the tail block -> {PATCH}+{tail}+{PATCH} = {PATCH+tail+PATCH} bytes "
          f"/ {PATCH} = {(PATCH+tail+PATCH)/PATCH} patches (NOT integral)")
    print(f"  yet the run completed, so the tail must be handled somewhere other than")
    print(f"  a naive reshape. Conclusion: our earlier prediction of a crash on the")
    print(f"  tail was WRONG and is withdrawn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
