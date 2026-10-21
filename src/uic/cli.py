"""Command-line entry point for the LMCompress reproduction.

Examples
--------
Audio (bGPT, the paper's headline 18.26 % on LJSpeech)::

    python -m uic.cli audio --limit 1 --verify-roundtrip

Text (Llama-3-8B + the shipped LoRA adapter), local subset::

    python -m uic.cli text --dataset medal --limit 3 --context-window 4096

Images (ImageGPT), including the honest lossless accounting::

    python -m uic.cli image --limit 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import List, Optional

def _default_repo_root() -> str:
    """Capsule checkout, from ``UIC_REPO_ROOT`` or the shared resolver."""
    env = os.environ.get("UIC_REPO_ROOT")
    if env:
        return env
    from .paths import capsule_root

    return capsule_root()


REPO_ROOT = _default_repo_root()
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _paths():
    return {
        "root": HERE,
        "results": os.path.join(HERE, "results"),
        "cursors": os.path.join(HERE, "runs", "cursors"),
        "logs": os.path.join(HERE, "logs"),
        "models": os.path.join(REPO_ROOT, "data", "model"),
        "data": os.path.join(REPO_ROOT, "data"),
    }


def _ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _variant_tag(**kwargs) -> str:
    """Build a filesystem-safe tag identifying a run configuration.

    Resumable cursors are keyed by this tag.  Without it, two runs that differ
    only by LoRA adapter or context length would share cursors and the second run
    would silently reuse the first run's numbers.
    """
    parts = []
    for key, value in sorted(kwargs.items()):
        if value is None or value is False:
            continue
        label = f"{key}={value}" if value is not True else key
        parts.append(str(label))
    return "__".join(parts) or "default"


# --------------------------------------------------------------------------- #
# Audio: bGPT over raw WAV bytes
# --------------------------------------------------------------------------- #


def run_audio(args) -> int:
    from .data import audio_units
    from .models import BGPTBytePredictor
    from .runner import run_batch, summarize

    p = _paths()
    weights = args.model_path or os.path.join(p["models"], "bgpt", "weights-audio.pth")
    units = audio_units(args.data_path, limit=args.limit, use_8bit=args.to_8bit)
    if not units:
        print(f"no .wav files under {args.data_path}")
        return 1
    print(f"loaded {len(units)} audio file(s), {sum(len(u.payload) for u in units)} bytes total")

    predictor = BGPTBytePredictor.load(
        REPO_ROOT, weights, device=args.device, extension=b"wav"
    )
    out = os.path.join(p["results"], args.out or f"audio_bgpt_{_ts()}.jsonl")
    try:
        records = run_batch(
            predictor,
            units,
            out,
            # The row mapping changed from the withdrawn contiguous-logit
            # selector to the reference-compatible 17-slot selector.  Keep its
            # checkpoints separate so a resumed run can never reuse old totals.
            os.path.join(p["cursors"], "audio_bgpt_rows17_tailpad_v1"),
            verify_roundtrip=args.verify_roundtrip,
            limit=args.limit,
        )
    finally:
        predictor.release()
    print(json.dumps(summarize(records or _read(out)), indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Image: ImageGPT over quantised pixel tokens
# --------------------------------------------------------------------------- #


def run_image(args) -> int:
    from .data import image_units
    from .models import ImageGPTPredictor
    from .runner import run_batch, summarize

    p = _paths()
    model_dir = args.model_path or os.path.join(p["models"], "imagegpt-small")
    units = image_units(
        args.data_path, model_dir, limit=args.limit, include_residual=True
    )
    if not units:
        print(f"no images under {args.data_path}")
        return 1
    print(f"loaded {len(units)} channel-unit(s) from {len(units)//3} image(s)")

    predictor = ImageGPTPredictor.load(model_dir, device=args.device)
    out = os.path.join(p["results"], args.out or f"image_igpt_{_ts()}.jsonl")
    try:
        records = run_batch(
            predictor,
            units,
            out,
            os.path.join(p["cursors"], "image_igpt"),
            verify_roundtrip=args.verify_roundtrip,
        )
    finally:
        predictor.release()
    summary = summarize(records or _read(out))

    # Honest accounting for the path this CLI implements.
    #
    # IMPORTANT SCOPE NOTE.  This CLI's image path is the *whole-image* one: it
    # runs an image through the ImageGPT processor once (which resizes to 32x32 and
    # quantises to 512 colours) and codes the resulting token grid.  That is NOT
    # the path the reference implementation uses -- the official path is
    # patch-wise and channel-wise -- and the residual figure this block reports was
    # withdrawn as evidence for C7 for exactly that reason (see
    # docs/AUDIT_FOR_REVIEW.md §4.4.3).  It is retained because the measurement is
    # still correctly labelled for *this* path, but it must not be quoted as a
    # property of the official pipeline or of the paper.
    if args.charge_residual and records:
        import zlib

        residual_total = 0
        for unit in units:
            raw = unit.meta.get("residual") or b""
            if raw:
                residual_total += len(zlib.compress(bytes(raw), 9))
        if residual_total:
            orig = summary["original_bytes"]
            model_bytes = summary["coded_bits"] / 8.0
            total = model_bytes + residual_total
            summary["exact_reconstruction_completion"] = {
                "WARNING": (
                    "Whole-image path only. NOT the official patch-wise path and "
                    "NOT evidence for the paper's image result. Withdrawn as a "
                    "claim; see docs/AUDIT_FOR_REVIEW.md section 4.4.3."
                ),
                "model_bytes": model_bytes,
                "residual_bytes_zlib9": residual_total,
                "total_bytes": total,
                "compression_rate_pct": 100.0 * total / orig,
                "compression_ratio": orig / total,
                "note": (
                    "Illustrative exact-reconstruction completion for THIS path: "
                    "de-quantise the token grid, upscale to the original size with "
                    "bilinear interpolation, code the signed residual with zlib -9. "
                    "This is a specific completion, not a lower bound; a better "
                    "residual model would do better."
                ),
            }
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Video: frames -> ImageGPT
# --------------------------------------------------------------------------- #


def run_video(args) -> int:
    from .data import video_units
    from .models import ImageGPTPredictor
    from .runner import run_batch, summarize

    p = _paths()
    model_dir = args.model_path or os.path.join(p["models"], "imagegpt-small")
    work = os.path.join(p["root"], "artifacts", "frames", os.path.basename(args.data_path))
    units = video_units(
        args.data_path, model_dir, work, frame_limit=args.frames, include_residual=False
    )
    if not units:
        print(f"no frames extracted from {args.data_path}")
        return 1
    print(f"loaded {len(units)} channel-unit(s) from {len(units)//3} frame(s)")

    predictor = ImageGPTPredictor.load(model_dir, device=args.device)
    out = os.path.join(p["results"], args.out or f"video_igpt_{_ts()}.jsonl")
    try:
        records = run_batch(
            predictor,
            units,
            out,
            os.path.join(p["cursors"], "video_igpt"),
            verify_roundtrip=args.verify_roundtrip,
        )
    finally:
        predictor.release()
    print(json.dumps(summarize(records or _read(out)), indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Text: Llama (+LoRA) over BPE tokens
# --------------------------------------------------------------------------- #


def run_text(args) -> int:
    from .data import text_units
    from .models import LlamaTokenPredictor, sparse_true_symbol_cdf  # noqa: F401
    from .runner import run_batch, summarize

    p = _paths()
    model_path = args.model_path or os.path.join(p["models"], "finetuned_llama", "Llama-3-8B")
    lora_path = args.lora_path
    if lora_path is None and not args.no_lora:
        lora_path = os.path.join(p["models"], "finetuned_llama", "Llama-3-8B_lora")

    data_path = args.data_path or os.path.join(
        p["data"], "text", "MEDAL_sample" if args.dataset == "medal" else "EURLEX_sample"
    )

    print(f"loading Llama from {model_path} (4bit={not args.no_4bit}) lora={lora_path}")
    predictor = LlamaTokenPredictor.load(
        model_path,
        lora_path=None if args.no_lora else lora_path,
        device=args.device,
        load_in_4bit=not args.no_4bit,
        context_window=args.context_window,
        merge_lora=not args.no_merge,
    )
    units = text_units(data_path, tokenizer=predictor, limit=args.limit)
    if not units:
        print(f"no text files under {data_path}")
        predictor.release()
        return 1
    total_tokens = sum(len(u.payload) for u in units)
    print(f"loaded {len(units)} text file(s), {total_tokens} tokens")

    # The rolling window needs the whole token sequence to look backwards.
    # Cursors are keyed by configuration, so a run with a different adapter or
    # context length cannot accidentally inherit another run's numbers.
    tag = _variant_tag(
        lora="none" if args.no_lora else "medal",
        q4=not args.no_4bit,
        ctx=args.context_window,
        ds=args.dataset,
    )
    out = os.path.join(p["results"], args.out or f"text_{args.dataset}_{tag}_{_ts()}.jsonl")
    records = []
    try:
        for i, unit in enumerate(units, 1):
            predictor.set_document(unit.payload)
            print(f"[{i}/{len(units)}] coding {unit.name} ({len(unit.payload)} tokens)")
            rec = run_batch(
                predictor,
                [unit],
                out,
                os.path.join(p["cursors"], f"text_llama_{tag}"),
                verify_roundtrip=args.verify_roundtrip,
            )
            records.extend(rec)
    finally:
        predictor.release()
    print(json.dumps(summarize(records or _read(out)), indent=2))
    return 0


def _read(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uic", description="LMCompress reproduction")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--data-path", type=str, default=None)
        sp.add_argument("--model-path", type=str, default=None)
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--device", type=str, default="cuda:0")
        sp.add_argument("--out", type=str, default=None)
        sp.add_argument("--verify-roundtrip", action="store_true")

    sp = sub.add_parser("audio", help="bGPT over WAV bytes")
    common(sp)
    sp.add_argument("--to-8bit", action="store_true", help="apply the paper's 8-bit reduction")
    sp.set_defaults(func=run_audio, data_path=os.path.join(REPO_ROOT, "data", "test_audios"))

    sp = sub.add_parser("image", help="ImageGPT over quantised pixels")
    common(sp)
    sp.add_argument("--charge-residual", action="store_true", default=True)
    sp.set_defaults(func=run_image, data_path=os.path.join(REPO_ROOT, "data", "test_images"))

    sp = sub.add_parser("video", help="ImageGPT over extracted frames")
    common(sp)
    sp.add_argument("--frames", type=int, default=3)
    sp.set_defaults(func=run_video, data_path=os.path.join(REPO_ROOT, "data", "test_videos"))

    sp = sub.add_parser("text", help="Llama(+LoRA) over BPE tokens")
    common(sp)
    sp.add_argument("--dataset", choices=["medal", "eurlex"], default="medal")
    sp.add_argument("--lora-path", type=str, default=None)
    sp.add_argument("--no-lora", action="store_true")
    sp.add_argument("--no-4bit", action="store_true")
    sp.add_argument("--no-merge", action="store_true")
    sp.add_argument("--context-window", type=int, default=4096)
    sp.set_defaults(func=run_text)

    sp = sub.add_parser("preflight", help="check environment and assets")
    sp.set_defaults(func=lambda a: __import__("uic.preflight", fromlist=["main"]).main())

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted -- progress is checkpointed, re-run to resume")
        return 130
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
