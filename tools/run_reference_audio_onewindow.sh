#!/usr/bin/env bash
# Run the AUTHORS' audio script unmodified, on one complete window.
#
# Why a copy: the capsule must stay byte-identical to what the authors published, so
# that its defects remain inspectable.  We therefore work in a temp tree that contains
# a copy of the capsule plus the reconstructed `samplings.py` (which the capsule fails
# to ship -- see tools/verify_reference_blockers.py).
#
# Why one window: a full 82 220-byte file cannot be processed by the shipped script at
# all.  split(..., 8160) yields 10 full windows plus a 620-byte tail; byte_collate pads
# that tail to 652 bytes, which is not a multiple of PATCH_SIZE=16, so the model's
# reshape raises.  A single 8 160-byte window has no tail and is therefore the largest
# input the reference script can actually complete.

set -u

# Resolved relative to this script; override with LMCOMPRESS_CAPSULE.
: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
REPO="$LMCOMPRESS_CAPSULE"
WORK=$CAPSULE/repro_understanding_is_compression/artifacts/refrun
PY="${PYTHON:-python3}"

rm -rf "$WORK"
mkdir -p "$WORK/capsule"
echo "copying capsule -> $WORK/capsule (excluding .venv and our repro folder)"
tar -C "$REPO" --exclude=.venv --exclude=repro_understanding_is_compression \
    -cf - . | tar -C "$WORK/capsule" -xf -

cp "$REPO/repro_understanding_is_compression/replacement_modules/samplings.py" \
   "$WORK/capsule/samplings.py"
echo "installed reconstructed samplings.py"

# one complete window == exactly one split block
mkdir -p "$WORK/one"
python3 - "$REPO/data/test_audios" "$WORK/one" <<'PY'
import os, sys, wave
src, dst = sys.argv[1], sys.argv[2]
for name in sorted(os.listdir(src)):
    if not name.lower().endswith(".wav"):
        continue
    p = os.path.join(src, name)
    with wave.open(p, "rb") as w:
        info = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        frames = w.readframes(8160)
    out = os.path.join(dst, name)
    with wave.open(out, "wb") as w:
        w.setnchannels(info[0]); w.setsampwidth(info[1]); w.setframerate(info[2])
        w.writeframes(frames)
    print(f"  {name}: wrote {len(frames)} bytes (one context window)")
PY

echo
echo "=== running the AUTHORS' script, unmodified ==="
cd "$WORK/capsule"
# The venv python is the WINDOWS interpreter, so every path handed to it must be a
# Windows path; wslpath -w performs the conversion.
"$PY" compress_with_bgpt_or_llama.py \
      --data_path "$(wslpath -w "$WORK/one")" \
      --data_type wav \
      --model_path "$(wslpath -w "$REPO/data/model/bgpt/weights-audio.pth")" \
      --context_window 8160 \
      --batch_size 1 2>&1 | tail -40
