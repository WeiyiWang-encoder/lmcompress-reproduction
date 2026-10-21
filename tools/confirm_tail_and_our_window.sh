#!/usr/bin/env bash
# Two confirmations against the AUTHORS' unmodified script.
#
#  Part 1: the full 82 220-byte file cannot be processed.  split(..., 8160) gives 10
#          full windows plus a 620-byte tail; byte_collate pads the tail to 652 bytes,
#          which is not a multiple of PATCH_SIZE=16, so the model raises.  We expect a
#          runtime error here, and the exact error message is the evidence.
#
#  Part 2: our port, on the SAME 8 160-byte window the reference just completed, so the
#          two implementations can be compared on identical input.

set -u

# Resolved relative to this script; override with LMCOMPRESS_CAPSULE.
: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
REPO="$LMCOMPRESS_CAPSULE"
R="$REPO/repro_understanding_is_compression"
WORK="$R/artifacts/refrun"
PY="$REPO/.venv/Scripts/python.exe"

echo "==================================================================="
echo "Part 1: full file through the authors' script (expected to FAIL)"
echo "==================================================================="
cd "$WORK/capsule"
"$PY" compress_with_bgpt_or_llama.py \
      --data_path "$(wslpath -w "$REPO/data/test_audios")" \
      --data_type wav \
      --model_path "$(wslpath -w "$REPO/data/model/bgpt/weights-audio.pth")" \
      --context_window 8160 \
      --batch_size 1 2>&1 | tail -25
echo "(exit above is the script's own; a traceback is the expected outcome)"

echo
echo "==================================================================="
echo "Part 2: our port on the same single window"
echo "==================================================================="
cd "$R"
PYTHONPATH=src "$PY" tools/measure_one_window.py 2>&1 | tail -15
