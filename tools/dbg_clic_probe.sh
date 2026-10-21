#!/usr/bin/env bash
# Debug helper: why does dimension probing return nothing on CLIC images?
set -u
# Resolved relative to this script; override with LMCOMPRESS_CAPSULE.
: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
REPO="$LMCOMPRESS_CAPSULE"
TP=$REPO/third_party/ffmpeg-git-20240629-amd64-static
IMG=$REPO/repro_understanding_is_compression/artifacts/clic/raw

echo "--- binaries ---"
ls -la "$TP/ffprobe" "$TP/ffmpeg" 2>&1 | sed 's/^/  /'

f=$(find "$IMG" -type f -iname '*.png' | sort | head -1)
echo "--- first image ---"
echo "  $f"
ls -la "$f" | sed 's/^/  /'

echo "--- ffprobe plain ---"
"$TP/ffprobe" -v error -select_streams v:0 -show_entries stream=width,height \
  -of csv=p=0 "$f" 2>&1 | sed 's/^/  /'
echo "  exit=$?"

echo "--- ffprobe with space separator ---"
"$TP/ffprobe" -v error -select_streams v:0 -show_entries stream=width,height \
  -of csv=p=0:s=' ' "$f" 2>&1 | sed 's/^/  /'

echo "--- ffprobe json ---"
"$TP/ffprobe" -v error -select_streams v:0 -show_entries stream=width,height \
  -of json "$f" 2>&1 | sed 's/^/  /'

echo "--- ffmpeg can it decode? ---"
"$TP/ffmpeg" -v error -y -i "$f" -frames:v 1 -f null - 2>&1 | sed 's/^/  /'
echo "  exit=$?"
