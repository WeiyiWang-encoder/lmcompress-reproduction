#!/usr/bin/env bash
# Debug the zstd leg of the video baselines.

: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
CAPSULE="$LMCOMPRESS_CAPSULE"
set -u
FF=$CAPSULE/third_party/ffmpeg-git-20240629-amd64-static/ffmpeg
Z="${ZSTD:-zstd}"
V=$CAPSULE/data/test_videos/4k_aerial_first_second.y4m
W=$(mktemp -d)
echo "workdir $W"

echo "--- step 1: ffmpeg rawvideo export ---"
"$FF" -y -hide_banner -loglevel error -i "$V" -c:v rawvideo -f yuv4mpegpipe "$W/raw.yuv"
echo "ffmpeg exit=$?"
ls -la "$W/" | head -5

echo "--- step 2: zstd ---"
if [ -x "$Z" ]; then
  "$Z" --ultra -22 -q -f "$W/raw.yuv" -o "$W/raw.zst"
  echo "zstd exit=$?"
  ls -la "$W/"
else
  echo "zstd not executable"
fi

echo "--- step 3: compare with plain zstd on the original .y4m for sanity ---"
"$Z" --ultra -22 -q -f "$V" -o "$W/orig.zst" 2>&1 | tail -2
echo "exit=$?"
ls -la "$W/" 2>/dev/null | tail -4
rm -rf "$W"
