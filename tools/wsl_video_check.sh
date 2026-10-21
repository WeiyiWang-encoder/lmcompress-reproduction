#!/usr/bin/env bash
# Verify the video toolchain before a long encode.

: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
CAPSULE="$LMCOMPRESS_CAPSULE"
set -u
R=$CAPSULE/repro_understanding_is_compression
V=$CAPSULE/data/test_videos/4k_aerial_first_second.y4m

echo "=== y4m_info strict frame count ==="
python3 "$R/tools/y4m_info.py" "$V" 0
echo "exit=$?"

echo
echo "=== zstd (Windows exe reached from WSL) ==="
Z="${ZSTD:-zstd}"
if [ -x "$Z" ]; then
  echo "executable: yes"
  "$Z" --version 2>&1 | head -2
else
  echo "executable: NO"
  ls -la "$Z" 2>&1 | head -2
  echo "-- fallback: WSL-side zstd? --"
  command -v zstd || echo "none in WSL"
fi

echo
echo "=== ffmpeg encoders present ==="
FF=$CAPSULE/third_party/ffmpeg-git-20240629-amd64-static/ffmpeg
"$FF" -hide_banner -encoders 2>/dev/null | grep -E "libx264|libx265|ffv1" | head -5
