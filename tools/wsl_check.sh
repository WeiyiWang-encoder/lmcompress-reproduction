#!/usr/bin/env bash
# Debug helper: verify the manifests resolve and the baseline binaries run.

: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
CAPSULE="$LMCOMPRESS_CAPSULE"
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

echo "=== manifest sample (audio) ==="
head -2 results/manifest_audio.txt | cat -A
p=$(head -1 results/manifest_audio.txt)
echo "p=[$p]"
if [ -f "$p" ]; then echo "EXISTS"; else echo "NOTFOUND"; fi
ls -la "$p" 2>&1 | head -2

echo
echo "=== manifest sample (text) ==="
head -1 results/manifest_text_medal.txt | cat -A
q=$(head -1 results/manifest_text_medal.txt)
if [ -f "$q" ]; then echo "EXISTS"; else echo "NOTFOUND"; fi

echo
echo "=== tool availability ==="
TP=$CAPSULE/third_party
for t in "$TP/ffmpeg-git-20240629-amd64-static/ffmpeg" \
         "$TP/OptimFROG_Linux_x64_5100/ofr" \
         "$TP/WavPack/cli/wavpack"; do
  if [ -x "$t" ]; then echo "OK   $t"; else echo "MISS $t"; fi
done
for c in zstd 7z 7zz zpaq brotli; do
  if command -v "$c" >/dev/null 2>&1; then echo "OK   $c -> $(command -v $c)"; else echo "MISS $c"; fi
done
python3 -c "import zlib,lzma,bz2,json; print('py stdlib ok')"
python3 -c "import brotli; print('py brotli ok')" 2>&1 | tail -1
