#!/usr/bin/env bash
# Video baseline: zstd over raw YUV, using the Windows zstd binary from WSL.
#
# Two traps this script exists to avoid, both hit during development:
#
#  1. ffmpeg infers the muxer from the *output* extension.  Naming the
#     intermediate ``raw.y4m`` while also passing ``-f yuv4mpegpipe`` made the leg
#     produce nothing, and because the failure was inside a ``$( ... )`` capture it
#     was silent.
#  2. ``zstd.exe`` is a **Windows** binary.  WSL paths such as ``/tmp/x`` or
#     ``a POSIX path`` must be converted with ``wslpath -w`` before being passed to it,
#     otherwise zstd reports "No such file or directory" for a file that plainly
#     exists.
#
# Usage: baselines_video_zstd.sh <manifest> <frames> <out.jsonl>

set -u

# Resolved relative to this script; override with LMCOMPRESS_CAPSULE.
: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
REPO="$LMCOMPRESS_CAPSULE"
FF="$REPO/third_party/ffmpeg-git-20240629-amd64-static/ffmpeg"
ZSTD="${ZSTD:-zstd}"
# A Windows-visible scratch dir, so zstd.exe can read what ffmpeg wrote.
WORK="$REPO/repro_understanding_is_compression/artifacts/tmp_video"
mkdir -p "$WORK"

MANIFEST="$1"
FRAMES="$2"
OUT="$3"

# wslpath -w converts a POSIX path into a Windows path for the Windows binary.  Spaces must
# survive, so every use is quoted.
winpath() { wslpath -w "$1"; }

emit() {
  python3 - "$@" <<'PY'
import json, sys
codec, unit, orig, comp, secs, frames, extra = (sys.argv[1], sys.argv[2],
    int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]),
    sys.argv[7] if len(sys.argv) > 7 else "{}")
row = {"codec": codec, "unit": unit, "original_bytes": orig,
       "compressed_bytes": comp,
       "compression_rate_pct": round(100.0*comp/orig, 6) if orig else None,
       "compression_ratio": round(orig/comp, 6) if comp else None,
       "seconds": round(secs, 3), "frames": frames}
row.update(json.loads(extra))
print(json.dumps(row, sort_keys=True), flush=True)
PY
}

while IFS= read -r path; do
  path=${path%$'\r'}
  [ -z "$path" ] && continue
  [ -f "$path" ] || { echo "MISSING $path" >&2; continue; }
  unit=$(basename "$path")

  read -r W H NF NTOT < <(python3 "$(dirname "$0")/y4m_info.py" "$path" "$FRAMES")
  ORIG=$(( W * H * 3 / 2 * NF ))
  EXTRA="{\"width\":$W,\"height\":$H,\"frames_in_file\":$NTOT,\"path_form\":\"raw YUV420\"}"
  echo "  $unit: ${W}x${H}, $NTOT frames in file, encoding $NF" >&2

  LIMIT=()
  [ "$FRAMES" -gt 0 ] && LIMIT=(-frames:v "$FRAMES")

  RAW="$WORK/raw_${unit}.yuv"
  ZST="$WORK/raw_${unit}.zst"

  t0=$(date +%s.%N)
  if ! "$FF" -y -hide_banner -loglevel error -i "$path" "${LIMIT[@]}" \
        -c:v rawvideo -f yuv4mpegpipe "$RAW"; then
    echo "  ffmpeg raw export FAILED" >&2
    continue
  fi
  t1=$(date +%s.%N)

  if ! "$ZSTD" --ultra -22 -q -f "$(winpath "$RAW")" -o "$(winpath "$ZST")" >/dev/null 2>&1; then
    echo "  zstd FAILED (windows paths: $(winpath "$RAW") -> $(winpath "$ZST"))" >&2
    rm -f "$RAW"
    continue
  fi
  t2=$(date +%s.%N)

  total=$(awk -v a="$t2" -v b="$t0" 'BEGIN{printf "%.3f", a-b}')
  emit "zstd --ultra -22 (raw yuv)" "$unit" "$ORIG" "$(stat -c %s "$ZST")" \
       "$total" "$NF" "$EXTRA"
  echo "  zstd: $(stat -c %s "$ZST") bytes, ${total}s" >&2
  rm -f "$RAW" "$ZST"
done < "$MANIFEST"
