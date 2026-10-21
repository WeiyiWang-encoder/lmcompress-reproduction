#!/usr/bin/env bash
# Video baselines from the capsule's static ffmpeg, run under WSL.
#
# The paper's video baselines are lossless H.264, H.265 and FFV1 (Supplementary
# Table 1-2: `crf: 0`; zstd with its own settings).  The shipped clip is raw Y4M,
# which is exactly what a codec wants, so we feed it directly and let the codec use
# interframe redundancy -- that is the fair comparison, and it is what the paper's
# H.264/H.265 numbers must have come from.
#
# Usage: baselines_video.sh <manifest> <frames> <out.jsonl>
#   manifest : one WSL path per line (Y4M files)
#   frames   : number of frames to encode (0 = all)
#
# Emits one JSON object per (codec, file).  Progress goes to stderr so the JSONL on
# stdout stays clean and append-only -- a power cut therefore leaves usable partial
# results.

set -u

# Resolved relative to this script; override with LMCOMPRESS_CAPSULE.
: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
REPO="$LMCOMPRESS_CAPSULE"
TP="$REPO/third_party"
FF="$TP/ffmpeg-git-20240629-amd64-static/ffmpeg"
ZSTD_WIN="${ZSTD_WIN:-$(command -v zstd)}"

MANIFEST="$1"
FRAMES="$2"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

emit() { # codec unit orig comp secs frames extra_json
  python3 - "$@" <<'PY'
import json, sys
codec, unit, orig, comp, secs, frames, extra = (sys.argv[1], sys.argv[2],
    int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]),
    sys.argv[7] if len(sys.argv) > 7 else "{}")
row = {"codec": codec, "unit": unit,
       "original_bytes": orig, "compressed_bytes": comp,
       "compression_rate_pct": round(100.0*comp/orig, 6) if orig else None,
       "compression_ratio": round(orig/comp, 6) if comp else None,
       "seconds": round(secs, 3), "frames": frames}
row.update(json.loads(extra))
print(json.dumps(row, sort_keys=True), flush=True)
PY
}

timed() {
  local t0 t1
  t0=$(date +%s.%N)
  "$@" >/dev/null 2>&1 || return 1
  t1=$(date +%s.%N)
  awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.3f", a-b}'
}

while IFS= read -r path; do
  path=${path%$'\r'}
  [ -z "$path" ] && continue
  [ -f "$path" ] || { echo "MISSING $path" >&2; continue; }
  unit=$(basename "$path")

  # raw YUV420 size from the Y4M header, times the number of frames we encode.
  # The frame count comes from STRICT parsing (tools/y4m_info.py): an arithmetic
  # heuristic (filesize // (frame_bytes + 6)) returns 59 for this 60-frame file
  # because the header is 60 bytes, and a wrong frame count makes every rate here
  # wrong by 1/60.
  read -r W H NF NTOT < <(python3 "$(dirname "$0")/y4m_info.py" "$path" "$FRAMES")
  ORIG=$(( W * H * 3 / 2 * NF ))
  EXTRA="{\"width\":$W,\"height\":$H,\"frames_in_file\":$NTOT,\"raw_bytes_per_frame\":$(( W * H * 3 / 2 ))}"
  echo "  $unit: ${W}x${H}, file has $NTOT frames, encoding $NF, raw = $ORIG bytes" >&2

  LIMIT=()
  [ "$FRAMES" -gt 0 ] && LIMIT=(-frames:v "$FRAMES")

  # ---- H.264 lossless ---------------------------------------------------- #
  if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" "${LIMIT[@]}" \
             -c:v libx264 -crf 0 -preset veryslow -pix_fmt yuv420p \
             "$WORK/h264.mkv"); then
    emit "h.264 crf 0 (x264 veryslow)" "$unit" "$ORIG" \
         "$(stat -c %s "$WORK/h264.mkv")" "$s" "$NF" "$EXTRA"
  else
    echo "  h.264 FAILED" >&2
  fi

  # ---- H.265 lossless ---------------------------------------------------- #
  if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" "${LIMIT[@]}" \
             -c:v libx265 -crf 0 -preset veryslow -pix_fmt yuv420p \
             -x265-params log-level=none "$WORK/h265.mkv"); then
    emit "h.265 crf 0 (x265 veryslow)" "$unit" "$ORIG" \
         "$(stat -c %s "$WORK/h265.mkv")" "$s" "$NF" "$EXTRA"
  else
    echo "  h.265 FAILED" >&2
  fi

  # ---- FFV1 lossless ----------------------------------------------------- #
  if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" "${LIMIT[@]}" \
             -c:v ffv1 -level 3 "$WORK/ffv1.mkv"); then
    emit "ffv1 level 3" "$unit" "$ORIG" \
         "$(stat -c %s "$WORK/ffv1.mkv")" "$s" "$NF" "$EXTRA"
  else
    echo "  ffv1 FAILED" >&2
  fi

  # ---- zstd over the raw YUV (the paper also lists zstd for video) ------- #
  if [ -x "$ZSTD_WIN" ]; then
    # NB: the intermediate must NOT be named *.y4m.  ffmpeg infers the muxer from
    # the output extension, so naming it raw.y4m made it pick the y4m muxer while
    # -f yuv4mpegpipe was also given, and the leg silently produced nothing.  That
    # is why the earlier run emitted no zstd row.
    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" "${LIMIT[@]}" \
               -c:v rawvideo -f yuv4mpegpipe "$WORK/raw.yuv"); then
      t0=$(date +%s.%N)
      if "$ZSTD_WIN" --ultra -22 -q -f "$WORK/raw.yuv" -o "$WORK/raw.zst" >/dev/null 2>&1; then
        t1=$(date +%s.%N)
        zs=$(awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.3f", a-b}')
        tot=$(awk -v a="$s" -v b="$zs" 'BEGIN{printf "%.3f", a+b}')
        emit "zstd --ultra -22 (raw yuv)" "$unit" "$ORIG" \
             "$(stat -c %s "$WORK/raw.zst")" "$tot" "$NF" "$EXTRA"
      else
        echo "  zstd FAILED on $WORK/raw.yuv" >&2
      fi
    else
      echo "  raw yuv export FAILED" >&2
    fi
  else
    echo "  zstd not executable at $ZSTD_WIN" >&2
  fi
done < "$MANIFEST"
