#!/usr/bin/env bash
# Classical baselines that require Linux binaries, run under WSL.
#
# These are the codecs the paper used that have no working Windows equivalent:
# the static ffmpeg build (png/webp/jpeg-xl/jpeg-2000/flac/ffv1/h.264/h.265),
# the WavPack CLI, and OptimFROG.  Python-native baselines (zstd, lzma/7z,
# brotli, zlib) live in tools/baselines_python.py.
#
# Tool settings follow the capsule's own run.sh and the paper's Supplementary
# Tables 1-2 verbatim.
#
# Usage: baselines_linux.sh <modality> <manifest>
# Manifest: one WSL path per line, LF endings.

set -u

# Resolved relative to this script; override with LMCOMPRESS_CAPSULE.
: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
REPO="$LMCOMPRESS_CAPSULE"
TP="$REPO/third_party"
FF="$TP/ffmpeg-git-20240629-amd64-static/ffmpeg"
FFPROBE="$TP/ffmpeg-git-20240629-amd64-static/ffprobe"
OFR="$TP/OptimFROG_Linux_x64_5100/ofr"
WV="$TP/WavPack/cli/wavpack"

MODALITY="$1"
MANIFEST="$2"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

emit() { # codec unit orig comp seconds extra
  python3 - "$@" <<'PY'
import json, sys
codec, unit, orig, comp, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else {}
row = {"codec": codec, "unit": unit, "original_bytes": orig,
       "compressed_bytes": comp,
       "compression_rate_pct": round(100.0*comp/orig, 6) if orig else None,
       "compression_ratio": round(orig/comp, 6) if comp else None,
       "seconds": round(secs, 3)}
row.update(extra)
print(json.dumps(row, sort_keys=True))
PY
}

timed() { # timed <cmd...>
  local t0 t1
  t0=$(date +%s.%N)
  "$@" >/dev/null 2>&1 || return 1
  t1=$(date +%s.%N)
  awk -v a="$t1" -v b="$t0" 'BEGIN{printf "%.3f", a-b}'
}

while IFS= read -r path; do
  path=${path%$'\r'}
  [ -z "$path" ] && continue
  if [ ! -f "$path" ]; then echo "MISSING $path" >&2; continue; fi
  unit=$(basename "$path")

  case "$MODALITY" in

  audio)
    # Denominator = raw PCM payload (not the container), matching the capsule's
    # own accounting for audio.
    orig=$(python3 -c "
import sys, wave
with wave.open(sys.argv[1],'rb') as f:
    print(f.getnframes()*f.getnchannels()*f.getsampwidth())" "$path")

    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" \
             -compression_level 12 "$WORK/o.flac"); then
      emit "flac -compression_level 12" "$unit" "$orig" "$(stat -c %s "$WORK/o.flac")" "$s" "{}"
    fi
    if s=$(timed "$WV" -y -hh -x3 -q "$path" -o "$WORK/o.wv"); then
      emit "wavpack -hh -x3" "$unit" "$orig" "$(stat -c %s "$WORK/o.wv")" "$s" "{}"
    fi
    # OptimFROG's CLI is `ofr [command] [options] src --output dst`; unlike FLAC
    # it has no `-o`, and passing one makes it treat the flag as an input file.
    if s=$(timed "$OFR" --preset max "$path" --output "$WORK/o.ofr"); then
      emit "optimfrog --preset max" "$unit" "$orig" "$(stat -c %s "$WORK/o.ofr")" "$s" "{}"
    fi
    ;;

  image)
    # Denominator = raw RGB pixels, the capsule's convention for images.
    # Dimensions come from ffprobe so this does not depend on PIL inside WSL.
    read -r w h < <("$FFPROBE" -v error -select_streams v:0 \
                      -show_entries stream=width,height -of csv=p=0 "$path" | tr ',' ' ')
    orig=$(( w * h * 3 ))
    echo "  image ${w}x${h} -> ${orig} raw bytes" >&2

    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" \
             -compression_level 100 "$WORK/o.png"); then
      emit "png (ffmpeg cl100)" "$unit" "$orig" "$(stat -c %s "$WORK/o.png")" "$s" "{}"
    fi
    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" \
             -c:v libwebp -lossless 1 -q:v 100 "$WORK/o.webp"); then
      emit "webp -lossless -q100" "$unit" "$orig" "$(stat -c %s "$WORK/o.webp")" "$s" "{}"
    fi
    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" \
             -c:v libjxl -distance 0 "$WORK/o.jxl"); then
      emit "jpeg-xl -distance 0" "$unit" "$orig" "$(stat -c %s "$WORK/o.jxl")" "$s" "{}"
    fi
    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" \
             -c:v jpeg2000 -pred 1 "$WORK/o.jp2"); then
      emit "jpeg-2000 lossless" "$unit" "$orig" "$(stat -c %s "$WORK/o.jp2")" "$s" "{}"
    fi
    ;;

  video)
    # Denominator = raw YUV420 frame bytes, computed from the Y4M header.
    read -r w h nf < <(python3 -c "
import os, sys
p = sys.argv[1]
with open(p,'rb') as f:
    head = f.readline().decode('ascii','replace').split()
w = h = 0
for t in head:
    if t.startswith('W'): w = int(t[1:])
    elif t.startswith('H'): h = int(t[1:])
frame = w*h*3//2
size = os.path.getsize(p)
print(w, h, max(1, size//(frame+6)))" "$path")
    orig=$(( w * h * 3 / 2 * nf ))

    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" \
             -c:v libx264 -crf 0 -preset veryslow -pix_fmt yuv420p "$WORK/o264.mkv"); then
      emit "h.264 crf 0 (x264 veryslow)" "$unit" "$orig" "$(stat -c %s "$WORK/o264.mkv")" "$s" \
           "{\"width\":$w,\"height\":$h,\"frames\":$nf}"
    fi
    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" \
             -c:v libx265 -crf 0 -preset veryslow -pix_fmt yuv420p "$WORK/o265.mkv"); then
      emit "h.265 crf 0 (x265 veryslow)" "$unit" "$orig" "$(stat -c %s "$WORK/o265.mkv")" "$s" \
           "{\"width\":$w,\"height\":$h,\"frames\":$nf}"
    fi
    if s=$(timed "$FF" -y -hide_banner -loglevel error -i "$path" \
             -c:v ffv1 -level 3 "$WORK/o.ffv1.mkv"); then
      emit "ffv1 level 3" "$unit" "$orig" "$(stat -c %s "$WORK/o.ffv1.mkv")" "$s" \
           "{\"width\":$w,\"height\":$h,\"frames\":$nf}"
    fi
    ;;

  *) echo "unknown modality '$MODALITY'" >&2; exit 2 ;;
  esac
done < "$MANIFEST"
