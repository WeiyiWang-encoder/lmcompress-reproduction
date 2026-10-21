#!/usr/bin/env bash
# Classical lossless image baselines on the CLIC professional validation set.
#
# Why: the report's image baseline table used a single 1714x1142 aerial photograph,
# which compresses very differently from CLIC2019 professional photography (PNG 84.4 %
# against a published 47.1 %).  That made the table non-like-for-like.  The CLIC
# professional validation set is downloadable (128.6 MiB, 41 images), almost exactly
# the corpus size the paper reports (197 images, 128 MB), so it removes the confound.
#
# The capsule's ffmpeg is a Linux ELF binary, so this runs under WSL.
#
# Usage: baselines_image_clic.sh <claude> [limit]
set -u

# Resolved relative to this script; override with LMCOMPRESS_CAPSULE.
: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
REPO="$LMCOMPRESS_CAPSULE"
TP="$REPO/third_party"
FF="$TP/ffmpeg-git-20240629-amd64-static/ffmpeg"
IMGDIR="$REPO/repro_understanding_is_compression/artifacts/clic/raw"
OUT="$REPO/repro_understanding_is_compression/results/baselines_image_clic.jsonl"
LIMIT="${1:-0}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

[ -x "$FF" ] || { echo "ffmpeg missing at $FF" >&2; exit 1; }

# Emit one row to both stdout and the result file: stdout is for watching progress,
# the file is the artefact. (An earlier version only printed, so the .jsonl stayed
# empty while the run looked successful.)
emit() {
  local row
  row=$(python3 - "$@" <<'PY'
import json, sys
codec, unit, w, h, raw, comp, secs = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
    int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7]))
print(json.dumps({"codec": codec, "unit": unit, "width": w, "height": h,
                  "raw_rgb_bytes": raw, "compressed_bytes": comp,
                  "rate_pct": round(100.0*comp/raw, 6),
                  "seconds": round(secs, 3)}, sort_keys=True))
PY
)
  printf '%s\n' "$row" | tee -a "$OUT"
}

mapfile -t FILES < <(find "$IMGDIR" -type f -iname '*.png' | sort)
total=${#FILES[@]}
[ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "$total" ] && total=$LIMIT
echo "CLIC professional: $total images" >&2

: > "$OUT"
i=0
for f in "${FILES[@]}"; do
  i=$((i+1))
  [ "$i" -gt "$total" ] && break
  unit=$(basename "$f")

  FFPROBE="$TP/ffmpeg-git-20240629-amd64-static/ffprobe"
  # This build rejects `-of csv=p=0:s=' '`; plain csv gives "W,H", so swap the comma.
  dims=$("$FFPROBE" -v error -select_streams v:0 -show_entries stream=width,height \
         -of csv=p=0 "$f" 2>/dev/null | tr ',' ' ')
  read -r W H <<< "$dims"
  [ -z "${W:-}" ] && { echo "  skip $unit (unreadable)" >&2; continue; }
  ORIG=$(( W * H * 3 ))

  # --- WebP lossless -------------------------------------------------------
  t0=$(date +%s.%N)
  if "$FF" -v error -y -i "$f" -c:v libwebp -lossless 1 -compression_level 6 \
        -quality 100 "$WORK/o.webp" 2>/dev/null; then
    t1=$(date +%s.%N)
    emit "webp -lossless -q100" "$unit" "$W" "$H" "$ORIG" \
         "$(stat -c %s "$WORK/o.webp")" "$(awk -v a="$t1" -v b="$t0" 'BEGIN{print a-b}')"
  else
    echo "  webp FAILED $unit" >&2
  fi

  # --- JPEG-2000 lossless --------------------------------------------------
  t0=$(date +%s.%N)
  if "$FF" -v error -y -i "$f" -c:v libopenjpeg -pred 1 "$WORK/o.jp2" 2>/dev/null; then
    t1=$(date +%s.%N)
    emit "jpeg-2000 lossless" "$unit" "$W" "$H" "$ORIG" \
         "$(stat -c %s "$WORK/o.jp2")" "$(awk -v a="$t1" -v b="$t0" 'BEGIN{print a-b}')"
  else
    echo "  jp2 FAILED $unit" >&2
  fi

  # --- PNG -----------------------------------------------------------------
  t0=$(date +%s.%N)
  if "$FF" -v error -y -i "$f" -c:v png -compression_level 100 "$WORK/o.png" 2>/dev/null; then
    t1=$(date +%s.%N)
    emit "png (ffmpeg, cl 100)" "$unit" "$W" "$H" "$ORIG" \
         "$(stat -c %s "$WORK/o.png")" "$(awk -v a="$t1" -v b="$t0" 'BEGIN{print a-b}')"
  else
    echo "  png FAILED $unit" >&2
  fi

  if [ $((i % 5)) -eq 0 ] || [ "$i" -eq "$total" ]; then
    echo "  [$i/$total] done" >&2
  fi
done

echo "wrote $OUT" >&2
