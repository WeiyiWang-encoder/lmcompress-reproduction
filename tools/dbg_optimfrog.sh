#!/usr/bin/env bash
# Debug: why does the optimfrog baseline row not appear?

: "${LMCOMPRESS_CAPSULE:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/capsule-4957991-code}"
CAPSULE="$LMCOMPRESS_CAPSULE"
set -u
TP=$CAPSULE/third_party
OFR="$TP/OptimFROG_Linux_x64_5100/ofr"
SRC=$CAPSULE/data/test_audios/LJ001-0004.wav
W=$(mktemp -d)
echo "ofr = $OFR"
echo "src = $SRC"
echo "--- preset syntax from help ---"
"$OFR" --help 2>&1 | grep -i -A2 "preset" | head -12
echo "--- attempt 1: --preset max -o OUT SRC ---"
"$OFR" --preset max -o "$W/o.ofr" "$SRC" 2>&1 | tail -5
echo "exit=$?"
ls -la "$W/" 2>&1 | head -5
echo "--- attempt 2: --preset max OUT SRC ---"
"$OFR" --preset max "$W/b.ofr" "$SRC" 2>&1 | tail -5
ls -la "$W/" 2>&1 | head -5
echo "--- attempt 3: -c (stdout) ---"
"$OFR" --preset max -c "$SRC" > "$W/c.ofr" 2>"$W/c.err"
echo "exit=$? size=$(stat -c %s "$W/c.ofr" 2>/dev/null)"
tail -3 "$W/c.err" 2>/dev/null
rm -rf "$W"
