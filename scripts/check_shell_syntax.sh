#!/usr/bin/env bash
# Syntax-check every shell script in tools/.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../tools" || exit 1
fail=0
n=0
for f in *.sh; do
  n=$((n + 1))
  if ! bash -n "$f" 2>/tmp/sh_syntax.err; then
    echo "FAIL $f"
    cat /tmp/sh_syntax.err
    fail=1
  fi
done
echo "checked $n shell scripts"
[ "$fail" -eq 0 ] && echo "all shell scripts parse"
exit "$fail"
