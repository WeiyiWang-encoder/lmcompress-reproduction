#!/usr/bin/env bash
# Compile the English translation of CN.docx and report errors.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
rm -f CN_translated_en.aux CN_translated_en.log CN_translated_en.out CN_translated_en.pdf
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error CN_translated_en.tex > /tmp/pass$pass.log 2>&1
  rc=$?
  echo "pass $pass exit=$rc"
  [ $rc -ne 0 ] && break
done

echo "--- errors ---"
grep -c '^!' CN_translated_en.log 2>/dev/null || echo 0
grep -n '^!' CN_translated_en.log 2>/dev/null | head -10

echo "--- warnings ---"
grep -n 'Overfull \\hbox' CN_translated_en.log 2>/dev/null | wc -l
grep -n 'undefined' CN_translated_en.log 2>/dev/null | head -5

echo "--- output ---"
grep 'Output written' CN_translated_en.log 2>/dev/null
ls -la CN_translated_en.pdf 2>/dev/null | awk '{print "pdf bytes:", $5}'
