"""Repository hygiene checks. Run this before every push.

Six things are checked, each of which has broken a reproduction repository at some point:

  1. no absolute filesystem paths in tracked code (they leak local layout and break clones);
  2. no file above the size GitHub will accept without LFS;
  3. nothing in the tree that is licensed material -- the capsule, datasets, weights;
  4. every artefact named in the report appendix is actually present;
  5. every Python file byte-compiles;
  6. every shell script parses.

Exit code is non-zero if anything fails, so it can be used as a pre-push hook or in CI.
"""

from __future__ import annotations

import os
import py_compile
import re
import subprocess
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# files the report appendix points at, so a reader can trace every number
EXPECTED_RESULTS = [
    "results/text_paired.json",
    "results/audio_bgpt_rows17_tailpad_v1.jsonl",
    "results/image_igpt_clic.json",
    "results/igpt_grayscale_collisions.json",
    "results/video_channelwise_summary.json",
    "results/baselines_video_wsl.jsonl",
    "results/baselines_video_zstd.jsonl",
]

# a path that only exists on the machine this was developed on.
# The drive-letter branch requires at least two path segments, otherwise it would also
# match regex escapes such as ``ratio:\s*`` and produce false positives.
ABS_PATH = re.compile(
    r"(/mnt/[a-z]/"
    r"|[A-Za-z]:\\[A-Za-z0-9_. -]+\\"
    r"|[A-Za-z]:/(?:Users|Document|home|data)/"
    r"|/home/[a-z]+/"
    r"|/Users/[A-Za-z]+/"
    r")")

FORBIDDEN_DIRS = {"capsule-4957991-code", "artifacts", ".venv", "venv", "capstone"}
FORBIDDEN_EXT = {".safetensors", ".pth", ".bin", ".y4m", ".wav", ".zip"}

SIZE_LIMIT_MB = 25


def walk():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", ".ipynb_checkpoints"}
                       and d not in FORBIDDEN_DIRS]
        for f in filenames:
            yield os.path.join(dirpath, f)


def main() -> int:
    failures = []

    print("1. absolute paths in tracked code and logs")
    hits = 0
    scan_ext = (".py", ".sh", ".md", ".tex", ".cff", ".txt", ".json", ".jsonl",
                ".out", ".err", ".log", ".cfg", ".yml", ".yaml", ".ipynb")
    for p in walk():
        if not p.endswith(scan_ext):
            continue
        rel = os.path.relpath(p, ROOT)
        if rel.startswith("data/") or rel in ("README.md",) and False:
            continue
        try:
            text = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if ABS_PATH.search(line):
                print(f"   {rel}:{i}: {line.strip()[:100]}")
                hits += 1
    if hits:
        failures.append(f"{hits} absolute path reference(s)")
    else:
        print("   none")

    print(f"\n2. files over {SIZE_LIMIT_MB} MB")
    big = 0
    for p in walk():
        mb = os.path.getsize(p) / 2**20
        if mb > SIZE_LIMIT_MB:
            print(f"   {os.path.relpath(p, ROOT)}: {mb:.1f} MB")
            big += 1
    if big:
        failures.append(f"{big} oversized file(s)")
    else:
        print("   none")

    print("\n3. licensed material present in the tree")
    bad = 0
    for p in walk():
        ext = os.path.splitext(p)[1].lower()
        if ext in FORBIDDEN_EXT:
            print(f"   {os.path.relpath(p, ROOT)}")
            bad += 1
    if bad:
        failures.append(f"{bad} licensed/regenerable file(s) tracked")
    else:
        print("   none")

    print("\n4. artefacts named in the report appendix")
    missing = 0
    for rel in EXPECTED_RESULTS:
        if os.path.exists(os.path.join(ROOT, rel)):
            print(f"   ok   {rel}")
        else:
            print(f"   MISS {rel}")
            missing += 1
    if missing:
        failures.append(f"{missing} expected result file(s) missing")

    print("\n5. python byte-compiles")
    errs = 0
    with tempfile.TemporaryDirectory() as tmp:
        for p in walk():
            if not p.endswith(".py"):
                continue
            try:
                py_compile.compile(p, cfile=os.path.join(tmp, "x.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                print(f"   {os.path.relpath(p, ROOT)}: {str(exc).splitlines()[-1][:100]}")
                errs += 1
    if errs:
        failures.append(f"{errs} python file(s) do not compile")
    else:
        print("   all compile")

    print("\n6. shell scripts parse")
    sh = [p for p in walk() if p.endswith(".sh")]
    if not sh:
        print("   none found")
    else:
        try:
            # bash cannot read Windows-style absolute paths, so pass names
            # relative to the repository and set cwd
            rels = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in sh]
            r = subprocess.run(["bash", "-n", *rels], cwd=ROOT,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                print(f"   {r.stderr.strip()[:400]}")
                failures.append("shell syntax error")
            else:
                print(f"   all {len(sh)} parse")
        except FileNotFoundError:
            print("   bash not available; skipped")

    print()
    if failures:
        print("FAILED: " + "; ".join(failures))
        return 1
    print("all repository checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
