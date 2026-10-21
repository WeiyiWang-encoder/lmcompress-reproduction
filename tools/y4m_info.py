#!/usr/bin/env python3
"""Print '<w> <h> <n>' for a Y4M file, counting frames by strict parsing.

Used by the video baseline shell driver.  An earlier inline heuristic
(``filesize // (frame_bytes + 6)``) returned 59 for a 60-frame file because the
header is 60 bytes, not the ~40 the heuristic assumed -- and a wrong frame count
silently makes every rate in the run wrong by 1/60.  Strict parsing leaves no
leftover, so this also verifies the file is not truncated.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    path = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    with open(path, "rb") as fh:
        header = fh.readline()
        parts = header.decode("ascii", "replace").split()
        w = h = 0
        for token in parts:
            if token.startswith("W"):
                w = int(token[1:])
            elif token.startswith("H"):
                h = int(token[1:])
        if not w or not h:
            print("cannot parse Y4M header", file=sys.stderr)
            return 1

        frame_bytes = w * h * 3 // 2  # 4:2:0 assumed
        n = 0
        while True:
            line = fh.readline()
            if not line.startswith(b"FRAME"):
                break
            if fh.seek(frame_bytes, os.SEEK_CUR) and False:
                break
            n += 1

    total = os.path.getsize(path)
    consumed = len(header) + n * (6 + frame_bytes)
    if consumed != total:
        print(f"WARNING: parsed {n} frames consuming {consumed} of {total} bytes",
              file=sys.stderr)

    take = n if want <= 0 else min(want, n)
    print(w, h, take, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
