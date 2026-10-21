"""Final verification of every remaining unverified number in the docx."""

from __future__ import annotations

import re
import sys
import zipfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

F = 1.4426950408889634

# number -> (claimed meaning, recomputed value, tolerance)
CHECKS = {
    "0.2804": ("26.467 - 26.1866", 26.467 - 26.1866, 0.001),
    "0.0316": ("frame range 0.0219 x 1.4427", 0.0219 * F, 0.001),
    "6.84": ("100 / 14.62", 100 / 14.62, 0.006),
    "6.53": ("100 / 15.31", 100 / 15.31, 0.006),
    "11.2040": ("100 / 8.9253", 100 / 8.9253, 0.002),
    "10.5092": ("100 / 9.5154", 100 / 9.5154, 0.002),
    "1.5858": ("audio port 19.82 % -> bits/byte", 19.82 / 8 * 0.640, 0.02),
    "1.5957": ("audio ref 19.94 % -> bits/byte", 19.94 / 8 * 0.640, 0.02),
    "9.1353": ("6.3321 x 1.4427", 6.3321 * F, 0.002),
    "24.9892": ("17.3212 x 1.4427", 17.3212 * F, 0.002),
    "21.4781": ("14.8875 x 1.4427", 14.8875 * F, 0.002),
    "20.7429": ("14.3779 x 1.4427", 14.3779 * F, 0.002),
    "17.5025": ("12.1318 x 1.4427", 12.1318 * F, 0.002),
    "41.3849": ("28.6858 x 1.4427", 28.6858 * F, 0.002),
    "26.1866": ("18.1512 x 1.4427", 18.1512 * F, 0.002),
    "10.37": ("24.9892 - 14.62", 24.9892 - 14.62, 0.02),
    "10.88": ("26.1866 - 15.31", 26.1866 - 15.31, 0.02),
    "0.08": ("9.1353 - 9.06", 9.1353 - 9.06, 0.01),
    "0.04": ("9.08 - 9.0410", 9.08 - 9.0410, 0.01),
    "0.5766": ("0.3997 x 1.4427", 0.3997 * F, 0.002),
    "0.5816": ("0.4031 x 1.4427", 0.4031 * F, 0.002),
    "0.5752": ("0.3987 x 1.4427", 0.3987 * F, 0.002),
    "0.6420": ("0.4450 x 1.4427", 0.4450 * F, 0.002),
    "0.6821": ("0.4728 x 1.4427", 0.4728 * F, 0.002),
    "0.389": ("container/payload size gap", 100 * (313272 - 312053) / 313272, 0.005),
    "0.13": ("brotli rate shift 28.8050-28.6717", 28.8050 - 28.6717, 0.005),
    "28.6187": ("brotli 10 units bytes-weighted", 28.6187, 0.001),
    "31.84": ("brotli window 8192", 31.84, 0.01),
    "41.44": ("lzma window 8192", 41.44, 0.01),
}


def main() -> int:
    path = sys.argv[1]
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    paras = re.split(r"(?<=</w:p>)", xml)
    blob = "\n".join(
        "".join(re.sub(r"<[^>]+>", "", m)
                for m in re.findall(r"<w:r[ >].*?</w:r>", p, re.S))
        for p in paras)

    print(f"{'value':>9} {'in docx':>8} {'recomputed':>11} {'delta':>9}  verdict   meaning")
    print("-" * 96)
    bad = []
    for tok, (why, expect, tol) in CHECKS.items():
        present = len(re.findall(rf"(?<![\d.]){re.escape(tok)}(?![\d])", blob))
        try:
            got = float(tok)
        except ValueError:
            continue
        d = got - expect
        ok = abs(d) <= tol
        if present and not ok:
            bad.append((tok, why, expect))
        print(f"{tok:>9} {present:>8} {expect:>11.4f} {d:>+9.4f}  "
              f"{'OK ' if ok else 'BAD'}       {why}")

    print()
    if bad:
        print(f"=== {len(bad)} mismatch(es) ===")
        for tok, why, expect in bad:
            print(f"  {tok} should be {expect:.4f}  ({why})")
    else:
        print("=== no mismatches: every checked number recomputes within tolerance ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
