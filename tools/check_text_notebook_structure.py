"""Inspect and validate the text notebook's structure and both window axes."""

from __future__ import annotations

import ast
import base64
import io
import json
import math
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402
REPO = _capsule_root()
MEDAL_DIR = os.path.join(REPO, "data", "text", "MEDAL_sample")
FILES = ["medal_102285.csv", "medal_103851.csv", "medal_108122.csv"]
NB = os.path.join(HERE, "colab", "Text_Paired_Context_Colab.ipynb")


def main() -> int:
    nb = json.load(open(NB, encoding="utf-8"))
    codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"cells {len(nb['cells'])}  code {len(codes)}  "
          f"size {os.path.getsize(NB)/1024:.0f} KB")
    print()
    n = 0
    for c in codes:
        n += 1
        src = "".join(c["source"])
        first = [l for l in src.splitlines() if l.strip()][:1]
        head = first[0][:58] if first else ""
        print(f"  {n:>2}. {len(src):>7,} chars  {head}")

    # syntax of every code cell, with IPython magics neutralised
    bad = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        lines = ["pass" if l.strip().startswith(("!", "%")) else l
                 for l in src.splitlines()]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:
            bad += 1
            print(f"  !! cell {i} SyntaxError line {exc.lineno}: {exc.msg}")
    print(f"\nsyntax errors: {bad}")
    assert bad == 0

    # ---- both window axes must match the asserted counts ------------------ #
    print("\n=== window axes ===")
    import pandas as pd
    from transformers import AutoTokenizer

    payload = None
    for c in codes:
        src = "".join(c["source"])
        if "PAYLOAD = " in src:
            i = src.find('PAYLOAD = """') + len('PAYLOAD = """')
            payload = src[i:src.find('"""', i)]
            break
    assert payload, "payload cell not found"

    tok = AutoTokenizer.from_pretrained(
        os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B"),
        local_files_only=True)

    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as zf:
        texts, tokenised = {}, {}
        for name in FILES:
            df = pd.read_csv(io.BytesIO(zf.read(name)))
            s = ""
            for _, row in df.iterrows():
                s += row["TEXT"]
            texts[name] = s
            tokenised[name] = tok(s, add_special_tokens=False)["input_ids"]

    # Axis A: character windows, tokenised per chunk (the reference's way)
    print(f"  {'char ctx':>9}{'windows':>9}{'mean tok':>10}")
    for ctx in (4096, 6144, 8192):
        wins = []
        for f in FILES:
            t = texts[f]
            for i in range(0, len(t), ctx):
                ids = tok(t[i:i+ctx], add_special_tokens=False)["input_ids"]
                if len(ids) >= 2:
                    wins.append(ids)
        mean = sum(len(w) for w in wins) / len(wins)
        print(f"  {ctx:>9}{len(wins):>9}{mean:>10.1f}")

    # Axis B: token windows
    print(f"  {'tok ctx':>9}{'windows':>9}{'mean tok':>10}")
    for ctx in (4096, 8192):
        wins = []
        for f in FILES:
            ids = tokenised[f]
            wins.extend(ids[i:i+ctx] for i in range(0, len(ids), ctx)
                        if len(ids[i:i+ctx]) >= 2)
        mean = sum(len(w) for w in wins) / len(wins)
        print(f"  {ctx:>9}{len(wins):>9}{mean:>10.1f}")

    a_counts = []
    for ctx in (4096, 6144, 8192):
        k = 0
        for f in FILES:
            t = texts[f]
            for i in range(0, len(t), ctx):
                if len(tok(t[i:i+ctx], add_special_tokens=False)["input_ids"]) >= 2:
                    k += 1
        a_counts.append(k)
    b_counts = []
    for ctx in (4096, 8192):
        k = 0
        for f in FILES:
            ids = tokenised[f]
            k += sum(1 for i in range(0, len(ids), ctx)
                     if len(ids[i:i+ctx]) >= 2)
        b_counts.append(k)

    print(f"\n  Axis A windows {a_counts}  (notebook asserts [78, 51, 39])")
    print(f"  Axis B windows {b_counts}  (notebook asserts [15, 9])")
    assert a_counts == [78, 51, 39], a_counts
    assert b_counts == [15, 9], b_counts
    print("\nALL STRUCTURE AND WINDOW CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
