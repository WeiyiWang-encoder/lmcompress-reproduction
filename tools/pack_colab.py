"""Pack everything Colab needs into one uploadable archive.

Colab cannot reach this machine's disk, so the runbook needs a single zip with:

* the capsule's LoRA adapter (so Part A2 can evaluate the shipped checkpoint),
* the local text samples, split into the paper's train/val/test proportions,
* the pure-python package source, so the notebook's numbers and the local
  numbers come from the same code rather than from two hand-copied variants,
* a short README stating what each part is and what is *not* included.

The 8B base weights are deliberately excluded: they are ~16 GB and Colab can pull
them from the Hub in a couple of minutes, which is far cheaper than uploading.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import zipfile
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

REPO = _capsule_root()
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

README = """LMCompress Colab bundle
=======================

Contents
--------
adapter/Llama-3-8B_lora/   the capsule's shipped LoRA adapter (MeDAL, 10 epochs).
                           Part A2 of the notebook evaluates this checkpoint.
                           NOTE: its own trainer_state.json shows best eval loss
                           at step 1020 (~2 epochs), worsening to step 5120.
data/train, data/val, data/test
                           local text samples split the way the paper splits the
                           corpus (train / validation / held-out test). Local
                           samples are small, so treat these as a smoke test for
                           the harness; Part A of the notebook downloads the full
                           corpus from the Hub for the real measurement.
data/audio, data/images    the audio and image samples used locally.
src/uic/                   the pure-python implementation. Copied so the
                           notebook can import exactly the code that produced the
                           local numbers instead of a re-typed variant.
tools/                     baseline drivers and table builders.

NOT included
------------
* Llama-3-8B base weights (~16 GB). The notebook pulls them from the Hub; you
  need a HuggingFace token with Llama-3 access.
* The full MeDAL / Pile of Law corpora. Also fetched by the notebook.

Quick start
-----------
1. Runtime -> Change runtime type -> H100 (or A100).
2. Upload this zip, then run the notebook top to bottom.
3. Set USE_DRIVE = True at the top if you want checkpoints to survive a reset.
"""


def add_tree(zf: zipfile.ZipFile, src: str, arc_prefix: str, patterns=("*.py",)) -> int:
    count = 0
    for pattern in patterns:
        for path in glob.glob(os.path.join(src, "**", pattern), recursive=True):
            if "__pycache__" in path:
                continue
            rel = os.path.relpath(path, src)
            zf.write(path, os.path.join(arc_prefix, rel))
            count += 1
    return count


def load_texts(paths) -> str:
    import pandas as pd

    parts = []
    for p in paths:
        df = pd.read_csv(p)
        if "TEXT" in df.columns:
            parts.append("".join(str(x) for x in df["TEXT"].tolist()))
    return "\n".join(parts)


def write_split(zf: zipfile.ZipFile, text: str, prefix: str, sizes_mb) -> dict:
    import pandas as pd

    b = text.encode("utf-8")
    total = len(b)
    if total == 0:
        return {}

    # Proportional split, so the layout still matches the paper's
    # train / validation / test intent even when the local corpus is smaller than
    # the paper's 64 MB / 16 MB / rest and a byte-exact split is impossible.
    want = [mb * 1024 * 1024 for mb in sizes_mb]
    want_sum = sum(want)
    if total >= want_sum:
        train_end = want[0]
        val_end = want[0] + want[1]
    else:
        train_end = int(total * want[0] / want_sum)
        val_end = int(total * (want[0] + want[1]) / want_sum)
    bounds = [(0, train_end), (train_end, val_end), (val_end, total)]

    info = {}
    names = ["train", "val", "test"]
    for name, (lo, hi) in zip(names, bounds):
        chunk = b[lo:hi].decode("utf-8", "ignore")
        # ~100 KB per file, matching the capsule's preprocess_text_dataset.py
        files, buf, size, i = 0, [], 0, 0
        for line in chunk.split("\n"):
            buf.append(line)
            size += len(line.encode()) + 1
            if size >= 100 * 1024:
                csv = pd.DataFrame({"TEXT": ["\n".join(buf)]}).to_csv(index=False)
                zf.writestr(f"data/{prefix}{name}/{name}_{i}.csv", csv)
                files += 1
                i += 1
                buf, size = [], 0
        if buf:
            csv = pd.DataFrame({"TEXT": ["\n".join(buf)]}).to_csv(index=False)
            zf.writestr(f"data/{prefix}{name}/{name}_{i}.csv", csv)
            files += 1
        info[name] = {"files": files, "bytes": hi - lo}
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "colab", "lmcompress_colab_bundle.zip"))
    ap.add_argument("--train-mb", type=int, default=64)
    ap.add_argument("--val-mb", type=int, default=16)
    args = ap.parse_args()

    src_text = sorted(
        glob.glob(os.path.join(REPO, "data", "text", "EURLEX_sample", "*.csv"))
        + glob.glob(os.path.join(REPO, "data", "text", "MEDAL_sample", "*.csv"))
    )
    if not src_text:
        print("no local text samples found")
        return 1
    text = load_texts(src_text)
    print(f"packing from {len(src_text)} local CSVs, {len(text.encode())/1e6:.2f} MB of text")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest: dict = {"sources": [os.path.basename(p) for p in src_text]}

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", README)

        # the shipped adapter
        adapter = os.path.join(
            REPO, "data", "model", "finetuned_llama", "Llama-3-8B_lora"
        )
        n_adapter = 0
        for root, _dirs, files in os.walk(adapter):
            for f in files:
                # skip the DeepSpeed optimiser shards: not needed to *use* the adapter
                if f.endswith(".pt") or f == "latest":
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, os.path.dirname(adapter))
                zf.write(full, os.path.join("adapter", rel))
                n_adapter += 1
        manifest["adapter_files"] = n_adapter

        # text splits
        manifest["splits"] = write_split(zf, text, "", (args.train_mb, args.val_mb))

        # code
        manifest["source_files"] = add_tree(
            zf, os.path.join(HERE, "src"), "src", ("*.py",)
        )
        manifest["tool_files"] = add_tree(
            zf, os.path.join(HERE, "tools"), "tools", ("*.py", "*.sh")
        )

        # small data samples
        n_data = 0
        for sub, arc in (
            (os.path.join(REPO, "data", "test_audios"), "data/audio"),
            (os.path.join(REPO, "data", "test_images"), "data/images"),
        ):
            for path in glob.glob(os.path.join(sub, "*")):
                if os.path.isfile(path) and os.path.getsize(path) < 8 * 1024 * 1024:
                    zf.write(path, os.path.join(arc, os.path.basename(path)))
                    n_data += 1
        manifest["sample_files"] = n_data

        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    size = os.path.getsize(args.out)
    print(f"wrote {args.out}  ({size/1e6:.1f} MB)")
    for name, info in manifest["splits"].items():
        print(f"  {name:5s} {info['files']:>4} files  {info['bytes']/1e6:>8.2f} MB")
    print(f"  adapter files: {manifest['adapter_files']}")
    print(f"  source files : {manifest['source_files']}")
    print(f"  sample files : {manifest['sample_files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
