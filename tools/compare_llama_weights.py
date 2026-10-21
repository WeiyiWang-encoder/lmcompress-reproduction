"""Are the capsule's Llama-3-8B weights the same tensors as the public checkpoint?

The paired text run on the A100 reported 6.33 % where our local port reports 9.04 % on
the same corpus and the same window count. The setup checks (corpus bytes, token counts,
window counts, adapter parameters) all pass, so the difference has to be in the weights,
the precision, or the harness.

This settles the weights by comparing tensor bytes, without downloading 16 GB: a
safetensors file starts with an 8-byte little-endian header length, then a JSON header
giving every tensor's dtype, shape and byte offsets. So one small tensor can be read out
of both files and hashed.

`model.norm.weight` is 4096 values; in bf16 that is 8 KB, which is cheap over HTTP Range.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import urllib.request
# Resolved through uic.paths so the repository can be cloned anywhere.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from uic.paths import capsule_root as _capsule_root  # noqa: E402

CAPSULE_DIR = os.path.join(_capsule_root(),
                           "data", "model", "finetuned_llama", "Llama-3-8B")
REPO = "unsloth/llama-3-8b"
TENSORS = ["model.norm.weight", "model.embed_tokens.weight"]


def capsule_shard(name: str) -> tuple[str, dict]:
    """Locate which shard of the capsule checkpoint holds `name`."""
    idx = json.load(open(os.path.join(CAPSULE_DIR, "model.safetensors.index.json"),
                        encoding="utf-8"))
    shard = idx["weight_map"][name]
    return os.path.join(CAPSULE_DIR, shard), idx


def read_header(fh_or_bytes, remote: bool):
    """Return (json_header, data_offset) for a safetensors file."""
    if remote:
        first = fh_or_bytes.read(8)
        n = struct.unpack("<Q", first)[0]
        hdr = json.loads(fh_or_bytes.read(n))
        return hdr, 8 + n
    with open(fh_or_bytes, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        return hdr, 8 + n


def local_tensor(path: str, name: str):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        if name not in hdr:
            return None
        info = hdr[name]
        base = 8 + n
        start, end = info["data_offsets"]
        fh.seek(base + start)
        return fh.read(end - start), info


def remote_tensor(repo: str, name: str):
    """Find which shard holds `name`, then range-read just that tensor."""
    idx_url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json"
    with urllib.request.urlopen(idx_url, timeout=60) as r:
        index = json.load(r)
    shard = index["weight_map"].get(name)
    if shard is None:
        return None, None
    url = f"https://huggingface.co/{repo}/resolve/main/{shard}"

    # header first
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=120) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(url, headers={"Range": f"bytes=8-{7+n}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        hdr = json.loads(r.read(n))
    info = hdr[name]
    base = 8 + n
    start, end = info["data_offsets"]
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={base+start}-{base+end-1}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read(end - start), info


def main() -> int:
    print(f"capsule : {CAPSULE_DIR}")
    print(f"remote  : {REPO}")
    print()

    for name in TENSORS:
        shard_path, _ = capsule_shard(name)
        loc = local_tensor(shard_path, name)
        if loc is None:
            print(f"  {name}: absent from the capsule file")
            continue
        lb, linfo = loc
        print(f"  {name}")
        print(f"    capsule  dtype={linfo['dtype']:<8} shape={linfo['shape']} "
              f"bytes={len(lb):,}")
        print(f"             sha256={hashlib.sha256(lb).hexdigest()[:40]}")

        try:
            rb, rinfo = remote_tensor(REPO, name)
        except Exception as exc:
            print(f"    remote   FAILED: {type(exc).__name__}: {exc}")
            continue
        if rb is None:
            print("    remote   tensor not in the index")
            continue
        print(f"    remote   dtype={rinfo['dtype']:<8} shape={rinfo['shape']} "
              f"bytes={len(rb):,}")
        print(f"             sha256={hashlib.sha256(rb).hexdigest()[:40]}")
        same = hashlib.sha256(lb).digest() == hashlib.sha256(rb).digest()
        print(f"    identical bytes: {same}")
        if not same:
            # byte-level difference, to distinguish a different-tensor case from a
            # re-serialisation (e.g. bf16 vs fp32 round-trip)
            diff = sum(1 for a, b in zip(lb, rb) if a != b)
            print(f"    differing bytes: {diff:,} / {min(len(lb), len(rb)):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
