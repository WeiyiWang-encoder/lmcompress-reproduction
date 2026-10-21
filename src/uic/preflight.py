"""Preflight and diagnostics.

Run this before any long coding job: it proves the models load, the modality
loaders work, and the ffmpeg dependency is present.  Every check is cheap
compared with the coding runs, and each one has, in practice, been the thing
that silently broke a multi-hour job.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List

def _default_repo_root() -> str:
    """Capsule checkout, from ``UIC_REPO_ROOT`` or the shared resolver."""
    env = os.environ.get("UIC_REPO_ROOT")
    if env:
        return env
    from .paths import capsule_root

    return capsule_root()


REPO_ROOT = _default_repo_root()


def _result(name: str, ok: bool, detail) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def check_environment() -> List[dict]:
    out: List[dict] = []
    import torch

    out.append(_result("python", True, sys.version.split()[0]))
    out.append(_result("torch", True, torch.__version__))
    cuda = torch.cuda.is_available()
    out.append(
        _result(
            "cuda",
            cuda,
            {
                "device": torch.cuda.get_device_name(0) if cuda else None,
                "capability": torch.cuda.get_device_capability(0) if cuda else None,
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                if cuda
                else None,
            },
        )
    )
    try:
        import bitsandbytes as bnb

        out.append(_result("bitsandbytes", True, bnb.__version__))
    except Exception as exc:  # noqa: BLE001
        out.append(_result("bitsandbytes", False, str(exc)))
    for mod in ("transformers", "peft", "accelerate", "pandas", "numpy"):
        try:
            m = __import__(mod)

            out.append(_result(mod, True, getattr(m, "__version__", "?")))
        except Exception as exc:  # noqa: BLE001
            out.append(_result(mod, False, str(exc)))
    return out


def check_assets(repo_root: str = REPO_ROOT) -> List[dict]:
    from .ffmpeg_tools import probe as ffmpeg_probe

    out: List[dict] = []
    assets = {
        "bgpt-audio": os.path.join(repo_root, "data", "model", "bgpt", "weights-audio.pth"),
        "bgpt-image": os.path.join(repo_root, "data", "model", "bgpt", "weights-image.pth"),
        "bgpt-text": os.path.join(repo_root, "data", "model", "bgpt", "weights-text.pth"),
        "imagegpt-small": os.path.join(repo_root, "data", "model", "imagegpt-small"),
        "llama-3-8b": os.path.join(repo_root, "data", "model", "finetuned_llama", "Llama-3-8B"),
        "llama-lora": os.path.join(repo_root, "data", "model", "finetuned_llama", "Llama-3-8B_lora"),
        "audio-sample": os.path.join(repo_root, "data", "test_audios"),
        "image-sample": os.path.join(repo_root, "data", "test_images"),
        "video-sample": os.path.join(repo_root, "data", "test_videos"),
        "text-medal": os.path.join(repo_root, "data", "text", "MEDAL_sample"),
        "text-eurlex": os.path.join(repo_root, "data", "text", "EURLEX_sample"),
    }
    for name, path in assets.items():
        exists = os.path.exists(path)
        size = None
        if exists and os.path.isfile(path):
            size = os.path.getsize(path)
        elif exists:
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(path)
                for f in fs
            )
        out.append(_result(f"asset:{name}", exists, {"path": path, "bytes": size}))
    out.append(_result("ffmpeg", ffmpeg_probe(repo_root).get("available", False), ffmpeg_probe(repo_root)))
    return out


def main() -> int:
    t0 = time.time()
    report = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": REPO_ROOT,
        "environment": check_environment(),
        "assets": check_assets(),
    }
    report["bad"] = [r["check"] for r in report["environment"] + report["assets"] if not r["ok"]]
    report["seconds"] = round(time.time() - t0, 2)

    print(json.dumps(report, indent=2))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "preflight.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")
    return 1 if report["bad"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
