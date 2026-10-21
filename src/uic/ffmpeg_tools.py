"""Locate the ffmpeg binary.

The capsule ships ffmpeg inside ``third_party``; we prefer that so the
reproduction does not depend on whatever happens to be on the machine's PATH.
"""

from __future__ import annotations

import glob
import os
import shutil

_CACHE: dict = {}


def find_ffmpeg(repo_root: str | None = None) -> str:
    if "ffmpeg" in _CACHE:
        return _CACHE["ffmpeg"]

    root = repo_root or os.environ.get("UIC_REPO_ROOT") or os.getcwd()
    candidates = [
        os.path.join(root, "third_party", "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(root, "third_party", "ffmpeg", "ffmpeg.exe"),
        os.path.join(root, "third_party", "ffmpeg", "ffmpeg"),
    ]
    candidates += glob.glob(os.path.join(root, "third_party", "**", "ffmpeg.exe"), recursive=True)
    candidates += glob.glob(os.path.join(root, "third_party", "**", "bin", "ffmpeg.exe"), recursive=True)

    for path in candidates:
        if os.path.isfile(path):
            _CACHE["ffmpeg"] = path
            return path

    on_path = shutil.which("ffmpeg")
    if on_path:
        _CACHE["ffmpeg"] = on_path
        return on_path

    raise FileNotFoundError(
        "ffmpeg not found: looked in third_party/ and on PATH. "
        "Video frame extraction needs it."
    )


def probe(path: str | None = None) -> dict:
    """Return basic availability info, for the preflight report."""
    import subprocess

    try:
        exe = find_ffmpeg(path)
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    try:
        out = subprocess.run(
            [exe, "-version"], capture_output=True, text=True, timeout=30
        ).stdout.splitlines()
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc), "path": exe}
    return {"available": True, "path": exe, "version": out[0] if out else "unknown"}
