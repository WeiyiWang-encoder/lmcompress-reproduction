"""Filesystem resolution for this repository.

Nothing here hard-codes an absolute path. The repository ships **code and measurement
results only**; datasets, model weights and the authors' capsule checkout are not
committed, because they are large and, for the models, separately licensed.

Two roots matter:

``repo_root``
    Where this checkout lives. Everything inside the repository is found relative to it.

``capsule_root``
    The authors' Code Ocean capsule (``capsule-4957991-code``), which supplies the
    datasets, the released weights and the reference implementation used for
    cross-checking. Resolved from the ``LMCOMPRESS_CAPSULE`` environment variable, and
    failing that from a sibling directory of the same name.

Typical use::

    export LMCOMPRESS_CAPSULE=/data/capsule-4957991-code
    python tools/baselines_text_windowed.py

If the variable is unset and the sibling directory does not exist, ``capsule_root``
raises with a message naming the variable rather than silently returning a wrong path.
"""

from __future__ import annotations

import os

__all__ = [
    "repo_root", "src_dir", "results_dir", "docs_dir", "report_dir",
    "capsule_root", "data_root", "model_dir", "test_image_dir", "test_audio_dir",
]

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SRC_DIR))

CAPSULE_ENV = "LMCOMPRESS_CAPSULE"
CAPSULE_DIRNAME = "capsule-4957991-code"


def repo_root() -> str:
    """Absolute path of this repository's root directory."""
    return _REPO_ROOT


def src_dir() -> str:
    return _SRC_DIR


def results_dir() -> str:
    """Directory holding the raw measurement files quoted in the report."""
    return os.path.join(_REPO_ROOT, "results")


def docs_dir() -> str:
    return os.path.join(_REPO_ROOT, "docs")


def report_dir() -> str:
    return os.path.join(_REPO_ROOT, "report")


def capsule_root() -> str:
    """Absolute path of the authors' capsule checkout.

    Order of resolution:
      1. the ``LMCOMPRESS_CAPSULE`` environment variable;
      2. a sibling directory of this repository named ``capsule-4957991-code``;
      3. an error that says how to fix it.
    """
    env = os.environ.get(CAPSULE_ENV)
    if env:
        path = os.path.abspath(os.path.expanduser(env))
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{CAPSULE_ENV}={env!r} does not point at a directory")
        return path

    sibling = os.path.join(os.path.dirname(_REPO_ROOT), CAPSULE_DIRNAME)
    if os.path.isdir(sibling):
        return sibling

    raise FileNotFoundError(
        "cannot locate the Code Ocean capsule.\n"
        f"  Set {CAPSULE_ENV} to the capsule checkout, e.g.\n"
        f"    export {CAPSULE_ENV}=/path/to/{CAPSULE_DIRNAME}\n"
        f"  or place the capsule next to this repository at {sibling}\n"
        "The capsule supplies the datasets, the released model weights and the "
        "reference implementation; see data/README.md.")


def data_root() -> str:
    """The capsule's ``data/`` directory (datasets and weights)."""
    return os.path.join(capsule_root(), "data")


def model_dir(name: str) -> str:
    """A model directory inside the capsule, e.g. ``model_dir("imagegpt-small")``.

    Known names: ``imagegpt-small``, ``bgpt``, and
    ``finetuned_llama/Llama-3-8B`` plus ``finetuned_llama/Llama-3-8B_lora``.
    """
    return os.path.join(data_root(), "model", name)


def test_image_dir() -> str:
    return os.path.join(data_root(), "test_images")


def test_audio_dir() -> str:
    return os.path.join(data_root(), "test_audios")
