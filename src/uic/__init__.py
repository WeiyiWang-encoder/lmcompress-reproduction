"""LMCompress reproduction -- clean-room implementation.

Package layout::

    uic.bitio    bit packing helpers
    uic.coder    arithmetic coder + probability quantisation
    uic.engine   resumable measurement driver
    uic.models   predictors (bGPT bytes, ImageGPT pixels, Llama tokens)
    uic.data     modality loaders
"""

__all__ = ["bitio", "coder", "engine", "models", "data"]
__version__ = "0.1.0"
