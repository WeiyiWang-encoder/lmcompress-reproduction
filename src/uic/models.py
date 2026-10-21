"""Neural next-symbol predictors used by LMCompress.

Three model families, as in the paper:

``bgpt``   byte-level hierarchical GPT (12-layer patch decoder + 3-layer byte
           decoder, 16-byte patches, 110 M parameters).  Codes bytes directly,
           so audio / image / video payloads go in with no tokenizer.
``igpt``   ImageGPT-small (24 layers, 512 dim, 513-token colour vocabulary).
           Codes quantised pixel tokens.
``llama``  Llama-3-8B causal LM, optionally with a LoRA adapter, optionally
           4-bit quantised.  Codes BPE tokens.

All predictors expose the same interface, so the engine stays model-agnostic::

    predictor.name
    predictor.n_symbols
    predictor.iter_steps(payload) -> Iterator[engine.Step]

The bGPT *architecture* is imported from the reference capsule so the weights
that ship with the paper load into provably the same module graph.  Everything
else here -- patch/byte alignment, chunking, masking, memory-bounded softmax,
probability quantisation -- is re-derived.  See ``docs/REPRODUCTION_NOTES.md``.
"""

from __future__ import annotations

import os
import sys
from typing import Iterator, List, Optional, Sequence

import numpy as np

from .coder import sparse_true_symbol_cdf
from .engine import Step

# --------------------------------------------------------------------------- #
# bGPT constants (must match data/model/bgpt/config.py)
# --------------------------------------------------------------------------- #

BGPT_PATCH_SIZE = 16
BGPT_PATCH_LENGTH = 512
BGPT_BYTE_LAYERS = 3
BGPT_PATCH_LAYERS = 12
BGPT_HIDDEN = 768
BGPT_VOCAB = 257  # 256 bytes + 1 separator/padding token
BGPT_PAD = 256

#: Cap on logits materialised by one softmax pass (~32 MB of float32).
_SOFTMAX_BUDGET_BYTES = 32 << 20

#: Tail mass for vocabularies too large to reconstruct densely (Llama).
_SPARSE_TAIL_MASS = 2.0 ** -30


def _gather_true_probs(logits, target_ids):
    """``p(target)`` under a softmax over the last dim, without a full softmax.

    Returns a float32 tensor shaped like ``target_ids``.  ``log_softmax`` is
    computed over slices of the vocabulary so peak memory stays bounded: a full
    softmax tensor for a 128 k vocabulary over 8 k tokens would be ~4 GB.
    """
    import torch

    vocab = int(logits.shape[-1])
    rows = int(np.prod(logits.shape[:-1]))
    if rows == 0:
        return torch.empty(target_ids.shape, dtype=torch.float32, device=logits.device)
    chunk_rows = max(1, min(rows, _SOFTMAX_BUDGET_BYTES // (vocab * 4)))

    flat = logits.reshape(rows, vocab)
    idx = target_ids.reshape(rows)
    out = torch.empty(rows, dtype=torch.float32, device=flat.device)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        block = flat[start:stop].to(torch.float32)
        logp = torch.log_softmax(block, dim=-1)
        out[start:stop] = logp.gather(-1, idx[start:stop].unsqueeze(-1)).squeeze(-1).exp()
    return out.reshape(target_ids.shape)


def _to_numpy(tensor) -> np.ndarray:
    arr = tensor.detach().to("cpu", dtype=None).float().numpy()
    np.clip(arr, 1e-38, 1.0, out=arr)
    return arr


def _steps_from_causal_logits(logits, target_ids, n_tokens: int):
    """Yield ``Step``s for ``n_tokens`` targets given ``(1, n, vocab)`` logits.

    ``logits[:, i, :]`` is the distribution for ``target_ids[:, i]``, i.e. the
    caller has already applied the causal shift.  For the small alphabets used by
    bGPT (257) and ImageGPT (513) we build the *exact* full distribution, so the
    resulting arithmetic code is a genuine lossless code of the model's own
    posterior rather than an approximation of it.
    """
    import torch

    rows = logits[:, :n_tokens, :].to(torch.float32)
    probs = torch.softmax(rows, dim=-1)
    targets = target_ids[:, :n_tokens]
    gathered = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    probs_np = _to_numpy(probs.reshape(-1, probs.shape[-1]))
    p_true_np = _to_numpy(gathered).reshape(-1)
    del logits, rows, probs, gathered
    n_tokens = int(targets.shape[1])
    for i in range(n_tokens):
        yield Step(
            symbol=int(targets[0, i]),
            p_true=float(p_true_np[i]),
            probs=probs_np[i],
        )


# --------------------------------------------------------------------------- #
# bGPT
# --------------------------------------------------------------------------- #


def build_bgpt(repo_root: str, weights: str, device: str):
    """Instantiate bGPT and load a released checkpoint.

    ``strict=False`` is required: current ``transformers`` versions no longer
    register the legacy GPT-2 ``attn.bias`` / ``masked_bias`` buffers that the
    checkpoint still carries.  We verify that no *learnable* tensor is missing,
    which proves the weights were loaded completely.
    """
    import torch
    from transformers import GPT2Config

    for candidate in (os.path.join(repo_root, "data", "model", "bgpt"), repo_root):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from data.model.bgpt.utils import bGPTLMHeadModel  # type: ignore

    patch_config = GPT2Config(
        num_hidden_layers=BGPT_PATCH_LAYERS,
        max_length=BGPT_PATCH_LENGTH,
        max_position_embeddings=BGPT_PATCH_LENGTH,
        hidden_size=BGPT_HIDDEN,
        n_head=BGPT_HIDDEN // 64,
        vocab_size=1,
    )
    byte_config = GPT2Config(
        num_hidden_layers=BGPT_BYTE_LAYERS,
        max_length=BGPT_PATCH_SIZE + 1,
        max_position_embeddings=BGPT_PATCH_SIZE + 1,
        hidden_size=BGPT_HIDDEN,
        n_head=BGPT_HIDDEN // 64,
        vocab_size=BGPT_VOCAB,
    )
    model = bGPTLMHeadModel(patch_config, byte_config)

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if "attn.bias" not in k and "masked_bias" not in k]
    if bad:
        raise RuntimeError(f"bGPT checkpoint is missing tensors: {bad[:5]}")

    del checkpoint, state
    model = model.to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return model


def bgpt_offsets(n_items: int, capacity: int, patch_size: int = BGPT_PATCH_SIZE) -> List[int]:
    """Compatibility placeholders for bGPT windows.

    bGPT rows are not a contiguous span: every 17th byte-level row is the
    cross-patch prediction.  Callers retain ``Window.offset`` for a uniform
    runner interface, but selection is performed by
    :func:`bgpt_prediction_indices`, so the only valid placeholder is zero.
    """
    offsets: List[int] = []
    idx = 0
    while idx < n_items:
        idx = min(n_items, idx + capacity)
        offsets.append(0)
    return offsets


def bgpt_prediction_indices(n_data: int, patch_size: int = BGPT_PATCH_SIZE) -> List[int]:
    """Return the flattened bGPT rows that predict each payload byte.

    Derivation.  ``bGPTLMHeadModel.forward`` sets ``masks[:, 0] = 0`` and shifts
    the patch mask left before selecting patches, so the returned logits cover
    patches ``1 .. P`` where ``P`` is the number of patches carrying payload.
    Each returned patch contributes 17 slot predictions.  The reference path
    drops slot 16 in every patch before flattening, so its payload position
    ``k = 16 * p + j`` uses original logit row ``17 * p + j`` and target grid
    position ``16 + k``.  Selecting ``16 + k`` from the *untrimmed* flattened
    logits is a different mapping: it incorrectly includes cross-patch rows.

    This explicit, non-contiguous mapping is intentionally kept separate from
    target selection.  A target-value assertion alone cannot prove that a chosen
    logit row predicts that target.
    """
    if n_data < 0:
        raise ValueError("n_data must be non-negative")
    if patch_size != BGPT_PATCH_SIZE:
        raise ValueError(f"expected patch_size={BGPT_PATCH_SIZE}, got {patch_size}")
    return [17 * (k // patch_size) + (k % patch_size) for k in range(n_data)]


def plan_bgpt_windows(n_data_bytes: int, patch_size: int = BGPT_PATCH_SIZE) -> List[int]:
    """Split a payload into per-forward-pass window sizes.

    Two patches of the 512-patch grid are structural: the leading extension
    patch (BOS) and the trailing separator patch (EOS), so capacity is
    ``patch_size * (PATCH_LENGTH - 2)`` = 8 160 bytes, matching the paper's
    8 192-byte context window.
    """
    if n_data_bytes <= 0:
        return []
    capacity = patch_size * (BGPT_PATCH_LENGTH - 2)
    return [min(capacity, n_data_bytes - i) for i in range(0, n_data_bytes, capacity)]


class BGPTBytePredictor:
    """Codes a byte payload with bGPT."""

    name = "bgpt"
    window_capacity = BGPT_PATCH_SIZE * (BGPT_PATCH_LENGTH - 2)

    def __init__(self, model, extension: bytes = b"bin", device: str = "cuda:0"):
        self.model = model
        self.device = device
        self.n_symbols = BGPT_VOCAB
        ext = list(extension)[:BGPT_PATCH_SIZE]
        self.ext = ext + [BGPT_PAD] * (BGPT_PATCH_SIZE - len(ext))

    # -- layout ------------------------------------------------------------ #

    def build_batch(self, data: Sequence[int]):
        """Build the patch grid and patch-level attention mask for one window."""
        import torch

        patch_size = BGPT_PATCH_SIZE
        n_data = len(data)
        if n_data > self.window_capacity:
            raise ValueError(f"window of {n_data} exceeds capacity {self.window_capacity}")

        full = n_data // patch_size
        remainder = n_data % patch_size
        data_patches = full + (1 if remainder else 0)
        n_patches = 1 + data_patches + 1

        seq = list(self.ext) + list(data) + [BGPT_PAD] * patch_size

        masks = [1] * n_patches + [0] * (BGPT_PATCH_LENGTH - n_patches)
        pad = BGPT_PATCH_LENGTH * patch_size - len(seq)
        if pad < 0:
            raise ValueError("window overflows the bGPT context")
        seq += [BGPT_PAD] * pad

        patches = torch.tensor([seq], dtype=torch.long, device=self.device)
        mask_t = torch.tensor([masks], dtype=torch.long, device=self.device)
        return patches, mask_t, n_patches

    def iter_steps(self, data: Sequence[int], offset: Optional[int] = None) -> Iterator[Step]:
        """Yield one ``Step`` per payload byte, in payload order.

        ``offset`` is retained for compatibility with saved runner state.  bGPT
        prediction rows are non-contiguous, so only ``None`` and ``0`` are valid;
        :func:`bgpt_prediction_indices` owns the actual mapping.
        """
        import torch

        if not data:
            return
        patch_size = BGPT_PATCH_SIZE
        n_data = len(data)
        if offset not in (None, 0):
            raise ValueError("bGPT rows are non-contiguous; offset must be 0 or None")
        row_indices = bgpt_prediction_indices(n_data)

        patches, mask_t, n_patches = self.build_batch(data)
        with torch.no_grad():
            logits = self.model(patches=patches, masks=mask_t).logits
        if logits.shape[0] != n_patches - 1:
            raise RuntimeError(
                f"expected {n_patches - 1} decoded patches, model returned {logits.shape[0]}"
            )

        flat_logits = logits.reshape(1, -1, logits.shape[-1])
        if row_indices and row_indices[-1] >= flat_logits.shape[1]:
            raise RuntimeError(
                f"need logit row {row_indices[-1]} but only "
                f"{flat_logits.shape[1]} rows exist"
            )
        row_index_t = torch.tensor(row_indices, dtype=torch.long, device=flat_logits.device)
        rows = flat_logits.index_select(1, row_index_t)

        grid = patches.reshape(1, BGPT_PATCH_LENGTH, patch_size).reshape(1, -1)
        target_ids = grid[:, BGPT_PATCH_SIZE : BGPT_PATCH_SIZE + n_data]
        expected_ids = torch.tensor([list(data)], dtype=target_ids.dtype, device=target_ids.device)
        if not torch.equal(target_ids, expected_ids):
            raise RuntimeError(
                "bGPT target alignment check failed: "
                "payload slice differs from the original window"
            )

        yield from _steps_from_causal_logits(rows, target_ids, n_data)
        del logits, flat_logits, rows, row_index_t

    @classmethod
    def load(cls, repo_root: str, weights: str, device: str = "cuda:0", extension: bytes = b"bin"):
        return cls(build_bgpt(repo_root, weights, device), extension=extension, device=device)

    def release(self):
        import torch

        self.model = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# ImageGPT
# --------------------------------------------------------------------------- #


class ImageGPTPredictor:
    """Codes ImageGPT colour tokens, one per pixel, in raster order."""

    name = "igpt"

    def __init__(self, model, device: str = "cuda:0"):
        self.model = model
        self.device = device
        self.n_symbols = int(getattr(model.config, "vocab_size", 513))
        self.window_capacity = int(getattr(model.config, "n_positions", 1024))

    @classmethod
    def load(cls, model_path: str, device: str = "cuda:0"):
        import torch
        from transformers import ImageGPTForCausalImageModeling

        model = ImageGPTForCausalImageModeling.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        return cls(model, device=device)

    def iter_steps(self, tokens: Sequence[int]) -> Iterator[Step]:
        """Code one segment of at most ``n_positions`` tokens.

        Position ``i`` predicts token ``i + 1`` (standard causal shift), so the
        first token of a segment is never scored -- the caller codes every token
        exactly once by overlapping segments by one token.
        """
        import torch

        if len(tokens) < 2:
            return
        ids = torch.tensor([list(tokens)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(ids).logits

        # causal shift: logits[:, i, :] predicts tokens[i + 1]
        shifted = logits[:, :-1, :]
        targets = ids[:, 1:]
        yield from _steps_from_causal_logits(shifted, targets, len(tokens) - 1)
        del logits, shifted

    def release(self):
        import torch

        self.model = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Llama
# --------------------------------------------------------------------------- #


class LlamaTokenPredictor:
    """Codes a BPE token sequence with Llama-3-8B (+ optional LoRA / 4-bit)."""

    name = "llama"

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda:0",
        context_window: int = 8192,
        prepend_bos: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_window = int(context_window)
        self.n_symbols = int(model.config.vocab_size)
        self.window_capacity = self.context_window
        self._all_tokens: List[int] = []
        self.bos_id: Optional[int] = None
        self.sentinel_id: Optional[int] = None
        if prepend_bos:
            bos = getattr(tokenizer, "bos_token_id", None)
            if bos is None:
                bos = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
            if isinstance(bos, int) and bos >= 0:
                self.bos_id = int(bos)
        # A natural-looking filler for the trailing sentinel slot.
        for candidate in ("<|end_of_text|>", "<|eot_id|>"):
            tok = tokenizer.convert_tokens_to_ids(candidate)
            if isinstance(tok, int) and 0 <= tok < self.n_symbols:
                self.sentinel_id = int(tok)
                break

    @classmethod
    def load(
        cls,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "cuda:0",
        load_in_4bit: bool = True,
        context_window: int = 8192,
        merge_lora: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        target = 0 if device.startswith("cuda") else "cpu"
        kwargs: dict = {"local_files_only": True, "device_map": {"": target}}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if lora_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
            if merge_lora:
                # Merging removes the per-layer adapter overhead, which roughly
                # doubles decoding throughput for the long coding loops.
                model = model.merge_and_unload()
        model.eval()
        return cls(model, tokenizer, device=device, context_window=context_window)

    def calibrate(self) -> int:
        """Measure how many tokens one maximal forward pass can score.

        This is deliberately empirical.  The causal-shift accounting is easy to
        get wrong by one, and every off-by-one here silently changes which bytes
        are compressed -- exactly the failure mode that made the reference
        implementation's audio numbers meaningless.  So instead of trusting a
        derivation, we run one throwaway forward pass at the maximum context
        length and count the predictions that actually come back.

        Returns the number of tokens this model can score in ONE window.
        """
        import torch

        b = 1 if self.bos_id is not None else 0
        seq_len = self.context_window
        # A plausible filler; the values do not matter, only the shapes do.
        filler = self.sentinel_id if self.sentinel_id is not None else 1
        seq = ([self.bos_id] if b else []) + [filler] * (seq_len - b)
        ids = torch.tensor([seq], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(ids).logits
        available = max(0, int(logits.shape[1]) - b - 1)
        del logits, ids
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return available

    def window_budget(self, has_bos: bool = True) -> int:
        """Tokens this model can score in one window.

        Derived from :meth:`calibrate`, which measures the model rather than
        assuming the arithmetic.  To score ``n`` tokens a window must feed
        ``n + 1`` of them, because the last one needs a successor to be predicted
        from; so a window uses ``1 (BOS) + n + 1`` positions on the first window
        and ``n + 1`` afterwards.  Both cases therefore score the same number of
        tokens, and BOS simply consumes the slack that the causal shift would
        otherwise waste.  (BOS is prepended only on the first window, so
        ``has_bos`` is accepted for call-site clarity but does not change the
        budget.)

        The value is cached after the first call so the planning path does not
        pay for a forward pass more than once.
        """
        cached = getattr(self, "_budget_cache", None)
        if cached is None:
            cached = self.calibrate()
            self._budget_cache = cached
        return max(1, int(cached))

    # -- text <-> tokens --------------------------------------------------- #

    def encode_text(self, text: str) -> List[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode_tokens(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def set_document(self, tokens: Sequence[int]) -> None:
        """Register the whole token sequence so windows can look backwards."""
        self._all_tokens = list(tokens)

    # -- coding ------------------------------------------------------------ #

    def iter_steps(
        self,
        tokens: Sequence[int],
        prior: int = 0,
        stop_at: Optional[int] = None,
    ) -> Iterator[Step]:
        """Code ``tokens`` given ``prior`` already-decoded tokens of context.

        Args:
          tokens: the tokens to code, in order.
          prior: how many tokens of the document precede ``tokens``.
          stop_at: exclusive end index inside the document.  Defaults to
            ``prior + len(tokens)``.  When it is *larger* the window is
            understood to need a trailing sentinel (the final window of a
            document, where the last real token would otherwise have no successor
            to be predicted from); when it is *smaller* the extra tokens are
            context only.

        Exactly ``min(len(tokens), stop_at - prior)`` steps are yielded, so the
        caller's accounting is exact.  Context for the forward pass is
        ``[BOS][preceding document tokens][window][sentinel?]``, truncated to
        ``context_window`` from the left.
        """
        import torch

        if not tokens:
            return
        if stop_at is None:
            stop_at = prior + len(tokens)
        # ``tokens`` is the document slice starting at ``prior``.
        doc_start = prior
        n_target = max(0, min(len(tokens), stop_at - doc_start))
        if n_target == 0:
            return

        # Goal: score exactly the document tokens at [doc_start, doc_start + n_target).
        #
        # A causal LM predicts token j from the position holding token j-1, so the
        # scored region needs one token of history in FRONT of it and one of
        # lookahead AFTER it:
        #
        #     feed = [ predecessor , scored tokens ... , successor ]
        #
        # where the predecessor may be absent only for the very first window of a
        # document (nothing precedes it) and the successor may be absent only at
        # the very end of the document.  Both are context: neither is scored.
        #
        # Earlier versions of this function got this wrong in two ways, and both
        # are worth recording because they are easy to repeat:
        #   * the target slice began one position late, so every token was scored
        #     against the distribution for its predecessor (a "one-byte lag");
        #   * the compensating "drop the first prediction" then silently discarded
        #     the final token of every document -- a loss that was mis-attributed
        #     to causal LMs, which in fact predict their last token fine.
        successor = (
            [int(self._all_tokens[doc_start + n_target])]
            if self._all_tokens and (doc_start + n_target) < len(self._all_tokens)
            else []
        )
        lead: List[int] = [int(self._all_tokens[doc_start - 1])] if doc_start > 0 else []
        feed = lead + list(tokens[:n_target]) + successor
        if len(feed) < len(lead) + n_target:
            return

        window_len = len(feed)
        history = max(0, self.context_window - window_len)
        pfx_end = doc_start - len(lead)
        pfx_start = max(0, pfx_end - history)
        prefix = self._all_tokens[pfx_start:pfx_end] if pfx_end > 0 else []
        bos: List[int] = [self.bos_id] if (self.bos_id is not None and not prefix and not lead) else []
        seq = bos + prefix + feed

        ids = torch.tensor([seq], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(ids).logits

        # Index of the first SCORED token inside ``ids``; its predictor sits one
        # position earlier, so the scope starts at ``offset - 1``.
        offset = len(bos) + len(prefix) + len(lead)
        seq_len = int(ids.shape[1])
        n_avail = max(0, seq_len - offset)
        if n_avail < n_target:
            raise RuntimeError(
                f"window needs {n_target} predictions but the model supplied "
                f"{n_avail} (seq={seq_len} offset={offset} ctx={self.context_window} "
                f"prefix={len(prefix)} lead={len(lead)} bos={len(bos)})"
            )
        scope = logits[:, offset - 1 : offset - 1 + n_target, :]
        targets = ids[:, offset : offset + n_target]
        if int(scope.shape[1]) != n_target or int(targets.shape[1]) != n_target:
            raise RuntimeError(
                f"scope/target mismatch: {int(scope.shape[1])} / "
                f"{int(targets.shape[1])} vs {n_target}"
            )
        # Hard alignment check against the document.  This is what catches a
        # one-position lag, which is otherwise invisible: the code runs, the
        # numbers look plausible, and every byte is scored against the wrong
        # distribution.
        for i in range(n_target):
            expected = int(self._all_tokens[doc_start + i])
            if int(targets[0, i]) != expected:
                raise RuntimeError(
                    f"target misalignment at {i}: model target {int(targets[0, i])} "
                    f"!= document token {expected} "
                    f"(doc_start={doc_start} n_target={n_target} offset={offset} "
                    f"seq_len={seq_len} lead={len(lead)} successor={len(successor)})"
                )
        if os.environ.get("UIC_DEBUG_SHAPES"):
            print(
                f"[shapes] ids={tuple(ids.shape)} logits={tuple(logits.shape)} "
                f"offset={offset} target={n_target} window={window_len} "
                f"prefix={len(prefix)} lead={len(lead)} successor={len(successor)} "
                f"bos={len(bos)}"
            )
        p_true = _gather_true_probs(scope.to(torch.float32), targets)
        del logits, scope, targets
        probs = _to_numpy(p_true).reshape(-1)

        tail = _SPARSE_TAIL_MASS
        for i in range(n_target):
            tok = int(tokens[i])
            p = float(probs[i])
            cdf = sparse_true_symbol_cdf(p, self.n_symbols, tok, tail_mass=tail)
            yield Step(symbol=tok, p_true=p, cdf=cdf)

    def release(self):
        import torch

        self.model = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
