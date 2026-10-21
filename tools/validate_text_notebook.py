"""Validate the text Colab notebook's logic locally, before it is run.

Four things can silently ruin that notebook, and three of them can be tested here:

 1. **The embedded payload** must decode to exactly the corpus the 9.0410 % baseline was
    measured on (312 053 bytes, 57 547 tokens, 15/9 windows). Tested.
 2. **The scorer** must agree with the model's own loss. A mis-shifted scorer is what
    invalidated an earlier audio finding in this project, so it is checked on a tiny
    Llama of the same architecture rather than assumed. Tested.
 3. **`disable_adapter()`** must actually change the model's output. If it silently did
    nothing, the base arm and the LoRA arm would be identical and the 2x2 would be
    meaningless. Tested on a tiny model with a tiny adapter.
 4. The full 8B run -- that is what Colab is for, and it is not attempted here.

Nothing in this script touches the GPU or the 8B checkpoint.
"""

from __future__ import annotations

import base64
import importlib.util
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


def load_payload() -> str:
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "PAYLOAD = " in src:
            i = src.find('PAYLOAD = """') + len('PAYLOAD = """')
            j = src.find('"""', i)
            return src[i:j]
    raise SystemExit("payload cell not found")


def check_payload() -> None:
    print("=== 1. embedded payload ===")
    payload = load_payload()
    print(f"  base64 length: {len(payload):,}")

    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as zf:
        names = sorted(zf.namelist())
        print(f"  members: {names}")
        assert names == sorted(FILES), f"unexpected members {names}"

        total_bytes = 0
        for name in FILES:
            blob = zf.read(name)
            disk = open(os.path.join(MEDAL_DIR, name), "rb").read()
            assert blob == disk, f"{name} differs from the on-disk file"
            total_bytes += len(blob)
        print(f"  identical to the on-disk files, total {total_bytes:,} bytes")

    # Reproduce the reference loader's concatenation and the token counts.
    import pandas as pd
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B"),
        local_files_only=True)

    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as zf:
        texts = {}
        for name in FILES:
            df = pd.read_csv(io.BytesIO(zf.read(name)))
            s = ""
            for _, row in df.iterrows():
                s += row["TEXT"]
            texts[name] = s

    raw = sum(len(t.encode()) for t in texts.values())
    n_tok = 0
    w4 = w8 = 0
    print(f"  {'file':<22}{'tokens':>8}{'win@4096':>10}{'win@8192':>10}")
    for name, s in texts.items():
        ids = tok(s, add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        w4 += math.ceil(len(ids) / 4096)
        w8 += math.ceil(len(ids) / 8192)
        print(f"  {name:<22}{len(ids):>8,}{math.ceil(len(ids)/4096):>10}"
              f"{math.ceil(len(ids)/8192):>10}")

    print(f"  raw bytes {raw:,} | tokens {n_tok:,} | windows {w4}/{w8}")
    assert raw == 312053, f"raw bytes {raw} != 312053"
    assert n_tok == 57547, f"tokens {n_tok} != 57547"
    assert (w4, w8) == (15, 9), f"windows {(w4, w8)} != (15, 9)"
    print("  PASS -- assertions in the notebook will hold")


def check_scorer() -> None:
    print("\n=== 2. scorer agrees with output.loss (tiny Llama) ===")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      max_position_embeddings=256)
    model = LlamaForCausalLM(cfg).eval()

    def window_bits(w, slice_len=64):
        t = torch.tensor([w], dtype=torch.long)
        with torch.no_grad():
            logits = model(t).logits
        L = t.shape[1]
        total = 0.0
        for a in range(0, L - 1, slice_len):
            b = min(a + slice_len, L - 1)
            seg = logits[:, a:b, :].float()
            lp = torch.log_softmax(seg, dim=-1)
            tgt = t[:, a + 1:b + 1]
            total += float(-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())
        return total

    torch.manual_seed(1)
    w = torch.randint(0, 256, (200,)).tolist()
    got = window_bits(w)
    t = torch.tensor([w], dtype=torch.long)
    with torch.no_grad():
        out = model(t, labels=t)
    ref = float(out.loss) * (len(w) - 1)
    rel = abs(got - ref) / got
    print(f"  window {len(w)} tokens")
    print(f"  gather   = {got:.6f} bits")
    print(f"  loss*(L-1) = {ref:.6f} bits")
    print(f"  rel diff = {rel:.3e}")
    assert rel < 1e-4, "scorer disagrees with output.loss"
    print("  PASS -- the notebook's assertion threshold (1e-4) is met")

    # And confirm the shift direction matters: scoring token i with row i must differ.
    with torch.no_grad():
        logits = model(t).logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    tgt = t[0]
    same = float(-lp[0].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())
    print(f"\n  for contrast, row i scoring token i (the wrong convention): "
          f"{same:.4f} bits")
    print(f"  difference from the correct convention: {abs(same-got):.4f} bits "
          f"({100*abs(same-got)/got:.1f} %)")
    assert abs(same - got) > 1e-3, "the two conventions coincide; the probe is blind"
    print("  PASS -- the probe can distinguish the two conventions")


def check_disable_adapter() -> None:
    print("\n=== 3. disable_adapter() actually changes the output ===")
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      max_position_embeddings=256)
    base = LlamaForCausalLM(cfg).eval()

    # Give the adapter non-zero effect: init_lora_weights=False leaves the B matrix
    # random instead of zero, so the delta is actually visible on a fresh adapter.
    # (This is a LoraConfig field, not a get_peft_model kwarg.)
    lcfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", init_lora_weights=False,
                      target_modules=["k_proj", "v_proj", "q_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lcfg).eval()

    n = sum(v.numel() for k, v in model.named_parameters() if "lora_" in k)
    print(f"  adapter parameters: {n:,}")
    assert n > 0

    w = torch.randint(0, 256, (64,)).tolist()
    t = torch.tensor([w], dtype=torch.long)

    def bits(m):
        with torch.no_grad():
            lg = m(t).logits.float()
        lp = torch.log_softmax(lg[:, :-1, :], dim=-1)
        tg = t[:, 1:]
        return float(-lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1).sum())

    with_lora = bits(model)
    with model.disable_adapter():
        without = bits(model)
    print(f"  with adapter    : {with_lora:.6f} bits")
    print(f"  adapter disabled: {without:.6f} bits")
    delta = abs(with_lora - without)
    print(f"  difference      : {delta:.6f} bits")
    assert delta > 1e-4, ("disable_adapter() had no effect -- the base arm would be "
                          "identical to the LoRA arm and the 2x2 would be vacuous")
    print("  PASS -- the two arms are distinguishable")


def main() -> int:
    if not os.path.exists(NB):
        print(f"notebook missing: {NB}")
        return 1
    check_payload()
    check_scorer()
    check_disable_adapter()
    print("\nALL LOCAL CHECKS PASSED")
    print("The 8B run itself is the Colab step; nothing here substitutes for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
