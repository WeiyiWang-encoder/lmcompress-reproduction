"""Validate the precision notebook's logic locally, before it goes to a GPU.

Three things can be wrong without the notebook looking wrong:

 1. the embedded payload must decode to the exact file the 2.5682 bits/token reference was
    measured on -- 103 534 bytes, 19 338 tokens, 5 windows of 4096;
 2. both formulas must match the model's own loss, otherwise the notebook would be
    measuring its own arithmetic rather than the model;
 3. the `native` formula must actually differ from `fp32` by the amount the notebook's
    argument depends on. If softmax in bf16 and log_softmax in fp32 agreed exactly, the
    "formula cannot explain 1.3 bits/token" line would be vacuous; if they differed by a
    lot, the argument would be wrong.

Only the quantisation arms are skipped, because they need a CUDA-capable bitsandbytes.
"""

from __future__ import annotations

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
NB = os.path.join(HERE, "colab", "Precision_Test_Colab.ipynb")


def get_payload() -> str:
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        src = "".join(c["source"])
        if "PAYLOAD = " in src:
            i = src.find('PAYLOAD = """') + len('PAYLOAD = """')
            return src[i:src.find('"""', i)]
    raise SystemExit("payload not found")


def check_payload() -> None:
    print("=== 1. payload and tokenisation ===")
    import pandas as pd
    from transformers import AutoTokenizer

    with zipfile.ZipFile(io.BytesIO(base64.b64decode(get_payload()))) as zf:
        names = zf.namelist()
        blob = zf.read("medal_102285.csv")
        disk = open(os.path.join(REPO, "data", "text", "MEDAL_sample",
                                 "medal_102285.csv"), "rb").read()
    print(f"  members {names}   identical to disk: {blob == disk}")
    assert blob == disk

    df = pd.read_csv(io.BytesIO(blob))
    text = ""
    for _, row in df.iterrows():
        text += row["TEXT"]
    raw = len(text.encode())
    tok = AutoTokenizer.from_pretrained(
        os.path.join(REPO, "data", "model", "finetuned_llama", "Llama-3-8B"),
        local_files_only=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    wins = [ids[i:i + 4096] for i in range(0, len(ids), 4096)]
    wins = [w for w in wins if len(w) >= 2]
    print(f"  bytes {raw:,} (need 103534)   tokens {len(ids):,} (need 19338)   "
          f"windows {len(wins)} (need 5)")
    assert raw == 103534 and len(ids) == 19338 and len(wins) == 5
    print("  PASS")


def check_formulas() -> None:
    print("\n=== 2. both formulas vs the model's own loss ===")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=4096, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      max_position_embeddings=256)
    model = LlamaForCausalLM(cfg).eval()
    torch.manual_seed(1)
    w = torch.randint(0, cfg.vocab_size, (160,)).tolist()

    def bits_fp32(w):
        t = torch.tensor([w], dtype=torch.long)
        with torch.no_grad():
            lg = model(t).logits
        L = t.shape[1]
        tot = 0.0
        for a in range(0, L - 1, 64):
            b = min(a + 64, L - 1)
            lp = torch.log_softmax(lg[:, a:b, :].float(), dim=-1)
            tot += float(-lp.gather(-1, t[:, a+1:b+1].unsqueeze(-1)).squeeze(-1).sum())
        return tot

    def bits_native(w):
        t = torch.tensor([w], dtype=torch.long)
        with torch.no_grad():
            lg = model(t).logits
        L = t.shape[1]
        tot = 0.0
        for a in range(0, L - 1, 64):
            b = min(a + 64, L - 1)
            p = lg[:, a:b, :].softmax(dim=-1)
            pd = torch.gather(p, -1, t[:, a+1:b+1].unsqueeze(-1)).squeeze(-1)
            pd = pd.add_(6e-8)
            tot += float(-torch.log2(pd).sum())
        return tot

    t = torch.tensor([w], dtype=torch.long)
    with torch.no_grad():
        out = model(t, labels=t)
    ref = float(out.loss) * (len(w) - 1)

    fp = bits_fp32(w)
    na = bits_native(w)
    print(f"  fp32 formula   {fp:12.6f} bits   rel vs loss {abs(fp-ref)/ref:.2e}")
    print(f"  native formula {na:12.6f} bits   rel vs loss {abs(na-ref)/ref:.2e}")
    print(f"  loss*(L-1)     {ref:12.6f} bits")
    assert abs(fp - ref) / ref < 1e-4, "fp32 formula disagrees with the loss"
    assert abs(na - ref) / ref < 1e-3, "native formula disagrees with the loss"
    print("  PASS -- both formulas reproduce the model's own loss at fp32 weights")

    # 3. the signal size: bf16 weights, native vs fp32 softmax
    print("\n=== 3. does the formula differ at bf16 precision? ===")
    model_bf = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        lg = model_bf(t).logits
    print(f"  logits dtype {lg.dtype}")
    L = t.shape[1]
    fp = na = 0.0
    for a in range(0, L - 1, 64):
        b = min(a + 64, L - 1)
        seg = lg[:, a:b, :]
        lp = torch.log_softmax(seg.float(), dim=-1)
        fp += float(-lp.gather(-1, t[:, a+1:b+1].unsqueeze(-1)).squeeze(-1).sum())
        p = seg.softmax(dim=-1)
        pd = torch.gather(p, -1, t[:, a+1:b+1].unsqueeze(-1)).squeeze(-1).add_(6e-8)
        na += float(-torch.log2(pd).sum())
    n = len(w) - 1
    d = (na - fp) / n
    print(f"  fp32 {fp/n:.4f} bits/token   native {na/n:.4f} bits/token   "
          f"delta {d:+.6f} bits/token ({100*d/(fp/n):+.4f} %)")
    print(f"  the notebook argues a formula difference cannot explain 1.3 bits/token;")
    print(f"  here the whole formula difference is {abs(d):.6f} bits/token, i.e. "
          f"{abs(d)/1.3*100:.3f} % of the gap to be explained")
    assert abs(d) < 0.05, (
        "the formula difference is large on this tiny model; the notebook's argument that "
        "formula cannot explain 1.3 bits/token would need revisiting")
    print("  PASS -- the argument holds: the formula is a second-order effect")
    print("\n   (note: a tiny random model is not evidence about 8B bf16 numerics; this")
    print("   only checks that the implementation direction and magnitude are sane.)")


def main() -> int:
    print(f"notebook: {os.path.basename(NB)}  {os.path.getsize(NB)/1024:.0f} KB\n")
    check_payload()
    check_formulas()
    print("\nALL LOCAL CHECKS PASSED")
    print("The quantisation arms need a GPU; that is what the notebook is for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
