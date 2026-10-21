"""Record the paired text result: the confound is resolved, and a new anomaly appears.

PART 1 -- the question the experiment was built for is SETTLED.

The report could not separate the LoRA penalty (+0.312 pp at context 4096) from the
context effect (~0.30 pp per SI Table 9), because they were the same size. Ten cells on
two window axes now settle it:

  adapter penalty is POSITIVE at all five contexts: +0.3987 .. +0.4728 pp
  context effect (4096 -> 8192) is at most 0.2001 pp, and only 0.0634 pp on the token axis

So the context effect is smaller than the penalty and the penalty is present regardless of
context. Context length is not the explanation.

PART 2 -- a new and larger anomaly, which blocks the absolute numbers.

Every cell is 2.6-3.0 pp BETTER than three independent references:

  notebook base A@8192 chars  6.3321 %   paper 9.06 %   authors' harness 8.9253 %
  notebook base B@4096 tokens 6.1657 %   our port 9.0410 %

and the four setup checks all pass (adapter 20,971,520 params; corpus 312,053 bytes;
windows 78/51/39 and 15/9; bf16 with 20.44 GiB peak VRAM, so genuinely an 8B model).
Bits per token are 2.65-3.01 here against ~3.9 everywhere else.

This is recorded as an open contradiction rather than resolved by preference. The penalty
finding survives it because a constant offset does not change the sign of a difference.
"""

from __future__ import annotations

import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

AUDIT_SECTION = """### 5.7 The paired run at ten cells: confound resolved, absolute level not [M]

The ten-cell run (`results/text_paired.json`, Colab A100,
`colab/Text_Paired_Context_Colab.ipynb`) measured both arms on both window axes.

**Setup checks, all passing** -- these are what make the run trustworthy as a *difference*:

| check | value | expected |
|---|---|---|
| adapter parameters | 20 971 520 | 20 971 520 |
| corpus bytes | 312 053 | 312 053 |
| character windows | 78 / 51 / 39 | 78 / 51 / 39 |
| token windows | 15 / 9 | 15 / 9 |
| dtype | torch.bfloat16 | bf16 |
| peak VRAM | 20.44 GiB | consistent with an 8 B model in bf16 |

**Rates (%)**

| arm | A: 4096 c | A: 6144 c | A: 8192 c | B: 4096 t | B: 8192 t |
|---|---|---|---|---|---|
| base | 6.5322 | 6.3927 | 6.3321 | 6.1657 | 6.1023 |
| +LoRA | 6.9319 | 6.7958 | 6.7308 | 6.6107 | 6.5751 |

**The adapter penalty**

| context | penalty (pp) |
|---|---|
| A 4096 chars | **+0.3997** |
| A 6144 chars | **+0.4031** |
| A 8192 chars | **+0.3987** |
| B 4096 tokens | **+0.4450** |
| B 8192 tokens | **+0.4728** |

Positive at **every** context, range only **0.0741 pp**. The context effect over the same
span is at most **0.2001 pp** (character axis) and **0.0634 pp** (token axis).

**So the confound is resolved.** Context length is *smaller* than the penalty and the
penalty is present regardless of it. The earlier caveat -- "the +0.312 pp and the ~0.30 pp
context effect cannot be separated" -- is now a measurement rather than an open question:
the adapter costs **0.40-0.47 pp** here, and context can account for at most **0.20 pp**.
The direction also agrees with the authors' own harness, where the penalty at 8192
characters was **+0.5901 pp** -- larger than, not smaller than, our 4096-token figure.

**An anomaly that blocks the absolute numbers.** Every cell is 2.6-3.0 pp *better* than
three independent references:

| measurement | rate | instrument |
|---|---|---|
| notebook, base, A@8192 chars | **6.3321 %** | this run, A100, bf16 |
| notebook, base, B@4096 tokens | **6.1657 %** | this run |
| paper, SI Table 9 @8192 | 9.06 % | published |
| authors' unmodified harness @8192 chars | **8.9253 %** | reference script |
| our port @4096 tokens | **9.0410 %** | `uic` |

In bits per token: **2.65-3.01** here against **~3.9** for all three others (perplexity
6.3-8.0 against ~15). The setup checks above rule out the mundane explanations -- wrong
corpus, wrong tokenizer, wrong window counts, a small model, an unloaded adapter, a
mis-shifted scorer (the notebook asserts the scorer against `output.loss` to 1e-4 and it
matched to 1e-8).

Two candidate explanations remain and this audit does not pick between them:

1. **The other three share a degradation.** The paper states the base model is loaded in
   **8-bit** for text. If that path is worse than nominal, or if the reference script's
   CPU/RAM path behaves differently, then paper, harness and our port would all be
   pessimistic together -- which is exactly the pattern in the table.
2. **This run is optimistic by some information leak not covered by the checks.** The
   scorer is verified, but a leak could sit elsewhere, e.g. in how the window tensor is
   presented to the model.

**The discriminating test** is cheap and specific: run the authors' unmodified harness on
the same A100, in the same bf16, on the same HF weights as this notebook. If it then
reports ~6.3 %, source 1 is confirmed and the notebook is right; if it still reports
~8.9 %, source 2 is confirmed and this run has a defect. Until that is done, **no absolute
text rate from this audit should be quoted**, and the report says so.

The penalty finding is unaffected: a constant offset does not change the sign or the size
of a difference, which is why it can be stated firmly while the level cannot.
"""


REPORT_SECTION = """### 5.7 十格配对实验：混淆项解决了，绝对量级还没解决

`results/text_paired.json`（Colab A100）在两条窗口轴上各测了双臂。

**装置校验（全部通过）**：适配器 **20 971 520** 参数、语料 **312 053** 字节、
字符窗口 **78/51/39**、token 窗口 **15/9**、dtype **bf16**、峰值显存 **20.44 GiB**（与 8B 模型相符）。

**率 (%)**

| 臂 | A: 4096 字符 | A: 6144 字符 | A: 8192 字符 | B: 4096 token | B: 8192 token |
|---|---|---|---|---|---|
| base | 6.5322 | 6.3927 | 6.3321 | 6.1657 | 6.1023 |
| +LoRA | 6.9319 | 6.7958 | 6.7308 | 6.6107 | 6.5751 |

**适配器惩罚**

| 上下文 | 惩罚 (pp) |
|---|---|
| A 4096 字符 | **+0.3997** |
| A 6144 字符 | **+0.4031** |
| A 8192 字符 | **+0.3987** |
| B 4096 token | **+0.4450** |
| B 8192 token | **+0.4728** |

**在每一个上下文下都为正，极差只有 0.0741 pp。** 同一跨度上的上下文效应最多
**0.2001 pp**（字符轴）与 **0.0634 pp**（token 轴）。

**所以混淆项解决了。** 上下文效应比惩罚**更小**，且惩罚在任何上下文下都存在。
原先"无法分离 +0.312 pp 与约 0.30 pp 上下文效应"的限定，
现在变成了一个测量结果：适配器在这里值 **0.40–0.47 pp**，上下文最多贡献 **0.20 pp**。
方向也与作者原始 harness 一致 —— 它在 8192 字符下的惩罚是 **+0.5901 pp**，比我们 4096 token 的更大而非更小。

**但出现了一个阻断绝对数字的异常。** 每一格都比三个独立来源**好 2.6–3.0 pp**：

| 测量 | 率 | 装置 |
|---|---|---|
| 本 notebook，base，A@8192 字符 | **6.3321 %** | 本次，A100，bf16 |
| 本 notebook，base，B@4096 token | **6.1657 %** | 本次 |
| 论文 SI Table 9 @8192 | 9.06 % | 已发表 |
| 作者原始 harness @8192 字符 | **8.9253 %** | 参考脚本 |
| 我们端口 @4096 token | **9.0410 %** | `uic` |

折成 bits/token：本次 **2.65–3.01**，其余三者都是 **~3.9**（困惑度 6.3–8.0 对 ~15）。
上面的校验排除了常见解释 —— 语料错、分词器错、窗口数错、模型被换小、适配器没加载、
评分器位移错（notebook 用 `output.loss` 断言，实测吻合到 1e-8）。

还剩两种可能，本报告不选边：

1. **其余三者共享同一处退化。** 论文明确说文本的基座模型按 **8-bit** 加载。若该路径差于名义值，
   或参考脚本的 CPU/RAM 路径行为不同，则论文、harness、我们端口会**一起偏悲观** —— 正是表里的形态。
2. **本次运行因某处未被校验覆盖的信息泄漏而偏乐观。** 评分器已验证，但泄漏可能在别处，
   例如窗口张量喂给模型的方式。

**判别实验很便宜且具体**：在**同一台 A100、同样 bf16、同样的 HF 权重**上跑作者原始 harness。
若它给出约 6.3 %，则可能 1 成立、本 notebook 正确；若仍是约 8.9 %，则可能 2 成立、本次运行有缺陷。
**在此之前，本报告不引用任何文本绝对码率**，正文已如此声明。

惩罚的结论不受影响：常数偏移不改变差值的符号与大小，所以**惩罚能确定地说，量级不能**。
"""


def main() -> int:
    d = json.load(open(os.path.join(HERE, "results", "text_paired.json"),
                       encoding="utf-8"))
    pens = d["penalties"]
    print(f"  penalties: {min(pens.values()):+.4f} .. {max(pens.values()):+.4f} pp, "
          f"all positive = {all(v > 0 for v in pens.values())}")

    a = io.open(AUDIT, encoding="utf-8").read()
    if "5.7 The paired run at ten cells" not in a:
        anchor = "### 5.6 The experiment that would settle it"
        if anchor in a:
            a = a.replace(anchor, AUDIT_SECTION + "\n" + anchor, 1)
        else:
            a = a.rstrip() + "\n\n" + AUDIT_SECTION + "\n"
        io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(a)
        print("  ok  audit: added 5.7")
    else:
        print("  --  audit already records 5.7")

    r = io.open(REPORT, encoding="utf-8").read()
    if "5.7 十格配对实验" not in r:
        anchor = "### 5.5 两个"
        if anchor in r:
            r = r.replace(anchor, REPORT_SECTION + "\n" + anchor, 1)
        else:
            r = r.rstrip() + "\n\n" + REPORT_SECTION + "\n"
        io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
        print("  ok  report: added 5.7")
    else:
        print("  --  report already records 5.7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
