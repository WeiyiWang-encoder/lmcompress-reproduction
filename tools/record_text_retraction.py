"""Retract the text reproduction figure: our port's causal code length is ~48 % too large.

Measurement (tools/isolate_harness.py, one 4096-token window of medal_102285.csv, the
capsule's own Llama-3-8B on CPU in bf16):

    plain HuggingFace forward, causal shift    10 516.775 bits  =  2.5682 bits/token
    the same figure from output.loss*(L-1)     10 516.776 bits  (rel diff 5.6e-08)

The scorer is therefore correct by construction: it equals the model's own loss. And the
capsule's weights are byte-identical to the public checkpoint (two tensors checked,
including the 1 051 MB embedding), so this is not a weights question.

Our port's recorded result for the same file is 73 410.306 bits over 19 333 scored tokens
= **3.797 bits/token**. On identical weights, the port reports 48 % more bits than the
model's own loss implies (3.797 / 2.5682 = 1.478).

Consequences, in order of severity:

1. Our text rate of 9.0410 % is wrong. A correct causal code length on this corpus is
   ~6.0 %, which is what the A100 paired run reported (6.1657 % at 4096 tokens).
2. The claim that text base "reproduces the paper to 0.04 pp" is withdrawn: it was two
   different inflations landing near the same place.
3. The paper's 9.06-9.36 % cannot be a plain causal code length on this model either,
   since the reference's own `forward_next` is correctly shifted (`logits[:, :-1]` against
   `input_ids[:, 1:]`) and would give ~6 % on this corpus.
4. The adapter penalty survives qualitatively -- the paired run finds it positive at all
   five contexts -- but the specific figure +0.312 pp is replaced by +0.40 to +0.47 pp.
"""

from __future__ import annotations

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")
REPORT = os.path.join(HERE, "docs", "REPORT_CN.md")

AUDIT_SECTION = """### 5.8 RETRACTED: our text code length is ~48 % too large [M]

**What was measured.** `tools/isolate_harness.py`, one 4096-token window of
`medal_102285.csv`, on the capsule's own Llama-3-8B, CPU, bf16:

| quantity | bits | bits/token |
|---|---|---|
| plain HuggingFace forward, causal shift | **10 516.775** | **2.5682** |
| the same via `output.loss * (L-1)` | 10 516.776 | 2.5682 |
| relative difference | **5.6e-08** | -- |
| our port's recorded result, same file | **73 410.306** (19 333 scored) | **3.797** |

The first two agreeing to 6e-08 makes the scorer correct *by construction* -- it equals the
model's own loss. And the weights are not in question: the capsule checkpoint is
**byte-identical** to the public one (`tools/compare_llama_weights.py` checked
`model.norm.weight` and the 1 051 MB `model.embed_tokens.weight`; both hashes matched).

**So on identical weights our port reports 48 % more bits than the model's loss implies**
(3.797 / 2.5682 = 1.478).

**What is withdrawn.**

1. **The text base rate of 9.0410 % is wrong.** A correct causal code length on this
   corpus is about **6.0 %**, which is what the paired A100 run measured (6.1657 % at
   4096 tokens, 6.3321 % at 8192 characters).
2. **The claim that text base "reproduces the paper to 0.04 pp" is withdrawn.** It was
   two different inflations landing near the same place, which is the least informative
   kind of agreement.
3. **The specific adapter penalty of +0.312 pp is replaced.** The paired run measures
   **+0.3987 to +0.4728 pp**, positive at all five contexts, on the correct harness. The
   qualitative finding survives; the number does not.
4. **The paper's 9.06-9.36 % cannot be a plain causal code length on this model either.**
   The reference's own `forward_next` is correctly shifted (`shifted_logits =
   logits[:, :-1, :]` against `shifted_input_ids = input_ids[:, 1:]`), so a run of that
   code on this corpus should also land near 6 %, not 9 %. Either the paper's numbers come
   from a different corpus or setting, or the 8.9253 % previously attributed to the
   authors' harness is not measuring what we assumed. This audit does not guess which.

**The defect is bounded but not yet located.** The ratio 1.478 sits close to
`log2(e) = 1.4427` but is not equal to it, so this is **not** a simple natural-log/log2
confusion; a 2.4 % gap remains that a log-base error would not produce. The port's
`iter_steps` was driven directly for this test and raised `IndexError: list index out of
range` at `self._all_tokens[doc_start + i]` because `_all_tokens` is populated by the
runner, so the port arm of the isolation probe did not complete -- the comparison above
rests on the port's *recorded* aggregate, not on a fresh port run. Locating the defect
needs one more pass over `LlamaTokenPredictor.iter_steps`.

**How this survived until now.** The port agreed with the authors' harness to 0.0622 pp,
and both agreed with the paper. Three-way agreement was read as validation. It was in fact
one shared convention being counted three times -- the same failure mode this audit already
named once in section 4.1 ("we searched for an artefact, searched in the wrong place, and
reported the absence of evidence as evidence"). The A100 run is the first measurement in
this project that was validated against the model's own loss rather than against another
implementation of the same idea.
"""


REPORT_SECTION = """### 5.8 【撤回】我们的文本码长偏大约 48%

**实测**（`tools/isolate_harness.py`，`medal_102285.csv` 的第一个 4096-token 窗口，
用**胶囊自己的** Llama-3-8B，CPU，bf16）：

| 量 | bits | bits/token |
|---|---|---|
| 普通 HuggingFace 前向，因果位移 | **10 516.775** | **2.5682** |
| 同一数值经 `output.loss * (L-1)` | 10 516.776 | 2.5682 |
| 相对差 | **5.6e-08** | — |
| 我们端口记录的同文件结果 | **73 410.306**（19 333 计分 token） | **3.797** |

前两者吻合到 6e-08，意味着该评分器**构造上就是对的** —— 它等于模型自己的 loss。
而权重也不成问题：胶囊权重与公开权重**逐字节相同**
（`model.norm.weight` 与 1 051 MB 的 `model.embed_tokens.weight` 两个哈希都一致）。

**所以，在相同权重上，我们端口比模型 loss 所隐含的码长多报了 48%**（3.797 / 2.5682 = 1.478）。

**撤回什么**

1. **文本 base 的 9.0410 % 是错的。** 本语料上正确的因果码长约 **6.0 %**，
   这正是配对 A100 实验测到的（4096 token 下 6.1657 %，8192 字符下 6.3321 %）。
2. **"文本 base 与论文吻合到 0.04 pp"这句话撤回。** 那是两个不同的偏大值恰好落在相近位置，
   是最没有信息量的一种"吻合"。
3. **+0.312 pp 这个具体惩罚数值被替换。** 配对实验在正确装置上测得
   **+0.3987 至 +0.4728 pp**，五个上下文全为正。定性结论保留，数字不保留。
4. **论文的 9.06–9.36 % 也不可能是该模型上的普通因果码长。** 参考实现自己的
   `forward_next` 位移是正确的（`logits[:, :-1, :]` 对 `input_ids[:, 1:]`），
   所以它在本语料上跑应当也落在 6 % 附近而不是 9 %。
   要么论文的数字来自不同语料或设置，要么此前归给"作者原始 harness"的 8.9253 % 并非我们以为的那个量。
   **本报告不猜是哪一种。**

**缺陷已定位范围但尚未定位代码。** 比值 1.478 接近 `log2(e) = 1.4427` 但并不相等，
所以**不是**简单的自然对数/log2 混用 —— 还差 2.4%，对数底错误不会产生这个残差。
本次尝试直接驱动端口的 `iter_steps` 时，它在 `self._all_tokens[doc_start + i]` 处
抛出 `IndexError`（该列表由 runner 填充），所以探针的端口臂没有跑完；
上面的比较依据的是端口的**已记录汇总值**，不是一次新的端口运行。
定位缺陷还需要再过一遍 `LlamaTokenPredictor.iter_steps`。

**这个错误为什么能活到现在**：端口与作者 harness 吻合到 0.0622 pp，两者又与论文吻合。
三方一致被当成了验证。实际上那只是**同一个约定被数了三遍** ——
正是本报告在 §4.1 已经命名过一次的失效模式。A100 那次运行是本项目第一次
**用模型自己的 loss** 而不是用"另一个同想法的实现"来做校验的测量。
"""


def main() -> int:
    a = io.open(AUDIT, encoding="utf-8").read()
    if "5.8 RETRACTED" not in a:
        anchor = "### 5.7 The training-amount ablation"
        if anchor in a:
            a = a.replace(anchor, AUDIT_SECTION + "\n" + anchor, 1)
            io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(a)
            print("  ok  audit: added 5.8 retraction")
        else:
            print("  !! audit anchor missing")
    else:
        print("  --  audit already has 5.8")

    r = io.open(REPORT, encoding="utf-8").read()
    if "5.8 【撤回】" not in r:
        anchor = "### 5.7 十格配对实验"
        if anchor in r:
            r = r.replace(anchor, REPORT_SECTION + "\n" + anchor, 1)
            io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
            print("  ok  report: added 5.8 retraction")
        else:
            print("  !! report anchor missing")
    else:
        print("  --  report already has 5.8")

    # status table: the text rows must no longer claim reproduction
    r = io.open(REPORT, encoding="utf-8").read()
    old = ("| 文本 | Llama-3-8B（base） | 是 | **9.0410%** vs 论文 9.08%，差 **−0.04 pp**")
    if old in r:
        new = ("| 文本 | Llama-3-8B（base） | **已撤回（§5.8）** | 原 9.0410 %，"
               "比模型自身 loss 隐含的码长**偏大 48 %**；正确值约 **6.0 %**"
               "（配对 A100 实测 6.1657 %）。与论文的\"吻合\"是两处偏大值巧合")
        r = r.replace(old, new, 1)
        io.open(REPORT, "w", encoding="utf-8", newline="\n").write(r)
        print("  ok  report: text status row updated")
    else:
        print("  --  text status row pattern not found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
